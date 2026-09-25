import io
import logging
import tempfile
import unittest
import warnings
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lma_scripts import lma_flash, lma_plot
from lma_scripts.log_output import LibraryDiagnosticFilter, readable_library_output


class StageLoggingTests(unittest.TestCase):
    def setUp(self):
        self.messages = io.StringIO()
        self.logger = logging.getLogger("lma_scripts.grid")
        self.plot_logger = logging.getLogger("lma_scripts.plot")
        self.handler = logging.StreamHandler(self.messages)
        self.logger.addHandler(self.handler)
        self.plot_logger.addHandler(self.handler)
        self.previous_level = self.logger.level
        self.previous_plot_level = self.plot_logger.level
        self.logger.setLevel(logging.INFO)
        self.plot_logger.setLevel(logging.INFO)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.plot_logger.removeHandler(self.handler)
        self.logger.setLevel(self.previous_level)
        self.plot_logger.setLevel(self.previous_plot_level)

    def test_library_output_keeps_progress_but_hides_numeric_diagnostics(self):
        with readable_library_output(self.logger):
            print("(14,) (11,)")
            print("100")
            print("collection times = [datetime.datetime(2025, 6, 15)]")
            print("sorting 2539 total points")
            print("total flashes: 2497")

        self.assertEqual(
            self.messages.getvalue(),
            "sorting 2539 total points\ntotal flashes: 2497\n",
        )

        library_filter = LibraryDiagnosticFilter()
        self.assertFalse(library_filter.filter(logging.LogRecord("lib", 20, "", 0, "100", (), None)))
        self.assertTrue(
            library_filter.filter(logging.LogRecord("lib", 20, "", 0, "total flashes: 2497", (), None))
        )

    def test_grid_logs_frame_and_completion(self):
        with (
            patch.object(lma_flash, "dlonlat_at_grid_center", return_value=(1, 1, (0, 1), (0, 1))),
            patch.object(lma_flash, "grid_h5flashfiles") as make_grid,
            patch.object(
                lma_flash.glob,
                "glob",
                return_value=[
                    "/tmp/grids/MALMA_20250615_200000_600_10src_0.01deg-dx_source.nc",
                    "/tmp/grids/MALMA_20250615_200000_600_10src_0.01deg-dx_source_3d.nc",
                ],
            ),
            patch.object(lma_flash.os.path, "isfile", return_value=True),
            patch.object(lma_flash, "add_time_altitude_counts"),
        ):
            def produce_files(*args, **kwargs):
                print("(12,) (6,)")
                self.assertIs(kwargs["output_writer"], lma_flash.write_cf_netcdf_latlon)
                self.assertIs(kwargs["output_writer_3d"], lma_flash.write_cf_netcdf_3d_latlon)
                self.assertNotIn("dz", kwargs)

            make_grid.side_effect = produce_files
            lma_flash.grid(
                ["MALMA_250615_200000_0600.dat.flash.h5"],
                "/tmp/grids",
                frame_interval=600,
            )

        output = self.messages.getvalue()
        self.assertIn("Creating 1 time frame(s) for MALMA", output)
        self.assertIn("2025-06-15 20:00:00 to 2025-06-15 20:10:00 UTC", output)
        self.assertIn("Completed NetCDF grids in /tmp/grids", output)
        self.assertIn("2 NetCDF file(s) present", output)
        self.assertIn("NetCDF output: /tmp/grids/MALMA_20250615_200000_600_10src_0.01deg-dx_source_3d.nc", output)
        self.assertNotIn("(12,) (6,)", output)

    def test_grid_passes_scalar_coordinates_to_lmatools(self):
        with (
            warnings.catch_warnings(),
            patch.object(lma_flash, "grid_h5flashfiles") as make_grid,
            patch.object(lma_flash.glob, "glob", return_value=[]),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lma_flash.grid(
                ["MALMA_250615_200000_0600.dat.flash.h5"],
                "/tmp/grids",
            )

        kwargs = make_grid.call_args.kwargs
        for axis, bounds in (("dx", "x_bnd"), ("dy", "y_bnd")):
            self.assertIsInstance(kwargs[axis], float)
            self.assertTrue(all(isinstance(value, float) for value in kwargs[bounds]))
            self.assertGreater(
                len(np.arange(kwargs[bounds][0], kwargs[bounds][1] + kwargs[axis], kwargs[axis])),
                1,
            )

    def test_plot_logs_input_summary_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("sys.argv", ["lma_plot", directory, directory]),
                patch.object(
                    lma_plot.FileBrowser,
                    "find",
                    return_value=[SimpleNamespace(path="/tmp/MALMA_20250615_200000_600_10src_0.0115deg-dx_source_3d.nc", prefix="MALMA")],
                ),
                patch.object(
                    lma_plot,
                    "get_data",
                    return_value={"start_time": datetime(2025, 6, 15, 20), "frame_interval": 600, "total": 2539},
                ),
                patch.object(lma_plot, "make_plot", return_value="/tmp/plot.png") as make_plot,
            ):
                lma_plot.main()

        output = self.messages.getvalue()
        self.assertIn("Found 1 gridded NetCDF file(s)", output)
        self.assertIn("Frame starts 2025-06-15 20:00:00 UTC", output)
        self.assertIn("2539 sources in selected grid", output)
        self.assertIn("Completed 1 image(s)", output)
        self.assertIn("Finished image 1/1 in", output)
        make_plot.assert_called_once()

    def test_plot_filename_preserves_resolution_decimal(self):
        input_file = "/data/MALMA_20220804_230000_600_10src_0.0115deg-dx_source_3d.nc"
        self.assertEqual(
            lma_plot._plot_filename(input_file, "/plots", "png"),
            "/plots/MALMA_20220804_230000_600_10src_0.0115deg-dx_source_3d.png",
        )


if __name__ == "__main__":
    unittest.main()
