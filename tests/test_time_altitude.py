import os
import tempfile
import unittest
from datetime import datetime

import numpy as np
import tables
from netCDF4 import Dataset

from lma_scripts.lma_flash import add_time_altitude_counts
from lma_scripts.lma_plot import create_parser


class TimeAltitudeTests(unittest.TestCase):
    def test_counts_include_only_retained_flashes_in_frame_and_altitude(self):
        with tempfile.TemporaryDirectory() as directory:
            grid_path = os.path.join(directory, "grid.nc")
            flash_path = os.path.join(directory, "flashes.h5")
            with Dataset(grid_path, "w") as grid:
                grid.createDimension("alt", 3)
                altitude = grid.createVariable("altitude", "f4", ("alt",))
                altitude[:] = [500, 1500, 2500]
            event_dtype = np.dtype([
                ("time", "f8"), ("alt", "f4"), ("flash_id", "i4")
            ])
            flash_dtype = np.dtype([("flash_id", "i4"), ("n_points", "i4")])
            with tables.open_file(flash_path, "w") as flashes:
                events = flashes.create_group("/", "events")
                flash_group = flashes.create_group("/", "flashes")
                event_table = flashes.create_table(events, "frame", event_dtype)
                event_table.attrs.start_time = (2024, 8, 4, 21, 0, 0)
                event_table.append(np.array([
                    (75600.1, 0.4, 1),
                    (75600.7, 1.4, 1),
                    (75601.2, 2.4, 1),
                    (75600.2, 0.4, 2),
                    (75602.0, 0.4, 1),
                    (75600.3, 3.5, 1),
                ], dtype=event_dtype))
                flash_table = flashes.create_table(flash_group, "frame", flash_dtype)
                flash_table.append(np.array([(1, 10), (2, 2)], dtype=flash_dtype))
            add_time_altitude_counts(
                grid_path, flash_path, datetime(2024, 8, 4, 21), 2, 10
            )
            with Dataset(grid_path) as grid:
                counts = grid.variables["time_altitude_count"][:]
                np.testing.assert_array_equal(counts, [[1, 1, 0], [0, 0, 1]])

    def test_cli_opt_in(self):
        parser = create_parser()
        self.assertFalse(parser.parse_args(["in", "out"]).time_altitude)
        self.assertTrue(parser.parse_args(["in", "out", "--time-altitude"]).time_altitude)


if __name__ == "__main__":
    unittest.main()
