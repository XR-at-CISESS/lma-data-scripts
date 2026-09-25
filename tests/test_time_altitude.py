import os
import tempfile
import unittest
from datetime import datetime

import numpy as np
import tables
from netCDF4 import Dataset

from lma_scripts.lma_flash import add_time_altitude_counts
from lma_scripts.lma_plot import _axis_extent, create_parser, get_data


class TimeAltitudeTests(unittest.TestCase):
    def test_counts_include_only_retained_flashes_in_frame_and_altitude(self):
        with tempfile.TemporaryDirectory() as directory:
            grid_path = os.path.join(
                directory, "DCLMA_20240804_210000_2_10src_source_3d.nc"
            )
            flash_path = os.path.join(directory, "flashes.h5")
            with Dataset(grid_path, "w") as grid:
                grid.createDimension("ntimes", 1)
                grid.createDimension("lon", 3)
                grid.createDimension("lat", 3)
                grid.createDimension("alt", 3)
                altitude = grid.createVariable("altitude", "f4", ("alt",))
                altitude[:] = [500, 1500, 2500]
                grid.createVariable("longitude", "f4", ("lon",))[:] = [-77, -76, -75]
                grid.createVariable("latitude", "f4", ("lat",))[:] = [38, 39, 40]
                time = grid.createVariable("time", "f4", ("ntimes",))
                time.units = "seconds since 2024-08-04 00:00:00"
                time[:] = [75600]
                source = grid.createVariable(
                    "lma_source", "i4", ("ntimes", "lon", "lat", "alt")
                )
                source.units = "sources"
                source[0, :, :, :] = 1
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
                    (75600.1, 400, 1),
                    (75600.7, 1400, 1),
                    (75601.2, 2400, 1),
                    (75600.2, 400, 2),
                    (75602.0, 400, 1),
                    (75600.3, 3500, 1),
                ], dtype=event_dtype))
                flash_table = flashes.create_table(flash_group, "frame", flash_dtype)
                flash_table.append(np.array([(1, 10), (2, 2)], dtype=flash_dtype))
            add_time_altitude_counts(
                grid_path, flash_path, datetime(2024, 8, 4, 21), 2, 10
            )
            with Dataset(grid_path) as grid:
                counts = grid.variables["time_altitude_count"][:]
                self.assertEqual(counts.shape, (2, 15))
                self.assertEqual(int(counts.sum()), 3)
                self.assertEqual(int(counts[0, 2]), 1)
                self.assertEqual(int(counts[0, 7]), 1)
                self.assertEqual(int(counts[1, 12]), 1)
                self.assertEqual(grid.variables["time_altitude_altitude"].units, "m")
            data = get_data(grid_path, time_altitude=True)
            self.assertEqual(data["time_altitude_count"].shape, (2, 15))
            self.assertEqual(int(data["time_altitude_count"].sum()), 3)

    def test_cli_opt_in(self):
        parser = create_parser()
        self.assertFalse(parser.parse_args(["in", "out"]).time_altitude)
        self.assertTrue(parser.parse_args(["in", "out", "--time-altitude"]).time_altitude)

    def test_focus_extent_ignores_sparse_outliers(self):
        coordinates = np.linspace(-80, -72, 801)
        weights = np.zeros(801)
        weights[200] = 1
        weights[300:500] = 10
        weights[700] = 1
        low, high = _axis_extent(coordinates, weights, 1.8)
        self.assertGreater(low, -78)
        self.assertLess(high, -74)

    def test_populated_grid_rejects_empty_time_altitude_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "DCLMA_20220805_000000_600_10src_source_3d.nc")
            with Dataset(path, "w") as grid:
                for name, length in (("ntimes", 1), ("lon", 3), ("lat", 3), ("alt", 3), ("second", 600)):
                    grid.createDimension(name, length)
                grid.createVariable("longitude", "f4", ("lon",))[:] = [-77, -76, -75]
                grid.createVariable("latitude", "f4", ("lat",))[:] = [38, 39, 40]
                grid.createVariable("altitude", "f4", ("alt",))[:] = [500, 1500, 2500]
                time = grid.createVariable("time", "f4", ("ntimes",))
                time.units = "seconds since 2022-08-05 00:00:00"
                time[:] = [0]
                sources = grid.createVariable("lma_source", "i4", ("ntimes", "lon", "lat", "alt"))
                sources.units = "sources"
                sources[0, :, :, :] = 1
                grid.createVariable("time_altitude_count", "i4", ("second", "alt"))[:, :] = 0
            with self.assertRaisesRegex(ValueError, "no usable time-altitude counts"):
                get_data(path, time_altitude=True)
            flash_path = os.path.join(directory, "DCLMA_220805_000000_0600.dat.flash.h5")
            with tables.open_file(flash_path, "w") as flashes:
                events = flashes.create_group("/", "events")
                flash_group = flashes.create_group("/", "flashes")
                event_table = flashes.create_table(
                    events, "frame", np.dtype([("time", "f8"), ("alt", "f4"), ("flash_id", "i4")])
                )
                event_table.attrs.start_time = (2022, 8, 5, 0, 0, 0)
                event_table.append(np.array([(0.5, 500, 1)], dtype=event_table.dtype))
                flash_table = flashes.create_table(
                    flash_group, "frame", np.dtype([("flash_id", "i4"), ("n_points", "i4")])
                )
                flash_table.append(np.array([(1, 10)], dtype=flash_table.dtype))
            data = get_data(path, time_altitude=True)
            self.assertEqual(int(data["time_altitude_count"].sum()), 1)
            self.assertEqual(len(data["time_altitude_alts"]), 15)


if __name__ == "__main__":
    unittest.main()
