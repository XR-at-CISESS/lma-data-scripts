"""
Use lmatools to sort LMA ASCII data into flashes
Create gridded imagery from those flashes.
The params dictionary controls the flash sorting parameters.
USAGE:
lma-flash {network} {year} {month} {days}

network: DCLMA, WFFLMA, MALMA
year: 20XX
month: XX
days: XX (multiple days separated by spaces)
"""

import sys, os, glob, pathlib, argparse
from datetime import datetime, timedelta
import subprocess
from time import perf_counter
from netCDF4 import Dataset

from lmatools.io.LMA import LMADataset
from lmatools.flashsort.gen_autorun import logger_setup, sort_files
from lmatools.flashsort.gen_sklearn import DBSCANFlashSorter
from lma_data.browser.file_browser import FileBrowser
from lma_data.LMA_filters import LMAFilters
from lma_data.lma_analysis_data_file import LMAAnalysisDataFile
from lma_data.lmatools_file import LMAToolsFile
from lma_data.browser.filters.non_empty import NonEmptyFileFilter
from lma_data.LMA_util import batch
from lma_scripts.log_output import (
    configure_stage_logging,
    filter_library_logger,
    readable_library_output,
)
from lma_scripts.grid_coordinates import scalar_grid_coordinates
from lma_scripts.time_altitude import compute_time_altitude_counts

from lmatools.grid.make_grids import (
    grid_h5flashfiles,
    dlonlat_at_grid_center,
    write_cf_netcdf_latlon,
    write_cf_netcdf_3d_latlon,
)
from six.moves import map

import logging, logging.handlers

from lma_data.LMA_info import info

LOG = logging.getLogger("lma_scripts.grid")


def add_time_altitude_counts(grid_path, flash_path, start_time, frame_interval, min_points):
    """Store one-second source counts by altitude in the 3D grid product."""
    with Dataset(grid_path, "r+") as grid:
        altitudes = grid.variables["altitude"][:]
        if len(altitudes) < 2:
            return
        counts = compute_time_altitude_counts(
            altitudes, flash_path, start_time, frame_interval, min_points
        )
        seconds = counts.shape[0]
        if "time_altitude_second" not in grid.dimensions:
            grid.createDimension("time_altitude_second", seconds)
        variable = grid.variables.get("time_altitude_count")
        if variable is None:
            variable = grid.createVariable(
                "time_altitude_count", "i4",
                ("time_altitude_second", grid.variables["altitude"].dimensions[0]),
                zlib=True,
            )
        variable.long_name = "Retained source count by second and altitude"
        variable.units = "sources"
        variable[:, :] = counts


def tfromfile(name):
    parts = name.split("_")
    y, m, d = list(map(int, (parts[-3][0:2], parts[-3][2:4], parts[-3][4:6])))
    H, M, S = list(map(int, (parts[-2][0:2], parts[-2][2:4], parts[-2][4:6])))
    return y + 2000, m, d, H, M, S


def sort_flashes(files, outdir, params):
    """Given a list of LMA ASCII data files, created HDF5 flash-sorted data
    files in base_sort_dir/h5_files.

    params is a dictionary with the following format

    params = {'stations':(5,99), # range of allowable numbers of contributing stations
          'chi2':(0,25.0),    # range of allowable chi-sq values
          'distance':3000.0, 'thresh_critical_time':0.15, # space and time grouping thresholds
          'thresh_duration':3.0, # maximum expected flash duration
          'ctr_lat':38.8894, 'ctr_lon':-77.03517, #center lat/lon to use for flash sorting, gridding
          'mask_length':55, # length of the hexadecimal station mask column in the LMA ASCII files
          }

    """
    LOG.info("Sorting %d analyzed file(s) into flashes", len(files))
    # -------------------- Setup Log File -----------------------
    logger_setup(outdir)
    filter_library_logger()
    LOG.info(
        "Flash filters: %d-%d stations, chi-squared %g-%g, "
        "maximum source separation %g m, minimum %s sources per flash",
        *params["stations"],
        *params["chi2"],
        params["distance"],
        params.get("min_points", "default"),
    )

    # ----------- Write 'param' settings to file ----------------
    info = open(os.path.join(outdir, "input_params.py"), "w")
    info.write(str(params))
    info.close()

    if True:
        # -----------------------------------------------------------
        # --------- Create Clustering Function/Object ? ? ? ? ? ? ?
        cluster = DBSCANFlashSorter(params).cluster

        # ===========================================================
        # ---------------- sort_files SubFunction -------------------
        # sort_files(files, outdir, cluster)
        # def sort_files(files, output_path, clusterer):
        output_path = outdir
        clusterer = cluster

        # --------------------- update logs ------------------------
        logger = logging.getLogger("FlashAutorunLogger")
        now = datetime.now().strftime("Flash autosort started %Y%m%d-%H%M%S")
        logger.info(now)

        h5_outfiles = []
        for index, a_file in enumerate(files, 1):
            file_started = perf_counter()
            LOG.info("Sorting file %d/%d: %s", index, len(files), os.path.basename(a_file))
            try:
                # ---------- create filename with .flash extention ----------
                file_base_name = os.path.split(a_file)[-1].replace(".gz", "")
                outfile = os.path.join(output_path, file_base_name + ".flash")

                # ===========================================================
                # *********** Create LMADataset and Cluster Data ************
                # lmadata = LMADataset(a_file,file_mask_length=params['mask_length']) (mask length param doesn't exist anymore)
                lmadata = LMADataset(a_file)
                with readable_library_output(LOG):
                    clusterer(lmadata)
                try:
                    flashes = lmadata.flashes
                    retained_sources = sum(flash.pointCount for flash in flashes)
                except (AttributeError, TypeError):
                    LOG.debug("Flash source counts are unavailable", exc_info=True)
                else:
                    LOG.info(
                        "Retained %d sources in %d flashes from %s",
                        retained_sources, len(flashes), os.path.basename(a_file),
                    )

                # ----------- create filename with .h5 extention ------------
                outfile_with_extension = outfile + ".h5"
                h5_outfiles.append(outfile_with_extension)

                with readable_library_output(LOG):
                    lmadata.write_h5_output(outfile_with_extension, a_file)
                LOG.info(
                    "Saved flash data to %s (%.1f s)",
                    outfile_with_extension, perf_counter() - file_started,
                )
            except:
                LOG.exception("Failed while sorting %s", a_file)
                logger.error(
                    "Did not successfully sort %s \n Error was: %s"
                    % (a_file, sys.exc_info()[1])
                )
                raise
        # return h5_outfiles

        # ------------- delete subfunction variables ----------------
        del (files, output_path, clusterer, file_base_name, outfile, lmadata)
        del (outfile_with_extension, logger, now, a_file)

        # -----------------------------------------------------------
        # ===========================================================

    # ------------------------------------------------------------
    # ------------- List HDF5 files that were created ------------
    h5_filenames = glob.glob(os.path.join(outdir, "*.dat.flash.h5"))
    h5_filenames.sort()
    LOG.info("%d flash file(s) ready for gridding", len(h5_filenames))
    return h5_filenames


def grid(
    h5_filenames,
    outpath,
    min_points=1,
    dx=1.0e3,
    dy=1.0e3,
    dz=1.0e3,
    frame_interval=60.0,
    x_bnd=(-200.0e3, 200.0e3),
    y_bnd=(-200.0e3, 200.0e3),
    z_bnd=(0.0e3, 20.0e3),
    ctr_lat=38.5,
    ctr_lon=-76.3,
    center_ID="MALMA",
    base_date=None,
):
    """Given a list of HDF5 filenames (sorted by time order) in h5_filenames,
    create 2D and 3D NetCDF grids with spacing dx, dy, dz in meters,
    frame_interval in seconds, and tuples of grid edges
    x_bnd, y_bnd, and z_bnd in meters

    The actual grids are in regular lat,lon coordinates, with spacing at the
    grid center matched to the dx, dy values given.

    Grids and plots are written to base_sort_dir/grid_files/ and  base_sort_dir/plots/
    base_date is used to optionally set a common reference time for each of the NetCDF grids.
    """
    if not h5_filenames:
        LOG.info("No flash files to grid")
    LOG.info(
        "Creating %d time frame(s) for %s at %g-second intervals; output: %s",
        len(h5_filenames), center_ID, frame_interval, outpath,
    )
    LOG.info(
        "Grid area: center lat %.4f°, lon %.4f°; %.0f × %.0f m horizontal cells; "
        "east/west %.0f to %.0f km, north/south %.0f to %.0f km, "
        "altitude %.0f to %.0f km; minimum %d sources per flash",
        ctr_lat, ctr_lon, dx, dy,
        x_bnd[0] / 1000, x_bnd[1] / 1000,
        y_bnd[0] / 1000, y_bnd[1] / 1000,
        z_bnd[0] / 1000, z_bnd[1] / 1000,
        min_points,
    )
    # not really in km, just a different name to distinguish from similar variables below.
    dx_km = dx
    dy_km = dy
    x_bnd_km = x_bnd
    y_bnd_km = y_bnd
    z_bnd_km = z_bnd

    # There are similar functions in lmatools to grid on a regular x,y grid in some map projection.
    dx, dy, x_bnd, y_bnd = dlonlat_at_grid_center(
        ctr_lat, ctr_lon, dx=dx_km, dy=dy_km, x_bnd=x_bnd_km, y_bnd=y_bnd_km
    )
    dx, dy, x_bnd, y_bnd = scalar_grid_coordinates(dx, dy, x_bnd, y_bnd)

    for index, f in enumerate(h5_filenames, 1):
        frame_started = perf_counter()
        y, m, d, H, M, S = tfromfile(f)
        start_time = datetime(y, m, d, H, M, S)
        end_time = start_time + timedelta(0, frame_interval)
        LOG.info(
            "Frame %d/%d covers %s to %s UTC",
            index, len(h5_filenames),
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        try:
            with readable_library_output(LOG):
                grid_h5flashfiles(
                    h5_filenames,
                    start_time,
                    end_time,
                    min_points_per_flash=min_points,
                    frame_interval=frame_interval,
                    proj_name="latlong",
                    base_date=base_date,
                    energy_grids="",
                    dx=dx,
                    dy=dy,
                    x_bnd=x_bnd,
                    y_bnd=y_bnd,
                    z_bnd=z_bnd_km,
                    ctr_lon=ctr_lon,
                    ctr_lat=ctr_lat,
                    outpath=outpath,
                    output_writer=write_cf_netcdf_latlon,
                    output_writer_3d=write_cf_netcdf_3d_latlon,
                    output_filename_prefix=center_ID,
                    spatial_scale_factor=1.0,
                )
        except Exception:
            LOG.exception(
                "Failed to create frame %d/%d from %s",
                index, len(h5_filenames), os.path.basename(f),
            )
            raise
        output_files = sorted(
            path
            for path in glob.glob(
                os.path.join(outpath, f"{center_ID}_{start_time:%Y%m%d_%H%M%S}_*.nc")
            )
            if os.path.isfile(path)
        )
        for output_file in output_files:
            if output_file.endswith("_source_3d.nc"):
                add_time_altitude_counts(
                    output_file, f, start_time, frame_interval, min_points,
                )
        LOG.info(
            "Finished frame %d/%d: %d NetCDF file(s) present in %.1f s",
            index, len(h5_filenames), len(output_files), perf_counter() - frame_started,
        )
        for output_file in output_files:
            LOG.info("NetCDF output: %s", output_file)
    LOG.info("Completed NetCDF grids in %s", outpath)


# ===========================================================
#      ================ MAIN SCRIPT ===================
# ===========================================================
def create_parser():
    parser = argparse.ArgumentParser(
        prog="lma_flash",
        description="Grid LMA data files and process them",
    )
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")

    return parser


def create_cache_filter():
    browser = FileBrowser(LMAToolsFile.try_parse)
    cache_filter = LMAFilters[LMAAnalysisDataFile].create_cache_filter_from_browser(
        browser,
        lambda _, file: file.datetime,
        lambda _, file: file.datetime,
        lambda args: args["out_dir"],
    )

    return cache_filter


def main():
    parser = create_parser()
    date_filter = LMAFilters[LMAAnalysisDataFile].create_date_filter(
        lambda _, file: file.datetime
    )
    network_filter = LMAFilters[LMAAnalysisDataFile].create_network_filter(
        lambda _, file: file.network
    )
    non_empty_filter = NonEmptyFileFilter()
    cache_filter = create_cache_filter()
    filters = [date_filter, network_filter, non_empty_filter, cache_filter]
    LMAFilters.apply_filters_to_argparser(parser, *filters)
    args = parser.parse_args()
    configure_stage_logging()

    data_dir: str = args.data_dir
    out_dir: str = args.out_dir

    # Ensure data out directory exists
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    browser = FileBrowser(LMAAnalysisDataFile.try_parse, "**/*.gz")
    LMAFilters.add_filters_to_browser(browser, *filters)
    files = browser.find(data_dir, **vars(args))
    LOG.info(
        "Found %d analyzed file(s) in %s; writing results to %s",
        len(files), data_dir, out_dir,
    )
    network_batches = batch(files, lambda f: f.network)

    for network_batch in network_batches:
        network = network_batch[0].network
        LOG.info("Processing %s (%d input file(s))", network, len(network_batch))
        lma_info = info(network)
        params = {
            "stations": (6, 99),  # range of allowable numbers of contributing stations
            "chi2": (0, 5),  # range of allowable chi-sq values
            "distance": 3000.0,  # space grouping thresholds
            "thresh_critical_time": 0.15,  # time grouping thresholds
            "thresh_duration": 3.0,  # maximum expected flash duration
            "ctr_lat": 38.5,  # center lat to use for flash sorting, gridding
            "ctr_lon": -76.3,  # center lon to use for flash sorting, gridding
            "mask_length": lma_info[
                4
            ],  # length of hexadec station mask column in the LMA ASCII files
            "min_points": 10,  # minimum number of points per flash
            "altitude": (0, 20000),
        }  # range of allowable altitude

        # -------------------------------------------------------
        # ----- Cluster Sources into Flashes (HDF5 files) -------
        paths = list(map(lambda f: f.path, network_batch))
        h5_filenames = sort_flashes(paths, out_dir, params)

        # -------------------------------------------------------
        # ---------- Produce Gridded NetCDF files ---------------
        grid(
            h5_filenames,
            out_dir,
            min_points=params["min_points"],
            dx=1.0e3,
            dy=1.0e3,
            dz=1.0e3,
            frame_interval=600,
            x_bnd=(-400.0e3, 400.0e3),
            y_bnd=(-400.0e3, 400.0e3),
            z_bnd=(0.0e3, 20.0e3),
            ctr_lat=params["ctr_lat"],
            ctr_lon=params["ctr_lon"],
            center_ID=str(network),
            base_date=datetime(2012, 1, 1),
        )


if __name__ == "__main__":
    main()
