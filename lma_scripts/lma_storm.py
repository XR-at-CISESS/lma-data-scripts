"""
Use lmatools to sort LMA ASCII data into flashes
Create gridded imagery from those flashes.
The params dictionary controls the flash sorting parameters.
Sum LMA ASCII data within 24 hours
USAGE:
python lma-storm {network} {year} {month} {day} {start_time} {end_time}

network: DCLMA, WFFLMA, MALMA
year: 20XX
month: XX
day: XX
start_time: HHMMSS
end_time: HHMMSS
"""
import sys, os, glob, pathlib
from datetime import datetime, timedelta
import subprocess, argparse

from lmatools.io.LMA import LMADataset
from lmatools.flashsort.gen_autorun import logger_setup, sort_files
from lmatools.flashsort.gen_sklearn import DBSCANFlashSorter

from lmatools.grid.make_grids import (
    grid_h5flashfiles,
    dlonlat_at_grid_center,
    write_cf_netcdf_latlon,
    write_cf_netcdf_3d_latlon,
)
from six.moves import map
from lma_data.LMA_util import get_lma_data_dir, get_lma_out_dir
from lma_scripts.grid_coordinates import scalar_grid_coordinates

import logging, logging.handlers

from lma_data.LMA_info import info


def tfromfile(name):
    parts = name.split("_")
    y, m, d = list(map(int, (parts[-3][0:2], parts[-3][2:4], parts[-3][4:6])))
    H, M, S = list(map(int, (parts[-2][0:2], parts[-2][2:4], parts[-2][4:6])))
    return y + 2000, m, d, H, M, S


def sort_flashes(files, base_sort_dir, params):
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
    # -------------------- Setup Log File -----------------------
    # base_sort_dir = outpath
    logger_setup(base_sort_dir)
    h5_dir = os.path.join(base_sort_dir, "storms")
    y, m, d, H, M, S = tfromfile(files[0])
    date = datetime(y, m, d, H, M, S)

    # ---------- Create Directory for HDF5 Files ----------------
    # Create HDF5 flash files
    base_out_dir = h5_dir + "/20%s" % (date.strftime("%y/%m/%d"))
    if os.path.exists(base_out_dir) == False:
        os.makedirs(base_out_dir)
        subprocess.call(
            [
                "chmod",
                "a+w",
                base_out_dir,
                h5_dir + "/20%s" % (date.strftime("%y/%m")),
                h5_dir + "/20%s" % (date.strftime("%y")),
            ]
        )

    # ----------- Add 'tag' to directory path (?) ---------------
    tag = ""
    outdir = os.path.join(base_out_dir, tag)

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
        files = files
        output_path = outdir
        clusterer = cluster

        # --------------------- update logs? ------------------------
        logger = logging.getLogger("FlashAutorunLogger")
        now = datetime.now().strftime("Flash autosort started %Y%m%d-%H%M%S")
        logger.info(now)

        h5_outfiles = []
        for a_file in files:
            try:
                # ---------- create filename with .flash extention ----------
                file_base_name = os.path.split(a_file)[-1].replace(".gz", "")
                outfile = os.path.join(output_path, file_base_name + ".flash")

                # ===========================================================
                # *********** Create LMADataset and Cluster Data ************
                lmadata = LMADataset(a_file, file_mask_length=params["mask_length"])
                clusterer(lmadata)

                # ----------- create filename with .h5 extention ------------
                outfile_with_extension = outfile + ".h5"
                h5_outfiles.append(outfile_with_extension)

                lmadata.write_h5_output(outfile_with_extension, a_file)
            except:
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
    h5_filenames = glob.glob(
        h5_dir + "/20%s/*.dat.flash.h5" % (date.strftime("%y/%m/%d"))
    )
    h5_filenames.sort()
    return h5_filenames


def grid(
    h5_filenames,
    base_sort_dir,
    min_points=1,
    dx=1.0e3,
    dy=1.0e3,
    dz=1.0e3,
    frame_interval=60.0,
    x_bnd=(-400.0e3, 400.0e3),
    y_bnd=(-400.0e3, 400.0e3),
    z_bnd=(0.0e3, 20.0e3),
    ctr_lat=38.52,
    ctr_lon=-76.42,
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
    # not really in km, just a different name to distinguish from similar variables below.
    dx_km = dx
    dy_km = dy
    x_bnd_km = x_bnd
    y_bnd_km = y_bnd
    z_bnd_km = z_bnd

    grid_dir = os.path.join(base_sort_dir, "storm_grid_files")

    # There are similar functions in lmatools to grid on a regular x,y grid in some map projection.
    dx, dy, x_bnd, y_bnd = dlonlat_at_grid_center(
        ctr_lat, ctr_lon, dx=dx_km, dy=dy_km, x_bnd=x_bnd_km, y_bnd=y_bnd_km
    )
    dx, dy, x_bnd, y_bnd = scalar_grid_coordinates(dx, dy, x_bnd, y_bnd)

    for f in h5_filenames:
        y, m, d, H, M, S = tfromfile(f)
        start_time = datetime(y, m, d, H, M, S)
        end_time = start_time + timedelta(0, frame_interval)
        date = start_time
        outpath = grid_dir + "/20%s" % (date.strftime("%y/%m/%d"))
        if os.path.exists(outpath) == False:
            os.makedirs(outpath)
            subprocess.call(
                [
                    "chmod",
                    "a+w",
                    outpath,
                    grid_dir + "/20%s" % (date.strftime("%y/%m")),
                    grid_dir + "/20%s" % (date.strftime("%y")),
                ]
            )
        if True:
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
        break


# ===========================================================
#      ================ MAIN SCRIPT ===================
# ===========================================================

def create_parser():
    parser = argparse.ArgumentParser(
        prog="lma_storm",
        description="Grid LMA data files and process them",
    )
    parser.add_argument("network")
    parser.add_argument("year", type=str)
    parser.add_argument("month", type=str)
    parser.add_argument("day", type=str)
    parser.add_argument("start_time", type=str)
    parser.add_argument("end_time", type=str)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # ---- Assign Directory/File Paths from User Input -------
    network = args.network
    year = args.year
    month = args.month
    day = args.day
    start_time = args.start_time
    end_time = args.end_time

    data_dir = get_lma_data_dir()
    network_data_dir = os.path.join(data_dir, network)
    out_dir = get_lma_out_dir()
    data_out = os.path.join(out_dir, network)

    # Ensure data out directory exists
    pathlib.Path(data_out).mkdir(parents=True, exist_ok=True)

    filenames = glob.glob(
        network_data_dir + "/" + "data/" + year + "/" + month + "/" + day + "/" + "*.gz"
    )

    # ---- Define storm length and calculate frame interval -------
    start_date = str(year + month + day)

    s_date = datetime.strptime(start_date, "%Y%m%d")
    s_time = datetime.strptime(start_time, "%H%M%S").time()
    start_datetime = datetime.combine(s_date, s_time)

    if end_time < start_time:
        e_date = s_date + timedelta(days=1)
        e_time = datetime.strptime(end_time, "%H%M%S").time()
        end_datetime = datetime.combine(e_date, e_time)
        frame_interval = int((end_datetime - start_datetime).total_seconds())
    else:
        e_time = datetime.strptime(end_time, "%H%M%S").time()
        end_datetime = datetime.combine(s_date, e_time)
        frame_interval = int((end_datetime - start_datetime).total_seconds())

    # ---- Parameters -------
    lma_info = info(str(network))
    center_ID = str(network)

    params = {
        "stations": (5, 99),  # range of allowable numbers of contributing stations
        "chi2": (0, 5),  # range of allowable chi-sq values
        "distance": 3000.0,  # space grouping thresholds
        "thresh_critical_time": 0.15,  # time grouping thresholds
        "thresh_duration": 3.0,  # maximum expected flash duration
        "ctr_lat": 38.5,  # center lat to use for flash sorting, gridding #lma_info[0]
        "ctr_lon": -76.3,  # center lon to use for flash sorting, gridding #lma_info[1]
        "mask_length": lma_info[
            4
        ],  # length of hexadec station mask column in the LMA ASCII files
        "min_points": 1,
    }  # minimum number of points per flash

    # ------- Create List of non-empty Data Files -----------
    data_files = []
    for file in filenames:
        fname = os.path.basename(file)
        f_time = int(float((fname.split("_")[2])))
        if os.stat(file).st_size > 2048:
            if end_time < start_time:
                if int(start_time) <= f_time:
                    data_files.append(file)
            else:
                if int(start_time) <= f_time <= int(end_time):
                    data_files.append(file)
        else:
            print("No data in data file")
            continue

    if end_time < start_time:
        filenames = glob.glob(
            data_dir
            + network
            + "/"
            + "data/"
            + year
            + "/"
            + month
            + "/"
            + str(int(day) + 1)
            + "/"
            + "*.gz"
        )
        for file in filenames:
            fname = os.path.basename(file)
            f_time = int(float((fname.split("_")[2])))
            if os.stat(file).st_size > 2048:
                if f_time <= int(end_time):
                    data_files.append(file)
            else:
                print("No data in data file")
                continue
    # -------------------------------------------------------
    # ----- Cluster Sources into Flashes (HDF5 files) -------
    h5_filenames = sort_flashes(data_files, data_out, params)

    # -------------------------------------------------------
    # ---------- Produce Gridded NetCDF files ---------------
    grid(
        h5_filenames,
        data_out,
        min_points=params["min_points"],
        dx=1.0e3,
        dy=1.0e3,
        dz=1.0e3,
        frame_interval=frame_interval,  # end_frame-start_frame
        x_bnd=(-400.0e3, 400.0e3),
        y_bnd=(-400.0e3, 400.0e3),
        z_bnd=(0.0e3, 20.0e3),
        ctr_lat=params["ctr_lat"],
        ctr_lon=params["ctr_lon"],
        center_ID=center_ID,
        base_date=datetime(2012, 1, 1),
    )


if __name__ == "__main__":
    main()
