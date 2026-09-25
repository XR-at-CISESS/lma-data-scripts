"""
Plots gridded 3D source products produced by LMA_flash_sort_and_grid.py
USAGE:
python lma-plot {network} {year} {month} {day}
"""

import os
import argparse
import logging
from time import perf_counter

import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib import gridspec, colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.geoaxes import GeoAxes

from lma_data.LMA_info import info
from lma_data.LMA_util import get_lma_shapes_dir, get_lma_out_dir
from lma_data.browser.file_browser import FileBrowser
from lma_data.lmatools_file import LMAToolsFile
from lma_data.LMA_filters import LMAFilters
from lma_scripts.log_output import configure_stage_logging

LOG = logging.getLogger("lma_scripts.plot")

GeoAxes._pcolormesh_patched = Axes.pcolormesh


def _plot_filename(file, outpath, image_type):
    return os.path.join(
        outpath,
        f"{os.path.splitext(os.path.basename(file))[0]}.{image_type}",
    )


def draw_map(
    ax,
    network="DCLMA",
    lat_0=38.8894,
    lon_0=-77.03517,
    plotrgba=[0.94, 0.96, 0.98, 1],
    continentrgba=[0.95, 0.93, 0.91, 1],
    countiesrgba=[0.9, 0.9, 0.9, 1],
    statesrgba=[0.7, 0.8, 1, 1],
    coastrgba=[0.2, 0.2, 0.2, 1],
    stationrgba=[0, 0.8, 0, 1],
    **kwargs,
):

    LOG.debug("Generating map")
    projection = ccrs.PlateCarree()

    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="50m",
        facecolor="none",
    )

    # --add features, such as lakes, river, borders, coastlines, etc.
    reader = shpreader.Reader(os.path.join(get_lma_shapes_dir(), "countyl010g.shp"))
    LOG.debug("Loaded county shapes")

    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, projection)

    ax.add_feature(cfeature.LAND, color=continentrgba)
    ax.add_feature(COUNTIES, facecolor="none", edgecolor=countiesrgba)
    ax.add_feature(states_provinces, edgecolor=statesrgba)
    ax.add_feature(cfeature.LAKES, color=plotrgba)
    ax.add_feature(cfeature.OCEAN, color=plotrgba)
    ax.coastlines(resolution="10m", color=coastrgba)

    LOG.debug("Added map features")

    # -------------- Gather network specific variables  --------------
    lma_info = info(network)
    station_lats = lma_info[2]
    station_lons = lma_info[3]
    lat_extents = lma_info[5]
    lon_extents = lma_info[6]

    LOG.debug("Loaded network information")

    extents = [*lon_extents, *lat_extents]
    ax.set_extent(extents, crs=projection)
    LOG.debug("Set map extent")
    ax.set_aspect("auto")

    LOG.debug("Set map aspect")

    # ------------------- Plot LMA Stations -------------------
    ax.scatter(
        station_lons,
        station_lats,
        s=6,
        marker="s",
        color=stationrgba,
        facecolors="none",
        transform=projection,
        zorder=10,
    )

    LOG.debug("Finished map")

    return ax


def round_time(dt, round_to=60):
    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def get_data(file, lon_index=(0, 800), lat_index=(0, 800), alt_index=(0, 20), time_altitude=False):

    LOG.debug("Reading 3D NetCDF grid from %s", file)
    # -------- Import Variables from Gridded NetCDF file ------------
    data = Dataset(file).variables
    lons = data["longitude"][:]
    lats = data["latitude"][:]
    grid_type = data["lma_source"][0, :, :, :]
    grid_units = data["lma_source"].units
    time = data["time"][:]
    time_units = data["time"].units
    alts = data["altitude"][:]
    time_altitude_count = None
    if time_altitude:
        if "time_altitude_count" not in data:
            raise ValueError(
                f"{file} has no time-altitude data; regenerate its grid with the current lma_flash"
            )
        time_altitude_count = data["time_altitude_count"][:, alt_index[0] : alt_index[1]]

    # ------------ Index Data in Region of Interest ------------
    if len(alt_index) > 0:
        lons = data["longitude"][lon_index[0] : lon_index[1]]
        lats = data["latitude"][lat_index[0] : lat_index[1]]
        alts = data["altitude"][alt_index[0] : alt_index[1]]
        grid_type = data["lma_source"][
            0,
            lon_index[0] : lon_index[1],
            lat_index[0] : lat_index[1],
            alt_index[0] : alt_index[1],
        ]

    # ------------- Extract date from Gridded NetCDF file -------------
    base_date = datetime.strptime(time_units, "seconds since %Y-%m-%d %H:%M:%S")
    time_delta = timedelta(0, float(time[0]), 0)
    start_time = base_date + time_delta

    fname = os.path.basename(file)
    frame_interval = int(fname.split("_")[3])  # split filename to get frame interval

    # start_time = round_time(start_time, round_to=frame_interval)
    start_time = round_time(start_time, round_to=60 * 1)

    LOG.debug("Calculating source totals and projections")
    # ------------------- Sum TOTAL Source Count -------------------
    total = np.sum(grid_type)

    # ---------- Create lat/lon mesh -------------
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)

    # ------------------ Initiate Summation Arrays ------------------
    lmalon = np.zeros((grid_type.shape[0], grid_type.shape[2]))
    lmalat = np.zeros((grid_type.shape[1], grid_type.shape[2]))
    lmalonlat = np.zeros((grid_type.shape[0], grid_type.shape[1]))

    # --------------- Sum Horizontal Source Counts ------------------
    lmalon = np.sum(grid_type, axis=1)
    lmalat = np.sum(grid_type, axis=0)
    lmaalt = np.sum(grid_type, axis=(0, 1))

    # ---------------- Sum Vertical Source Counts -------------------
    lmalonlat = np.sum(grid_type, axis=2)

    # ------------------ Mask source values <= 0 --------------------
    lmalon = np.ma.masked_where(lmalon <= 0, lmalon)
    lmalat = np.ma.masked_where(lmalat <= 0, lmalat)
    lmalonlat = np.ma.masked_where(lmalonlat <= 0, lmalonlat)

    DATA = dict(
        mesh_lon=mesh_lon,
        mesh_lat=mesh_lat,
        alts=alts,
        lmalon=lmalon,
        lmalat=lmalat,
        lmalonlat=lmalonlat,
        lmaalt=lmaalt,
        total=total,
        grid_units=grid_units,
        grid_type=grid_type,
        start_time=start_time,
        frame_interval=frame_interval,
        time_altitude_count=time_altitude_count,
    )

    return DATA


def make_plot(
    data,
    file,
    grid_name="gridded_product",
    network="DCLMA",
    outpath="./",
    do_save=True,
    theme="normal",
    image_type="png",
    time_altitude=False,
):
    lma_info = info(network)
    lat_0 = lma_info[0]
    lon_0 = lma_info[1]
    lat_extents = lma_info[5]
    lon_extents = lma_info[6]

    # --------------------- Setup ColorMap --------------------------
    vmin = data["grid_type"][:].min()
    vmax = 500  # data['grid_type'].max()
    cmap = colors.ListedColormap(
        [
            "#7e00ff",
            "#0000ff",
            "#007eff",
            "#00ffff",
            "#00ff00",
            "#007e3e",
            "#3e7e7e",
            "#ffff00",
            "#ffbe7e",
            "#ff7e00",
            "#ff007e",
            "#ff0000",
            "#be0000",
            "#7e0000",
            "#000000",
            "#7e7e7e",
            "#bebebe",
            "#ffffff",
        ]
    )
    cticks = [1, 2, 4, 6, 9, 11, 16, 22, 30, 42, 59, 78, 107, 145, 197, 269, 367, 500]
    cbounds = [1, 2, 4, 6, 9, 11, 16, 22, 30, 42, 59, 78, 107, 145, 197, 269, 367, 500]
    norm = colors.BoundaryNorm(cbounds, cmap.N)

    # ---------------- Setup Figure Color Theme ---------------------
    if theme == "diagnostics":
        fbgrgba = [0.95, 0.95, 0.95, 1]
        edgergba = [0.5, 0.5, 0.5, 1]
        meshrgba = [0.7, 0.5, 0.7, 0.01]
        borderrgba = edgergba
        plotrgba = [0.9, 0.9, 0.95, 1]
        gridrgba = [1, 0.4, 0.8, 0.5]
        stationrgba = [0, 1, 0, 1]
        textrgba = [0, 0, 0, 1]
        continentrgba = [0.95, 0.95, 0.95, 1]
        countiesrgba = [0.9, 0.9, 0.9, 1]
        statesrgba = [0.7, 0.8, 1, 1]
        coastrgba = [0.2, 0.2, 0.2, 1]
    elif theme == "dark":
        fbgrgba = [0, 0, 0, 1]
        edgergba = [0.9, 0.9, 0.9, 1]
        meshrgba = [0.7, 0.5, 0.7, 0]
        borderrgba = edgergba
        plotrgba = [0, 0, 0, 1]
        gridrgba = [1, 0.4, 0.8, 0.2]
        stationrgba = [0, 0.7, 0, 1]
        textrgba = [0.9, 0.9, 0.9, 1]
        continentrgba = [0.1, 0.05, 0.05, 1]
        countiesrgba = [0.15, 0.15, 0.15, 1]
        statesrgba = [0.4, 0.5, 0.7, 1]
        coastrgba = [0.8, 0.8, 0.8, 1]
    else:
        fbgrgba = [1, 1, 1, 1]
        edgergba = [0, 0, 0, 1]
        meshrgba = [0.7, 0.5, 0.7, 0]
        borderrgba = edgergba
        plotrgba = [0.94, 0.96, 0.98, 1]
        gridrgba = [1, 0.4, 0.8, 0.1]
        stationrgba = [0, 0.8, 0, 1]
        textrgba = [0, 0, 0, 1]
        continentrgba = [0.95, 0.93, 0.91, 1]
        countiesrgba = [0.9, 0.9, 0.9, 1]
        statesrgba = [0.7, 0.8, 1, 1]
        coastrgba = [0.2, 0.2, 0.2, 1]

    # ==============================================================
    # ------------- Generate Figure with Subplot Specs -------------
    LOG.debug("Creating figure")

    # -------- Calcualte Figure Size from Subplot Aspect Ratios --------
    FA = 1.2 if time_altitude else 0
    F0 = 4.5  # main panel is F0 times the size of the cross section panel
    F1 = 0.25  # cbar panel is F1 times the size of the cross section panel
    F2 = 0.6  # Gap above cbar is F2 times the size of the cross section panel

    fwin = 6  # figure width in inches
    dpi = 300

    mpl.pyplot.rc("axes", linewidth=1, edgecolor=edgergba)  # assign axes edge colors

    fig = plt.figure(
        figsize=(fwin, fwin * (FA + 1 + F0 + F2 + F1) / (1 + F0)),
        dpi=dpi,
        facecolor=fbgrgba,
        edgecolor=borderrgba,
        linewidth=2,
        frameon=True,
        tight_layout=True,
    )

    gs = fig.add_gridspec(
        5,
        2,
        width_ratios=(F0, 1),
        height_ratios=(FA, 1, F0, F2, F1),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0,
        hspace=0.12 if time_altitude else 0,
    )

    # ===============================================================
    # ----------------- Plan View Composite (main) ------------------
    ax0 = fig.add_subplot(
        gs[2, 0],
        facecolor=plotrgba,
        frame_on=True,
        xscale="linear",
        yscale="linear",
        projection=ccrs.PlateCarree(),
    )

    map = draw_map(
        ax0,
        network=network,
        lat_0=lat_0,
        lon_0=lon_0,
        plotrgba=[0.94, 0.96, 0.98, 1],
        continentrgba=[0.95, 0.93, 0.91, 1],
        countiesrgba=[0.9, 0.9, 0.9, 1],
        statesrgba=[0.7, 0.8, 1, 1],
        coastrgba=[0.2, 0.2, 0.2, 1],
        stationrgba=[0, 0.8, 0, 1],
    )

    # -------------------- Create x/y mesh -------------------------
    mesh_xyx = np.zeros(data["mesh_lon"].shape)
    mesh_xyy = np.zeros(data["mesh_lat"].shape)
    mesh_xyx, mesh_xyy = data["mesh_lon"], data["mesh_lat"]

    map.pcolormesh(
        data["mesh_lon"],
        data["mesh_lat"],
        data["lmalonlat"].T[:-1, :-1],
        transform=ccrs.PlateCarree(),
        edgecolor=meshrgba,
        cmap=cmap,
        norm=norm,
        shading="flat",
        zorder=10,
    )

    # ===============================================================
    # ---------- East-West Composite Cross Section (top) ------------

    # ----------------------- Create x/z mesh -----------------------
    row = np.floor(mesh_xyx.shape[0] / 2)  # find middle row of mesh_xyx
    row = np.array(row, dtype="int")  # convert row index to integer value
    xvec = mesh_xyx[row, :]  # index the middle row of mesh_xyx

    mesh_xzx, mesh_xzz = np.meshgrid(xvec, data["alts"])

    # ------------------ Add subplot Axis and Plot ------------------
    ax1 = fig.add_subplot(
        gs[1, 0],
        sharex=ax0,
        facecolor=plotrgba,
        frame_on=True,
        xscale="linear",
        yscale="linear",
    )
    plot_xz = ax1.pcolormesh(
        mesh_xzx,
        mesh_xzz / 1e3,
        data["lmalon"].T[:-1, :-1],
        edgecolor=meshrgba,
        cmap=cmap,
        norm=norm,
        shading="flat",
        zorder=10,
    )
    # plot_xz.cmap.set_over('w')

    # ===============================================================
    # --------- North-South Composite Cross Section (right) ---------
    ax2 = fig.add_subplot(
        gs[2, 1:2],
        sharey=ax0,
        facecolor=plotrgba,
        frame_on=True,
        xmargin=0,
        ymargin=0,
        xscale="linear",
        yscale="linear",
    )

    # -------------------- Create z/y mesh -------------------------
    col = np.floor(mesh_xyy.shape[1] / 2)  # find middle column of mesh_xyy
    col = np.array(col, dtype="int")  # convert column to integer value
    yvec = mesh_xyy[:, col]  # index the middle row of mesh_xyy

    mesh_zyz, mesh_zyy = np.meshgrid(data["alts"], yvec)

    # --------------- Add subplot Axis and Plot --------------------
    plot_zy = ax2.pcolormesh(
        mesh_zyz / 1e3,
        mesh_zyy,
        data["lmalat"][:-1, :-1],
        edgecolor=meshrgba,
        cmap=cmap,
        norm=norm,
        shading="flat",
        zorder=10,
    )
    # plot_zy.cmap.set_over('w')

    # ===============================================================
    # -------------- Total Sounce Count (top-right) -----------------
    ax3 = fig.add_subplot(
        gs[1, 1],
        facecolor=plotrgba,
        frame_on=True,
        xmargin=0,
        ymargin=0,
        xscale="linear",
        yscale="linear",
    )

    ax3.plot(
        data["lmaalt"],
        data["alts"] / 1e3,
        linestyle="-",
        linewidth=1,
        marker="None",
        color=borderrgba,
    )
    if data["lmaalt"].max() > 0:
        ax3.set_xlim([0, 1.1 * data["lmaalt"].max()])
    else:
        ax3.set_xlim([0, 1.1])

    txtx = ax3.get_xlim()
    txty = ax3.get_ylim()
    ax3.text(
        txtx[1] * 0.05,
        txty[1] * 0.95,
        "N: " + str(data["total"]),
        horizontalalignment="left",
        verticalalignment="top",
        fontdict=dict(family="PT Sans", size=10),
        color=textrgba,
        alpha=textrgba[3],
    )

    # ===============================================================
    # ------------------- Add Altitude/Time -------------------------

    if time_altitude:
        counts = data["time_altitude_count"]
        if counts is None:
            raise ValueError("Time-altitude counts were not loaded")
        altitude_time_ax = fig.add_subplot(gs[0, :], facecolor=plotrgba)
        seconds, altitude_bins = np.nonzero(counts > 0)
        timestamps = [
            data["start_time"] + timedelta(seconds=int(second) + 0.5)
            for second in seconds
        ]
        time_scatter = altitude_time_ax.scatter(
            timestamps,
            data["alts"][altitude_bins] / 1e3,
            c=counts[seconds, altitude_bins],
            s=3,
            marker="s",
            cmap=cmap,
            norm=norm,
            linewidths=0,
        )
        altitude_time_ax.set_xlim(
            data["start_time"],
            data["start_time"] + timedelta(seconds=data["frame_interval"]),
        )
        altitude_time_ax.set_ylim(0, 20)
        altitude_time_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        altitude_time_ax.set_ylabel("Alt (km)", color=textrgba)
        time_colorbar = fig.colorbar(
            time_scatter, ax=altitude_time_ax, orientation="vertical",
            pad=0.01, fraction=0.025,
        )
        time_colorbar.set_label("Sources / second / altitude bin", color=textrgba)
        time_colorbar.ax.tick_params(labelsize=7, labelcolor=textrgba)
        altitude_time_ax.grid(color=gridrgba, linewidth=1)
        altitude_time_ax.tick_params(labelsize=8, labelcolor=textrgba, color=edgergba)

    # ===============================================================
    # ------------------- Add/Format Colorbar -----------------------
    ax4 = fig.add_subplot(gs[4, :])
    cbar = fig.colorbar(
        ax=ax4,
        cax=ax4,
        ticks=cticks,
        orientation="horizontal",
        mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    )
    cbar.set_label(
        data["grid_units"],
        fontdict=dict(family="PT Sans", size=12),
        color=textrgba,
        alpha=textrgba[3],
    )

    # Plot Title
    fig.suptitle(
        network
        + ": Source Density\n"
        + str(data["start_time"])
        + " - "
        + str(
            (data["start_time"] + timedelta(seconds=data["frame_interval"])).strftime(
                "%H:%M:%S"
            )
        )
        + " UTC",
        x=0.5,
        y=0.98,
        fontdict=dict(family="PT Sans"),
        fontsize=12,
        color=textrgba,
        alpha=textrgba[3],
    )

    # ===============================================================
    # ------------- Finish Setting the Layout Parameters ------------
    # ----------------------- x/y tick labels -----------------------
    ax0.tick_params(
        axis="both",
        which="both",
        direction="in",
        pad=3,
        labelsize=10,
        color=edgergba,
        labelcolor=textrgba,
        labeltop=False,
        labelright=False,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        pad=3,
        labelsize=10,
        color=edgergba,
        labelcolor=textrgba,
        labelbottom=False,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax2.tick_params(
        axis="both",
        which="both",
        direction="in",
        pad=3,
        labelsize=10,
        color=edgergba,
        labelcolor=textrgba,
        labelleft=False,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax3.tick_params(
        axis="y",
        which="both",
        direction="in",
        pad=3,
        labelsize=10,
        color=edgergba,
        labelcolor=textrgba,
        bottom=False,
        top=False,
        left=True,
        right=True,
    )
    ax4.tick_params(
        axis="x",
        which="major",
        direction="in",
        pad=3,
        labelsize=10,
        color=edgergba,
        labelcolor=textrgba,
        bottom=True,
        top=True,
        left=False,
        right=False,
    )

    # Place tick marks every 100 kilometers
    # ticklabels = np.arange(-400,500,100)
    ticklabels = ["", "-300", "-200", "-100", "0", "100", "200", "300", ""]
    ticklocationx = np.linspace(*lon_extents, num=9)
    ticklocationy = np.linspace(*lat_extents, num=9)

    ax0.set_xticks(ticklocationx)
    ax0.set_yticks(ticklocationy)
    ax0.set_xticklabels(ticklabels)
    ax0.set_yticklabels(ticklabels)

    # Remove tick labels from cross section shared axis
    ax3.set_xticks([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    # Change tick label font
    for tick in ax0.get_xticklabels():
        tick.set_fontname("PT Sans")
    for tick in ax0.get_yticklabels():
        tick.set_fontname("PT Sans")

    for tick in ax1.get_xticklabels():
        tick.set_fontname("PT Sans")
    for tick in ax1.get_yticklabels():
        tick.set_fontname("PT Sans")

    for tick in ax2.get_xticklabels():
        tick.set_fontname("PT Sans")
    for tick in ax2.get_yticklabels():
        tick.set_fontname("PT Sans")

    for tick in ax4.get_xticklabels():
        tick.set_fontname("PT Sans")

    # -------------- Turn plot gridlines on ---------------------
    ax0.grid(
        visible=True,
        which="both",
        axis="both",
        linestyle="-",
        linewidth=1,
        color=gridrgba,
        alpha=gridrgba[3],
    )

    ax1.grid(
        visible=True,
        which="both",
        axis="both",
        linestyle="-",
        linewidth=1,
        color=gridrgba,
        alpha=gridrgba[3],
    )
    ax2.grid(
        visible=True,
        which="both",
        axis="both",
        linestyle="-",
        linewidth=1,
        color=gridrgba,
        alpha=gridrgba[3],
    )
    ax3.grid(
        visible=True,
        which="both",
        axis="y",
        linestyle="-",
        linewidth=1,
        color=gridrgba,
        alpha=gridrgba[3],
    )

    # -------------------- axis labels --------------------------
    ax0.set_xlabel(
        "East-West Distance (km)",
        fontdict=dict(family="PT Sans", size=12),
        color=textrgba,
        alpha=textrgba[3],
    )
    ax2.set_xlabel(
        "Alt (km)",
        fontdict=dict(family="PT Sans", size=12),
        color=textrgba,
        alpha=textrgba[3],
    )

    ax0.set_ylabel(
        "North-South Distance (km)",
        fontdict=dict(family="PT Sans", size=12),
        color=textrgba,
        alpha=textrgba[3],
    )
    ax1.set_ylabel(
        "Alt (km)",
        fontdict=dict(family="PT Sans", size=12),
        color=textrgba,
        alpha=textrgba[3],
    )

    # ===============================================================
    # ----------------------- Save file ------------------------------
    LOG.debug("Saving plot")
    filename = _plot_filename(file, outpath, image_type)
    if do_save:
        plt.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fbgrgba,
            edgecolor=edgergba,
        )
        LOG.info("Saved image to %s", filename)
    plt.close(fig)


# ===========================================================
# ----------------------- MAIN SCRIPT -----------------------
# ===========================================================


def create_parser():
    parser = argparse.ArgumentParser(
        prog="lma_plot",
        description="Create plots of LMA data",
    )

    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--time-altitude", action="store_true",
        help="show source counts by time and altitude above the map",
    )

    return parser


def create_cache_filter():
    cache_browser = FileBrowser(LMAToolsFile.try_parse)
    cache_filter = LMAFilters[LMAToolsFile].create_cache_filter_from_browser(
        cache_browser,
        lambda _, file: file.datetime,
        lambda _, file: file.datetime,
        lambda args: args["out_dir"],
    )

    return cache_filter


def main():
    parser = create_parser()
    date_filter = LMAFilters[LMAToolsFile].create_date_filter(
        lambda _, file: file.datetime
    )
    network_filter = LMAFilters[LMAToolsFile].create_network_filter(
        lambda _, file: file.prefix
    )
    cache_filter = create_cache_filter()

    filters = [date_filter, network_filter, cache_filter]
    LMAFilters.apply_filters_to_argparser(parser, *filters)

    # ---------- Assign Date Variables from User Input ----------
    args = parser.parse_args()
    data_dir = args.data_dir
    outpath = args.out_dir
    verbose = args.verbose
    configure_stage_logging(verbose)

    os.makedirs(outpath, exist_ok=True)

    # --------------- User input parameters --------------------
    params = {"lon_index": (0, 800), "lat_index": (0, 800), "alt_index": (0, 20)}  # km

    # ------------- Get List of 3D Gridded Files ----------------
    browser = FileBrowser(LMAToolsFile.try_parse, glob_pathname="**/*source_3d.nc")
    LMAFilters.add_filters_to_browser(browser, *filters)

    # -----------------------------------------------------------
    # -------------- Generate and Save the Plots ----------------
    filepaths = browser.find(data_dir, **vars(args))
    LOG.info(
        "Found %d gridded NetCDF file(s) in %s; writing images to %s",
        len(filepaths), data_dir, outpath,
    )
    if not filepaths:
        LOG.info("No gridded files matched the selected filters")

    for index, file in enumerate(filepaths, 1):
        file_started = perf_counter()
        # ------------ Get Data from Gridded NetCDF Files ------------
        path = file.path
        network = file.prefix
        LOG.info(
            "Rendering file %d/%d for %s: %s",
            index, len(filepaths), network, os.path.basename(path),
        )
        try:
            data = get_data(
                path,
                lon_index=params["lon_index"],
                lat_index=params["lat_index"],
                alt_index=params["alt_index"],
                time_altitude=args.time_altitude,
            )
            LOG.info(
                "Frame starts %s UTC; interval %d seconds; %s sources in selected grid",
                data["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                data["frame_interval"], data["total"],
            )
            if not np.ma.is_masked(data["total"]) and data["total"] == 0:
                LOG.warning("Selected grid has no sources; the image may be empty")
            # --------------------- Generate Figure ---------------------
            make_plot(
                data,
                path,
                grid_name="src_density",
                network=network,
                outpath=outpath,
                do_save=True,
                theme="norm",
                image_type="png",
                time_altitude=args.time_altitude,
            )
            LOG.info(
                "Finished image %d/%d in %.1f s",
                index, len(filepaths), perf_counter() - file_started,
            )
        except Exception:
            LOG.exception("Failed to render %s", path)
            raise
    if filepaths:
        LOG.info("Completed %d image(s) in %s", len(filepaths), outpath)


if __name__ == "__main__":
    main()
