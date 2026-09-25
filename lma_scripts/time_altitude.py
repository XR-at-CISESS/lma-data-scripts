"""Source counts by second and altitude from flash-sorted LMA events."""

from datetime import datetime

import numpy as np
import tables


def compute_time_altitude_counts(altitudes, flash_path, start_time, frame_interval, min_points):
    """Count retained sources in numeric NetCDF altitude bins and one-second bins.

    HDF5 event altitudes and numeric NetCDF altitude coordinates are both meters.
    The legacy NetCDF coordinate's units attribute incorrectly says km.
    """
    altitudes = np.asarray(altitudes, dtype=float)
    seconds = int(np.ceil(frame_interval))
    counts = np.zeros((seconds, len(altitudes)), dtype=np.int32)
    if len(altitudes) < 2:
        return counts
    altitude_edges = np.concatenate((
        [altitudes[0] - (altitudes[1] - altitudes[0]) / 2],
        (altitudes[:-1] + altitudes[1:]) / 2,
        [altitudes[-1] + (altitudes[-1] - altitudes[-2]) / 2],
    ))
    day_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    frame_start = (start_time - day_start).total_seconds()
    with tables.open_file(flash_path) as flashes:
        for name, event_table in flashes.root.events._v_children.items():
            flash_table = flashes.root.flashes._v_children[name]
            retained = flash_table.read_where(
                f"n_points >= {min_points}", field="flash_id"
            )
            if len(retained) == 0:
                continue
            table_start = datetime(*event_table.attrs.start_time)
            day_offset = (table_start.replace(hour=0, minute=0, second=0) - day_start).total_seconds()
            for offset in range(0, event_table.nrows, 100_000):
                events = event_table.read(offset, min(offset + 100_000, event_table.nrows))
                selected = np.isin(events["flash_id"], retained)
                second = np.floor(day_offset + events["time"][selected] - frame_start).astype(int)
                altitude = np.searchsorted(
                    altitude_edges, events["alt"][selected], side="right"
                ) - 1
                valid = (
                    (second >= 0) & (second < seconds)
                    & (altitude >= 0) & (altitude < len(altitudes))
                )
                np.add.at(counts, (second[valid], altitude[valid]), 1)
    return counts
