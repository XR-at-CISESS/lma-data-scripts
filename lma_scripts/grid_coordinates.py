"""Normalize coordinates returned by lmatools for NumPy grid creation."""

import numpy as np


def scalar_grid_coordinates(dx, dy, x_bnd, y_bnd):
    """Convert lmatools' singleton arrays to scalar degree values."""

    def scalar(value):
        return float(np.asarray(value).item())

    return (
        scalar(dx),
        scalar(dy),
        tuple(scalar(value) for value in x_bnd),
        tuple(scalar(value) for value in y_bnd),
    )
