import math

import numpy as np


def resolve_init_rect(bounds):
    low, high = bounds
    if low >= high:
        raise ValueError("Bounds low must be < bounds high.")
    return float(low), float(high), float(low), float(high)


def make_uniform_points(count, *, bounds=(-515, 515)):
    if count < 1:
        raise ValueError("count must be >= 1.")

    x_low, x_high, y_low, y_high = resolve_init_rect(bounds)
    n_cols = math.ceil(math.sqrt(count))
    n_rows = math.ceil(count / n_cols)

    xs = np.linspace(x_low, x_high, n_cols)
    ys = np.linspace(y_low, y_high, n_rows)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    return points[:count].astype(float, copy=False)
