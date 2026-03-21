import math

import numpy as np


def resolve_init_rect(bounds):
    low, high = bounds
    if low >= high:
        raise ValueError("Bounds low must be < bounds high.")
    return float(low), float(high), float(low), float(high)


def make_uniform_points(count, *, bounds=(-512, 512)):
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


def make_random_points(count, *, bounds=(-512, 512), rng=None):
    if count < 1:
        raise ValueError("count must be >= 1.")

    if rng is None:
        rng = np.random.default_rng()

    x_low, x_high, y_low, y_high = resolve_init_rect(bounds)
    low = np.array([x_low, y_low], dtype=float)
    high = np.array([x_high, y_high], dtype=float)
    return rng.uniform(low, high, size=(count, 2))


def make_initial_points(count, *, bounds=(-512, 512), mode="grid", rng=None):
    if mode == "grid":
        return make_uniform_points(count, bounds=bounds)
    if mode == "random":
        return make_random_points(count, bounds=bounds, rng=rng)
    raise ValueError("mode must be either 'grid' or 'random'.")
