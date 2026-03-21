import numpy as np

try:
    from .ui import run_interface
except ImportError:
    from ui import run_interface


def f_xy(x, y):
    return -(y + 47.0) * np.sin(np.sqrt(np.abs(x / 2.0 + (y + 47.0)))) \
           - x * np.sin(np.sqrt(np.abs(x - (y + 47.0))))


def fitness(points):
    return f_xy(points[:, 0], points[:, 1])


def main():
    run_interface(
        fitness=fitness,
        objective_function=f_xy,
        bounds=(-512, 512),
        interval=12,
    )


if __name__ == "__main__":
    main()
