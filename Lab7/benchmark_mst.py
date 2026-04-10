from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

from boruvka import boruvka_mst
from graph_utils import load_graph
from kruskal import kruskal_mst
from prim_binary_heap import prim_binary_heap_mst
from prim_fibonacci_heap import prim_fibonacci_heap_mst


ALGORITHMS = {
    "kruskal": kruskal_mst,
    "prim_binary_heap": prim_binary_heap_mst,
    "prim_fibonacci_heap": prim_fibonacci_heap_mst,
    "boruvka": boruvka_mst,
}

DENSITY_PERCENTS = (5, 15, 25, 35, 45, 55, 65, 75, 85)
GRAPH_SIZES = (10, 100, 250, 500, 750, 1000, 1500, 2000, 5000)


def benchmark_graph(
    path: Path,
    repeats: int,
) -> list[dict[str, str | int | float]]:
    n, edges = load_graph(path)
    mst_weights: dict[str, int] = {}
    rows: list[dict[str, str | int | float]] = []

    for name, algorithm in ALGORITHMS.items():
        timings: list[float] = []
        total_weight = 0
        mst_size = 0

        for _ in range(repeats):
            start = perf_counter()
            total_weight, mst = algorithm(n, edges)
            timings.append(perf_counter() - start)
            mst_size = len(mst)

        mst_weights[name] = total_weight
        rows.append(
            {
                "graph": path.name,
                "n": n,
                "m": len(edges),
                "algorithm": name,
                "mst_weight": total_weight,
                "mst_edges": mst_size,
                "avg_time_ms": mean(timings) * 1000,
            }
        )

    if len(set(mst_weights.values())) != 1:
        raise ValueError(f"MST weights differ for {path.name}: {mst_weights}")

    return rows


def get_graph_paths(directory: Path) -> list[Path]:
    paths = []
    for size in GRAPH_SIZES:
        for percent in DENSITY_PERCENTS:
            paths.append(directory / f"n{size}_p{percent:02d}.json")
    return paths


def save_results(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["graph", "n", "m", "algorithm", "mst_weight", "mst_edges", "avg_time_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows: list[dict[str, str | int | float]]) -> None:
    print(f"{'graph':18} {'algorithm':22} {'m':>8} {'weight':>8} {'time_ms':>12}")
    for row in rows:
        print(
            f"{row['graph']:18} {row['algorithm']:22} {row['m']:8} "
            f"{row['mst_weight']:8} {row['avg_time_ms']:12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("generated_graphs"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"))
    args = parser.parse_args()

    rows: list[dict[str, str | int | float]] = []
    for path in get_graph_paths(args.dir):
        rows.extend(benchmark_graph(path, args.repeats))

    rows.sort(
        key=lambda row: (
            int(row["n"]),
            int(row["graph"].split("_p")[1].split(".")[0]),
            row["algorithm"],
        )
    )
    save_results(args.output, rows)
    print_results(rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
