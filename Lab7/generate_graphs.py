from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from graph_utils import save_graph
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from graph_utils import save_graph


DENSITY_PERCENTS = (5, 15, 25, 35, 45, 55, 65, 75, 85)
GRAPH_SIZES = (10, 100, 250, 500, 750, 1000, 1500, 2000, 5000)


def generate_connected_graph(
    n: int,
    m: int,
    min_weight: int = 1,
    max_weight: int = 100,
    seed: int | None = None,
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    edges: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for v in range(1, n):
        u = rng.randrange(v)
        a, b = (u, v) if u < v else (v, u)
        seen.add((a, b))
        edges.append((a, b, rng.randint(min_weight, max_weight)))

    while len(edges) < m:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b, rng.randint(min_weight, max_weight)))

    return edges


def _generate_spanning_tree(n: int, rng: random.Random) -> list[tuple[int, int]]:
    tree_edges: list[tuple[int, int]] = []
    for v in range(1, n):
        u = rng.randrange(v)
        a, b = (u, v) if u < v else (v, u)
        tree_edges.append((a, b))
    return tree_edges


def _write_graph_stream(
    path: str | Path,
    n: int,
    tree_edges: list[tuple[int, int]],
    extra_needed: int,
    min_weight: int,
    max_weight: int,
    rng: random.Random,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree_set = set(tree_edges)
    total_candidates = n * (n - 1) // 2 - len(tree_edges)
    remaining_extra = extra_needed
    remaining_candidates = total_candidates
    first_edge = True

    with output_path.open("w", encoding="utf-8") as file:
        file.write('{\n  "n": ')
        file.write(str(n))
        file.write(',\n  "edges": [\n')

        for u, v in tree_edges:
            if not first_edge:
                file.write(",\n")
            weight = rng.randint(min_weight, max_weight)
            file.write(f"    [{u}, {v}, {weight}]")
            first_edge = False

        for u in range(n - 1):
            for v in range(u + 1, n):
                if (u, v) in tree_set:
                    continue
                if remaining_extra > 0 and rng.randrange(remaining_candidates) < remaining_extra:
                    if not first_edge:
                        file.write(",\n")
                    weight = rng.randint(min_weight, max_weight)
                    file.write(f"    [{u}, {v}, {weight}]")
                    first_edge = False
                    remaining_extra -= 1
                remaining_candidates -= 1

        file.write("\n  ]\n}\n")


def generate_connected_graph_file(
    path: str | Path,
    n: int,
    m: int,
    min_weight: int = 1,
    max_weight: int = 100,
    seed: int | None = None,
) -> None:
    rng = random.Random(seed)
    tree_edges = _generate_spanning_tree(n, rng)
    extra_needed = m - (n - 1)
    if n <= 2000:
        edge_weights = [(u, v, rng.randint(min_weight, max_weight)) for u, v in tree_edges]
        seen = set(tree_edges)
        while len(edge_weights) < m:
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edge_weights.append((a, b, rng.randint(min_weight, max_weight)))
        save_graph(path, n, edge_weights)
        return

    _write_graph_stream(path, n, tree_edges, extra_needed, min_weight, max_weight, rng)


def edge_count_for_percent(n: int, percent: int) -> int:
    max_edges = n * (n - 1) // 2
    min_edges = n - 1
    span = max_edges - min_edges
    return min_edges + (span * percent) // 100


def generate_lab_suite(
    out_dir: str | Path = Path(__file__).resolve().parent / "generated_graphs",
    seed: int | None = None,
) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for size in GRAPH_SIZES:
        for idx, percent in enumerate(DENSITY_PERCENTS):
            graph_seed = None if seed is None else seed + size * 100 + idx
            edge_count = edge_count_for_percent(size, percent)
            file_path = out_path / f"n{size}_p{percent:02d}.json"
            generate_connected_graph_file(file_path, size, edge_count, seed=graph_seed)
            generated.append(file_path)

    return generated


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.n is not None and args.m is not None:
        out_path = args.out
        if out_path is None:
            out_path = Path(__file__).resolve().parent / "generated_graphs" / f"n{args.n}_m{args.m}.json"
        generate_connected_graph_file(out_path, args.n, args.m, seed=args.seed)
        print(out_path)
    else:
        generated = generate_lab_suite(seed=args.seed)
        for path in generated:
            print(path)


if __name__ == "__main__":
    _main()
