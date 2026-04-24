from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from graph_utils import load_graph
    from kruskal import kruskal_mst
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from graph_utils import load_graph
    from kruskal import kruskal_mst


Edge = tuple[int, int, int]
PathLike = str | Path


@dataclass(frozen=True)
class ExpansionStep:
    parent: int
    new_vertex: int
    weight: int
    path_from_start: tuple[int, ...]
    source_edge_index: int


def save_expansion_order_csv(path: PathLike, steps: list[ExpansionStep]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "parent", "new_vertex", "weight", "path_from_start", "source_edge_index"],
        )
        writer.writeheader()
        for number, step in enumerate(steps, 1):
            writer.writerow(
                {
                    "step": number,
                    "parent": step.parent,
                    "new_vertex": step.new_vertex,
                    "weight": step.weight,
                    "path_from_start": " -> ".join(map(str, step.path_from_start)),
                    "source_edge_index": step.source_edge_index,
                }
            )


def harmonic_expansion_order(
    n: int,
    tree_edges: list[Edge],
    start: int = 0,
) -> list[ExpansionStep]:
    if n == 0:
        return []
    if start < 0 or start >= n:
        raise ValueError(f"Start vertex {start} is outside 0..{n - 1}")

    for u, v, _ in tree_edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError(f"Edge ({u}, {v}) contains a vertex outside 0..{n - 1}")

    known = {start}
    path_by_vertex: dict[int, tuple[int, ...]] = {start: (start,)}
    remaining = list(enumerate(tree_edges))
    steps: list[ExpansionStep] = []

    while len(known) < n:
        added_index: int | None = None

        for index, (source_edge_index, (u, v, weight)) in enumerate(remaining):
            u_is_known = u in known
            v_is_known = v in known
            if u_is_known == v_is_known:
                continue

            parent, new_vertex = (u, v) if u_is_known else (v, u)
            path = (*path_by_vertex[parent], new_vertex)
            steps.append(
                ExpansionStep(
                    parent=parent,
                    new_vertex=new_vertex,
                    weight=weight,
                    path_from_start=path,
                    source_edge_index=source_edge_index,
                )
            )
            known.add(new_vertex)
            path_by_vertex[new_vertex] = path
            added_index = index
            break

        if added_index is None:
            missing = sorted(set(range(n)) - known)
            raise ValueError(
                "Selected edges do not connect every vertex from "
                f"start {start}; unreachable vertices: {missing}"
            )

        remaining.pop(added_index)

    return steps


def format_expansion_order(
    steps: list[ExpansionStep],
    limit: int | None = None,
) -> list[str]:
    visible_steps = steps if limit is None else steps[:limit]
    lines = [
        (
            f"{number}. {step.parent} -> {step.new_vertex} "
            f"(w={step.weight}), путь: {' -> '.join(map(str, step.path_from_start))}"
        )
        for number, step in enumerate(visible_steps, 1)
    ]
    if limit is not None and len(steps) > limit:
        lines.append(f"... показано {limit} из {len(steps)} шагов")
    return lines


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    n, edges = load_graph(args.path)
    total, mst = kruskal_mst(n, edges)
    steps = harmonic_expansion_order(n, mst, start=args.start)

    print(f"graph: {args.path.name}")
    print(f"start: {args.start}")
    print(f"mst_weight: {total}")
    print(f"steps: {len(steps)}")
    for line in format_expansion_order(steps, limit=args.limit):
        print(line)


if __name__ == "__main__":
    _main()
