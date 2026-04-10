from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from graph_utils import load_graph
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from graph_utils import load_graph


def boruvka_mst(n: int, edges: list[tuple[int, int, int]]) -> tuple[int, list[tuple[int, int, int]]]:
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    total = 0
    mst: list[tuple[int, int, int]] = []
    components = n

    while components > 1:
        best: dict[int, tuple[int, int, int]] = {}
        for u, v, w in edges:
            ru = find(u)
            rv = find(v)
            if ru == rv:
                continue
            edge = (u, v, w)
            if ru not in best or w < best[ru][2]:
                best[ru] = edge
            if rv not in best or w < best[rv][2]:
                best[rv] = edge

        if not best:
            break

        for u, v, w in best.values():
            if union(u, v):
                total += w
                mst.append((u, v, w))
                components -= 1
                if components == 1:
                    break

    return total, mst


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    n, edges = load_graph(args.path)
    total, mst = boruvka_mst(n, edges)
    print(total)
    print(len(mst))


if __name__ == "__main__":
    _main()
