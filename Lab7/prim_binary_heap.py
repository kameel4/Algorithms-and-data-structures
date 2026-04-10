from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

try:
    from graph_utils import load_graph
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from graph_utils import load_graph


def prim_binary_heap_mst(n: int, edges: list[tuple[int, int, int]]) -> tuple[int, list[tuple[int, int, int]]]:
    if n == 0:
        return 0, []

    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    used = [False] * n
    mst: list[tuple[int, int, int]] = []
    total = 0
    heap: list[tuple[int, int, int]] = [(0, 0, -1)]

    while heap and len(mst) < n - 1:
        w, v, parent = heapq.heappop(heap)
        if used[v]:
            continue
        used[v] = True
        if parent != -1:
            mst.append((parent, v, w))
            total += w
        for to, cost in adj[v]:
            if not used[to]:
                heapq.heappush(heap, (cost, to, v))

    return total, mst


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    n, edges = load_graph(args.path)
    total, mst = prim_binary_heap_mst(n, edges)
    print(total)
    print(len(mst))


if __name__ == "__main__":
    _main()
