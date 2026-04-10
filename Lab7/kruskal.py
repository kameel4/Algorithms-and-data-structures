from __future__ import annotations

import sys

from graph_utils import load_graph


def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> tuple[int, list[tuple[int, int, int]]]:
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

    for u, v, w in sorted(edges, key=lambda edge: edge[2]):
        if union(u, v):
            total += w
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break

    return total, mst


if __name__ == "__main__":
    n, edges = load_graph(sys.argv[1])
    total, mst = kruskal_mst(n, edges)
    print(total)
    print(len(mst))
