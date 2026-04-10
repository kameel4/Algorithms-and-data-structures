from __future__ import annotations

import sys

from graph_utils import load_graph


class FibNode:
    def __init__(self, key: int, vertex: int) -> None:
        self.key = key
        self.vertex = vertex
        self.degree = 0
        self.mark = False
        self.parent: FibNode | None = None
        self.child: FibNode | None = None
        self.left = self
        self.right = self


class FibHeap:
    def __init__(self) -> None:
        self.min_node: FibNode | None = None
        self.size = 0

    def insert(self, key: int, vertex: int) -> FibNode:
        node = FibNode(key, vertex)
        self._add_root(node)
        self.size += 1
        return node

    def extract_min(self) -> FibNode | None:
        z = self.min_node
        if z is None:
            return None

        if z.child is not None:
            for child in list(self._iterate(z.child)):
                child.parent = None
                child.mark = False
                child.left = child
                child.right = child
                self._add_root(child)

        if z.right is z:
            self.min_node = None
        else:
            z.left.right = z.right
            z.right.left = z.left
            self.min_node = z.right
            self._consolidate()

        self.size -= 1
        return z

    def decrease_key(self, node: FibNode, new_key: int) -> None:
        node.key = new_key
        parent = node.parent
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        if self.min_node is None or node.key < self.min_node.key:
            self.min_node = node

    def _add_root(self, node: FibNode) -> None:
        node.parent = None
        if self.min_node is None:
            node.left = node
            node.right = node
            self.min_node = node
            return

        node.left = self.min_node
        node.right = self.min_node.right
        self.min_node.right.left = node
        self.min_node.right = node
        if node.key < self.min_node.key:
            self.min_node = node

    def _consolidate(self) -> None:
        roots = list(self._iterate(self.min_node))
        trees: dict[int, FibNode] = {}
        self.min_node = None

        for node in roots:
            node.left = node
            node.right = node
            degree = node.degree
            while degree in trees:
                other = trees.pop(degree)
                if other.key < node.key:
                    node, other = other, node
                self._link(other, node)
                degree = node.degree
            trees[degree] = node

        for node in trees.values():
            self._add_root(node)

    def _link(self, child: FibNode, parent: FibNode) -> None:
        child.parent = parent
        child.mark = False
        if parent.child is None:
            child.left = child
            child.right = child
            parent.child = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right.left = child
            parent.child.right = child
        parent.degree += 1

    def _cut(self, node: FibNode, parent: FibNode) -> None:
        if node.right is node:
            parent.child = None
        else:
            node.left.right = node.right
            node.right.left = node.left
            if parent.child is node:
                parent.child = node.right
        parent.degree -= 1
        node.left = node
        node.right = node
        node.mark = False
        self._add_root(node)

    def _cascading_cut(self, node: FibNode) -> None:
        parent = node.parent
        if parent is None:
            return
        if not node.mark:
            node.mark = True
            return
        self._cut(node, parent)
        self._cascading_cut(parent)

    def _iterate(self, start: FibNode | None):
        if start is None:
            return
        node = start
        while True:
            yield node
            node = node.right
            if node is start:
                break


def prim_fibonacci_heap_mst(
    n: int, edges: list[tuple[int, int, int]]
) -> tuple[int, list[tuple[int, int, int]]]:
    adjacency = [[] for _ in range(n)]
    for u, v, w in edges:
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))

    heap = FibHeap()
    nodes: list[FibNode | None] = [None] * n
    parent = [-1] * n
    used = [False] * n
    total = 0
    mst: list[tuple[int, int, int]] = []

    for start in range(n):
        if used[start]:
            continue
        nodes[start] = heap.insert(0, start)

        while heap.size:
            node = heap.extract_min()
            if node is None:
                break
            u = node.vertex
            if used[u]:
                continue

            used[u] = True
            nodes[u] = None
            if parent[u] != -1:
                total += node.key
                mst.append((parent[u], u, node.key))

            for v, w in adjacency[u]:
                if used[v]:
                    continue
                if nodes[v] is None:
                    parent[v] = u
                    nodes[v] = heap.insert(w, v)
                elif w < nodes[v].key:
                    parent[v] = u
                    heap.decrease_key(nodes[v], w)

    return total, mst


if __name__ == "__main__":
    n, edges = load_graph(sys.argv[1])
    total, mst = prim_fibonacci_heap_mst(n, edges)
    print(total)
    print(len(mst))
