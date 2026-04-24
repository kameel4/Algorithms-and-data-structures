from __future__ import annotations

import unittest
import csv
from pathlib import Path

from harmonic_expansion import ExpansionStep, harmonic_expansion_order, save_expansion_order_csv


class HarmonicExpansionOrderTest(unittest.TestCase):
    def test_reorders_tree_edges_from_start(self) -> None:
        tree_edges = [
            (2, 3, 1),
            (0, 1, 2),
            (1, 2, 3),
        ]

        steps = harmonic_expansion_order(4, tree_edges, start=0)

        self.assertEqual(
            [(step.parent, step.new_vertex, step.weight) for step in steps],
            [(0, 1, 2), (1, 2, 3), (2, 3, 1)],
        )
        self.assertEqual(
            [step.path_from_start for step in steps],
            [(0, 1), (0, 1, 2), (0, 1, 2, 3)],
        )

    def test_uses_selected_start_vertex(self) -> None:
        tree_edges = [
            (2, 3, 1),
            (0, 1, 2),
            (1, 2, 3),
        ]

        steps = harmonic_expansion_order(4, tree_edges, start=2)

        self.assertEqual(
            [(step.parent, step.new_vertex, step.weight) for step in steps],
            [(2, 3, 1), (2, 1, 3), (1, 0, 2)],
        )
        self.assertEqual(
            [step.path_from_start for step in steps],
            [(2, 3), (2, 1), (2, 1, 0)],
        )

    def test_disconnected_edges_raise_error(self) -> None:
        with self.assertRaises(ValueError):
            harmonic_expansion_order(4, [(0, 1, 1), (2, 3, 1)], start=0)

    def test_empty_graph_has_empty_order(self) -> None:
        self.assertEqual(harmonic_expansion_order(0, [], start=0), [])

    def test_save_expansion_order_csv(self) -> None:
        path = Path(__file__).resolve().parent / "ui_results" / "_test_expansion_order.csv"
        steps = [
            ExpansionStep(parent=0, new_vertex=2, weight=4, path_from_start=(0, 2), source_edge_index=0),
            ExpansionStep(parent=2, new_vertex=1, weight=8, path_from_start=(0, 2, 1), source_edge_index=1),
        ]

        try:
            save_expansion_order_csv(path, steps)
            with path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(rows[0]["step"], "1")
        self.assertEqual(rows[0]["parent"], "0")
        self.assertEqual(rows[0]["new_vertex"], "2")
        self.assertEqual(rows[0]["weight"], "4")
        self.assertEqual(rows[0]["path_from_start"], "0 -> 2")
        self.assertEqual(rows[1]["path_from_start"], "0 -> 2 -> 1")


if __name__ == "__main__":
    unittest.main()
