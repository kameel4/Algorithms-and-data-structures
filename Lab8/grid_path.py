from __future__ import annotations

import csv
import json
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from subprocess import TimeoutExpired
from time import perf_counter
from typing import Any

from models import SearchMetrics, SearchResult


Coord = tuple[int, int]
GridPathSolution = dict[str, object]

ALGORITHMS: dict[str, str] = {
    "baseline": "DFS Backtracking",
    "warnsdorff": "Warnsdorff Heuristic",
    "connectivity": "Connectivity Pruning",
    "backjumping": "Conflict-Directed Backjumping",
}

_NEIGHBOR_DELTAS: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True, slots=True)
class GridPathProblem:
    rows: int
    cols: int
    start: Coord
    end: Coord
    blocked: frozenset[Coord] = frozenset()


@dataclass(frozen=True, slots=True)
class _SearchConfig:
    ordering: str
    connectivity_pruning: bool
    backjumping: bool


_CONFIGS: dict[str, _SearchConfig] = {
    "baseline": _SearchConfig(ordering="fixed", connectivity_pruning=False, backjumping=False),
    "warnsdorff": _SearchConfig(ordering="warnsdorff", connectivity_pruning=False, backjumping=False),
    "connectivity": _SearchConfig(ordering="fixed", connectivity_pruning=True, backjumping=False),
    "backjumping": _SearchConfig(ordering="warnsdorff", connectivity_pruning=True, backjumping=True),
}

_CPP_SOLVER = Path(__file__).with_name("grid_path_solver.exe")


def solve_problem(
    problem: GridPathProblem,
    algorithm: str,
    max_solutions: int = 1,
) -> SearchResult[GridPathSolution]:
    started_at = perf_counter()

    if algorithm not in ALGORITHMS:
        return _finalize(
            SearchResult(
                algorithm=algorithm,
                found=False,
                solution=None,
                message=f"Unknown algorithm: {algorithm}",
            ),
            started_at,
        )

    cpp_result = _solve_with_cpp(problem, algorithm, max_solutions)
    if cpp_result is not None:
        return cpp_result

    try:
        solver = _GridPathSolver(problem, algorithm, max_solutions)
    except ValueError as exc:
        return _finalize(
            SearchResult(
                algorithm=algorithm,
                found=False,
                solution=None,
                message=str(exc),
            ),
            started_at,
        )

    result = solver.solve()
    result.algorithm = algorithm
    return _finalize(result, started_at)


def export_paths(
    problem: GridPathProblem,
    algorithm: str,
    output_path: str | Path,
) -> SearchResult[GridPathSolution]:
    cpp_result = _solve_with_cpp(problem, algorithm, max_solutions=0, export_path=output_path)
    if cpp_result is not None:
        return cpp_result

    started_at = perf_counter()
    result = solve_problem(problem, algorithm, max_solutions=0)
    if not result.found or result.solution is None:
        return result

    paths = result.solution.get("paths", [])
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["algorithm", result.algorithm])
        writer.writerow(["rows", problem.rows])
        writer.writerow(["cols", problem.cols])
        writer.writerow(["start", problem.start[0], problem.start[1]])
        writer.writerow(["end", problem.end[0], problem.end[1]])
        writer.writerow(["blocked", " ".join(f"{row},{col}" for row, col in sorted(problem.blocked))])
        writer.writerow(["path_count", len(paths)])
        writer.writerow([])
        writer.writerow(["path_index", "step", "row", "col"])
        for path_index, path in enumerate(paths, start=1):
            for step, (row, col) in enumerate(path):  # type: ignore[misc]
                writer.writerow([path_index, step, row, col])

    result.metrics.elapsed_ms = (perf_counter() - started_at) * 1000
    return result


def compare_algorithms(
    problem: GridPathProblem,
    algorithms: list[str] | None = None,
) -> list[SearchResult[GridPathSolution]]:
    selected = algorithms or list(ALGORITHMS)
    return [solve_problem(problem, algorithm) for algorithm in selected]


def _solve_with_cpp(
    problem: GridPathProblem,
    algorithm: str,
    max_solutions: int,
    export_path: str | Path | None = None,
) -> SearchResult[GridPathSolution] | None:
    if not _CPP_SOLVER.exists():
        return None
    if problem.rows * problem.cols > 64:
        return None

    blocked = ";".join(f"{row},{col}" for row, col in sorted(problem.blocked))
    command = [
        str(_CPP_SOLVER),
        "--rows",
        str(problem.rows),
        "--cols",
        str(problem.cols),
        "--start",
        str(problem.start[0]),
        str(problem.start[1]),
        "--end",
        str(problem.end[0]),
        str(problem.end[1]),
        "--algorithm",
        algorithm,
        "--max-solutions",
        str(max_solutions),
    ]
    if blocked:
        command.extend(["--blocked", blocked])
    if export_path is not None:
        command.extend(["--export", str(export_path)])

    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=None,
            check=False,
        )
    except (OSError, TimeoutExpired):
        return None

    stdout = completed.stdout.strip()
    if not stdout:
        return SearchResult(
            algorithm=algorithm,
            found=False,
            solution=None,
            message=completed.stderr.strip() or "C++ solver did not return a result.",
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return SearchResult(
            algorithm=algorithm,
            found=False,
            solution=None,
            message=f"C++ solver returned invalid output: {stdout[:200]}",
        )

    return _cpp_payload_to_result(payload, algorithm)


def _cpp_payload_to_result(payload: dict[str, Any], algorithm: str) -> SearchResult[GridPathSolution]:
    metrics_payload = payload.get("metrics") or {}
    metrics = SearchMetrics(
        elapsed_ms=float(metrics_payload.get("elapsed_ms", 0.0)),
        nodes_expanded=int(metrics_payload.get("nodes_expanded", 0)),
        states_generated=int(metrics_payload.get("states_generated", 0)),
        pruned_states=int(metrics_payload.get("pruned_states", 0)),
        max_frontier=int(metrics_payload.get("max_frontier", 0)),
        max_depth=int(metrics_payload.get("max_depth", 0)),
        peak_memory_kb=float(metrics_payload.get("peak_memory_kb", 0.0)),
    )

    first_path = [tuple(point) for point in payload.get("first_path", [])]
    found = bool(payload.get("found", False))
    solution: GridPathSolution | None = None
    if found:
        solution = {
            "path": first_path,
            "path_count": int(payload.get("path_count", 0)),
            "exhaustive": bool(payload.get("exhaustive", False)),
            "cost": int(payload.get("cost", max(len(first_path) - 1, 0))),
            "visited_cells": int(payload.get("visited_cells", len(first_path))),
            "engine": "cpp",
        }

    return SearchResult(
        algorithm=algorithm,
        found=found,
        solution=solution,
        metrics=metrics,
        message=str(payload.get("message", "")),
    )


class _GridPathSolver:
    def __init__(self, problem: GridPathProblem, algorithm: str, max_solutions: int) -> None:
        self.problem = problem
        self.config = _CONFIGS[algorithm]
        self.metrics = SearchMetrics(max_frontier=1)
        self.solution_limit = None if max_solutions <= 0 else max_solutions
        self.solutions: list[list[Coord]] = []
        self._validate_problem()

        self.free_cells = frozenset(
            (row, col)
            for row in range(problem.rows)
            for col in range(problem.cols)
            if (row, col) not in problem.blocked
        )
        self.adjacency = {
            cell: tuple(neighbor for neighbor in self._neighbors(cell) if neighbor in self.free_cells)
            for cell in self.free_cells
        }
        self.path: list[Coord] = [problem.start]
        self.visited: set[Coord] = {problem.start}
        self.depth_of: dict[Coord, int] = {problem.start: 0}

    def solve(self) -> SearchResult[GridPathSolution]:
        early_failure = self._check_static_constraints()
        if early_failure is not None:
            return SearchResult(
                algorithm="",
                found=False,
                solution=None,
                metrics=self.metrics,
                message=early_failure,
            )

        if len(self.free_cells) == 1:
            self._record_solution()
            return self._success(exhaustive=True)

        if self.config.connectivity_pruning:
            conflict = self._structural_conflict(self.problem.start)
            if conflict is not None:
                return SearchResult(
                    algorithm="",
                    found=False,
                    solution=None,
                    metrics=self.metrics,
                    message="Initial state is structurally infeasible.",
                )

        if self.config.backjumping and self.solution_limit == 1:
            stopped, _ = self._dfs_backjump(self.problem.start, 0)
            limit_reached = stopped == "stop"
        else:
            limit_reached = self._dfs(self.problem.start, 0)

        if self.solutions:
            return self._success(exhaustive=not limit_reached)

        return SearchResult(
            algorithm="",
            found=False,
            solution=None,
            metrics=self.metrics,
            message="No Hamiltonian path was found for the selected configuration.",
        )

    def _validate_problem(self) -> None:
        if self.problem.rows <= 0 or self.problem.cols <= 0:
            raise ValueError("Grid dimensions must be positive.")

        for point in (self.problem.start, self.problem.end):
            if not self._in_bounds(point):
                raise ValueError("Start and end cells must be inside the grid.")

        invalid_blocks = [cell for cell in self.problem.blocked if not self._in_bounds(cell)]
        if invalid_blocks:
            raise ValueError("Blocked cells must be inside the grid.")

        if self.problem.start in self.problem.blocked or self.problem.end in self.problem.blocked:
            raise ValueError("Start and end cells cannot be blocked.")

    def _check_static_constraints(self) -> str | None:
        free_count = len(self.free_cells)
        if free_count == 0:
            return "There are no free cells in the grid."

        if self.problem.start not in self.free_cells or self.problem.end not in self.free_cells:
            return "Start and end cells must be free."

        if self.problem.start == self.problem.end and free_count > 1:
            return "Start and end may coincide only for a single free cell."

        if not self._is_connected(self.free_cells, self.problem.start):
            return "Free cells must form a single connected component."

        even_count = sum(1 for row, col in self.free_cells if (row + col) % 2 == 0)
        odd_count = free_count - even_count
        parity_delta = even_count - odd_count
        start_even = (self.problem.start[0] + self.problem.start[1]) % 2 == 0
        end_even = (self.problem.end[0] + self.problem.end[1]) % 2 == 0

        if abs(parity_delta) > 1:
            return "Parity constraint makes a Hamiltonian path impossible."
        if parity_delta == 0 and start_even == end_even:
            return "For an even number of free cells, endpoints must have opposite colors."
        if parity_delta == 1 and not (start_even and end_even):
            return "Parity requires both endpoints on even-colored cells."
        if parity_delta == -1 and (start_even or end_even):
            return "Parity requires both endpoints on odd-colored cells."
        return None

    def _dfs(self, current: Coord, depth: int) -> bool:
        self.metrics.max_depth = max(self.metrics.max_depth, depth)
        self.metrics.max_frontier = max(self.metrics.max_frontier, len(self.path))

        if len(self.path) == len(self.free_cells):
            if current == self.problem.end:
                return self._record_solution()
            self.metrics.pruned_states += 1
            return False
        if current == self.problem.end:
            self.metrics.pruned_states += 1
            return False

        self.metrics.nodes_expanded += 1
        candidates = self._ordered_candidates(current)
        if not candidates:
            self.metrics.pruned_states += 1
            return False

        for next_cell in candidates:
            self.metrics.states_generated += 1
            self._push(next_cell)
            if self.config.connectivity_pruning and self._structural_conflict(next_cell) is not None:
                self._pop(next_cell)
                continue
            limit_reached = self._dfs(next_cell, depth + 1)
            self._pop(next_cell)
            if limit_reached:
                return True
        return False

    def _dfs_backjump(self, current: Coord, depth: int) -> tuple[str, set[int]]:
        self.metrics.max_depth = max(self.metrics.max_depth, depth)
        self.metrics.max_frontier = max(self.metrics.max_frontier, len(self.path))

        if len(self.path) == len(self.free_cells):
            if current == self.problem.end:
                if self._record_solution():
                    return "stop", set()
                return "exhausted", set()
            self.metrics.pruned_states += 1
            return "jump", {depth}
        if current == self.problem.end:
            self.metrics.pruned_states += 1
            return "jump", {depth}

        conflict = self._structural_conflict(current)
        if conflict is not None:
            return "jump", conflict

        self.metrics.nodes_expanded += 1
        candidates = self._ordered_candidates(current)
        if not candidates:
            self.metrics.pruned_states += 1
            return "jump", {depth}

        absorbed_conflict: set[int] = set()
        for next_cell in candidates:
            self.metrics.states_generated += 1
            self._push(next_cell)
            status, child_conflict = self._dfs_backjump(next_cell, depth + 1)
            self._pop(next_cell)
            if status == "stop":
                return "stop", set()
            if status == "exhausted":
                continue
            if depth in child_conflict:
                absorbed_conflict.update(child_conflict - {depth})
                continue
            absorbed_conflict.update(child_conflict)

        return "jump", absorbed_conflict or {depth}

    def _ordered_candidates(self, current: Coord) -> list[Coord]:
        candidates = [neighbor for neighbor in self.adjacency[current] if neighbor not in self.visited]
        if self.config.ordering == "fixed":
            return candidates

        last_depth = len(self.path)

        def key(cell: Coord) -> tuple[int, int, int, int]:
            onward_degree = sum(
                1
                for neighbor in self.adjacency[cell]
                if neighbor not in self.visited and neighbor != current
            )
            end_penalty = 1 if cell == self.problem.end and last_depth < len(self.free_cells) - 1 else 0
            distance_to_end = abs(cell[0] - self.problem.end[0]) + abs(cell[1] - self.problem.end[1])
            return (end_penalty, onward_degree, distance_to_end, cell[0] * self.problem.cols + cell[1])

        candidates.sort(key=key)
        return candidates

    def _structural_conflict(self, current: Coord) -> set[int] | None:
        remaining = self.free_cells - self.visited
        current_depth = self.depth_of[current]
        if not remaining:
            if current == self.problem.end:
                return None
            self.metrics.pruned_states += 1
            return {current_depth}
        if current == self.problem.end:
            self.metrics.pruned_states += 1
            return {current_depth}

        relevant = remaining | {current}
        reachable = self._reachable_from(current, relevant)
        if reachable != relevant:
            self.metrics.pruned_states += 1
            isolated = relevant - reachable
            return self._build_conflict_set(isolated | {current})

        conflict_nodes: set[Coord] = set()
        for cell in relevant:
            degree = sum(1 for neighbor in self.adjacency[cell] if neighbor in relevant)
            if cell in {current, self.problem.end}:
                if degree == 0:
                    conflict_nodes.add(cell)
            elif degree < 2:
                conflict_nodes.add(cell)

        if conflict_nodes:
            self.metrics.pruned_states += 1
            return self._build_conflict_set(conflict_nodes | {current})
        return None

    def _build_conflict_set(self, cells: set[Coord]) -> set[int]:
        conflict = set()
        for cell in cells:
            if cell in self.depth_of:
                conflict.add(self.depth_of[cell])
            for neighbor in self.adjacency.get(cell, ()):
                if neighbor in self.depth_of:
                    conflict.add(self.depth_of[neighbor])
        return conflict or {len(self.path) - 1}

    def _reachable_from(self, start: Coord, allowed: set[Coord]) -> set[Coord]:
        seen = {start}
        queue_cells = deque([start])
        while queue_cells:
            current = queue_cells.popleft()
            for neighbor in self.adjacency[current]:
                if neighbor not in allowed or neighbor in seen:
                    continue
                seen.add(neighbor)
                queue_cells.append(neighbor)
        return seen

    def _is_connected(self, allowed: frozenset[Coord], start: Coord) -> bool:
        if start not in allowed:
            return False
        return len(self._reachable_from(start, set(allowed))) == len(allowed)

    def _push(self, cell: Coord) -> None:
        self.path.append(cell)
        self.visited.add(cell)
        self.depth_of[cell] = len(self.path) - 1

    def _pop(self, cell: Coord) -> None:
        self.path.pop()
        self.visited.remove(cell)
        self.depth_of.pop(cell, None)

    def _record_solution(self) -> bool:
        if self.solution_limit is None or len(self.solutions) < self.solution_limit:
            self.solutions.append(self.path.copy())
        return self.solution_limit is not None and len(self.solutions) >= self.solution_limit

    def _success(self, exhaustive: bool) -> SearchResult[GridPathSolution]:
        first_path = self.solutions[0]
        message = f"Found {len(self.solutions)} Hamiltonian path(s)."
        if not exhaustive:
            message += " Search stopped after reaching the configured limit."
        return SearchResult(
            algorithm="",
            found=True,
            solution={
                "path": first_path,
                "paths": self.solutions,
                "path_count": len(self.solutions),
                "exhaustive": exhaustive,
                "cost": len(first_path) - 1,
                "visited_cells": len(first_path),
            },
            metrics=self.metrics,
            message=message,
        )

    def _neighbors(self, cell: Coord) -> tuple[Coord, ...]:
        row, col = cell
        neighbors: list[Coord] = []
        for d_row, d_col in _NEIGHBOR_DELTAS:
            candidate = (row + d_row, col + d_col)
            if self._in_bounds(candidate):
                neighbors.append(candidate)
        return tuple(neighbors)

    def _in_bounds(self, cell: Coord) -> bool:
        row, col = cell
        return 0 <= row < self.problem.rows and 0 <= col < self.problem.cols


def _finalize(
    result: SearchResult[GridPathSolution],
    started_at: float,
) -> SearchResult[GridPathSolution]:
    result.metrics.elapsed_ms = (perf_counter() - started_at) * 1000
    result.metrics.peak_memory_kb = 0.0
    return result


__all__ = ["ALGORITHMS", "GridPathProblem", "compare_algorithms", "export_paths", "solve_problem"]
