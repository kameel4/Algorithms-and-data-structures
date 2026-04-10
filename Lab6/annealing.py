from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TSPGraph:
    """In-memory representation of a TSP instance."""

    name: str
    node_labels: list[str]
    distance_matrix: np.ndarray
    coordinates: np.ndarray | None
    directed: bool
    source_path: Path

    @property
    def size(self) -> int:
        return len(self.node_labels)


def load_tsp_graph(path: str | Path) -> TSPGraph:
    """Load a TSP graph from either `.edgelist` or `.stp`."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Graph file not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".edgelist":
        return _load_edgelist_graph(source_path)
    if suffix == ".stp":
        return _load_stp_graph(source_path)

    raise ValueError(f"Unsupported graph format: {source_path.suffix}")


def run_simulated_annealing(
    graph: TSPGraph,
    *,
    initial_temperature: float,
    iterations_per_temperature: int,
    cooling_mode: str = "geometric",
    alpha: float = 0.95,
    min_temperature: float = 0.1,
    seed: int | None = None,
) -> dict[str, object]:
    """
    Run simulated annealing for a TSP graph.

    The algorithm follows the lab constraints:
    - DFS-based initial Hamiltonian cycle,
    - energy = route length,
    - neighbor = swap two random cities,
    - geometric or Cauchy cooling.
    """

    _validate_graph(graph)
    normalized_mode = _normalize_cooling_mode(cooling_mode)
    _validate_sa_params(
        initial_temperature=initial_temperature,
        iterations_per_temperature=iterations_per_temperature,
        cooling_mode=normalized_mode,
        alpha=alpha,
        min_temperature=min_temperature,
    )

    rng = np.random.default_rng(seed)
    evaluations = 0
    accepted_moves = 0
    improving_moves = 0

    start_time = time.perf_counter()

    current_route, current_length, initial_search_evaluations = _make_dfs_initial_route(graph, rng)
    evaluations += initial_search_evaluations

    best_route = current_route.copy()
    best_length = current_length
    initial_route = current_route.copy()
    initial_length = current_length

    temperature = float(initial_temperature)
    temperature_steps = 0
    cauchy_step = 1

    history_temperatures = [temperature]
    history_best_lengths = [float(best_length)]
    history_current_lengths = [float(current_length)]

    while temperature > min_temperature:
        temperature_steps += 1

        for _ in range(iterations_per_temperature):
            candidate_route = _swap_two_cities(current_route, rng)
            candidate_length = _route_length(candidate_route, graph.distance_matrix)
            evaluations += 1

            delta = candidate_length - current_length
            if delta <= 0:
                accept = True
            else:
                accept_probability = math.exp(-delta / temperature)
                accept = bool(rng.random() < accept_probability)

            if accept:
                current_route = candidate_route
                current_length = candidate_length
                accepted_moves += 1

                if current_length < best_length:
                    best_route = current_route.copy()
                    best_length = current_length
                    improving_moves += 1

        history_temperatures.append(float(temperature))
        history_best_lengths.append(float(best_length))
        history_current_lengths.append(float(current_length))

        if normalized_mode == "geometric":
            temperature *= alpha
        else:
            cauchy_step += 1
            temperature = initial_temperature / cauchy_step

    elapsed_seconds = time.perf_counter() - start_time

    return {
        "best_route_indices": _closed_cycle_indices(best_route),
        "best_route_labels": _closed_cycle_labels(best_route, graph.node_labels),
        "best_length": float(best_length),
        "initial_route_labels": _closed_cycle_labels(initial_route, graph.node_labels),
        "initial_length": float(initial_length),
        "history_temperatures": history_temperatures,
        "history_best_lengths": history_best_lengths,
        "history_current_lengths": history_current_lengths,
        "evaluations": int(evaluations),
        "temperature_steps": int(temperature_steps),
        "accepted_moves": int(accepted_moves),
        "improving_moves": int(improving_moves),
        "elapsed_seconds": float(elapsed_seconds),
        "cooling_mode": normalized_mode,
    }


def _load_edgelist_graph(path: Path) -> TSPGraph:
    edges: list[tuple[str, str, float]] = []
    seen_labels: dict[str, None] = {}

    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid edgelist row at {path}:{line_number}. "
                    "Expected: source target weight"
                )

            source, target, weight_text = parts
            try:
                weight = float(weight_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid edge weight at {path}:{line_number}: {weight_text}"
                ) from exc

            edges.append((source, target, weight))
            seen_labels.setdefault(source, None)
            seen_labels.setdefault(target, None)

    if not edges:
        raise ValueError(f"Edgelist graph is empty: {path}")

    node_labels = list(seen_labels)
    label_to_index = {label: idx for idx, label in enumerate(node_labels)}
    size = len(node_labels)

    distance_matrix = np.full((size, size), np.inf, dtype=float)
    np.fill_diagonal(distance_matrix, 0.0)

    for source, target, weight in edges:
        distance_matrix[label_to_index[source], label_to_index[target]] = weight

    return TSPGraph(
        name=path.stem,
        node_labels=node_labels,
        distance_matrix=distance_matrix,
        coordinates=None,
        directed=True,
        source_path=path,
    )


def _load_stp_graph(path: Path) -> TSPGraph:
    name = path.stem
    node_count: int | None = None
    distance_matrix: np.ndarray | None = None
    coordinates: np.ndarray | None = None

    current_section: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line or line == "EOF":
                continue

            if line.startswith("Section "):
                current_section = line.removeprefix("Section ").strip().lower()
                continue

            if line == "End":
                current_section = None
                continue

            parts = line.split()
            keyword = parts[0]

            if keyword == "Name" and len(parts) >= 2:
                name = parts[1].strip('"')
                continue

            if current_section == "graph":
                if keyword == "Nodes":
                    node_count = int(parts[1])
                    distance_matrix = np.full((node_count, node_count), np.inf, dtype=float)
                    np.fill_diagonal(distance_matrix, 0.0)
                    continue

                if keyword == "E":
                    if distance_matrix is None:
                        raise ValueError(
                            f"Found edge before node count in {path}:{line_number}"
                        )

                    source = int(parts[1]) - 1
                    target = int(parts[2]) - 1
                    weight = float(parts[3])
                    distance_matrix[source, target] = weight
                    distance_matrix[target, source] = weight
                    continue

            if current_section == "coordinates":
                if keyword == "DD":
                    if node_count is None:
                        raise ValueError(
                            f"Found coordinates before node count in {path}:{line_number}"
                        )
                    if coordinates is None:
                        coordinates = np.full((node_count, 2), np.nan, dtype=float)

                    vertex = int(parts[1]) - 1
                    x_coord = float(parts[2])
                    y_coord = float(parts[3])
                    coordinates[vertex] = (x_coord, y_coord)
                    continue

    if node_count is None or distance_matrix is None:
        raise ValueError(f"Could not parse graph section from STP file: {path}")

    node_labels = [str(index) for index in range(1, node_count + 1)]
    if coordinates is not None and np.isnan(coordinates).any():
        coordinates = None

    return TSPGraph(
        name=name,
        node_labels=node_labels,
        distance_matrix=distance_matrix,
        coordinates=coordinates,
        directed=False,
        source_path=path,
    )


def _validate_graph(graph: TSPGraph) -> None:
    if not isinstance(graph, TSPGraph):
        raise TypeError("graph must be a TSPGraph instance")

    if graph.size < 2:
        raise ValueError("graph must contain at least two vertices")

    matrix = np.asarray(graph.distance_matrix, dtype=float)
    if matrix.shape != (graph.size, graph.size):
        raise ValueError("distance_matrix shape must match the number of nodes")


def _validate_sa_params(
    *,
    initial_temperature: float,
    iterations_per_temperature: int,
    cooling_mode: str,
    alpha: float,
    min_temperature: float,
) -> None:
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be > 0")
    if min_temperature < 0:
        raise ValueError("min_temperature must be >= 0")
    if initial_temperature <= min_temperature:
        raise ValueError("initial_temperature must be greater than min_temperature")
    if iterations_per_temperature <= 0:
        raise ValueError("iterations_per_temperature must be > 0")
    if cooling_mode not in {"geometric", "cauchy"}:
        raise ValueError("cooling_mode must be either 'geometric' or 'cauchy'")
    if cooling_mode == "geometric" and not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1 for geometric cooling")


def _normalize_cooling_mode(cooling_mode: str) -> str:
    normalized = str(cooling_mode).strip().lower()
    aliases = {
        "geometric": "geometric",
        "base": "geometric",
        "basic": "geometric",
        "cauchy": "cauchy",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported cooling mode: {cooling_mode}")
    return aliases[normalized]


def _make_dfs_initial_route(
    graph: TSPGraph,
    rng: np.random.Generator,
    *,
    max_states: int | None = None,
) -> tuple[np.ndarray, float, int]:
    distance_matrix = np.asarray(graph.distance_matrix, dtype=float)
    ordered_neighbors = _build_dfs_neighbor_order(distance_matrix, rng)
    out_degrees = np.array([len(neighbors) for neighbors in ordered_neighbors], dtype=int)

    if np.any(out_degrees == 0):
        raise RuntimeError(
            "Could not find a Hamiltonian cycle with DFS because at least one vertex "
            "has no outgoing edges."
        )

    search_limit = max_states or max(100_000, graph.size * graph.size * 25)
    route = np.empty(graph.size, dtype=int)
    visited = np.zeros(graph.size, dtype=bool)
    explored_states = 0
    search_exhausted = False

    def dfs(depth: int, current: int) -> bool:
        nonlocal explored_states, search_exhausted

        explored_states += 1
        if explored_states > search_limit:
            search_exhausted = True
            return False

        if depth == graph.size:
            return math.isfinite(distance_matrix[current, route[0]])

        for next_vertex in ordered_neighbors[current]:
            if visited[next_vertex]:
                continue

            if depth == graph.size - 1 and not math.isfinite(distance_matrix[next_vertex, route[0]]):
                continue

            visited[next_vertex] = True
            route[depth] = next_vertex

            if dfs(depth + 1, next_vertex):
                return True

            visited[next_vertex] = False

            if search_exhausted:
                return False

        return False

    for start_vertex in _ordered_start_vertices(out_degrees, rng):
        visited.fill(False)
        visited[start_vertex] = True
        route[0] = start_vertex

        if dfs(1, start_vertex):
            cycle = route.copy()
            return cycle, _route_length(cycle, distance_matrix), explored_states

        if search_exhausted:
            break

    if search_exhausted:
        raise RuntimeError(
            "Could not find a Hamiltonian cycle with DFS within the search budget. "
            "The graph may be too constrained or may not contain any feasible TSP tour."
        )

    raise RuntimeError(
        "Could not find a Hamiltonian cycle with DFS. "
        "The graph may not contain any feasible TSP tour."
    )


def _build_dfs_neighbor_order(
    distance_matrix: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    finite_edges = np.isfinite(distance_matrix)
    np.fill_diagonal(finite_edges, False)
    out_degrees = finite_edges.sum(axis=1).astype(int)
    ordered_neighbors: list[np.ndarray] = []

    for vertex in range(distance_matrix.shape[0]):
        neighbors = np.flatnonzero(finite_edges[vertex])
        if neighbors.size <= 1:
            ordered_neighbors.append(neighbors)
            continue

        # Prefer constrained neighbors first, then lighter edges, then random tie-breaking.
        tie_breaker = rng.permutation(neighbors.size)
        order = np.lexsort((tie_breaker, distance_matrix[vertex, neighbors], out_degrees[neighbors]))
        ordered_neighbors.append(neighbors[order])

    return ordered_neighbors


def _ordered_start_vertices(out_degrees: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vertices = np.arange(out_degrees.size, dtype=int)
    if vertices.size <= 1:
        return vertices

    tie_breaker = rng.permutation(vertices.size)
    order = np.lexsort((tie_breaker, out_degrees))
    return vertices[order]


def _swap_two_cities(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if route.size < 2:
        return route.copy()

    left, right = rng.choice(route.size, size=2, replace=False)
    candidate = route.copy()
    candidate[left], candidate[right] = candidate[right], candidate[left]
    return candidate


def _route_length(route: np.ndarray, distance_matrix: np.ndarray) -> float:
    ordered = np.asarray(route, dtype=int)
    next_vertices = np.roll(ordered, -1)
    weights = distance_matrix[ordered, next_vertices]
    if np.isinf(weights).any():
        return math.inf
    return float(np.sum(weights))


def _closed_cycle_labels(route: np.ndarray, node_labels: Iterable[str]) -> list[str]:
    labels = list(node_labels)
    cycle = [labels[index] for index in route.tolist()]
    cycle.append(cycle[0])
    return cycle


def _closed_cycle_indices(route: np.ndarray) -> list[int]:
    cycle = route.astype(int).tolist()
    cycle.append(cycle[0])
    return cycle
