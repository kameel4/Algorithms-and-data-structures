from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .annealing import TSPGraph, load_tsp_graph
except ImportError:
    from annealing import TSPGraph, load_tsp_graph


DEFAULT_ITERATIONS = 100
DEFAULT_INITIAL_PHEROMONE = 1.0
DEFAULT_ELITE_ANTS = 5


def run_ant_colony(
    graph: TSPGraph,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    pheromone_importance: float = 1.0,
    distance_importance: float = 2.0,
    pheromone_deposit: float = 100.0,
    evaporation_intensity: float = 3.0,
    mode: str = "basic",
    ant_count: int | None = None,
    elite_ants: int = DEFAULT_ELITE_ANTS,
    initial_pheromone: float = DEFAULT_INITIAL_PHEROMONE,
    seed: int | None = None,
) -> dict[str, object]:
    """
    Run Ant System or Elitist Ant System for a TSP graph.

    The implementation follows the currently fixed lab assumptions:
    - ants are distributed uniformly across start vertices;
    - transition probabilities use pheromone importance and inverse distance importance;
    - pheromone evaporation intensity is provided on a 0..10 scale.
    """

    _validate_graph(graph)
    normalized_mode = _normalize_mode(mode)
    _validate_ant_params(
        iterations=iterations,
        pheromone_importance=pheromone_importance,
        distance_importance=distance_importance,
        pheromone_deposit=pheromone_deposit,
        evaporation_intensity=evaporation_intensity,
        ant_count=ant_count,
        elite_ants=elite_ants,
        initial_pheromone=initial_pheromone,
    )

    rng = np.random.default_rng(seed)
    evaporation_rate = evaporation_intensity / 10.0
    ant_count = ant_count or graph.size

    complete_graph = _is_complete_graph(graph)
    neighbor_lists = None if complete_graph else _build_neighbor_lists(graph.distance_matrix)
    pheromone = _initialize_pheromone_matrix(graph, initial_pheromone)
    heuristic = _build_weighted_heuristic_matrix(
        graph.distance_matrix,
        distance_importance=distance_importance,
    )

    best_route: np.ndarray | None = None
    best_length = math.inf
    best_iteration = 0

    evaluations = 0
    successful_tours = 0
    failed_tours = 0

    history_iteration_best_lengths: list[float] = []
    history_global_best_lengths: list[float] = []
    history_successful_ants: list[int] = []

    start_time = time.perf_counter()

    for iteration_index in range(1, iterations + 1):
        start_vertices = _assign_ant_starts(
            graph.size,
            ant_count,
            iteration_index=iteration_index,
            rng=rng,
        )
        iteration_routes: list[np.ndarray] = []
        iteration_lengths: list[float] = []

        for start_vertex in start_vertices:
            route = _construct_ant_route(
                graph,
                pheromone,
                heuristic,
                start_vertex=int(start_vertex),
                pheromone_importance=pheromone_importance,
                complete_graph=complete_graph,
                neighbor_lists=neighbor_lists,
                rng=rng,
            )

            if route is None:
                failed_tours += 1
                continue

            route_length = _route_length(route, graph.distance_matrix)
            if not math.isfinite(route_length):
                failed_tours += 1
                continue

            evaluations += 1
            successful_tours += 1
            iteration_routes.append(route)
            iteration_lengths.append(route_length)

            if route_length < best_length:
                best_route = route.copy()
                best_length = route_length
                best_iteration = iteration_index

        pheromone = _evaporate_pheromone(pheromone, evaporation_rate, graph)

        if not iteration_routes:
            if best_route is None:
                raise RuntimeError(
                    "No feasible Hamiltonian cycle was constructed. "
                    "The graph may be too sparse or may not contain a valid TSP tour."
                )

            history_iteration_best_lengths.append(math.inf)
            history_global_best_lengths.append(float(best_length))
            history_successful_ants.append(0)
            continue

        _deposit_iteration_pheromone(
            pheromone,
            graph,
            routes=iteration_routes,
            lengths=iteration_lengths,
            pheromone_deposit=pheromone_deposit,
        )

        if normalized_mode == "elitist" and best_route is not None and math.isfinite(best_length):
            _deposit_best_route_pheromone(
                pheromone,
                graph,
                best_route=best_route,
                best_length=best_length,
                pheromone_deposit=pheromone_deposit,
                elite_ants=elite_ants,
            )

        iteration_best = min(iteration_lengths)
        history_iteration_best_lengths.append(float(iteration_best))
        history_global_best_lengths.append(float(best_length))
        history_successful_ants.append(len(iteration_routes))

    elapsed_seconds = time.perf_counter() - start_time

    if best_route is None or not math.isfinite(best_length):
        raise RuntimeError("The ant colony algorithm did not find any feasible Hamiltonian cycle.")

    return {
        "best_route_indices": _closed_cycle_indices(best_route),
        "best_route_labels": _closed_cycle_labels(best_route, graph.node_labels),
        "best_length": float(best_length),
        "evaluations": int(evaluations),
        "iterations": int(iterations),
        "best_iteration": int(best_iteration),
        "successful_tours": int(successful_tours),
        "failed_tours": int(failed_tours),
        "history_iteration_best_lengths": history_iteration_best_lengths,
        "history_global_best_lengths": history_global_best_lengths,
        "history_successful_ants": history_successful_ants,
        "elapsed_seconds": float(elapsed_seconds),
        "mode": normalized_mode,
        "ant_count": int(ant_count),
        "pheromone_importance": float(pheromone_importance),
        "distance_importance": float(distance_importance),
        "pheromone_deposit": float(pheromone_deposit),
        "evaporation_intensity": float(evaporation_intensity),
        "evaporation_rate": float(evaporation_rate),
        "elite_ants": int(elite_ants if normalized_mode == "elitist" else 0),
    }


def run_basic_ant_colony(
    graph: TSPGraph,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    pheromone_importance: float = 1.0,
    distance_importance: float = 2.0,
    pheromone_deposit: float = 100.0,
    evaporation_intensity: float = 3.0,
    ant_count: int | None = None,
    initial_pheromone: float = DEFAULT_INITIAL_PHEROMONE,
    seed: int | None = None,
) -> dict[str, object]:
    return run_ant_colony(
        graph,
        iterations=iterations,
        pheromone_importance=pheromone_importance,
        distance_importance=distance_importance,
        pheromone_deposit=pheromone_deposit,
        evaporation_intensity=evaporation_intensity,
        mode="basic",
        ant_count=ant_count,
        initial_pheromone=initial_pheromone,
        seed=seed,
    )


def run_elitist_ant_colony(
    graph: TSPGraph,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    pheromone_importance: float = 1.0,
    distance_importance: float = 2.0,
    pheromone_deposit: float = 100.0,
    evaporation_intensity: float = 3.0,
    ant_count: int | None = None,
    elite_ants: int = DEFAULT_ELITE_ANTS,
    initial_pheromone: float = DEFAULT_INITIAL_PHEROMONE,
    seed: int | None = None,
) -> dict[str, object]:
    return run_ant_colony(
        graph,
        iterations=iterations,
        pheromone_importance=pheromone_importance,
        distance_importance=distance_importance,
        pheromone_deposit=pheromone_deposit,
        evaporation_intensity=evaporation_intensity,
        mode="elitist",
        ant_count=ant_count,
        elite_ants=elite_ants,
        initial_pheromone=initial_pheromone,
        seed=seed,
    )


def _validate_graph(graph: TSPGraph) -> None:
    if not isinstance(graph, TSPGraph):
        raise TypeError("graph must be a TSPGraph instance")

    if graph.size < 2:
        raise ValueError("graph must contain at least two vertices")

    matrix = np.asarray(graph.distance_matrix, dtype=float)
    if matrix.shape != (graph.size, graph.size):
        raise ValueError("distance_matrix shape must match the number of nodes")


def _validate_ant_params(
    *,
    iterations: int,
    pheromone_importance: float,
    distance_importance: float,
    pheromone_deposit: float,
    evaporation_intensity: float,
    ant_count: int | None,
    elite_ants: int,
    initial_pheromone: float,
) -> None:
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if pheromone_importance < 0:
        raise ValueError("pheromone_importance must be >= 0")
    if distance_importance < 0:
        raise ValueError("distance_importance must be >= 0")
    if pheromone_deposit <= 0:
        raise ValueError("pheromone_deposit must be > 0")
    if not (0 <= evaporation_intensity <= 10):
        raise ValueError("evaporation_intensity must be between 0 and 10")
    if ant_count is not None and ant_count <= 0:
        raise ValueError("ant_count must be > 0 when provided")
    if elite_ants < 0:
        raise ValueError("elite_ants must be >= 0")
    if initial_pheromone <= 0:
        raise ValueError("initial_pheromone must be > 0")


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "basic": "basic",
        "base": "basic",
        "classic": "basic",
        "standard": "basic",
        "elitist": "elitist",
        "elite": "elitist",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported ant colony mode: {mode}")
    return aliases[normalized]


def _initialize_pheromone_matrix(graph: TSPGraph, initial_pheromone: float) -> np.ndarray:
    pheromone = np.zeros_like(graph.distance_matrix, dtype=float)
    finite_mask = np.isfinite(graph.distance_matrix)
    np.fill_diagonal(finite_mask, False)
    pheromone[finite_mask] = initial_pheromone
    return pheromone


def _is_complete_graph(graph: TSPGraph) -> bool:
    return (not graph.directed) and bool(np.isfinite(graph.distance_matrix).all())


def _build_neighbor_lists(distance_matrix: np.ndarray) -> list[np.ndarray]:
    size = distance_matrix.shape[0]
    finite_mask = np.isfinite(distance_matrix)
    neighbor_lists: list[np.ndarray] = []
    for vertex in range(size):
        row_mask = finite_mask[vertex].copy()
        row_mask[vertex] = False
        neighbor_lists.append(np.flatnonzero(row_mask))
    return neighbor_lists


def _build_weighted_heuristic_matrix(
    distance_matrix: np.ndarray,
    *,
    distance_importance: float,
) -> np.ndarray:
    visibility = np.zeros_like(distance_matrix, dtype=float)
    finite_mask = np.isfinite(distance_matrix) & (distance_matrix > 0)
    if distance_importance == 0:
        visibility[finite_mask] = 1.0
        return visibility

    inverse_distances = 1.0 / distance_matrix[finite_mask]
    if distance_importance == 1:
        visibility[finite_mask] = inverse_distances
    else:
        visibility[finite_mask] = np.power(inverse_distances, distance_importance)
    return visibility


def _assign_ant_starts(
    node_count: int,
    ant_count: int,
    *,
    iteration_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if ant_count >= node_count:
        base_cycle = np.arange(node_count, dtype=int)
        repeats = int(math.ceil(ant_count / node_count))
        starts = np.tile(base_cycle, repeats)[:ant_count].copy()
        rng.shuffle(starts)
        return starts

    start_offset = (iteration_index - 1) % node_count
    starts = (np.arange(ant_count, dtype=int) + start_offset) % node_count
    rng.shuffle(starts)
    return starts


def _construct_ant_route(
    graph: TSPGraph,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    *,
    start_vertex: int,
    pheromone_importance: float,
    complete_graph: bool,
    neighbor_lists: list[np.ndarray] | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if complete_graph:
        return _construct_route_once(
            graph,
            pheromone,
            heuristic,
            start_vertex=start_vertex,
            pheromone_importance=pheromone_importance,
            complete_graph=complete_graph,
            neighbor_lists=neighbor_lists,
            rng=rng,
        )

    attempts = max(12, graph.size * 4)
    for _ in range(attempts):
        route = _construct_route_once(
            graph,
            pheromone,
            heuristic,
            start_vertex=start_vertex,
            pheromone_importance=pheromone_importance,
            complete_graph=complete_graph,
            neighbor_lists=neighbor_lists,
            rng=rng,
        )
        if route is not None:
            return route

    if graph.size <= 12:
        return _construct_route_backtracking(
            graph,
            pheromone,
            heuristic,
            start_vertex=start_vertex,
            pheromone_importance=pheromone_importance,
            neighbor_lists=neighbor_lists,
            rng=rng,
        )

    return None


def _construct_route_once(
    graph: TSPGraph,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    *,
    start_vertex: int,
    pheromone_importance: float,
    complete_graph: bool,
    neighbor_lists: list[np.ndarray] | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    route = np.empty(graph.size, dtype=int)
    visited = np.zeros(graph.size, dtype=bool)

    route[0] = start_vertex
    visited[start_vertex] = True
    current = start_vertex

    for step in range(1, graph.size):
        if complete_graph:
            candidates = np.flatnonzero(~visited)
        else:
            current_neighbors = neighbor_lists[current] if neighbor_lists is not None else np.empty(0, dtype=int)
            candidates = current_neighbors[~visited[current_neighbors]]
        if candidates.size == 0:
            return None

        desirability = _transition_desirability(
            pheromone[current, candidates],
            heuristic[current, candidates],
            pheromone_importance=pheromone_importance,
        )
        next_vertex = _select_candidate(candidates, desirability, rng)
        route[step] = next_vertex
        visited[next_vertex] = True
        current = next_vertex

    if not math.isfinite(graph.distance_matrix[current, start_vertex]):
        return None

    return route


def _construct_route_backtracking(
    graph: TSPGraph,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    *,
    start_vertex: int,
    pheromone_importance: float,
    neighbor_lists: list[np.ndarray] | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    route = [start_vertex]
    visited = np.zeros(graph.size, dtype=bool)
    visited[start_vertex] = True

    def dfs(current: int) -> bool:
        if len(route) == graph.size:
            return math.isfinite(graph.distance_matrix[current, start_vertex])

        current_neighbors = neighbor_lists[current] if neighbor_lists is not None else np.empty(0, dtype=int)
        candidates = current_neighbors[~visited[current_neighbors]]
        if candidates.size == 0:
            return False

        desirability = _transition_desirability(
            pheromone[current, candidates],
            heuristic[current, candidates],
            pheromone_importance=pheromone_importance,
        )
        ordered_candidates = _weighted_candidate_order(candidates, desirability, rng)

        for next_vertex in ordered_candidates:
            visited[next_vertex] = True
            route.append(int(next_vertex))
            if dfs(int(next_vertex)):
                return True
            route.pop()
            visited[next_vertex] = False

        return False

    if not dfs(start_vertex):
        return None

    return np.asarray(route, dtype=int)


def _transition_desirability(
    pheromone_values: np.ndarray,
    heuristic_values: np.ndarray,
    *,
    pheromone_importance: float,
) -> np.ndarray:
    if pheromone_importance == 0:
        return heuristic_values.copy()
    if pheromone_importance == 1:
        desirability = pheromone_values * heuristic_values
        desirability[~np.isfinite(desirability)] = 0.0
        return desirability

    pheromone_component = np.power(pheromone_values, pheromone_importance, dtype=float)
    desirability = pheromone_component * heuristic_values
    desirability[~np.isfinite(desirability)] = 0.0
    return desirability


def _select_candidate(
    candidates: np.ndarray,
    desirability: np.ndarray,
    rng: np.random.Generator,
) -> int:
    total = float(np.sum(desirability))
    if total <= 0 or not math.isfinite(total):
        return int(candidates[int(rng.integers(0, candidates.size))])

    threshold = float(rng.random()) * total
    cumulative = 0.0
    for index, weight in enumerate(desirability):
        cumulative += float(weight)
        if cumulative >= threshold:
            return int(candidates[index])

    return int(candidates[-1])


def _weighted_candidate_order(
    candidates: np.ndarray,
    desirability: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    remaining_candidates = candidates.astype(int).tolist()
    remaining_weights = desirability.astype(float).tolist()
    ordered: list[int] = []

    while remaining_candidates:
        weight_sum = sum(remaining_weights)
        if weight_sum <= 0 or not math.isfinite(weight_sum):
            index = int(rng.integers(0, len(remaining_candidates)))
        else:
            probabilities = [weight / weight_sum for weight in remaining_weights]
            index = int(rng.choice(len(remaining_candidates), p=probabilities))

        ordered.append(remaining_candidates.pop(index))
        remaining_weights.pop(index)

    return ordered


def _evaporate_pheromone(
    pheromone: np.ndarray,
    evaporation_rate: float,
    graph: TSPGraph,
) -> np.ndarray:
    updated = pheromone * (1.0 - evaporation_rate)
    finite_mask = np.isfinite(graph.distance_matrix)
    np.fill_diagonal(finite_mask, False)
    updated[~finite_mask] = 0.0
    return updated


def _deposit_iteration_pheromone(
    pheromone: np.ndarray,
    graph: TSPGraph,
    *,
    routes: list[np.ndarray],
    lengths: list[float],
    pheromone_deposit: float,
) -> None:
    for route, route_length in zip(routes, lengths):
        _deposit_route_pheromone(
            pheromone,
            graph,
            route,
            pheromone_deposit / route_length,
        )


def _deposit_best_route_pheromone(
    pheromone: np.ndarray,
    graph: TSPGraph,
    *,
    best_route: np.ndarray,
    best_length: float,
    pheromone_deposit: float,
    elite_ants: int,
) -> None:
    if elite_ants <= 0:
        return

    _deposit_route_pheromone(
        pheromone,
        graph,
        best_route,
        (elite_ants * pheromone_deposit) / best_length,
    )


def _deposit_route_pheromone(
    pheromone: np.ndarray,
    graph: TSPGraph,
    route: np.ndarray,
    amount: float,
) -> None:
    cycle = np.concatenate([route, route[:1]])
    for source, target in zip(cycle, cycle[1:]):
        pheromone[source, target] += amount
        if not graph.directed:
            pheromone[target, source] += amount


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


def load_graph(path: str | Path) -> TSPGraph:
    return load_tsp_graph(path)
