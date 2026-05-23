from __future__ import annotations

import json
import subprocess
from collections import deque
from heapq import heappop, heappush
from pathlib import Path
from random import Random
from subprocess import TimeoutExpired
from time import perf_counter
from typing import Any, Iterable, Sequence

from models import SearchMetrics, SearchResult


BoardState = tuple[int, ...]
Move = str
PuzzleSolution = dict[str, object]

BOARD_SIZE = 4
BOARD_LEN = BOARD_SIZE * BOARD_SIZE
GOAL_STATE: BoardState = tuple(range(1, BOARD_LEN)) + (0,)
ALGORITHMS: dict[str, str] = {
    "astar": "A* (Manhattan)",
    "bfs": "Breadth-First Search",
    "ida_star": "IDA*",
    "backjumping": "Intelligent Backtracking",
}

_MOVE_DELTAS: dict[Move, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
_REVERSE_MOVE: dict[Move, Move] = {
    "U": "D",
    "D": "U",
    "L": "R",
    "R": "L",
}
_MOVE_ALIASES: dict[str, Move] = {
    "U": "U",
    "UP": "U",
    "D": "D",
    "DOWN": "D",
    "L": "L",
    "LEFT": "L",
    "R": "R",
    "RIGHT": "R",
}
_GOAL_INDEX: dict[int, int] = {tile: idx for idx, tile in enumerate(GOAL_STATE)}
_CPP_SOLVER = Path(__file__).with_name("puzzle15_solver.exe")
_NEIGHBORS: tuple[tuple[tuple[Move, int], ...], ...] = tuple(
    tuple(
        (move, nr * BOARD_SIZE + nc)
        for move, (dr, dc) in _MOVE_DELTAS.items()
        if 0 <= (nr := row + dr) < BOARD_SIZE and 0 <= (nc := col + dc) < BOARD_SIZE
    )
    for row in range(BOARD_SIZE)
    for col in range(BOARD_SIZE)
)


def validate_state(state: Sequence[int] | Sequence[Sequence[int]]) -> BoardState:
    """Normalize a board representation to a flat 16-item tuple."""

    if len(state) == BOARD_SIZE and all(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        for item in state
    ):
        flat = tuple(value for row in state for value in row)  # type: ignore[arg-type]
    else:
        flat = tuple(int(value) for value in state)  # type: ignore[arg-type]

    if len(flat) != BOARD_LEN:
        raise ValueError("State must contain exactly 16 cells.")

    values = set(flat)
    if values != set(range(BOARD_LEN)):
        raise ValueError("State must contain each number from 0 to 15 exactly once.")

    return flat


def is_solvable(state: Sequence[int] | Sequence[Sequence[int]]) -> bool:
    board = validate_state(state)
    inversions = 0
    tiles = [value for value in board if value != 0]
    for idx, left in enumerate(tiles):
        inversions += sum(1 for right in tiles[idx + 1 :] if left > right)

    blank_row_from_bottom = BOARD_SIZE - (board.index(0) // BOARD_SIZE)
    return (inversions + blank_row_from_bottom) % 2 == 1


def generate_random_board(shuffles: int = 20, seed: int | None = None) -> BoardState:
    if shuffles < 0:
        raise ValueError("shuffles must be non-negative.")

    rng = Random(seed)
    state = GOAL_STATE
    blank_idx = BOARD_LEN - 1
    previous_move: Move | None = None

    for _ in range(shuffles):
        candidates = list(_NEIGHBORS[blank_idx])
        if previous_move is not None and len(candidates) > 1:
            reverse = _REVERSE_MOVE[previous_move]
            candidates = [candidate for candidate in candidates if candidate[0] != reverse]
        move, next_blank = rng.choice(candidates)
        state = _swap_positions(state, blank_idx, next_blank)
        blank_idx = next_blank
        previous_move = move

    return state


def solve_board(
    state: Sequence[int] | Sequence[Sequence[int]],
    algorithm: str,
) -> SearchResult[PuzzleSolution]:
    start_time = perf_counter()

    if algorithm not in ALGORITHMS:
        return _finalize_result(
            SearchResult(
                algorithm=algorithm,
                found=False,
                solution=None,
                message=f"Unknown algorithm: {algorithm}",
            ),
            start_time,
        )

    try:
        board = validate_state(state)
    except ValueError as exc:
        return _finalize_result(
            SearchResult(
                algorithm=algorithm,
                found=False,
                solution=None,
                message=str(exc),
            ),
            start_time,
        )

    if not is_solvable(board):
        return _finalize_result(
            SearchResult(
                algorithm=algorithm,
                found=False,
                solution=None,
                message="Board is unsolvable.",
            ),
            start_time,
        )

    cpp_result = _solve_with_cpp(board, algorithm)
    if cpp_result is not None:
        return cpp_result

    solvers = {
        "astar": _solve_astar,
        "bfs": _solve_bfs,
        "ida_star": _solve_ida_star,
        "backjumping": _solve_backjumping,
    }
    result = solvers[algorithm](board)
    result.algorithm = algorithm
    return _finalize_result(result, start_time)


def compare_algorithms(
    state: Sequence[int] | Sequence[Sequence[int]],
    algorithms: list[str] | None = None,
) -> list[SearchResult[PuzzleSolution]]:
    selected = algorithms or list(ALGORITHMS)
    return [solve_board(state, algorithm) for algorithm in selected]


def apply_moves(
    state: Sequence[int] | Sequence[Sequence[int]],
    moves: Iterable[str],
) -> list[BoardState]:
    board = validate_state(state)
    path = [board]

    for raw_move in moves:
        move = _normalize_move(raw_move)
        blank_idx = board.index(0)
        for candidate_move, next_blank in _NEIGHBORS[blank_idx]:
            if candidate_move == move:
                board = _swap_positions(board, blank_idx, next_blank)
                path.append(board)
                break
        else:
            raise ValueError(f"Illegal move '{raw_move}' for current board.")

    return path


def _solve_with_cpp(board: BoardState, algorithm: str) -> SearchResult[PuzzleSolution] | None:
    if not _CPP_SOLVER.exists():
        return None

    command = [
        str(_CPP_SOLVER),
        "--algorithm",
        algorithm,
        "--state",
        ",".join(str(value) for value in board),
    ]

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


def _cpp_payload_to_result(payload: dict[str, Any], algorithm: str) -> SearchResult[PuzzleSolution]:
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

    found = bool(payload.get("found", False))
    solution: PuzzleSolution | None = None
    if found:
        moves = [str(move) for move in payload.get("moves", [])]
        states = [tuple(int(value) for value in state) for state in payload.get("states", [])]
        final_state_payload = payload.get("final_state")
        final_state = (
            tuple(int(value) for value in final_state_payload)
            if isinstance(final_state_payload, list)
            else (states[-1] if states else GOAL_STATE)
        )
        solution = {
            "moves": moves,
            "path": states,
            "states": states,
            "cost": int(payload.get("cost", len(moves))),
            "final_state": final_state,
            "engine": "cpp",
        }

    return SearchResult(
        algorithm=algorithm,
        found=found,
        solution=solution,
        metrics=metrics,
        message=str(payload.get("message", "")),
    )


def _solve_bfs(board: BoardState) -> SearchResult[PuzzleSolution]:
    metrics = SearchMetrics(max_frontier=1)
    if board == GOAL_STATE:
        return _success_result("bfs", board, [], metrics)

    frontier: deque[tuple[BoardState, int]] = deque([(board, board.index(0))])
    parents: dict[BoardState, tuple[BoardState | None, Move | None]] = {
        board: (None, None)
    }
    depths: dict[BoardState, int] = {board: 0}

    while frontier:
        metrics.max_frontier = max(metrics.max_frontier, len(frontier))
        state, blank_idx = frontier.popleft()
        depth = depths[state]
        metrics.max_depth = max(metrics.max_depth, depth)

        if state == GOAL_STATE:
            return _success_result("bfs", board, _reconstruct_moves(state, parents), metrics)

        metrics.nodes_expanded += 1
        for move, next_blank in _NEIGHBORS[blank_idx]:
            child = _swap_positions(state, blank_idx, next_blank)
            metrics.states_generated += 1
            if child in parents:
                metrics.pruned_states += 1
                continue

            parents[child] = (state, move)
            depths[child] = depth + 1
            frontier.append((child, next_blank))
            metrics.max_frontier = max(metrics.max_frontier, len(frontier))

    return SearchResult(
        algorithm="bfs",
        found=False,
        solution=None,
        metrics=metrics,
        message="No solution found.",
    )


def _solve_astar(board: BoardState) -> SearchResult[PuzzleSolution]:
    metrics = SearchMetrics(max_frontier=1)
    initial_blank = board.index(0)
    initial_h = _manhattan(board)
    if initial_h == 0:
        return _success_result("astar", board, [], metrics)

    counter = 0
    frontier: list[tuple[int, int, int, int, BoardState, int]] = [
        (initial_h, initial_h, 0, counter, board, initial_blank)
    ]
    parents: dict[BoardState, tuple[BoardState | None, Move | None]] = {
        board: (None, None)
    }
    best_cost: dict[BoardState, int] = {board: 0}

    while frontier:
        metrics.max_frontier = max(metrics.max_frontier, len(frontier))
        f_score, _, depth, _, state, blank_idx = heappop(frontier)
        if depth != best_cost.get(state):
            metrics.pruned_states += 1
            continue

        metrics.max_depth = max(metrics.max_depth, depth)
        if state == GOAL_STATE:
            return _success_result(
                "astar",
                board,
                _reconstruct_moves(state, parents),
                metrics,
            )

        metrics.nodes_expanded += 1
        for move, next_blank in _NEIGHBORS[blank_idx]:
            child = _swap_positions(state, blank_idx, next_blank)
            metrics.states_generated += 1
            next_depth = depth + 1
            if next_depth >= best_cost.get(child, next_depth + 1):
                metrics.pruned_states += 1
                continue

            best_cost[child] = next_depth
            parents[child] = (state, move)
            counter += 1
            child_h = _manhattan(child)
            heappush(
                frontier,
                (next_depth + child_h, child_h, next_depth, counter, child, next_blank),
            )
            metrics.max_frontier = max(metrics.max_frontier, len(frontier))

    return SearchResult(
        algorithm="astar",
        found=False,
        solution=None,
        metrics=metrics,
        message="No solution found.",
    )


def _solve_ida_star(board: BoardState) -> SearchResult[PuzzleSolution]:
    metrics = SearchMetrics(max_frontier=1)
    initial_blank = board.index(0)
    initial_h = _manhattan(board)
    if initial_h == 0:
        return _success_result("ida_star", board, [], metrics)

    path = [board]
    moves: list[Move] = []
    active_path = {board}
    bound = initial_h

    while True:
        best_depth_seen: dict[BoardState, int] = {board: 0}
        next_bound = _ida_dfs(
            state=board,
            blank_idx=initial_blank,
            depth=0,
            bound=bound,
            last_move=None,
            moves=moves,
            path=path,
            active_path=active_path,
            best_depth_seen=best_depth_seen,
            metrics=metrics,
        )
        if next_bound == -1:
            return _success_result("ida_star", board, moves.copy(), metrics)
        if next_bound == float("inf"):
            return SearchResult(
                algorithm="ida_star",
                found=False,
                solution=None,
                metrics=metrics,
                message="No solution found.",
            )
        bound = int(next_bound)


def _solve_backjumping(board: BoardState) -> SearchResult[PuzzleSolution]:
    metrics = SearchMetrics(max_frontier=1)
    initial_blank = board.index(0)
    initial_h = _manhattan(board)
    if initial_h == 0:
        return _success_result("backjumping", board, [], metrics)

    path = [board]
    moves: list[Move] = []
    active_path = {board}
    dead_cache: set[tuple[BoardState, int]] = set()
    bound = initial_h

    while True:
        best_depth_seen: dict[BoardState, int] = {board: 0}
        next_bound = _backjump_dfs(
            state=board,
            blank_idx=initial_blank,
            depth=0,
            bound=bound,
            last_move=None,
            moves=moves,
            path=path,
            active_path=active_path,
            best_depth_seen=best_depth_seen,
            dead_cache=dead_cache,
            metrics=metrics,
        )
        if next_bound == -1:
            return _success_result("backjumping", board, moves.copy(), metrics)
        if next_bound == float("inf"):
            return SearchResult(
                algorithm="backjumping",
                found=False,
                solution=None,
                metrics=metrics,
                message="No solution found.",
            )
        bound = int(next_bound)


def _backjump_dfs(
    *,
    state: BoardState,
    blank_idx: int,
    depth: int,
    bound: int,
    last_move: Move | None,
    moves: list[Move],
    path: list[BoardState],
    active_path: set[BoardState],
    best_depth_seen: dict[BoardState, int],
    dead_cache: set[tuple[BoardState, int]],
    metrics: SearchMetrics,
) -> float:
    heuristic = _manhattan(state)
    score = depth + heuristic
    metrics.max_depth = max(metrics.max_depth, depth)
    metrics.max_frontier = max(metrics.max_frontier, len(path))

    if score > bound:
        metrics.pruned_states += 1
        return float(score)
    if state == GOAL_STATE:
        return -1
    dead_key = (state, bound - depth)
    if dead_key in dead_cache:
        metrics.pruned_states += 1
        return float("inf")

    metrics.nodes_expanded += 1
    next_jump = float("inf")
    candidates: list[tuple[int, int, Move, BoardState, int]] = []
    for move, next_blank in _ordered_neighbors(blank_idx, last_move):
        child = _swap_positions(state, blank_idx, next_blank)
        metrics.states_generated += 1
        if child in active_path:
            metrics.pruned_states += 1
            continue

        next_depth = depth + 1
        if next_depth >= best_depth_seen.get(child, next_depth + 1):
            metrics.pruned_states += 1
            continue

        child_h = _manhattan(child)
        candidates.append((next_depth + child_h, child_h, move, child, next_blank))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _, _, move, child, next_blank in candidates:
        next_depth = depth + 1
        previous_depth = best_depth_seen.get(child)
        best_depth_seen[child] = next_depth
        active_path.add(child)
        path.append(child)
        moves.append(move)

        result = _backjump_dfs(
            state=child,
            blank_idx=next_blank,
            depth=next_depth,
            bound=bound,
            last_move=move,
            moves=moves,
            path=path,
            active_path=active_path,
            best_depth_seen=best_depth_seen,
            dead_cache=dead_cache,
            metrics=metrics,
        )

        if result == -1:
            return -1
        next_jump = min(next_jump, result)

        moves.pop()
        path.pop()
        active_path.remove(child)
        if previous_depth is None:
            best_depth_seen.pop(child, None)
        else:
            best_depth_seen[child] = previous_depth

    dead_cache.add(dead_key)
    return next_jump


def _ida_dfs(
    *,
    state: BoardState,
    blank_idx: int,
    depth: int,
    bound: int,
    last_move: Move | None,
    moves: list[Move],
    path: list[BoardState],
    active_path: set[BoardState],
    best_depth_seen: dict[BoardState, int],
    metrics: SearchMetrics,
) -> float:
    heuristic = _manhattan(state)
    score = depth + heuristic
    metrics.max_depth = max(metrics.max_depth, depth)
    metrics.max_frontier = max(metrics.max_frontier, len(path))

    if score > bound:
        metrics.pruned_states += 1
        return float(score)
    if state == GOAL_STATE:
        return -1

    metrics.nodes_expanded += 1
    next_threshold = float("inf")
    candidates: list[tuple[int, Move, BoardState, int]] = []
    for move, next_blank in _ordered_neighbors(blank_idx, last_move):
        child = _swap_positions(state, blank_idx, next_blank)
        metrics.states_generated += 1
        if child in active_path:
            metrics.pruned_states += 1
            continue

        next_depth = depth + 1
        if next_depth >= best_depth_seen.get(child, next_depth + 1):
            metrics.pruned_states += 1
            continue

        candidates.append((_manhattan(child), move, child, next_blank))

    candidates.sort(key=lambda item: (item[0], item[1]))
    for _, move, child, next_blank in candidates:
        next_depth = depth + 1
        previous_depth = best_depth_seen.get(child)
        best_depth_seen[child] = next_depth
        active_path.add(child)
        path.append(child)
        moves.append(move)

        result = _ida_dfs(
            state=child,
            blank_idx=next_blank,
            depth=next_depth,
            bound=bound,
            last_move=move,
            moves=moves,
            path=path,
            active_path=active_path,
            best_depth_seen=best_depth_seen,
            metrics=metrics,
        )

        if result == -1:
            return -1
        next_threshold = min(next_threshold, result)

        moves.pop()
        path.pop()
        active_path.remove(child)
        if previous_depth is None:
            best_depth_seen.pop(child, None)
        else:
            best_depth_seen[child] = previous_depth

    return next_threshold


def _ordered_neighbors(blank_idx: int, last_move: Move | None) -> list[tuple[Move, int]]:
    neighbors = list(_NEIGHBORS[blank_idx])
    if last_move is None:
        return neighbors
    reverse = _REVERSE_MOVE[last_move]
    return [neighbor for neighbor in neighbors if neighbor[0] != reverse]


def _success_result(
    algorithm: str,
    initial_state: BoardState,
    moves: list[Move],
    metrics: SearchMetrics,
) -> SearchResult[PuzzleSolution]:
    path = apply_moves(initial_state, moves)
    return SearchResult(
        algorithm=algorithm,
        found=True,
        solution={
            "moves": moves,
            "path": path,
            "cost": len(moves),
            "final_state": path[-1],
        },
        metrics=metrics,
        message="Solved.",
    )


def _finalize_result(
    result: SearchResult[PuzzleSolution],
    started_at: float,
) -> SearchResult[PuzzleSolution]:
    result.metrics.elapsed_ms = (perf_counter() - started_at) * 1000
    result.metrics.peak_memory_kb = 0.0
    return result


def _reconstruct_moves(
    state: BoardState,
    parents: dict[BoardState, tuple[BoardState | None, Move | None]],
) -> list[Move]:
    moves: list[Move] = []
    current = state
    while True:
        previous, move = parents[current]
        if previous is None or move is None:
            break
        moves.append(move)
        current = previous
    moves.reverse()
    return moves


def _swap_positions(state: BoardState, left: int, right: int) -> BoardState:
    items = list(state)
    items[left], items[right] = items[right], items[left]
    return tuple(items)


def _manhattan(state: BoardState) -> int:
    distance = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        goal_idx = _GOAL_INDEX[tile]
        row, col = divmod(idx, BOARD_SIZE)
        goal_row, goal_col = divmod(goal_idx, BOARD_SIZE)
        distance += abs(row - goal_row) + abs(col - goal_col)
    return distance


def _normalize_move(move: str) -> Move:
    normalized = _MOVE_ALIASES.get(move.strip().upper())
    if normalized is None:
        raise ValueError(f"Unsupported move: {move}")
    return normalized


__all__ = [
    "ALGORITHMS",
    "GOAL_STATE",
    "apply_moves",
    "compare_algorithms",
    "generate_random_board",
    "is_solvable",
    "solve_board",
    "validate_state",
]
