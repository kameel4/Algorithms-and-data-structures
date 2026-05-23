from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any

import puzzle15
from puzzle15 import ALGORITHMS, GOAL_STATE, apply_moves, generate_random_board, validate_state


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "benchmark_results"
RAW_CSV = RESULT_DIR / "puzzle15_benchmark_raw.csv"
SUMMARY_CSV = RESULT_DIR / "puzzle15_benchmark_summary.csv"

PUZZLE_SHUFFLES = (0, 5, 10, 15, 20, 25, 30)
BFS_SHUFFLES = {0, 5, 10, 15}
CASES: list[tuple[str, int, int, tuple[int, ...]]] = [
    (f"shuffle_{shuffles}", shuffles, 42, generate_random_board(shuffles=shuffles, seed=42))
    for shuffles in PUZZLE_SHUFFLES
]

FIELDNAMES = [
    "case",
    "shuffles",
    "seed",
    "state",
    "engine",
    "algorithm",
    "repeat",
    "status",
    "found",
    "cost",
    "valid_solution",
    "time_ms",
    "peak_memory_kb",
    "nodes_expanded",
    "states_generated",
    "pruned_states",
    "max_depth",
    "max_frontier",
    "message",
]


def worker(engine: str, algorithm: str, raw_state: str) -> None:
    board = validate_state([int(value) for value in raw_state.split(",")])
    started = perf_counter()

    if engine == "python":
        solvers = {
            "astar": puzzle15._solve_astar,  # noqa: SLF001
            "bfs": puzzle15._solve_bfs,  # noqa: SLF001
            "ida_star": puzzle15._solve_ida_star,  # noqa: SLF001
            "backjumping": puzzle15._solve_backjumping,  # noqa: SLF001
        }
        tracemalloc.start()
        try:
            result = solvers[algorithm](board)
            result.algorithm = algorithm
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        result.metrics.elapsed_ms = (perf_counter() - started) * 1000
        result.metrics.peak_memory_kb = peak / 1024
    elif engine == "cpp":
        result = puzzle15._solve_with_cpp(board, algorithm)  # noqa: SLF001
        if result is None:
            raise RuntimeError("C++ solver is unavailable.")
    else:
        raise RuntimeError(f"Unsupported engine: {engine}")

    moves = []
    valid_solution = False
    if result.found and result.solution is not None:
        moves = [str(move) for move in result.solution.get("moves", [])]
        valid_solution = apply_moves(board, moves)[-1] == GOAL_STATE

    payload = {
        "status": "ok",
        "found": result.found,
        "cost": len(moves) if result.found else "",
        "valid_solution": valid_solution,
        "time_ms": result.metrics.elapsed_ms,
        "peak_memory_kb": result.metrics.peak_memory_kb,
        "nodes_expanded": result.metrics.nodes_expanded,
        "states_generated": result.metrics.states_generated,
        "pruned_states": result.metrics.pruned_states,
        "max_depth": result.metrics.max_depth,
        "max_frontier": result.metrics.max_frontier,
        "message": result.message,
    }
    print(json.dumps(payload, ensure_ascii=False))


def run_worker(
    *,
    engine: str,
    algorithm: str,
    case_name: str,
    shuffles: int,
    seed: int,
    state: tuple[int, ...],
    repeat: int,
) -> dict[str, Any]:
    state_text = ",".join(str(value) for value in state)
    command = [
        sys.executable,
        str(Path(__file__).name),
        "--worker",
        "--engine",
        engine,
        "--algorithm",
        algorithm,
        "--state",
        state_text,
    ]
    base = {
        "case": case_name,
        "shuffles": shuffles,
        "seed": seed,
        "state": state_text,
        "engine": engine,
        "algorithm": algorithm,
        "repeat": repeat,
    }

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        return {
            **base,
            "status": "error",
            "found": "",
            "cost": "",
            "valid_solution": "",
            "time_ms": "",
            "peak_memory_kb": "",
            "nodes_expanded": "",
            "states_generated": "",
            "pruned_states": "",
            "max_depth": "",
            "max_frontier": "",
            "message": (completed.stderr or completed.stdout).strip()[:500],
        }

    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return {
            **base,
            "status": "error",
            "found": "",
            "cost": "",
            "valid_solution": "",
            "time_ms": "",
            "peak_memory_kb": "",
            "nodes_expanded": "",
            "states_generated": "",
            "pruned_states": "",
            "max_depth": "",
            "max_frontier": "",
            "message": completed.stdout.strip()[:500],
        }

    return {**base, **payload}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["case"], row["engine"], row["algorithm"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (case, engine, algorithm), items in sorted(groups.items()):
        ok_items = [item for item in items if item["status"] == "ok"]
        first = items[0]
        if not ok_items:
            summary.append(
                {
                    "case": case,
                    "shuffles": first["shuffles"],
                    "seed": first["seed"],
                    "engine": engine,
                    "algorithm": algorithm,
                    "status": first["status"],
                    "runs": len(items),
                    "cost": "",
                    "valid_solution": "",
                    "avg_time_ms": "",
                    "min_time_ms": "",
                    "avg_peak_memory_kb": "",
                    "avg_nodes_expanded": "",
                    "message": first["message"],
                }
            )
            continue

        times = [float(item["time_ms"]) for item in ok_items]
        memory = [float(item["peak_memory_kb"]) for item in ok_items]
        nodes = [int(item["nodes_expanded"]) for item in ok_items]
        summary.append(
            {
                "case": case,
                "shuffles": first["shuffles"],
                "seed": first["seed"],
                "engine": engine,
                "algorithm": algorithm,
                "status": "ok",
                "runs": len(ok_items),
                "cost": ok_items[0]["cost"],
                "valid_solution": ok_items[0]["valid_solution"],
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "avg_peak_memory_kb": sum(memory) / len(memory),
                "avg_nodes_expanded": sum(nodes) / len(nodes),
                "message": ok_items[0]["message"],
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(repeats: int) -> None:
    rows: list[dict[str, Any]] = []
    for case_name, shuffles, seed, state in CASES:
        for engine in ("python", "cpp"):
            for algorithm in ALGORITHMS:
                if algorithm == "bfs" and shuffles not in BFS_SHUFFLES:
                    continue
                for repeat in range(1, repeats + 1):
                    print(f"{case_name} {engine} {algorithm} repeat {repeat}", flush=True)
                    rows.append(
                        run_worker(
                            engine=engine,
                            algorithm=algorithm,
                            case_name=case_name,
                            shuffles=shuffles,
                            seed=seed,
                            state=state,
                            repeat=repeat,
                        )
                    )

    summary = summarize(rows)
    write_csv(RAW_CSV, rows, FIELDNAMES)
    write_csv(
        SUMMARY_CSV,
        summary,
        [
            "case",
            "shuffles",
            "seed",
            "engine",
            "algorithm",
            "status",
            "runs",
            "cost",
            "valid_solution",
            "avg_time_ms",
            "min_time_ms",
            "avg_peak_memory_kb",
            "avg_nodes_expanded",
            "message",
        ],
    )
    print(f"raw={RAW_CSV}")
    print(f"summary={SUMMARY_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--engine", choices=["python", "cpp"], default="python")
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="astar")
    parser.add_argument("--state", default=",".join(str(value) for value in GOAL_STATE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args.engine, args.algorithm, args.state)
    else:
        run_benchmark(args.repeats)


if __name__ == "__main__":
    main()
