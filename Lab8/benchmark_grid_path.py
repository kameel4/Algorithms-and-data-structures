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

import grid_path
from grid_path import ALGORITHMS, GridPathProblem


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "benchmark_results"
RAW_CSV = RESULT_DIR / "grid_path_benchmark_raw.csv"
SUMMARY_CSV = RESULT_DIR / "grid_path_benchmark_summary.csv"

CASES: list[tuple[str, int, int, tuple[int, int], tuple[int, int]]] = [
    ("3x3", 3, 3, (0, 0), (2, 2)),
    ("3x4", 3, 4, (0, 0), (2, 3)),
    ("4x4", 4, 4, (0, 0), (3, 2)),
    ("4x5", 4, 5, (0, 0), (3, 4)),
    ("5x5", 5, 5, (0, 0), (4, 4)),
    ("6x6", 6, 6, (0, 0), (5, 4)),
]

FIELDNAMES = [
    "case",
    "rows",
    "cols",
    "cells",
    "start",
    "end",
    "engine",
    "algorithm",
    "repeat",
    "status",
    "found",
    "path_count",
    "exhaustive",
    "time_ms",
    "peak_memory_kb",
    "nodes_expanded",
    "states_generated",
    "pruned_states",
    "max_depth",
    "max_frontier",
    "message",
]


def worker(engine: str, algorithm: str, rows: int, cols: int, end_row: int, end_col: int) -> None:
    problem = GridPathProblem(rows=rows, cols=cols, start=(0, 0), end=(end_row, end_col))
    started = perf_counter()

    if engine == "python":
        solver = grid_path._GridPathSolver(problem, algorithm, max_solutions=0)  # noqa: SLF001
        tracemalloc.start()
        try:
            result = solver.solve()
            result.algorithm = algorithm
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        result.metrics.elapsed_ms = (perf_counter() - started) * 1000
        result.metrics.peak_memory_kb = peak / 1024
    elif engine == "cpp":
        result = grid_path._solve_with_cpp(problem, algorithm, max_solutions=0)  # noqa: SLF001
        if result is None:
            raise RuntimeError("C++ solver is unavailable.")
    else:
        raise RuntimeError(f"Unsupported engine: {engine}")

    solution = result.solution or {}
    payload = {
        "status": "ok",
        "found": result.found,
        "path_count": int(solution.get("path_count", 0)),
        "exhaustive": bool(solution.get("exhaustive", False)),
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
    rows: int,
    cols: int,
    start: tuple[int, int],
    end: tuple[int, int],
    repeat: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).name),
        "--worker",
        "--engine",
        engine,
        "--algorithm",
        algorithm,
        "--rows",
        str(rows),
        "--cols",
        str(cols),
        "--end-row",
        str(end[0]),
        "--end-col",
        str(end[1]),
    ]
    base = {
        "case": case_name,
        "rows": rows,
        "cols": cols,
        "cells": rows * cols,
        "start": f"{start[0]},{start[1]}",
        "end": f"{end[0]},{end[1]}",
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
            "path_count": "",
            "exhaustive": "",
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
            "path_count": "",
            "exhaustive": "",
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
                    "rows": first["rows"],
                    "cols": first["cols"],
                    "cells": first["cells"],
                    "engine": engine,
                    "algorithm": algorithm,
                    "status": first["status"],
                    "runs": len(items),
                    "path_count": "",
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
                "rows": first["rows"],
                "cols": first["cols"],
                "cells": first["cells"],
                "engine": engine,
                "algorithm": algorithm,
                "status": "ok",
                "runs": len(ok_items),
                "path_count": ok_items[0]["path_count"],
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
    for case_name, case_rows, case_cols, start, end in CASES:
        for engine in ("python", "cpp"):
            for algorithm in ALGORITHMS:
                for repeat in range(1, repeats + 1):
                    print(f"{case_name} {engine} {algorithm} repeat {repeat}", flush=True)
                    rows.append(
                        run_worker(
                            engine=engine,
                            algorithm=algorithm,
                            case_name=case_name,
                            rows=case_rows,
                            cols=case_cols,
                            start=start,
                            end=end,
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
            "rows",
            "cols",
            "cells",
            "engine",
            "algorithm",
            "status",
            "runs",
            "path_count",
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
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="baseline")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--end-row", type=int, default=2)
    parser.add_argument("--end-col", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args.engine, args.algorithm, args.rows, args.cols, args.end_row, args.end_col)
    else:
        run_benchmark(args.repeats)


if __name__ == "__main__":
    main()
