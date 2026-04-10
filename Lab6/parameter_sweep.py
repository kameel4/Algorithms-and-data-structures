from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .annealing import load_tsp_graph, run_simulated_annealing
    from .ant_colony import run_basic_ant_colony, run_elitist_ant_colony
except ImportError:
    from annealing import load_tsp_graph, run_simulated_annealing
    from ant_colony import run_basic_ant_colony, run_elitist_ant_colony


RUNNERS = {
    "annealing": run_simulated_annealing,
    "ant": run_basic_ant_colony,
    "elitist-ant": run_elitist_ant_colony,
}


def _normalize_algorithm(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "annealing": "annealing",
        "simulated-annealing": "annealing",
        "sa": "annealing",
        "ant": "ant",
        "aco": "ant",
        "ant-colony": "ant",
        "basic-ant": "ant",
        "elitist-ant": "elitist-ant",
        "elitist": "elitist-ant",
        "elite-ant": "elitist-ant",
        "elite-ant-colony": "elitist-ant",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported algorithm: {name}")
    return aliases[normalized]


def _clean_number(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _expand_range_spec(spec: Any) -> list[Any]:
    if isinstance(spec, list):
        return [_clean_number(value) for value in spec]

    if not isinstance(spec, dict):
        return [_clean_number(spec)]

    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ValueError("'values' must be a non-empty list.")
        return [_clean_number(value) for value in values]

    required = {"start", "stop", "step"}
    if not required.issubset(spec):
        raise ValueError("Range spec must contain either 'values' or 'start'/'stop'/'step'.")

    start = spec["start"]
    stop = spec["stop"]
    step = spec["step"]
    if step == 0:
        raise ValueError("Range 'step' must be non-zero.")

    values = []
    current = start
    epsilon = abs(step) * 1e-9

    if step > 0:
        condition = lambda x: x <= stop + epsilon
    else:
        condition = lambda x: x >= stop - epsilon

    while condition(current):
        values.append(_clean_number(current))
        current += step

    if not values:
        raise ValueError("Range spec produced no values.")

    return values


def _expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    if not grid:
        return [dict()]

    items = list(grid.items())
    keys = [key for key, _ in items]
    value_lists = [_expand_range_spec(spec) for _, spec in items]
    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    return combinations


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def _resolve_seeds(spec: Any) -> list[int]:
    values = _expand_range_spec(spec)
    seeds = [int(value) for value in values]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _filter_runner_kwargs(runner: Any, params: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(runner)
    allowed = set(signature.parameters)
    filtered = {key: value for key, value in params.items() if key in allowed}
    missing = [
        key
        for key, value in signature.parameters.items()
        if value.default is inspect._empty and key not in filtered
    ]
    if missing:
        raise ValueError(f"Missing required runner parameters: {missing}")
    return filtered


def _mean_or_none(values: list[Any]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _std_or_none(values: list[Any]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.std(filtered))


def _analyze_result(result: dict[str, Any], *, reference_best_length: float | None) -> dict[str, Any]:
    best_length = float(result["best_length"])
    gap = None if reference_best_length is None else best_length - reference_best_length
    hit_reference = None
    if reference_best_length is not None:
        hit_reference = 1 if math.isclose(best_length, reference_best_length, rel_tol=0.0, abs_tol=1e-9) else 0

    analysis = {
        "best_length": best_length,
        "gap_to_reference": gap,
        "evaluations": int(result.get("evaluations", 0)),
        "elapsed_ms": float(result.get("elapsed_seconds", 0.0)) * 1000.0,
    }
    if hit_reference is not None:
        analysis["hit_reference"] = hit_reference

    for key in (
        "accepted_moves",
        "improving_moves",
        "temperature_steps",
        "successful_tours",
        "failed_tours",
        "best_iteration",
        "ant_count",
        "elite_ants",
    ):
        value = result.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            analysis[key] = _clean_number(value)

    return analysis


def _build_summary(
    experiment_name: str,
    algorithm: str,
    params: dict[str, Any],
    graph_name: str,
    reference_best_length: float | None,
    seed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "experiment": experiment_name,
        "algorithm": algorithm,
        "graph": graph_name,
        "params": params,
        "reference_best_length": reference_best_length,
        "mean_best_length": _mean_or_none([item["analysis"]["best_length"] for item in seed_runs]),
        "std_best_length": _std_or_none([item["analysis"]["best_length"] for item in seed_runs]),
        "best_best_length": min(item["analysis"]["best_length"] for item in seed_runs),
        "worst_best_length": max(item["analysis"]["best_length"] for item in seed_runs),
        "mean_gap_to_reference": _mean_or_none(
            [item["analysis"]["gap_to_reference"] for item in seed_runs]
        ),
        "mean_evaluations": _mean_or_none([item["analysis"]["evaluations"] for item in seed_runs]),
        "mean_runtime_ms": _mean_or_none([item["analysis"]["elapsed_ms"] for item in seed_runs]),
        "runs": seed_runs,
    }

    extra_analysis_keys = sorted(
        {
            key
            for item in seed_runs
            for key in item["analysis"]
            if key not in {"best_length", "gap_to_reference", "evaluations", "elapsed_ms"}
        }
    )
    for key in extra_analysis_keys:
        summary[f"mean_{key}"] = _mean_or_none(
            [item["analysis"].get(key) for item in seed_runs]
        )

    return summary


def _summary_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    primary = (
        item["mean_gap_to_reference"]
        if item["mean_gap_to_reference"] is not None
        else item["mean_best_length"]
    )
    return (
        math.inf if primary is None else float(primary),
        math.inf if item["mean_runtime_ms"] is None else float(item["mean_runtime_ms"]),
        math.inf if item["mean_evaluations"] is None else float(item["mean_evaluations"]),
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _flatten_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_param_keys = sorted({key for item in summaries for key in item["params"]})
    all_metric_keys = [
        key
        for key in sorted({key for item in summaries for key in item})
        if key not in {"experiment", "algorithm", "graph", "params", "runs"}
    ]
    rows = []
    for item in summaries:
        row = {
            "experiment": item["experiment"],
            "algorithm": item["algorithm"],
            "graph": item["graph"],
        }
        for key in all_metric_keys:
            row[key] = item.get(key)
        for key in all_param_keys:
            row[key] = item["params"].get(key)
        rows.append(row)
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(config: dict[str, Any]) -> dict[str, Any]:
    default_graph = config.get("graph")
    if not default_graph:
        raise ValueError("Top-level 'graph' is required.")

    seeds = _resolve_seeds(config["seeds"])
    reference_best_length = config.get("reference_best_length")
    if reference_best_length is not None:
        reference_best_length = float(reference_best_length)

    graph_cache: dict[Path, Any] = {}
    summaries = []

    expanded_experiments = []
    for experiment in config["experiments"]:
        algorithm = _normalize_algorithm(experiment["algorithm"])
        combinations = _expand_grid(experiment.get("grid", {}))
        for params in combinations:
            expanded_experiments.append(
                {
                    "name": experiment["name"],
                    "algorithm": algorithm,
                    "graph": experiment.get("graph", default_graph),
                    "fixed": dict(experiment.get("fixed", {})),
                    "params": params,
                }
            )

    total = len(expanded_experiments)
    for index, experiment in enumerate(expanded_experiments, start=1):
        graph_path = Path(experiment["graph"]).expanduser()
        if not graph_path.is_absolute():
            graph_path = (Path(config.get("_config_dir", ".")).resolve() / graph_path).resolve()
        else:
            graph_path = graph_path.resolve()

        if graph_path not in graph_cache:
            graph_cache[graph_path] = load_tsp_graph(graph_path)
        graph = graph_cache[graph_path]

        runner = RUNNERS[experiment["algorithm"]]
        base_params = {
            **experiment["fixed"],
            **experiment["params"],
            "graph": graph,
        }
        filtered_params = _filter_runner_kwargs(runner, base_params)
        if "seed" in filtered_params:
            raise ValueError("Do not set 'seed' inside experiment params. Use top-level 'seeds' instead.")

        seed_runs = []
        for seed in seeds:
            runtime_start = time.perf_counter()
            result = runner(**filtered_params, seed=seed)
            runtime_ms = (time.perf_counter() - runtime_start) * 1000.0
            analysis = _analyze_result(
                result,
                reference_best_length=reference_best_length,
            )
            analysis["elapsed_ms"] = runtime_ms
            seed_runs.append(
                {
                    "seed": seed,
                    "runtime_ms": runtime_ms,
                    "analysis": analysis,
                    "best_route_labels": result.get("best_route_labels"),
                }
            )

        saved_params = {key: value for key, value in filtered_params.items() if key != "graph"}
        summary = _build_summary(
            experiment["name"],
            experiment["algorithm"],
            saved_params,
            graph.name,
            reference_best_length,
            seed_runs,
        )
        summaries.append(summary)

        primary_metric = (
            summary["mean_gap_to_reference"]
            if summary["mean_gap_to_reference"] is not None
            else summary["mean_best_length"]
        )
        print(
            f"[{index}/{total}] {experiment['name']} | {experiment['algorithm']} | "
            f"primary={_format_metric(primary_metric)} | "
            f"runtime_ms={_format_metric(summary['mean_runtime_ms'])} | "
            f"evals={_format_metric(summary['mean_evaluations'])}"
        )

    summaries.sort(key=_summary_sort_key)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "graph": str(default_graph),
        "reference_best_length": reference_best_length,
        "seeds": seeds,
        "summary": summaries,
    }


def _print_top(summaries: list[dict[str, Any]], top_n: int) -> None:
    print()
    print(f"Top {min(top_n, len(summaries))} configurations:")
    for index, item in enumerate(summaries[:top_n], start=1):
        params = ", ".join(f"{key}={value}" for key, value in item["params"].items())
        primary_metric = (
            item["mean_gap_to_reference"]
            if item["mean_gap_to_reference"] is not None
            else item["mean_best_length"]
        )
        print(
            f"{index}. {item['experiment']} | {item['algorithm']} | "
            f"primary={_format_metric(primary_metric)} | "
            f"runtime_ms={_format_metric(item['mean_runtime_ms'])} | "
            f"evals={_format_metric(item['mean_evaluations'])}"
        )
        print(f"   {params}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep for Lab6 TSP heuristics.")
    parser.add_argument("--config", required=True, help="Path to JSON config with experiments.")
    parser.add_argument("--output-dir", help="Directory for summary.json and summary.csv.")
    parser.add_argument("--top", type=int, default=10, help="How many top configurations to print.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_json(config_path)
    config["_config_dir"] = str(config_path.parent)

    result = run_sweep(config)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        config_path.parent / "benchmark_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, ensure_ascii=False, indent=2)

    csv_rows = _flatten_summary_rows(result["summary"])
    _write_csv(csv_path, csv_rows)

    _print_top(result["summary"], args.top)
    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
