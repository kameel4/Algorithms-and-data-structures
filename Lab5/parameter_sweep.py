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

import numpy as np

try:
    from .ga import run_ga, run_ga_bitwise
    from .main import f_xy, fitness
    from .pso import run_pso
except ImportError:
    from ga import run_ga, run_ga_bitwise
    from main import f_xy, fitness
    from pso import run_pso


RUNNERS = {
    "ga": run_ga,
    "ga-bit": run_ga_bitwise,
    "pso": run_pso,
}


def _normalize_algorithm(name):
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "ga": "ga",
        "ga-arithmetic": "ga",
        "ga-bit": "ga-bit",
        "ga-bitwise": "ga-bit",
        "ga-gray": "ga-bit",
        "pso": "pso",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported algorithm: {name}")
    return aliases[normalized]


def _clean_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _expand_range_spec(spec):
    if isinstance(spec, list):
        return [_clean_number(value) for value in spec]

    if not isinstance(spec, dict):
        return [spec]

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


def _expand_grid(grid):
    if not grid:
        return [dict()]

    items = list(grid.items())
    keys = [key for key, _ in items]
    value_lists = [_expand_range_spec(spec) for _, spec in items]
    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    return combinations


def _load_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def _resolve_seeds(spec):
    values = _expand_range_spec(spec)
    seeds = [int(value) for value in values]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _resolve_reference_value(config):
    if "reference_value" in config:
        return float(config["reference_value"])

    reference_point = np.asarray(config["reference_point"], dtype=float)
    return float(f_xy(reference_point[0], reference_point[1]))


def _filter_runner_kwargs(runner, params):
    signature = inspect.signature(runner)
    allowed = set(signature.parameters)
    filtered = {key: value for key, value in params.items() if key in allowed}
    missing = [
        key for key, value in signature.parameters.items()
        if value.default is inspect._empty and key not in filtered and key != "fitness"
    ]
    if missing:
        raise ValueError(f"Missing required runner parameters: {missing}")
    return filtered


def _evaluate_frame(mode, *, positions, best_point, best_value, reference_point, reference_value, success_config):
    point_distances = np.linalg.norm(positions - reference_point, axis=1)
    best_distance = float(np.linalg.norm(best_point - reference_point))
    best_gap = float(best_value) - float(reference_value)
    best_ok = (
        best_distance <= float(success_config["best_distance_tol"])
        and best_gap <= float(success_config["best_gap_tol"])
    )

    if mode == "best":
        success = best_ok
    elif mode == "converged":
        success = best_ok and float(np.max(point_distances)) <= float(success_config["population_max_distance_tol"])
    else:
        raise ValueError(f"Unsupported success mode: {mode}")

    return {
        "success": success,
        "best_gap": best_gap,
        "best_distance": best_distance,
        "mean_distance": float(np.mean(point_distances)),
        "p90_distance": float(np.quantile(point_distances, 0.9)),
        "max_distance": float(np.max(point_distances)),
    }


def _analyze_result(result, *, reference_point, reference_value, success_config):
    mode = success_config["mode"]
    first_success_step = None
    final_metrics = None

    for step, (positions, best_value, best_point) in enumerate(
        zip(
            result["history_positions"],
            result["history_best"],
            result["history_best_point"],
        )
    ):
        metrics = _evaluate_frame(
            mode,
            positions=np.asarray(positions, dtype=float),
            best_point=np.asarray(best_point, dtype=float),
            best_value=float(best_value),
            reference_point=reference_point,
            reference_value=reference_value,
            success_config=success_config,
        )
        if metrics["success"] and first_success_step is None:
            first_success_step = step
        final_metrics = metrics

    population_size = int(np.asarray(result["history_positions"][0]).shape[0])
    total_steps = len(result["history_positions"])

    return {
        "final_success": bool(final_metrics["success"]),
        "first_success_step": first_success_step,
        "evaluations_to_success": None if first_success_step is None else population_size * (first_success_step + 1),
        "total_evaluations": population_size * total_steps,
        "final_best_gap": final_metrics["best_gap"],
        "final_best_distance": final_metrics["best_distance"],
        "final_mean_distance": final_metrics["mean_distance"],
        "final_p90_distance": final_metrics["p90_distance"],
        "final_max_distance": final_metrics["max_distance"],
    }


def _mean_or_none(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _build_summary(experiment_name, algorithm, params, seed_runs):
    success_runs = [item for item in seed_runs if item["analysis"]["final_success"]]
    return {
        "experiment": experiment_name,
        "algorithm": algorithm,
        "params": params,
        "success_count": len(success_runs),
        "success_rate": len(success_runs) / len(seed_runs),
        "mean_first_success_step": _mean_or_none(
            [item["analysis"]["first_success_step"] for item in success_runs]
        ),
        "mean_evaluations_to_success": _mean_or_none(
            [item["analysis"]["evaluations_to_success"] for item in success_runs]
        ),
        "mean_total_evaluations": _mean_or_none(
            [item["analysis"]["total_evaluations"] for item in seed_runs]
        ),
        "mean_runtime_ms": _mean_or_none([item["runtime_ms"] for item in seed_runs]),
        "mean_final_best_gap": _mean_or_none(
            [item["analysis"]["final_best_gap"] for item in seed_runs]
        ),
        "mean_final_best_distance": _mean_or_none(
            [item["analysis"]["final_best_distance"] for item in seed_runs]
        ),
        "mean_final_mean_distance": _mean_or_none(
            [item["analysis"]["final_mean_distance"] for item in seed_runs]
        ),
        "mean_final_p90_distance": _mean_or_none(
            [item["analysis"]["final_p90_distance"] for item in seed_runs]
        ),
        "mean_final_max_distance": _mean_or_none(
            [item["analysis"]["final_max_distance"] for item in seed_runs]
        ),
        "runs": seed_runs,
    }


def _summary_sort_key(item):
    return (
        -item["success_count"],
        math.inf if item["mean_first_success_step"] is None else item["mean_first_success_step"],
        math.inf if item["mean_evaluations_to_success"] is None else item["mean_evaluations_to_success"],
        math.inf if item["mean_runtime_ms"] is None else item["mean_runtime_ms"],
        math.inf if item["mean_final_max_distance"] is None else item["mean_final_max_distance"],
    )


def _format_metric(value):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _flatten_summary_rows(summaries):
    all_param_keys = sorted({key for item in summaries for key in item["params"]})
    rows = []
    for item in summaries:
        row = {
            "experiment": item["experiment"],
            "algorithm": item["algorithm"],
            "success_count": item["success_count"],
            "success_rate": item["success_rate"],
            "mean_first_success_step": item["mean_first_success_step"],
            "mean_evaluations_to_success": item["mean_evaluations_to_success"],
            "mean_total_evaluations": item["mean_total_evaluations"],
            "mean_runtime_ms": item["mean_runtime_ms"],
            "mean_final_best_gap": item["mean_final_best_gap"],
            "mean_final_best_distance": item["mean_final_best_distance"],
            "mean_final_mean_distance": item["mean_final_mean_distance"],
            "mean_final_p90_distance": item["mean_final_p90_distance"],
            "mean_final_max_distance": item["mean_final_max_distance"],
        }
        for key in all_param_keys:
            row[key] = item["params"].get(key)
        rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(config):
    reference_point = np.asarray(config["reference_point"], dtype=float)
    reference_value = _resolve_reference_value(config)
    seeds = _resolve_seeds(config["seeds"])
    bounds = tuple(float(value) for value in config["bounds"])
    init_mode = str(config.get("init_mode", "random"))
    success_config = dict(config["success"])
    success_config.setdefault("mode", "converged")
    success_config.setdefault("best_distance_tol", 1e-2)
    success_config.setdefault("best_gap_tol", 1e-6)
    if success_config["mode"] == "converged":
        success_config.setdefault("population_max_distance_tol", 5.0)

    expanded = []
    for experiment in config["experiments"]:
        algorithm = _normalize_algorithm(experiment["algorithm"])
        combinations = _expand_grid(experiment.get("grid", {}))
        for params in combinations:
            expanded.append(
                {
                    "name": experiment["name"],
                    "algorithm": algorithm,
                    "fixed": dict(experiment.get("fixed", {})),
                    "params": params,
                }
            )

    summaries = []
    total = len(expanded)
    for index, experiment in enumerate(expanded, start=1):
        runner = RUNNERS[experiment["algorithm"]]
        runner_params = {
            **experiment["fixed"],
            **experiment["params"],
            "bounds": bounds,
            "init_mode": init_mode,
            "fitness": fitness,
        }
        filtered_params = _filter_runner_kwargs(runner, runner_params)
        if "seed" in filtered_params:
            raise ValueError("Do not set 'seed' inside experiment params. Use top-level 'seeds' instead.")

        seed_runs = []
        for seed in seeds:
            runtime_start = time.perf_counter()
            result = runner(**filtered_params, seed=seed)
            runtime_ms = (time.perf_counter() - runtime_start) * 1000.0
            analysis = _analyze_result(
                result,
                reference_point=reference_point,
                reference_value=reference_value,
                success_config=success_config,
            )
            seed_runs.append(
                {
                    "seed": seed,
                    "runtime_ms": runtime_ms,
                    "analysis": analysis,
                }
            )

        saved_params = {key: value for key, value in filtered_params.items() if key != "fitness"}
        summary = _build_summary(
            experiment["name"],
            experiment["algorithm"],
            saved_params,
            seed_runs,
        )
        summaries.append(summary)

        print(
            f"[{index}/{total}] {experiment['name']} | {experiment['algorithm']} | "
            f"success={summary['success_count']}/{len(seeds)} | "
            f"first={_format_metric(summary['mean_first_success_step'])} | "
            f"evals={_format_metric(summary['mean_evaluations_to_success'])} | "
            f"max_dist={_format_metric(summary['mean_final_max_distance'])}"
        )

    summaries.sort(key=_summary_sort_key)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_point": reference_point.tolist(),
        "reference_value": reference_value,
        "bounds": list(bounds),
        "init_mode": init_mode,
        "seeds": seeds,
        "success": success_config,
        "summary": summaries,
    }


def _print_top(summaries, top_n):
    print()
    print(f"Top {min(top_n, len(summaries))} configurations:")
    for index, item in enumerate(summaries[:top_n], start=1):
        params = ", ".join(f"{key}={value}" for key, value in item["params"].items() if key != "fitness")
        print(
            f"{index}. {item['experiment']} | {item['algorithm']} | "
            f"success={item['success_count']} | first={_format_metric(item['mean_first_success_step'])} | "
            f"evals={_format_metric(item['mean_evaluations_to_success'])} | "
            f"runtime_ms={_format_metric(item['mean_runtime_ms'])}"
        )
        print(f"   {params}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for Lab5 optimization algorithms.")
    parser.add_argument("--config", required=True, help="Path to JSON config with experiments.")
    parser.add_argument("--output-dir", help="Directory for summary.json and summary.csv.")
    parser.add_argument("--top", type=int, default=10, help="How many top configurations to print.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_json(config_path)
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
