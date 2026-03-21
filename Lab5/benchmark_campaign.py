from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from parameter_sweep import _flatten_summary_rows, _write_csv, run_sweep


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "benchmark_results"
REFERENCE_POINT = [512.0, 404.2319]
COMMON_SUCCESS = {
    "mode": "converged",
    "best_distance_tol": 0.01,
    "best_gap_tol": 0.000001,
    "population_max_distance_tol": 5.0,
}


def _experiment(name, algorithm, *, init_mode, fixed, grid=None):
    payload = {
        "name": name,
        "algorithm": algorithm,
        "init_mode": init_mode,
        "fixed": dict(fixed),
    }
    if grid:
        payload["grid"] = grid
    return payload


def _base_config(seed_stop, experiments):
    return {
        "bounds": [-512, 512],
        "init_mode": "random",
        "reference_point": list(REFERENCE_POINT),
        "seeds": {
            "start": 0,
            "stop": seed_stop,
            "step": 1,
        },
        "success": dict(COMMON_SUCCESS),
        "experiments": experiments,
    }


def _stage1_config():
    experiments = [
        _experiment(
            "ga_real_random",
            "ga",
            init_mode="random",
            fixed={
                "pop_size": 1024,
                "crossover_prob": 0.8,
                "tournament_k": 3,
            },
            grid={
                "generations": {"values": [54, 56, 60, 62]},
                "mutation_prob": {"values": [0.35, 0.4]},
                "elite_size": {"values": [16, 24]},
                "sigma0": {"values": [60.0, 70.0, 80.0]},
            },
        ),
        _experiment(
            "ga_real_grid_probe",
            "ga",
            init_mode="grid",
            fixed={
                "pop_size": 1024,
                "crossover_prob": 0.8,
                "tournament_k": 3,
                "elite_size": 16,
            },
            grid={
                "generations": {"values": [54, 60]},
                "mutation_prob": {"values": [0.35, 0.4]},
                "sigma0": {"values": [60.0, 70.0]},
            },
        ),
        _experiment(
            "ga_bit_random",
            "ga-bit",
            init_mode="random",
            fixed={
                "pop_size": 1024,
                "crossover_prob": 0.95,
                "tournament_k": 5,
            },
            grid={
                "generations": {"values": [90, 100, 110]},
                "mutation_prob": {"values": [0.07, 0.08, 0.09]},
                "elite_size": {"values": [24, 32]},
                "bit_width": {"values": [16, 20]},
            },
        ),
        _experiment(
            "ga_bit_grid_probe",
            "ga-bit",
            init_mode="grid",
            fixed={
                "pop_size": 1024,
                "crossover_prob": 0.95,
                "tournament_k": 5,
                "bit_width": 16,
            },
            grid={
                "generations": {"values": [90, 100]},
                "mutation_prob": {"values": [0.08, 0.09]},
                "elite_size": {"values": [24, 32]},
            },
        ),
        _experiment(
            "pso_grid",
            "pso",
            init_mode="grid",
            fixed={
                "use_constriction": True,
            },
            grid={
                "swarm_size": {"values": [768, 1024, 1536]},
                "iterations": {"values": [360, 420, 480]},
                "c1": {"values": [1.4, 1.6]},
                "c2": {"values": [2.8, 3.0]},
                "vmax_ratio": {"values": [0.06, 0.08]},
            },
        ),
        _experiment(
            "pso_random_probe",
            "pso",
            init_mode="random",
            fixed={
                "use_constriction": True,
            },
            grid={
                "swarm_size": {"values": [768, 1024]},
                "iterations": {"values": [360, 420]},
                "c1": {"values": [1.4, 1.6]},
                "c2": {"values": [2.8, 3.0]},
                "vmax_ratio": {"values": [0.06, 0.08]},
            },
        ),
    ]
    return _base_config(seed_stop=2, experiments=experiments)


def _summary_rank_key(item):
    runtime = math.inf if item["mean_runtime_ms"] is None else item["mean_runtime_ms"]
    max_distance = math.inf if item["mean_final_max_distance"] is None else item["mean_final_max_distance"]
    first_step = math.inf if item["mean_first_success_step"] is None else item["mean_first_success_step"]
    return (-item["success_count"], runtime, max_distance, first_step)


def _group_top_by_algorithm(summary, *, top_n):
    grouped = {}
    for item in summary:
        grouped.setdefault(item["algorithm"], []).append(item)

    selected = []
    for algorithm, items in grouped.items():
        ordered = sorted(items, key=_summary_rank_key)
        selected.extend(ordered[:top_n])
        print()
        print(f"Top {top_n} for {algorithm}:")
        for index, candidate in enumerate(ordered[:top_n], start=1):
            print(
                f"  {index}. {candidate['experiment']} | success={candidate['success_count']} | "
                f"runtime_ms={candidate['mean_runtime_ms']:.3f} | "
                f"max_dist={candidate['mean_final_max_distance']:.6f}"
            )

    return selected


def _params_to_fixed(params):
    fixed = dict(params)
    fixed.pop("bounds", None)
    fixed.pop("init_mode", None)
    return fixed


def _build_followup_config(*, stage_name, seed_stop, selected):
    experiments = []
    for index, item in enumerate(selected, start=1):
        experiments.append(
            _experiment(
                f"{stage_name}_{item['algorithm']}_{index}",
                item["algorithm"],
                init_mode=item["params"]["init_mode"],
                fixed=_params_to_fixed(item["params"]),
            )
        )
    return _base_config(seed_stop=seed_stop, experiments=experiments)


def _save_stage(stage_name, config, result, output_dir):
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    config_path = stage_dir / "config.json"
    summary_path = stage_dir / "summary.json"
    csv_path = stage_dir / "summary.csv"

    with config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, ensure_ascii=False, indent=2)

    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, ensure_ascii=False, indent=2)

    csv_rows = _flatten_summary_rows(result["summary"])
    _write_csv(csv_path, csv_rows)
    return stage_dir


def _format_param_value(value):
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def _format_params(params):
    ordered_keys = [
        key for key in params
        if key not in {"bounds", "init_mode"}
    ]
    return ", ".join(f"{key}={_format_param_value(params[key])}" for key in ordered_keys)


def _collect_best_per_algorithm(summary):
    grouped = {}
    for item in summary:
        grouped.setdefault(item["algorithm"], []).append(item)

    best = []
    for algorithm, items in grouped.items():
        best.append(sorted(items, key=_summary_rank_key)[0])
    return sorted(best, key=_summary_rank_key)


def _write_final_report(*, output_dir, stage1_result, stage2_result, stage3_result):
    final_best = _collect_best_per_algorithm(stage3_result["summary"])
    report_path = output_dir / "final_report.md"
    table_path = output_dir / "final_best_configs.csv"

    rows = []
    for item in final_best:
        rows.append(
            {
                "algorithm": item["algorithm"],
                "success_rate": item["success_rate"],
                "success_count": item["success_count"],
                "mean_runtime_ms": item["mean_runtime_ms"],
                "mean_final_max_distance": item["mean_final_max_distance"],
                "mean_first_success_step": item["mean_first_success_step"],
                "init_mode": item["params"]["init_mode"],
                "params": _format_params(item["params"]),
            }
        )

    _write_csv(table_path, rows)

    lines = [
        "# Lab5 Benchmark Campaign",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Best Confirmed Configuration Per Algorithm",
        "",
        "| Algorithm | Success | Success Rate | Mean Runtime (ms) | Mean Final Max Distance | Mean First Success Step | Init Mode | Parameters |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in final_best:
        lines.append(
            "| "
            f"{item['algorithm']} | "
            f"{item['success_count']} | "
            f"{item['success_rate']:.3f} | "
            f"{item['mean_runtime_ms']:.3f} | "
            f"{item['mean_final_max_distance']:.6f} | "
            f"{'-' if item['mean_first_success_step'] is None else f'{item['mean_first_success_step']:.3f}'} | "
            f"{item['params']['init_mode']} | "
            f"{_format_params(item['params'])} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Stage 1 summary: `{output_dir / 'stage1' / 'summary.json'}`",
            f"- Stage 2 summary: `{output_dir / 'stage2' / 'summary.json'}`",
            f"- Stage 3 summary: `{output_dir / 'stage3' / 'summary.json'}`",
            f"- Final best CSV: `{table_path}`",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, table_path


def run_campaign(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: broad search (3 seeds) ===")
    stage1_config = _stage1_config()
    stage1_result = run_sweep(stage1_config)
    _save_stage("stage1", stage1_config, stage1_result, output_dir)

    print()
    print("=== Stage 2: finalists confirmation (20 seeds) ===")
    stage1_top = _group_top_by_algorithm(stage1_result["summary"], top_n=4)
    stage2_config = _build_followup_config(stage_name="stage2", seed_stop=19, selected=stage1_top)
    stage2_result = run_sweep(stage2_config)
    _save_stage("stage2", stage2_config, stage2_result, output_dir)

    print()
    print("=== Stage 3: final confirmation (50 seeds) ===")
    stage2_top = _group_top_by_algorithm(stage2_result["summary"], top_n=2)
    stage3_config = _build_followup_config(stage_name="stage3", seed_stop=49, selected=stage2_top)
    stage3_result = run_sweep(stage3_config)
    _save_stage("stage3", stage3_config, stage3_result, output_dir)

    report_path, table_path = _write_final_report(
        output_dir=output_dir,
        stage1_result=stage1_result,
        stage2_result=stage2_result,
        stage3_result=stage3_result,
    )

    print()
    print(f"Campaign output: {output_dir}")
    print(f"Final report   : {report_path}")
    print(f"Final table    : {table_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a multi-stage benchmark campaign for Lab5 algorithms.")
    parser.add_argument("--output-dir", help="Directory for campaign artifacts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        DEFAULT_OUTPUT_ROOT / f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_campaign(output_dir=output_dir)


if __name__ == "__main__":
    main()
