from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .annealing import load_tsp_graph, run_simulated_annealing
    from .ant_colony import run_basic_ant_colony, run_elitist_ant_colony
except ImportError:
    from annealing import load_tsp_graph, run_simulated_annealing
    from ant_colony import run_basic_ant_colony, run_elitist_ant_colony


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_RESULTS_DIR = ROOT_DIR / "benchmark_results"

RUNNERS = {
    "sa_geometric": run_simulated_annealing,
    "sa_cauchy": run_simulated_annealing,
    "ant_basic": run_basic_ant_colony,
    "ant_elitist": run_elitist_ant_colony,
}

GRAPH_SPECS = {
    "figure1.edgelist": {
        "path": DATA_DIR / "figure1.edgelist",
        "display_name": "figure1",
        "reference_best_length": 14.0,
        "seeds": list(range(1, 11)),
    },
    "berlin52.stp": {
        "path": DATA_DIR / "berlin52.stp",
        "display_name": "berlin52",
        "reference_best_length": None,
        "seeds": [1, 2, 3, 4, 5],
    },
    "world666.stp": {
        "path": DATA_DIR / "world666.stp",
        "display_name": "world666",
        "reference_best_length": None,
        "seeds": [1, 2, 3],
    },
}


def _variant_plans(graph_key: str) -> dict[str, dict[str, Any]]:
    common_sa_default = {
        "initial_temperature": 100.0,
        "iterations_per_temperature": 50,
        "min_temperature": 0.1,
    }
    figure1_sa_geo = [
        ("initial_temperature", [1, 2, 5, 10, 20, 50, 100]),
        ("iterations_per_temperature", [5, 10, 20, 50, 100]),
        ("alpha", [0.8, 0.85, 0.9, 0.95, 0.98]),
        ("min_temperature", [0.01, 0.05, 0.1]),
    ]
    berlin_sa_geo = [
        ("initial_temperature", [50, 100, 200, 400, 800]),
        ("iterations_per_temperature", [20, 50, 100, 150]),
        ("alpha", [0.9, 0.93, 0.95, 0.97, 0.975]),
        ("min_temperature", [0.05, 0.1, 0.2]),
    ]
    world_sa_geo = [
        ("initial_temperature", [100, 400, 1000, 3000]),
        ("iterations_per_temperature", [5, 10, 20, 50]),
        ("alpha", [0.95, 0.97, 0.99, 0.995]),
        ("min_temperature", [0.1, 5.0, 10.0, 20.0]),
    ]

    if graph_key == "figure1.edgelist":
        ant_default = {
            "iterations": 100,
            "ant_count": 6,
            "pheromone_importance": 1.0,
            "distance_importance": 2.0,
            "pheromone_deposit": 100.0,
            "evaporation_intensity": 3.0,
            "initial_pheromone": 1.0,
        }
        ant_sweeps = [
            ("ant_count", [3, 6, 9]),
            ("iterations", [10, 20, 40, 100]),
            ("distance_importance", [1.0, 2.0, 3.0, 4.0]),
            ("pheromone_importance", [0.5, 1.0, 1.5, 2.0]),
            ("evaporation_intensity", [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("pheromone_deposit", [50.0, 100.0, 200.0]),
        ]
        elite_sweeps = ant_sweeps + [("elite_ants", [1, 2, 5, 8])]
        sa_geo = figure1_sa_geo
        sa_cauchy = [
            ("initial_temperature", [1, 2, 5, 10, 20, 50, 100]),
            ("iterations_per_temperature", [5, 10, 20, 50, 100]),
            ("min_temperature", [0.01, 0.05, 0.1]),
        ]
    elif graph_key == "berlin52.stp":
        ant_default = {
            "iterations": 100,
            "ant_count": 52,
            "pheromone_importance": 1.0,
            "distance_importance": 2.0,
            "pheromone_deposit": 100.0,
            "evaporation_intensity": 3.0,
            "initial_pheromone": 1.0,
        }
        ant_sweeps = [
            ("ant_count", [20, 30, 40, 52, 80]),
            ("iterations", [20, 40, 60, 100, 150]),
            ("distance_importance", [1.0, 2.0, 3.0]),
            ("pheromone_importance", [0.5, 1.0, 1.5, 2.0]),
            ("evaporation_intensity", [2.0, 3.0, 4.0, 5.0]),
            ("pheromone_deposit", [50.0, 100.0, 200.0]),
        ]
        elite_sweeps = ant_sweeps + [("elite_ants", [1, 3, 5, 8])]
        sa_geo = berlin_sa_geo
        sa_cauchy = [
            ("initial_temperature", [50, 100, 200, 400, 800]),
            ("iterations_per_temperature", [20, 50, 100, 150]),
            ("min_temperature", [0.05, 0.1, 0.2]),
        ]
    else:
        ant_default = {
            "iterations": 8,
            "ant_count": 8,
            "pheromone_importance": 1.0,
            "distance_importance": 2.0,
            "pheromone_deposit": 100.0,
            "evaporation_intensity": 3.0,
            "initial_pheromone": 1.0,
        }
        ant_sweeps = [
            ("ant_count", [8, 16]),
            ("iterations", [4, 8, 12]),
            ("distance_importance", [1.0, 2.0, 3.0]),
            ("pheromone_importance", [0.5, 1.0, 1.5]),
            ("evaporation_intensity", [2.0, 3.0, 4.0]),
            ("pheromone_deposit", [50.0, 100.0]),
        ]
        elite_sweeps = ant_sweeps + [("elite_ants", [1, 3, 5])]
        sa_geo = world_sa_geo
        sa_cauchy = [
            ("initial_temperature", [100, 400, 1000, 3000]),
            ("iterations_per_temperature", [5, 10, 20, 50]),
            ("min_temperature", [0.1, 5.0, 10.0, 20.0]),
        ]

    return {
        "sa_geometric": {
            "label": "Simulated annealing (geometric)",
            "runner": "sa_geometric",
            "baseline": {
                **common_sa_default,
                "cooling_mode": "geometric",
                "alpha": 0.95,
            },
            "sweeps": sa_geo,
            "baseline_kind": "lab_default",
        },
        "sa_cauchy": {
            "label": "Simulated annealing (cauchy)",
            "runner": "sa_cauchy",
            "baseline": {
                **common_sa_default,
                "cooling_mode": "cauchy",
            },
            "sweeps": sa_cauchy,
            "baseline_kind": "lab_default",
        },
        "ant_basic": {
            "label": "Ant colony",
            "runner": "ant_basic",
            "baseline": ant_default,
            "sweeps": ant_sweeps,
            "baseline_kind": "lab_default" if graph_key != "world666.stp" else "budget_baseline",
        },
        "ant_elitist": {
            "label": "Elitist ant colony",
            "runner": "ant_elitist",
            "baseline": {
                **ant_default,
                "elite_ants": 5 if graph_key != "world666.stp" else 3,
            },
            "sweeps": elite_sweeps,
            "baseline_kind": "lab_default" if graph_key != "world666.stp" else "budget_baseline",
        },
    }


def _normalize_graphs(values: list[str] | None) -> list[str]:
    if not values:
        return list(GRAPH_SPECS)
    selected = []
    for value in values:
        token = value.strip().lower()
        if token in {"all", "*"}:
            return list(GRAPH_SPECS)
        matches = [
            graph_key
            for graph_key in GRAPH_SPECS
            if graph_key.lower() == token or Path(graph_key).stem.lower() == token
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown graph selection: {value}")
        selected.append(matches[0])
    return selected


def _normalize_families(values: list[str] | None) -> list[str]:
    if not values:
        return ["sa", "aco"]
    families = []
    for value in values:
        token = value.strip().lower()
        if token in {"all", "*"}:
            return ["sa", "aco"]
        aliases = {
            "sa": "sa",
            "annealing": "sa",
            "aco": "aco",
            "ant": "aco",
        }
        if token not in aliases:
            raise ValueError(f"Unknown family selection: {value}")
        family = aliases[token]
        if family not in families:
            families.append(family)
    return families


def _variants_for_families(families: list[str]) -> list[str]:
    variants = []
    if "sa" in families:
        variants.extend(["sa_geometric", "sa_cauchy"])
    if "aco" in families:
        variants.extend(["ant_basic", "ant_elitist"])
    return variants


def _sort_values(values: list[Any]) -> list[Any]:
    def sort_key(value: Any) -> tuple[int, Any]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, float(value))
        return (1, str(value))

    return sorted(values, key=sort_key)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _summary_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    hit_rate = summary.get("hit_rate")
    if hit_rate is None:
        hit_rate = 0.0
    return (
        float(summary["mean_best_length"]),
        -float(hit_rate),
        float(summary["mean_runtime_ms"]),
        float(summary["std_best_length"]),
    )


def _run_config(
    graph: Any,
    *,
    runner_key: str,
    params: dict[str, Any],
    seeds: list[int],
    reference_best_length: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runner = RUNNERS[runner_key]
    runs = []
    for seed in seeds:
        started = time.perf_counter()
        error_text = None
        best_length = math.inf
        try:
            result = runner(graph=graph, seed=seed, **params)
            best_length = float(result["best_length"])
        except Exception as exc:
            result = {}
            error_text = f"{type(exc).__name__}: {exc}"
        runtime_ms = (time.perf_counter() - started) * 1000.0
        runs.append(
            {
                "seed": seed,
                "best_length": best_length,
                "runtime_ms": runtime_ms,
                "hit_reference": (
                    None
                    if reference_best_length is None or not math.isfinite(best_length)
                    else int(math.isclose(best_length, reference_best_length, rel_tol=0.0, abs_tol=1e-9))
                ),
                "best_route_labels": result.get("best_route_labels"),
                "error": error_text,
            }
        )

    finite_runs = [item for item in runs if math.isfinite(item["best_length"])]
    if not finite_runs:
        summary = {
            "mean_best_length": math.inf,
            "std_best_length": math.inf,
            "median_best_length": math.inf,
            "best_best_length": math.inf,
            "worst_best_length": math.inf,
            "mean_runtime_ms": float(np.mean([item["runtime_ms"] for item in runs])),
            "hit_rate": 0.0,
            "success_rate": 0.0,
        }
        return summary, runs

    lengths = np.array([item["best_length"] for item in finite_runs], dtype=float)
    hit_values = [item["hit_reference"] for item in finite_runs if item["hit_reference"] is not None]
    summary = {
        "mean_best_length": float(np.mean(lengths)),
        "std_best_length": float(np.std(lengths)),
        "median_best_length": float(np.median(lengths)),
        "best_best_length": float(np.min(lengths)),
        "worst_best_length": float(np.max(lengths)),
        "mean_runtime_ms": float(np.mean([item["runtime_ms"] for item in runs])),
        "hit_rate": float(np.mean(hit_values)) if hit_values else None,
        "success_rate": float(len(finite_runs) / len(runs)),
    }
    return summary, runs


def _ensure_value_list(values: list[Any], baseline_value: Any) -> list[Any]:
    result = list(values)
    if baseline_value not in result:
        result.append(baseline_value)
    return _sort_values(result)


def _plot_parameter_sweep(
    *,
    title: str,
    parameter_name: str,
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    reference_best_length: float | None,
    baseline_value: Any,
    tuned_value: Any,
    output_path: Path,
) -> None:
    values = [row["value"] for row in summary_rows]
    labels = [_format_value(value) for value in values]
    positions = np.arange(len(values), dtype=float)
    grouped_lengths = [
        [item["best_length"] for item in raw_rows if item["value"] == value and math.isfinite(item["best_length"])]
        for value in values
    ]
    means = [row["mean_best_length"] for row in summary_rows]
    stds = [row["std_best_length"] for row in summary_rows]

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    axes[0].boxplot(
        grouped_lengths,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#d9e8fb", "edgecolor": "#24557a"},
        medianprops={"color": "#c45100", "linewidth": 1.8},
        whiskerprops={"color": "#24557a"},
        capprops={"color": "#24557a"},
    )
    for index, lengths in enumerate(grouped_lengths):
        if not lengths:
            continue
        jitter = np.linspace(-0.12, 0.12, num=len(lengths))
        axes[0].scatter(positions[index] + jitter, lengths, color="#24557a", alpha=0.75, s=24)
    axes[0].plot(positions, means, color="#cf3c2c", marker="o", linewidth=2.0, label="Mean over seeds")
    if reference_best_length is not None:
        axes[0].axhline(reference_best_length, color="#1d7a3a", linestyle="--", linewidth=1.4, label="Reference best")
    axes[0].set_ylabel("Best path length")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(loc="best")

    axes[1].errorbar(positions, means, yerr=stds, color="#cf3c2c", marker="o", linewidth=2.0, capsize=4)
    axes[1].set_ylabel("Mean best length")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].set_xticks(positions, labels, rotation=30, ha="right")
    axes[1].set_xlabel(parameter_name)

    if baseline_value in values:
        baseline_index = values.index(baseline_value)
        axes[1].scatter([positions[baseline_index]], [means[baseline_index]], color="#222222", s=70, label="Baseline")
    if tuned_value in values:
        tuned_index = values.index(tuned_value)
        axes[1].scatter([positions[tuned_index]], [means[tuned_index]], color="#1d7a3a", s=70, label="Selected")
    if axes[1].legend_ is None:
        axes[1].legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_variant_distribution(
    *,
    title: str,
    series: list[dict[str, Any]],
    output_path: Path,
    reference_best_length: float | None,
) -> None:
    labels = [item["label"] for item in series]
    positions = np.arange(len(series), dtype=float)
    grouped_lengths = [item["lengths"] for item in series]

    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.boxplot(
        grouped_lengths,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#f4dfc5", "edgecolor": "#7a4f1d"},
        medianprops={"color": "#24557a", "linewidth": 1.8},
    )
    for index, lengths in enumerate(grouped_lengths):
        jitter = np.linspace(-0.12, 0.12, num=len(lengths))
        axis.scatter(positions[index] + jitter, lengths, color="#7a4f1d", alpha=0.75, s=24)
    if reference_best_length is not None:
        axis.axhline(reference_best_length, color="#1d7a3a", linestyle="--", linewidth=1.4, label="Reference best")
        axis.legend(loc="best")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Best path length")
    axis.set_title(title)
    axis.grid(alpha=0.25, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _parameter_slug(parameter_name: str) -> str:
    return parameter_name.lower().replace("-", "_")


def _write_variant_report(variant_dir: Path, *, graph_name: str, variant_plan: dict[str, Any], result: dict[str, Any]) -> None:
    baseline = result["baseline_summary"]
    final = result["final_summary"]
    lines = [
        f"# {variant_plan['label']} on {graph_name}",
        "",
        f"- Baseline kind: `{variant_plan['baseline_kind']}`",
        f"- Baseline params: `{json.dumps(result['baseline_params'], ensure_ascii=False, sort_keys=True)}`",
        f"- Tuned params: `{json.dumps(result['tuned_params'], ensure_ascii=False, sort_keys=True)}`",
        f"- Baseline mean best length: `{baseline['mean_best_length']:.6f}`",
        f"- Tuned mean best length: `{final['mean_best_length']:.6f}`",
        f"- Improvement: `{baseline['mean_best_length'] - final['mean_best_length']:.6f}`",
        f"- Baseline best/worst: `{baseline['best_best_length']:.6f}` / `{baseline['worst_best_length']:.6f}`",
        f"- Tuned best/worst: `{final['best_best_length']:.6f}` / `{final['worst_best_length']:.6f}`",
        "",
        "## Parameter sweeps",
        "",
    ]
    for item in result["selection_history"]:
        lines.append(
            f"- `{item['parameter']}`: selected `{_format_value(item['selected_value'])}` "
            f"from baseline `{_format_value(item['baseline_value'])}`"
        )
    (variant_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_variant(
    *,
    graph_key: str,
    graph: Any,
    variant_key: str,
    variant_plan: dict[str, Any],
    seeds: list[int],
    reference_best_length: float | None,
    output_dir: Path,
) -> dict[str, Any]:
    variant_dir = output_dir / GRAPH_SPECS[graph_key]["display_name"] / variant_key
    plots_dir = variant_dir / "plots"
    variant_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_params = dict(variant_plan["baseline"])
    baseline_summary, baseline_runs = _run_config(
        graph,
        runner_key=variant_plan["runner"],
        params=baseline_params,
        seeds=seeds,
        reference_best_length=reference_best_length,
    )
    current_params = dict(baseline_params)
    selection_history = []

    for parameter_name, raw_values in variant_plan["sweeps"]:
        values = _ensure_value_list(list(raw_values), current_params[parameter_name])
        raw_rows = []
        summary_rows = []
        best_choice = None
        for value in values:
            candidate_params = dict(current_params)
            candidate_params[parameter_name] = value
            summary, runs = _run_config(
                graph,
                runner_key=variant_plan["runner"],
                params=candidate_params,
                seeds=seeds,
                reference_best_length=reference_best_length,
            )
            summary_row = {
                "parameter": parameter_name,
                "value": value,
                "mean_best_length": summary["mean_best_length"],
                "std_best_length": summary["std_best_length"],
                "best_best_length": summary["best_best_length"],
                "worst_best_length": summary["worst_best_length"],
                "mean_runtime_ms": summary["mean_runtime_ms"],
                "hit_rate": summary.get("hit_rate"),
                "success_rate": summary["success_rate"],
            }
            summary_rows.append(summary_row)
            for run in runs:
                raw_rows.append(
                    {
                        "parameter": parameter_name,
                        "value": value,
                        "seed": run["seed"],
                        "best_length": run["best_length"],
                        "runtime_ms": run["runtime_ms"],
                        "hit_reference": run["hit_reference"],
                        "error": run["error"],
                    }
                )
            if best_choice is None or _summary_key(summary) < _summary_key(best_choice["summary"]):
                best_choice = {"value": value, "summary": summary}

        current_params[parameter_name] = best_choice["value"]
        selection_history.append(
            {
                "parameter": parameter_name,
                "baseline_value": baseline_params[parameter_name],
                "selected_value": best_choice["value"],
            }
        )
        _write_csv(variant_dir / f"{_parameter_slug(parameter_name)}_raw.csv", raw_rows)
        _write_csv(variant_dir / f"{_parameter_slug(parameter_name)}_summary.csv", summary_rows)
        _plot_parameter_sweep(
            title=f"{GRAPH_SPECS[graph_key]['display_name']} | {variant_plan['label']}",
            parameter_name=parameter_name,
            raw_rows=raw_rows,
            summary_rows=summary_rows,
            reference_best_length=reference_best_length,
            baseline_value=baseline_params[parameter_name],
            tuned_value=best_choice["value"],
            output_path=plots_dir / f"{_parameter_slug(parameter_name)}.png",
        )

    final_summary, final_runs = _run_config(
        graph,
        runner_key=variant_plan["runner"],
        params=current_params,
        seeds=seeds,
        reference_best_length=reference_best_length,
    )

    result = {
        "graph": GRAPH_SPECS[graph_key]["display_name"],
        "variant": variant_key,
        "label": variant_plan["label"],
        "baseline_kind": variant_plan["baseline_kind"],
        "baseline_params": baseline_params,
        "baseline_summary": baseline_summary,
        "baseline_runs": baseline_runs,
        "tuned_params": current_params,
        "final_summary": final_summary,
        "final_runs": final_runs,
        "selection_history": selection_history,
    }
    with (variant_dir / "result.json").open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, ensure_ascii=False, indent=2)
    _write_csv(variant_dir / "baseline_runs.csv", baseline_runs)
    _write_csv(variant_dir / "final_runs.csv", final_runs)
    _write_variant_report(variant_dir, graph_name=GRAPH_SPECS[graph_key]["display_name"], variant_plan=variant_plan, result=result)
    return result


def _family_label(variant_key: str) -> str:
    return "SA" if variant_key.startswith("sa_") else "ACO"


def _write_graph_report(graph_dir: Path, *, graph_key: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    display_name = GRAPH_SPECS[graph_key]["display_name"]
    reference_best_length = GRAPH_SPECS[graph_key]["reference_best_length"]
    tuned_series = [
        {
            "label": result["label"],
            "lengths": [item["best_length"] for item in result["final_runs"] if math.isfinite(item["best_length"])],
        }
        for result in results
    ]
    baseline_vs_tuned_series = []
    for result in results:
        baseline_vs_tuned_series.append(
            {
                "label": f"{result['label']} baseline",
                "lengths": [item["best_length"] for item in result["baseline_runs"] if math.isfinite(item["best_length"])],
            }
        )
        baseline_vs_tuned_series.append(
            {
                "label": f"{result['label']} tuned",
                "lengths": [item["best_length"] for item in result["final_runs"] if math.isfinite(item["best_length"])],
            }
        )

    _plot_variant_distribution(
        title=f"{display_name}: tuned variant comparison",
        series=tuned_series,
        output_path=graph_dir / "tuned_variant_comparison.png",
        reference_best_length=reference_best_length,
    )
    _plot_variant_distribution(
        title=f"{display_name}: baseline vs tuned",
        series=baseline_vs_tuned_series,
        output_path=graph_dir / "baseline_vs_tuned.png",
        reference_best_length=reference_best_length,
    )

    best_overall = min(results, key=lambda item: _summary_key(item["final_summary"]))
    comparison_rows = []
    for result in results:
        baseline_mean = result["baseline_summary"]["mean_best_length"]
        tuned_mean = result["final_summary"]["mean_best_length"]
        improvement_abs = baseline_mean - tuned_mean
        improvement_pct = 0.0 if baseline_mean == 0 else (improvement_abs / baseline_mean) * 100.0
        comparison_rows.append(
            {
                "graph": display_name,
                "variant": result["variant"],
                "label": result["label"],
                "family": _family_label(result["variant"]),
                "baseline_kind": result["baseline_kind"],
                "baseline_mean_best_length": baseline_mean,
                "tuned_mean_best_length": tuned_mean,
                "improvement_absolute": improvement_abs,
                "improvement_percent": improvement_pct,
                "tuned_best_single_run": result["final_summary"]["best_best_length"],
                "tuned_worst_single_run": result["final_summary"]["worst_best_length"],
                "tuned_hit_rate": result["final_summary"].get("hit_rate"),
                "tuned_params_json": json.dumps(result["tuned_params"], ensure_ascii=False, sort_keys=True),
            }
        )
    _write_csv(graph_dir / "variant_comparison.csv", comparison_rows)

    lines = [
        f"# {display_name}",
        "",
        f"- Best overall variant: `{best_overall['label']}`",
        f"- Best overall mean best length: `{best_overall['final_summary']['mean_best_length']:.6f}`",
        f"- Best overall params: `{json.dumps(best_overall['tuned_params'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Variant comparison",
        "",
    ]
    for row in comparison_rows:
        lines.append(
            f"- {row['label']}: baseline={row['baseline_mean_best_length']:.6f}, "
            f"tuned={row['tuned_mean_best_length']:.6f}, "
            f"delta={row['improvement_absolute']:.6f}, "
            f"delta_pct={row['improvement_percent']:.2f}%"
        )
    (graph_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "graph": display_name,
        "winner_variant": best_overall["variant"],
        "winner_label": best_overall["label"],
        "winner_mean_best_length": best_overall["final_summary"]["mean_best_length"],
        "winner_params": best_overall["tuned_params"],
        "rows": comparison_rows,
    }


def _write_master_report(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    flat_rows = [row for summary in summaries for row in summary["rows"]]
    _write_csv(output_dir / "master_comparison.csv", flat_rows)

    lines = [
        "# Sensitivity benchmark",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Winners by graph",
        "",
    ]
    for summary in summaries:
        lines.append(
            f"- {summary['graph']}: {summary['winner_label']} with mean best length "
            f"{summary['winner_mean_best_length']:.6f} and params "
            f"`{json.dumps(summary['winner_params'], ensure_ascii=False, sort_keys=True)}`"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(*, graphs: list[str], families: list[str], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_summaries = []
    variants = _variants_for_families(families)

    for graph_key in graphs:
        graph_spec = GRAPH_SPECS[graph_key]
        graph = load_tsp_graph(graph_spec["path"])
        graph_dir = output_dir / graph_spec["display_name"]
        graph_dir.mkdir(parents=True, exist_ok=True)
        plans = _variant_plans(graph_key)
        variant_results = []
        for variant_key in variants:
            print(f"Running {variant_key} on {graph_spec['display_name']} with seeds={graph_spec['seeds']}")
            result = _run_variant(
                graph_key=graph_key,
                graph=graph,
                variant_key=variant_key,
                variant_plan=plans[variant_key],
                seeds=list(graph_spec["seeds"]),
                reference_best_length=graph_spec["reference_best_length"],
                output_dir=output_dir,
            )
            print(
                f"Selected for {graph_spec['display_name']} / {variant_key}: "
                f"{json.dumps(result['tuned_params'], ensure_ascii=False, sort_keys=True)} | "
                f"mean_best_length={result['final_summary']['mean_best_length']:.6f}"
            )
            variant_results.append(result)
        graph_summaries.append(_write_graph_report(graph_dir, graph_key=graph_key, results=variant_results))

    _write_master_report(output_dir, graph_summaries)
    return graph_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential one-factor sensitivity analysis with per-seed distributions for Lab6 heuristics."
    )
    parser.add_argument("--graphs", nargs="*", help="Graphs to test. Default: all.")
    parser.add_argument("--families", nargs="*", help="Families to test: sa, aco, or all.")
    parser.add_argument("--output-dir", help="Directory for generated artefacts.")
    args = parser.parse_args()

    graphs = _normalize_graphs(args.graphs)
    families = _normalize_families(args.families)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        DEFAULT_RESULTS_DIR / f"sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_analysis(graphs=graphs, families=families, output_dir=output_dir)
    print(f"Saved artefacts to {output_dir}")


if __name__ == "__main__":
    main()
