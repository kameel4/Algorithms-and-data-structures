from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .parameter_sweep import run_sweep
except ImportError:
    from parameter_sweep import run_sweep


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_RESULTS_DIR = ROOT_DIR / "benchmark_results"

GRAPH_SPECS = {
    "figure1.edgelist": {
        "path": DATA_DIR / "figure1.edgelist",
        "display_name": "figure1",
        "reference_best_length": 14.0,
    },
    "berlin52.stp": {
        "path": DATA_DIR / "berlin52.stp",
        "display_name": "berlin52",
        "reference_best_length": None,
    },
    "world666.stp": {
        "path": DATA_DIR / "world666.stp",
        "display_name": "world666",
        "reference_best_length": None,
    },
}

ALGORITHM_ALIASES = {
    "annealing": "annealing",
    "sa": "annealing",
    "ant": "ant",
    "aco": "ant",
    "elitist-ant": "elitist-ant",
    "elitist": "elitist-ant",
    "elite-ant": "elitist-ant",
}

PROFILE_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "quick": {
        "figure1.edgelist": {
            "seeds": [1, 2, 3, 4, 5],
            "annealing": {
                "geometric": {
                    "initial_temperature": [5, 10, 20],
                    "iterations_per_temperature": [10, 20],
                    "alpha": [0.85, 0.9, 0.95],
                },
                "cauchy": {
                    "initial_temperature": [5, 10, 20],
                    "iterations_per_temperature": [10, 20],
                },
            },
            "ant": {
                "iterations": [10, 20],
                "ant_count": [3, 6],
                "pheromone_importance": [0.5, 1.0, 2.0],
                "distance_importance": [1.0, 2.0],
                "pheromone_deposit": [50, 100],
                "evaporation_intensity": [2, 4],
            },
            "elitist-ant": {
                "iterations": [10, 20],
                "ant_count": [3, 6],
                "pheromone_importance": [0.5, 1.0, 2.0],
                "distance_importance": [1.0, 2.0],
                "pheromone_deposit": [50, 100],
                "evaporation_intensity": [2, 4],
                "elite_ants": [2, 5],
            },
        },
        "berlin52.stp": {
            "seeds": [1, 2, 3],
            "annealing": {
                "geometric": {
                    "initial_temperature": [100, 200, 300],
                    "iterations_per_temperature": [50, 100],
                    "alpha": [0.93, 0.95],
                },
                "cauchy": {
                    "initial_temperature": [100, 200, 300],
                    "iterations_per_temperature": [50, 100],
                },
            },
            "ant": {
                "iterations": [20, 30],
                "ant_count": [20, 30, 40],
                "pheromone_importance": [1.0, 1.5],
                "distance_importance": [1.5, 2.0],
                "pheromone_deposit": [100],
                "evaporation_intensity": [3, 4],
            },
            "elitist-ant": {
                "iterations": [20, 30],
                "ant_count": [20, 30, 40],
                "pheromone_importance": [1.0, 1.5],
                "distance_importance": [1.5, 2.0],
                "pheromone_deposit": [100],
                "evaporation_intensity": [3, 4],
                "elite_ants": [3, 5],
            },
        },
        "world666.stp": {
            "seeds": [1, 2],
            "annealing": {
                "geometric": {
                    "initial_temperature": [200, 400],
                    "iterations_per_temperature": [10, 20],
                    "alpha": [0.95, 0.97],
                },
                "cauchy": {
                    "initial_temperature": [200, 400],
                    "iterations_per_temperature": [10, 20],
                },
            },
            "ant": {
                "iterations": [4, 8],
                "ant_count": [8, 16],
                "pheromone_importance": [1.0, 1.5],
                "distance_importance": [1.5, 2.0],
                "pheromone_deposit": [100],
                "evaporation_intensity": [2, 4],
            },
            "elitist-ant": {
                "iterations": [4, 8],
                "ant_count": [8, 16],
                "pheromone_importance": [1.0, 1.5],
                "distance_importance": [1.5, 2.0],
                "pheromone_deposit": [100],
                "evaporation_intensity": [2, 4],
                "elite_ants": [3],
            },
        },
    },
    "full": {
        "figure1.edgelist": {
            "seeds": [1, 2, 3, 4, 5, 6, 7, 8],
            "annealing": {
                "geometric": {
                    "initial_temperature": [5, 10, 20, 40],
                    "iterations_per_temperature": [10, 20, 40],
                    "alpha": [0.8, 0.85, 0.9, 0.95],
                },
                "cauchy": {
                    "initial_temperature": [5, 10, 20, 40],
                    "iterations_per_temperature": [10, 20, 40],
                },
            },
            "ant": {
                "iterations": [10, 20, 40],
                "ant_count": [3, 6, 9],
                "pheromone_importance": [0.5, 1.0, 2.0],
                "distance_importance": [1.0, 2.0, 3.0],
                "pheromone_deposit": [25, 50, 100],
                "evaporation_intensity": [1, 3, 5],
            },
            "elitist-ant": {
                "iterations": [10, 20, 40],
                "ant_count": [3, 6, 9],
                "pheromone_importance": [0.5, 1.0, 2.0],
                "distance_importance": [1.0, 2.0, 3.0],
                "pheromone_deposit": [25, 50, 100],
                "evaporation_intensity": [1, 3, 5],
                "elite_ants": [2, 5, 8],
            },
        },
        "berlin52.stp": {
            "seeds": [1, 2, 3, 4, 5],
            "annealing": {
                "geometric": {
                    "initial_temperature": [200, 400, 600, 800],
                    "iterations_per_temperature": [100, 200, 300],
                    "alpha": [0.95, 0.97, 0.975],
                },
                "cauchy": {
                    "initial_temperature": [200, 400, 600],
                    "iterations_per_temperature": [100, 200, 300],
                },
            },
            "ant": {
                "iterations": [20, 30, 40],
                "ant_count": [20, 30, 40],
                "pheromone_importance": [0.5, 1.0, 1.5],
                "distance_importance": [1.0, 2.0, 3.0],
                "pheromone_deposit": [50, 100],
                "evaporation_intensity": [2, 3, 4],
            },
            "elitist-ant": {
                "iterations": [20, 30, 40],
                "ant_count": [20, 30, 40],
                "pheromone_importance": [0.5, 1.0, 1.5],
                "distance_importance": [1.0, 2.0, 3.0],
                "pheromone_deposit": [50, 100],
                "evaporation_intensity": [2, 3, 4],
                "elite_ants": [3, 5, 8],
            },
        },
        "world666.stp": {
            "seeds": [1, 2, 3],
            "annealing": {
                "geometric": {
                    "initial_temperature": [200, 400, 600],
                    "iterations_per_temperature": [20, 40, 60],
                    "alpha": [0.95, 0.97, 0.98],
                },
                "cauchy": {
                    "initial_temperature": [200, 400, 600],
                    "iterations_per_temperature": [20, 40, 60],
                },
            },
            "ant": {
                "iterations": [5, 10, 15],
                "ant_count": [8, 16, 24],
                "pheromone_importance": [1.0, 1.5, 2.0],
                "distance_importance": [1.5, 2.0, 3.0],
                "pheromone_deposit": [50, 100],
                "evaporation_intensity": [2, 4, 6],
            },
            "elitist-ant": {
                "iterations": [5, 10, 15],
                "ant_count": [8, 16, 24],
                "pheromone_importance": [1.0, 1.5, 2.0],
                "distance_importance": [1.5, 2.0, 3.0],
                "pheromone_deposit": [50, 100],
                "evaporation_intensity": [2, 4, 6],
                "elite_ants": [3, 5],
            },
        },
    },
}


def _normalize_graph_selection(values: list[str] | None) -> list[str]:
    if not values:
        return list(GRAPH_SPECS)

    normalized: list[str] = []
    for value in values:
        candidate = value.strip()
        if candidate in GRAPH_SPECS:
            normalized.append(candidate)
            continue

        matches = [
            graph_key
            for graph_key in GRAPH_SPECS
            if Path(graph_key).stem.lower() == candidate.lower()
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown graph selection: {value}")
        normalized.append(matches[0])
    return normalized


def _normalize_algorithms(values: list[str] | None) -> list[str]:
    if not values:
        return ["annealing", "ant", "elitist-ant"]

    normalized: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key not in ALGORITHM_ALIASES:
            raise ValueError(f"Unknown algorithm selection: {value}")
        normalized.append(ALGORITHM_ALIASES[key])
    return normalized


def _annealing_min_temperatures(graph_key: str, profile_name: str) -> list[float]:
    if graph_key == "figure1.edgelist":
        return [0.01, 0.05, 0.1] if profile_name == "full" else [0.05, 0.1]
    if graph_key == "berlin52.stp":
        return [0.05, 0.1, 0.2] if profile_name == "full" else [0.05, 0.1]
    return [5.0, 10.0, 20.0] if profile_name == "full" else [10.0, 20.0]


def _build_campaign_config(
    graph_key: str,
    algorithm_key: str,
    profile_name: str,
    seed_override: list[int] | None,
) -> dict[str, Any]:
    graph_spec = GRAPH_SPECS[graph_key]
    profile_spec = PROFILE_PRESETS[profile_name][graph_key]
    seeds = list(seed_override) if seed_override else list(profile_spec["seeds"])
    experiment_name_prefix = f"{graph_spec['display_name']}-{algorithm_key}"

    if algorithm_key == "annealing":
        annealing_spec = profile_spec["annealing"]
        min_temperatures = _annealing_min_temperatures(graph_key, profile_name)
        geometric_grid = dict(annealing_spec["geometric"])
        geometric_grid["min_temperature"] = min_temperatures
        cauchy_grid = dict(annealing_spec["cauchy"])
        cauchy_grid["min_temperature"] = min_temperatures
        experiments = [
            {
                "name": f"{experiment_name_prefix}-geometric",
                "algorithm": "annealing",
                "fixed": {
                    "cooling_mode": "geometric",
                },
                "grid": geometric_grid,
            },
            {
                "name": f"{experiment_name_prefix}-cauchy",
                "algorithm": "annealing",
                "fixed": {
                    "cooling_mode": "cauchy",
                },
                "grid": cauchy_grid,
            },
        ]
    else:
        algorithm_spec = dict(profile_spec[algorithm_key])
        fixed = {
            "initial_pheromone": 1.0,
        }
        experiments = [
            {
                "name": experiment_name_prefix,
                "algorithm": algorithm_key,
                "fixed": fixed,
                "grid": algorithm_spec,
            }
        ]

    return {
        "graph": str(graph_spec["path"]),
        "reference_best_length": graph_spec["reference_best_length"],
        "seeds": {"values": seeds},
        "experiments": experiments,
    }


def _primary_metric(item: dict[str, Any]) -> float:
    if item.get("mean_gap_to_reference") is not None:
        return float(item["mean_gap_to_reference"])
    return float(item["mean_best_length"])


def _summary_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _primary_metric(item),
        math.inf if item.get("mean_runtime_ms") is None else float(item["mean_runtime_ms"]),
        math.inf if item.get("mean_evaluations") is None else float(item["mean_evaluations"]),
    )


def _flatten_summary_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_param_keys = sorted({key for item in summary for key in item.get("params", {})})
    all_metric_keys = [
        key
        for key in sorted({key for item in summary for key in item})
        if key not in {"experiment", "algorithm", "graph", "params", "runs"}
    ]
    rows: list[dict[str, Any]] = []

    for item in summary:
        row = {
            "experiment": item["experiment"],
            "algorithm": item["algorithm"],
            "graph": item["graph"],
            "primary_metric": _primary_metric(item),
        }
        for key in all_metric_keys:
            row[key] = item.get(key)
        for key in all_param_keys:
            row[key] = item.get("params", {}).get(key)
        rows.append(row)

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sort_values(values: list[Any]) -> list[Any]:
    def sort_key(value: Any) -> tuple[int, Any]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, float(value))
        return (1, str(value))

    return sorted(values, key=sort_key)


def _plot_quality_runtime(summary: list[dict[str, Any]], destination: Path, title: str) -> None:
    runtimes = [float(item["mean_runtime_ms"]) for item in summary]
    primary_values = [_primary_metric(item) for item in summary]
    evaluations = [float(item["mean_evaluations"]) for item in summary]

    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        runtimes,
        primary_values,
        c=evaluations,
        cmap="viridis",
        s=60,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    axis.set_title(title)
    axis.set_xlabel("Mean runtime, ms")
    axis.set_ylabel("Primary metric (lower is better)")
    axis.grid(True, alpha=0.25)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Mean evaluations")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_parameter_effects(
    summary: list[dict[str, Any]],
    destination_dir: Path,
    title_prefix: str,
) -> list[str]:
    generated_files: list[str] = []
    param_keys = sorted({key for item in summary for key in item.get("params", {})})

    for param_key in param_keys:
        buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for item in summary:
            value = item.get("params", {}).get(param_key)
            if value is not None:
                buckets[value].append(item)

        values = _sort_values(list(buckets))
        if len(values) <= 1:
            continue

        mean_primary = []
        best_primary = []
        mean_runtime = []
        mean_evaluations = []
        for value in values:
            bucket = buckets[value]
            mean_primary.append(_mean([_primary_metric(item) for item in bucket]))
            best_primary.append(min(_primary_metric(item) for item in bucket))
            mean_runtime.append(_mean([float(item["mean_runtime_ms"]) for item in bucket]))
            mean_evaluations.append(_mean([float(item["mean_evaluations"]) for item in bucket]))

        positions = list(range(len(values)))
        labels = [str(value) for value in values]

        figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(positions, mean_primary, marker="o", label="Mean primary metric")
        axes[0].plot(positions, best_primary, marker="s", label="Best primary metric")
        axes[0].set_ylabel("Primary metric")
        axes[0].set_title(f"{title_prefix}: effect of {param_key}")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        runtime_axis = axes[1]
        runtime_axis.plot(positions, mean_runtime, color="#2a6f97", marker="o", label="Mean runtime, ms")
        runtime_axis.set_ylabel("Runtime, ms", color="#2a6f97")
        runtime_axis.tick_params(axis="y", labelcolor="#2a6f97")
        runtime_axis.grid(True, alpha=0.25)

        evaluations_axis = runtime_axis.twinx()
        evaluations_axis.plot(
            positions,
            mean_evaluations,
            color="#c96c00",
            marker="s",
            label="Mean evaluations",
        )
        evaluations_axis.set_ylabel("Evaluations", color="#c96c00")
        evaluations_axis.tick_params(axis="y", labelcolor="#c96c00")

        runtime_axis.set_xticks(positions, labels, rotation=30, ha="right")
        runtime_axis.set_xlabel(param_key)

        handles = runtime_axis.get_lines() + evaluations_axis.get_lines()
        labels_for_handles = [line.get_label() for line in handles]
        runtime_axis.legend(handles, labels_for_handles, loc="upper left")

        figure.tight_layout()
        output_path = destination_dir / f"effect_{param_key}.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        generated_files.append(output_path.name)

    return generated_files


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _write_report(root_dir: Path, bundle_results: list[dict[str, Any]]) -> None:
    report_path = root_dir / "report.md"
    lines = [
        "# Lab6 tuning report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for bundle in bundle_results:
        best_item = bundle["summary"][0]
        relative_dir = bundle["output_dir"].relative_to(root_dir)
        lines.extend(
            [
                f"## {bundle['graph_display']} / {bundle['algorithm']}",
                "",
                f"- Configurations tested: {len(bundle['summary'])}",
                f"- Best primary metric: {_primary_metric(best_item):.6f}",
                f"- Best mean best length: {float(best_item['mean_best_length']):.6f}",
                f"- Best single-run length: {float(best_item['best_best_length']):.6f}",
                f"- Worst single-run length: {float(best_item['worst_best_length']):.6f}",
                f"- Mean runtime of best config: {float(best_item['mean_runtime_ms']):.3f} ms",
                f"- Mean evaluations of best config: {float(best_item['mean_evaluations']):.3f}",
                f"- Best params: `{_format_params(best_item['params'])}`",
                f"- Output directory: `{relative_dir}`",
                f"- Main plots: `{relative_dir / 'plots' / 'quality_vs_runtime.png'}`",
                "",
            ]
        )
        if best_item.get("mean_hit_reference") is not None:
            lines.insert(
                len(lines) - 1,
                f"- Optimal-hit rate: {float(best_item['mean_hit_reference']) * 100.0:.1f}%",
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _run_bundle(
    graph_key: str,
    algorithm_key: str,
    profile_name: str,
    root_output_dir: Path,
    seed_override: list[int] | None,
) -> dict[str, Any]:
    config = _build_campaign_config(graph_key, algorithm_key, profile_name, seed_override)
    graph_display = GRAPH_SPECS[graph_key]["display_name"]
    output_dir = root_output_dir / graph_display / algorithm_key
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, ensure_ascii=False, indent=2)

    result = run_sweep(config)
    result["summary"].sort(key=_summary_sort_key)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, ensure_ascii=False, indent=2)

    rows = _flatten_summary_rows(result["summary"])
    _write_csv(output_dir / "summary.csv", rows)

    plot_title = f"{graph_display} / {algorithm_key} / {profile_name}"
    _plot_quality_runtime(result["summary"], plots_dir / "quality_vs_runtime.png", plot_title)
    effect_files = _plot_parameter_effects(result["summary"], plots_dir, plot_title)

    return {
        "graph_key": graph_key,
        "graph_display": graph_display,
        "algorithm": algorithm_key,
        "output_dir": output_dir,
        "plots": ["quality_vs_runtime.png", *effect_files],
        "summary": result["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-graph Lab6 tuning campaigns and generate visual reports."
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        help="Graph file names or stems. Defaults to all Lab6/data graphs in the preset catalog.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        help="Subset of algorithms: annealing, ant, elitist-ant.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS),
        default="quick",
        help="Preset grid profile. Use 'full' for the larger campaign.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        help="Optional explicit seed list that overrides the preset seeds.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for reports. Default: Lab6/benchmark_results/tuning_<timestamp>.",
    )
    args = parser.parse_args()

    selected_graphs = _normalize_graph_selection(args.graphs)
    selected_algorithms = _normalize_algorithms(args.algorithms)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (DEFAULT_RESULTS_DIR / f"tuning_{timestamp}").resolve()
    )
    root_output_dir.mkdir(parents=True, exist_ok=True)

    bundle_results = []
    for graph_key in selected_graphs:
        for algorithm_key in selected_algorithms:
            print(
                f"Running {algorithm_key} on {graph_key} "
                f"with profile={args.profile} and seeds={args.seeds or 'preset'}"
            )
            bundle = _run_bundle(
                graph_key=graph_key,
                algorithm_key=algorithm_key,
                profile_name=args.profile,
                root_output_dir=root_output_dir,
                seed_override=args.seeds,
            )
            best = bundle["summary"][0]
            print(
                f"Best {algorithm_key} on {graph_key}: "
                f"primary={_primary_metric(best):.6f}, "
                f"runtime_ms={float(best['mean_runtime_ms']):.3f}, "
                f"params={_format_params(best['params'])}"
            )
            bundle_results.append(bundle)

    _write_report(root_output_dir, bundle_results)
    print()
    print(f"Campaign report: {root_output_dir / 'report.md'}")
    print(f"Artifacts root : {root_output_dir}")


if __name__ == "__main__":
    main()
