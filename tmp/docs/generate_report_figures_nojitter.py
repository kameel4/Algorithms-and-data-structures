from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"Lab6")
BENCHMARK_ROOT = ROOT / "benchmark_results" / "sensitivity_final_v2_20260326"
OUTPUT_DIR = ROOT / "report_figures_nojitter"

LABEL_TRANSLATIONS = {
    "Simulated annealing (geometric)": "Отжиг (геометрический)",
    "Simulated annealing (cauchy)": "Отжиг Коши",
    "Ant colony": "Муравьиный алгоритм",
    "Elitist ant colony": "Элитные муравьи",
}

PARAMETER_TRANSLATIONS = {
    "alpha": "alpha",
    "initial_temperature": "Начальная температура",
    "iterations": "Число итераций",
    "elite_ants": "Число элитных муравьёв",
}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_float(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _translate_label(label: str) -> str:
    for source, target in LABEL_TRANSLATIONS.items():
        if label == source:
            return target
        if label == f"{source} baseline":
            return f"{target} базовый"
        if label == f"{source} tuned":
            return f"{target} настроенный"
    return label


def _plot_parameter_sweep_nojitter(
    *,
    title: str,
    parameter_name: str,
    raw_csv: Path,
    summary_csv: Path,
    baseline_value: Any,
    selected_value: Any,
    output_path: Path,
) -> None:
    raw_rows = _load_csv_rows(raw_csv)
    summary_rows = _load_csv_rows(summary_csv)

    values = [_try_float(row["value"]) for row in summary_rows]
    labels = [_format_value(value) for value in values]
    positions = np.arange(len(values), dtype=float)

    grouped_lengths: list[list[float]] = []
    means: list[float] = []
    stds: list[float] = []
    for row in summary_rows:
        value_text = row["value"]
        lengths = [
            float(item["best_length"])
            for item in raw_rows
            if item["value"] == value_text and item["best_length"] and math.isfinite(float(item["best_length"]))
        ]
        grouped_lengths.append(lengths)
        means.append(float(row["mean_best_length"]))
        stds.append(float(row["std_best_length"]))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

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
        axes[0].scatter(
            np.full(len(lengths), positions[index]),
            lengths,
            color="#24557a",
            alpha=0.75,
            s=24,
        )
    axes[0].plot(
        positions,
        means,
        color="#cf3c2c",
        marker="o",
        linewidth=2.0,
        label="Среднее по seed",
    )
    axes[0].set_ylabel("Длина лучшего пути")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(loc="best")

    axes[1].errorbar(
        positions,
        means,
        yerr=stds,
        color="#cf3c2c",
        marker="o",
        linewidth=2.0,
        capsize=4,
    )
    axes[1].set_ylabel("Средняя длина лучшего пути")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].set_xticks(positions, labels, rotation=30, ha="right")
    axes[1].set_xlabel(PARAMETER_TRANSLATIONS.get(parameter_name, parameter_name))

    if baseline_value in values:
        baseline_index = values.index(baseline_value)
        axes[1].scatter(
            [positions[baseline_index]],
            [means[baseline_index]],
            color="#222222",
            s=70,
            label="Базовое значение",
            zorder=5,
        )
    if selected_value in values:
        tuned_index = values.index(selected_value)
        axes[1].scatter(
            [positions[tuned_index]],
            [means[tuned_index]],
            color="#1d7a3a",
            s=70,
            label="Выбранное значение",
            zorder=6,
        )
    axes[1].legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_variant_distribution_nojitter(
    *,
    title: str,
    series: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [_translate_label(item["label"]) for item in series]
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
        if not lengths:
            continue
        axis.scatter(
            np.full(len(lengths), positions[index]),
            lengths,
            color="#7a4f1d",
            alpha=0.75,
            s=24,
        )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Длина лучшего пути")
    axis.set_title(title)
    axis.grid(alpha=0.25, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _series_from_result(result_path: Path, *, include_baseline: bool) -> list[dict[str, Any]]:
    result = _load_json(result_path)
    label = result["label"]
    if include_baseline:
        return [
            {
                "label": f"{label} baseline",
                "lengths": [
                    float(item["best_length"])
                    for item in result["baseline_runs"]
                    if math.isfinite(float(item["best_length"]))
                ],
            },
            {
                "label": f"{label} tuned",
                "lengths": [
                    float(item["best_length"])
                    for item in result["final_runs"]
                    if math.isfinite(float(item["best_length"]))
                ],
            },
        ]
    return [
        {
            "label": label,
            "lengths": [
                float(item["best_length"])
                for item in result["final_runs"]
                if math.isfinite(float(item["best_length"]))
            ],
        }
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _plot_variant_distribution_nojitter(
        title="berlin52: базовая и настроенная конфигурации",
        series=(
            _series_from_result(BENCHMARK_ROOT / "sa" / "berlin52" / "sa_geometric" / "result.json", include_baseline=True)
            + _series_from_result(BENCHMARK_ROOT / "sa" / "berlin52" / "sa_cauchy" / "result.json", include_baseline=True)
        ),
        output_path=OUTPUT_DIR / "fig08_sa_berlin52_baseline_vs_tuned.png",
    )

    _plot_parameter_sweep_nojitter(
        title="berlin52 | Имитация отжига (геометрическое охлаждение)",
        parameter_name="alpha",
        raw_csv=BENCHMARK_ROOT / "sa" / "berlin52" / "sa_geometric" / "alpha_raw.csv",
        summary_csv=BENCHMARK_ROOT / "sa" / "berlin52" / "sa_geometric" / "alpha_summary.csv",
        baseline_value=0.95,
        selected_value=0.975,
        output_path=OUTPUT_DIR / "fig09_sa_berlin52_alpha.png",
    )

    _plot_parameter_sweep_nojitter(
        title="world666 | Имитация отжига (охлаждение Коши)",
        parameter_name="initial_temperature",
        raw_csv=BENCHMARK_ROOT / "sa" / "world666" / "sa_cauchy" / "initial_temperature_raw.csv",
        summary_csv=BENCHMARK_ROOT / "sa" / "world666" / "sa_cauchy" / "initial_temperature_summary.csv",
        baseline_value=100.0,
        selected_value=3000.0,
        output_path=OUTPUT_DIR / "fig10_sa_world666_initial_temperature.png",
    )

    _plot_variant_distribution_nojitter(
        title="world666: сравнение настроенных вариантов",
        series=(
            _series_from_result(BENCHMARK_ROOT / "sa" / "world666" / "sa_geometric" / "result.json", include_baseline=False)
            + _series_from_result(BENCHMARK_ROOT / "sa" / "world666" / "sa_cauchy" / "result.json", include_baseline=False)
        ),
        output_path=OUTPUT_DIR / "fig11_sa_world666_tuned_variant_comparison.png",
    )

    _plot_variant_distribution_nojitter(
        title="berlin52: базовая и настроенная конфигурации",
        series=(
            _series_from_result(BENCHMARK_ROOT / "aco_fb" / "berlin52" / "ant_basic" / "result.json", include_baseline=True)
            + _series_from_result(BENCHMARK_ROOT / "aco_fb" / "berlin52" / "ant_elitist" / "result.json", include_baseline=True)
        ),
        output_path=OUTPUT_DIR / "fig12_aco_berlin52_baseline_vs_tuned.png",
    )

    _plot_parameter_sweep_nojitter(
        title="berlin52 | Муравьиный алгоритм",
        parameter_name="iterations",
        raw_csv=BENCHMARK_ROOT / "aco_fb" / "berlin52" / "ant_basic" / "iterations_raw.csv",
        summary_csv=BENCHMARK_ROOT / "aco_fb" / "berlin52" / "ant_basic" / "iterations_summary.csv",
        baseline_value=100.0,
        selected_value=150.0,
        output_path=OUTPUT_DIR / "fig13_aco_berlin52_iterations.png",
    )

    _plot_parameter_sweep_nojitter(
        title="berlin52 | Алгоритм элитных муравьёв",
        parameter_name="elite_ants",
        raw_csv=BENCHMARK_ROOT / "aco_fb" / "berlin52" / "ant_elitist" / "elite_ants_raw.csv",
        summary_csv=BENCHMARK_ROOT / "aco_fb" / "berlin52" / "ant_elitist" / "elite_ants_summary.csv",
        baseline_value=5.0,
        selected_value=8.0,
        output_path=OUTPUT_DIR / "fig14_aco_berlin52_elite_ants.png",
    )

    _plot_variant_distribution_nojitter(
        title="world666: базовая и настроенная конфигурации",
        series=(
            _series_from_result(BENCHMARK_ROOT / "aco_world" / "world666" / "ant_basic" / "result.json", include_baseline=True)
            + _series_from_result(BENCHMARK_ROOT / "aco_world" / "world666" / "ant_elitist" / "result.json", include_baseline=True)
        ),
        output_path=OUTPUT_DIR / "fig15_aco_world666_baseline_vs_tuned.png",
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
