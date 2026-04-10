from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def _primary_metric(row: dict[str, Any]) -> float:
    value = row.get("mean_gap_to_reference")
    if value is None:
        value = row.get("mean_best_length")
    if value is None:
        return math.inf
    return float(value)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _slugify(value: str) -> str:
    parts = []
    for char in str(value).lower():
        parts.append(char if char.isalnum() else "_")
    return "".join(parts).strip("_") or "plot"


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _value_sort_key(value: Any) -> tuple[int, str]:
    normalized = _normalize_value(value)
    if isinstance(normalized, (int, float)):
        return (0, f"{float(normalized):020.8f}")
    return (1, str(normalized))


def _discover_summary_files(path: str | Path) -> list[Path]:
    root = Path(path).resolve()
    if root.is_file():
        return [root]
    return sorted(root.rglob("summary.json"))


def _varying_param_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = sorted({key for row in rows for key in row.get("params", {})})
    varying = []
    for key in keys:
        values = {
            json.dumps(row["params"].get(key), sort_keys=True, default=str)
            for row in rows
            if key in row.get("params", {})
        }
        if len(values) > 1:
            varying.append(key)
    return varying


def _group_rows_by_param(rows: list[dict[str, Any]], param_name: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}

    for row in rows:
        params = row.get("params", {})
        if param_name not in params:
            continue

        value = _normalize_value(params[param_name])
        key = json.dumps(value, sort_keys=True, default=str)
        bucket = buckets.setdefault(
            key,
            {
                "value": value,
                "primary": [],
                "runtime_ms": [],
                "evaluations": [],
            },
        )
        bucket["primary"].append(_primary_metric(row))
        bucket["runtime_ms"].append(float(row.get("mean_runtime_ms") or 0.0))
        bucket["evaluations"].append(float(row.get("mean_evaluations") or 0.0))

    grouped = []
    for bucket in buckets.values():
        grouped.append(
            {
                "value": bucket["value"],
                "mean_primary": float(np.mean(bucket["primary"])),
                "min_primary": float(np.min(bucket["primary"])),
                "max_primary": float(np.max(bucket["primary"])),
                "mean_runtime_ms": float(np.mean(bucket["runtime_ms"])),
                "mean_evaluations": float(np.mean(bucket["evaluations"])),
                "samples": len(bucket["primary"]),
            }
        )

    grouped.sort(key=lambda item: _value_sort_key(item["value"]))
    return grouped


def _plot_metric_series(
    axis: Any,
    grouped: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    color: str,
) -> None:
    labels = [_format_metric(item["value"]) for item in grouped]
    values = [float(item[metric_key]) for item in grouped]
    numeric = all(isinstance(item["value"], (int, float)) for item in grouped)

    if numeric and len(grouped) > 1:
        x_values = [float(item["value"]) for item in grouped]
        axis.plot(x_values, values, marker="o", linewidth=2, color=color)
        axis.set_xticks(x_values)
    else:
        x_values = np.arange(len(grouped))
        axis.bar(x_values, values, color=color, alpha=0.8)
        axis.set_xticks(x_values, labels, rotation=25, ha="right")

    axis.set_title(title)
    axis.grid(alpha=0.25, linestyle="--")


def _plot_parameter_effect(
    summary_rows: list[dict[str, Any]],
    *,
    param_name: str,
    output_dir: Path,
    title_prefix: str,
) -> Path | None:
    grouped = _group_rows_by_param(summary_rows, param_name)
    if len(grouped) <= 1:
        return None

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    _plot_metric_series(
        axes[0],
        grouped,
        metric_key="mean_primary",
        title="Mean primary metric",
        color="#1f77b4",
    )
    axes[0].set_ylabel("Lower is better")

    primary_numeric = all(isinstance(item["value"], (int, float)) for item in grouped)
    if primary_numeric and len(grouped) > 1:
        x_values = [float(item["value"]) for item in grouped]
        axes[0].fill_between(
            x_values,
            [item["min_primary"] for item in grouped],
            [item["max_primary"] for item in grouped],
            color="#1f77b4",
            alpha=0.18,
        )

    _plot_metric_series(
        axes[1],
        grouped,
        metric_key="mean_runtime_ms",
        title="Mean runtime (ms)",
        color="#ff7f0e",
    )
    axes[1].set_ylabel("Lower is better")

    _plot_metric_series(
        axes[2],
        grouped,
        metric_key="mean_evaluations",
        title="Mean evaluations",
        color="#2ca02c",
    )
    axes[2].set_ylabel("Lower is better")

    figure.suptitle(f"{title_prefix}: parameter effect for '{param_name}'", fontsize=13)
    plot_path = output_dir / f"effect_{_slugify(param_name)}.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def _plot_overview(
    summary_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    title_prefix: str,
    top_n: int,
) -> Path:
    sorted_rows = sorted(summary_rows, key=_primary_metric)
    runtime = np.array([float(row.get("mean_runtime_ms") or 0.0) for row in sorted_rows], dtype=float)
    primary = np.array([_primary_metric(row) for row in sorted_rows], dtype=float)
    evaluations = np.array([float(row.get("mean_evaluations") or 0.0) for row in sorted_rows], dtype=float)

    figure, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    size = np.clip(np.sqrt(np.maximum(evaluations, 1.0)), 12.0, 80.0)
    scatter = axis.scatter(
        runtime,
        primary,
        s=size,
        c=np.arange(len(sorted_rows)),
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )

    for index, row in enumerate(sorted_rows[: min(top_n, len(sorted_rows))], start=1):
        axis.annotate(
            str(index),
            (float(row.get("mean_runtime_ms") or 0.0), _primary_metric(row)),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
            weight="bold",
        )

    axis.set_title(f"{title_prefix}: runtime vs quality")
    axis.set_xlabel("Mean runtime (ms)")
    axis.set_ylabel("Primary metric (lower is better)")
    axis.grid(alpha=0.25, linestyle="--")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Configuration rank")

    plot_path = output_dir / "overview_runtime_vs_quality.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def _write_markdown_report(
    summary_data: dict[str, Any],
    *,
    output_dir: Path,
    plot_paths: list[Path],
    top_n: int,
) -> Path:
    summary_rows = list(summary_data.get("summary", []))
    summary_rows.sort(key=_primary_metric)

    lines = [
        f"# {summary_data.get('algorithm', 'unknown')} on {summary_data.get('graph_name', 'unknown')}",
        "",
        f"- Graph: `{summary_data.get('graph_name', '-')}`",
        f"- Algorithm: `{summary_data.get('algorithm', '-')}`",
        f"- Profile: `{summary_data.get('profile', '-')}`",
        f"- Seeds: `{', '.join(str(seed) for seed in summary_data.get('seeds', []))}`",
        f"- Configurations tested: `{len(summary_rows)}`",
        "",
        "## Best configuration",
        "",
    ]

    if summary_rows:
        best = summary_rows[0]
        lines.extend(
            [
                f"- Primary metric: `{_format_metric(_primary_metric(best))}`",
                f"- Mean runtime (ms): `{_format_metric(best.get('mean_runtime_ms'))}`",
                f"- Mean evaluations: `{_format_metric(best.get('mean_evaluations'))}`",
                f"- Params: `{json.dumps(best.get('params', {}), ensure_ascii=False, sort_keys=True)}`",
                "",
                "## Top configurations",
                "",
            ]
        )

        for index, row in enumerate(summary_rows[: min(top_n, len(summary_rows))], start=1):
            lines.append(
                f"{index}. primary={_format_metric(_primary_metric(row))}; "
                f"runtime_ms={_format_metric(row.get('mean_runtime_ms'))}; "
                f"evals={_format_metric(row.get('mean_evaluations'))}; "
                f"params={json.dumps(row.get('params', {}), ensure_ascii=False, sort_keys=True)}"
            )

    if plot_paths:
        lines.extend(["", "## Generated plots", ""])
        for plot_path in plot_paths:
            lines.append(f"- `{plot_path.name}`")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def plot_summary_file(
    summary_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = 10,
) -> list[Path]:
    summary_file = Path(summary_path).resolve()
    summary_data = _load_summary(summary_file)
    summary_rows = list(summary_data.get("summary", []))
    if not summary_rows:
        return []

    plots_dir = Path(output_dir).resolve() if output_dir else summary_file.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = (
        f"{summary_data.get('algorithm', 'algorithm')} | "
        f"{summary_data.get('graph_name', summary_data.get('graph', 'graph'))}"
    )
    plot_paths: list[Path] = []
    plot_paths.append(
        _plot_overview(
            summary_rows,
            output_dir=plots_dir,
            title_prefix=title_prefix,
            top_n=top_n,
        )
    )

    for param_name in _varying_param_keys(summary_rows):
        plot_path = _plot_parameter_effect(
            summary_rows,
            param_name=param_name,
            output_dir=plots_dir,
            title_prefix=title_prefix,
        )
        if plot_path is not None:
            plot_paths.append(plot_path)

    plot_paths.append(
        _write_markdown_report(
            summary_data,
            output_dir=plots_dir,
            plot_paths=plot_paths,
            top_n=top_n,
        )
    )
    return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build plots for Lab6 tuning summaries.")
    parser.add_argument("path", help="Path to summary.json or to a directory containing summary files.")
    parser.add_argument("--top", type=int, default=10, help="How many top configurations to annotate and report.")
    args = parser.parse_args()

    files = _discover_summary_files(args.path)
    if not files:
        raise SystemExit("No summary.json files found.")

    for summary_file in files:
        plot_paths = plot_summary_file(summary_file, top_n=args.top)
        print(f"{summary_file}: generated {len(plot_paths)} artefacts")


if __name__ == "__main__":
    main()
