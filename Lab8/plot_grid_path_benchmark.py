from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "benchmark_results"
SUMMARY_CSV = RESULT_DIR / "grid_path_benchmark_summary.csv"

ALGORITHM_LABELS = {
    "baseline": "DFS",
    "warnsdorff": "Warnsdorff",
    "connectivity": "Connectivity",
    "backjumping": "Backjumping",
}

def load_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def ok_rows(rows: list[dict[str, str]], engine: str, algorithm: str) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["engine"] == engine and row["algorithm"] == algorithm and row["status"] == "ok"
    ]
    return sorted(selected, key=lambda row: int(row["cells"]))


def row_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row["case"], row["engine"], row["algorithm"]): row
        for row in rows
        if row["status"] == "ok"
    }


def percent_increase(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def apply_log_y(values: list[float], *, force: bool = False, threshold: float = 20.0) -> bool:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        plt.grid(True, alpha=0.3)
        return False

    should_log = force or max(positive_values) / min(positive_values) >= threshold
    if should_log:
        plt.yscale("log")
        plt.ylim(bottom=min(positive_values) * 0.7, top=max(positive_values) * 1.5)
        plt.grid(True, which="both", alpha=0.3)
        return True

    plt.grid(True, alpha=0.3)
    return False


def plot_time(rows: list[dict[str, str]], engine: str, filename: str) -> None:
    plt.figure(figsize=(9, 5))
    all_values: list[float] = []
    for algorithm, label in ALGORITHM_LABELS.items():
        selected = ok_rows(rows, engine, algorithm)
        x = [int(row["cells"]) for row in selected]
        y = [float(row["avg_time_ms"]) for row in selected]
        all_values.extend(y)
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Количество клеток")
    plt.ylabel("Время, мс")
    is_log = apply_log_y(all_values, force=True)
    scale_note = " (лог. шкала)" if is_log else ""
    plt.title(f"Время полного перебора путей по сетке, {engine.upper()}{scale_note}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory(rows: list[dict[str, str]], filename: str) -> None:
    plt.figure(figsize=(9, 5))
    all_values: list[float] = []
    for engine, marker in [("python", "o"), ("cpp", "s")]:
        selected = ok_rows(rows, engine, "connectivity")
        x = [int(row["cells"]) for row in selected]
        y = [float(row["avg_peak_memory_kb"]) for row in selected]
        all_values.extend(y)
        plt.plot(x, y, marker=marker, label=engine.upper())
    plt.xlabel("Количество клеток")
    plt.ylabel("Дополнительная рабочая память, KiB")
    is_log = apply_log_y(all_values)
    scale_note = " (лог. шкала)" if is_log else ""
    plt.title(f"Дополнительная память алгоритма отсечения по связности{scale_note}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory_all_algorithms(rows: list[dict[str, str]], engine: str, filename: str) -> None:
    plt.figure(figsize=(9, 5))
    all_values: list[float] = []
    for algorithm, label in ALGORITHM_LABELS.items():
        selected = ok_rows(rows, engine, algorithm)
        x = [int(row["cells"]) for row in selected]
        y = [float(row["avg_peak_memory_kb"]) for row in selected]
        all_values.extend(y)
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Количество клеток")
    plt.ylabel("Дополнительная рабочая память, KiB")
    is_log = apply_log_y(all_values)
    scale_note = " (лог. шкала)" if is_log else ""
    plt.title(f"Дополнительная память алгоритмов сетки, {engine.upper()}{scale_note}")
    if all_values and not is_log:
        plt.ylim(bottom=0, top=max(all_values) * 1.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory_overhead_percent(rows: list[dict[str, str]], engine: str, filename: str) -> None:
    data = row_by_key(rows)
    cases = sorted(
        {row["case"] for row in rows if row["engine"] == engine},
        key=lambda item: int(item.split("x")[0]) * int(item.split("x")[1]),
    )
    algorithms = ["warnsdorff", "connectivity", "backjumping"]
    labels = [ALGORITHM_LABELS[item] for item in algorithms]
    x_positions = list(range(len(cases)))
    width = 0.24

    plt.figure(figsize=(10, 5.5))
    for index, algorithm in enumerate(algorithms):
        values: list[float] = []
        for case in cases:
            baseline = float(data[(case, engine, "baseline")]["avg_peak_memory_kb"])
            current = float(data[(case, engine, algorithm)]["avg_peak_memory_kb"])
            values.append(percent_increase(current, baseline))
        offsets = [pos + (index - 1) * width for pos in x_positions]
        bars = plt.bar(offsets, values, width=width, label=labels[index])
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.axhline(0, color="#444444", linewidth=0.8)
    plt.xticks(x_positions, cases)
    plt.xlabel("Размер сетки")
    plt.ylabel("Прирост памяти относительно DFS, %")
    plt.title(f"Относительный прирост памяти модификаций, {engine.upper()}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory_source_breakdown(rows: list[dict[str, str]], filename: str) -> None:
    data = row_by_key(rows)
    case = "6x6"
    baseline = float(data[(case, "cpp", "baseline")]["avg_peak_memory_kb"])
    warnsdorff = float(data[(case, "cpp", "warnsdorff")]["avg_peak_memory_kb"])
    connectivity = float(data[(case, "cpp", "connectivity")]["avg_peak_memory_kb"])
    backjumping = float(data[(case, "cpp", "backjumping")]["avg_peak_memory_kb"])

    ordering_extra = max(0.0, warnsdorff - baseline)
    connectivity_extra = max(0.0, connectivity - baseline)
    backjumping_extra = max(0.0, backjumping - baseline - ordering_extra - connectivity_extra)

    algorithms = ["DFS", "Warnsdorff", "Connectivity", "Backjumping"]
    components = {
        "База DFS": [baseline, baseline, baseline, baseline],
        "Упорядочивание": [0.0, ordering_extra, 0.0, ordering_extra],
        "Проверка связности": [0.0, 0.0, connectivity_extra, connectivity_extra],
        "Конфликты Backjumping": [0.0, 0.0, 0.0, backjumping_extra],
    }

    x_positions = list(range(len(algorithms)))
    bottoms = [0.0 for _ in algorithms]
    plt.figure(figsize=(10, 5.5))
    for label, values in components.items():
        plt.bar(x_positions, values, bottom=bottoms, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for x, total in zip(x_positions, bottoms):
        plt.text(x, total, f"{total:.2f} KiB", ha="center", va="bottom", fontsize=9)

    plt.xticks(x_positions, algorithms)
    plt.ylabel("Дополнительная рабочая память, KiB")
    plt.title("Из чего складывается память алгоритмов сетки, C++ 6x6")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_speedup(rows: list[dict[str, str]], filename: str) -> None:
    by_key = row_by_key(rows)

    plt.figure(figsize=(9, 5))
    cases = sorted(
        {row["case"] for row in rows},
        key=lambda item: int(item.split("x")[0]) * int(item.split("x")[1]),
    )
    for algorithm, label in ALGORITHM_LABELS.items():
        points: list[tuple[int, float]] = []
        for case in cases:
            python_row = by_key.get((case, "python", algorithm))
            cpp_row = by_key.get((case, "cpp", algorithm))
            if python_row is None or cpp_row is None:
                continue
            speedup = float(python_row["avg_time_ms"]) / float(cpp_row["avg_time_ms"])
            points.append((int(python_row["cells"]), speedup))
        if points:
            x, y = zip(*points)
            plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Количество клеток")
    plt.ylabel("Ускорение C++ относительно Python")
    plt.title("Сравнение скорости Python и C++ для сетки")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def main() -> None:
    rows = load_rows()
    plot_time(rows, "python", "time_python.png")
    plot_time(rows, "cpp", "time_cpp.png")
    plot_memory(rows, "memory_connectivity.png")
    plot_memory_all_algorithms(rows, "python", "memory_python_all_algorithms.png")
    plot_memory_all_algorithms(rows, "cpp", "memory_cpp_all_algorithms.png")
    plot_memory_overhead_percent(rows, "cpp", "memory_overhead_percent_cpp.png")
    plot_memory_source_breakdown(rows, "memory_source_breakdown_cpp_6x6.png")
    plot_speedup(rows, "speedup_cpp_vs_python.png")
    for path in [
        RESULT_DIR / "time_python.png",
        RESULT_DIR / "time_cpp.png",
        RESULT_DIR / "memory_connectivity.png",
        RESULT_DIR / "memory_python_all_algorithms.png",
        RESULT_DIR / "memory_cpp_all_algorithms.png",
        RESULT_DIR / "memory_overhead_percent_cpp.png",
        RESULT_DIR / "memory_source_breakdown_cpp_6x6.png",
        RESULT_DIR / "speedup_cpp_vs_python.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()
