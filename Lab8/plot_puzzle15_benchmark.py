from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "benchmark_results"
SUMMARY_CSV = RESULT_DIR / "puzzle15_benchmark_summary.csv"

ALGORITHM_LABELS = {
    "astar": "A*",
    "bfs": "BFS",
    "ida_star": "IDA*",
    "backjumping": "Backjumping",
}
DETAILED_ALGORITHMS = ("astar", "ida_star", "backjumping")


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["case"], row["engine"], row["algorithm"]


def row_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {row_key(row): row for row in rows if row["status"] == "ok"}


def case_sort_key(case: str) -> int:
    return int(case.split("_")[1])


def ok_rows(rows: list[dict[str, str]], engine: str, algorithm: str) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["engine"] == engine and row["algorithm"] == algorithm and row["status"] == "ok"
    ]
    return sorted(selected, key=lambda row: int(row["shuffles"]))


def common_cases(rows: list[dict[str, str]], engine: str, algorithms: tuple[str, ...]) -> list[str]:
    available: list[set[str]] = []
    for algorithm in algorithms:
        available.append(
            {
                row["case"]
                for row in rows
                if row["engine"] == engine and row["algorithm"] == algorithm and row["status"] == "ok"
            }
        )
    if not available:
        return []
    return sorted(set.intersection(*available), key=case_sort_key)


def values_for_cases(
    rows: dict[tuple[str, str, str], dict[str, str]],
    cases: list[str],
    engine: str,
    algorithm: str,
    field: str,
) -> tuple[list[int], list[float]]:
    x: list[int] = []
    y: list[float] = []
    for case in cases:
        row = rows[(case, engine, algorithm)]
        x.append(int(row["shuffles"]))
        y.append(float(row[field]))
    return x, y


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
    data = row_by_key(rows)
    cases = common_cases(rows, engine, tuple(ALGORITHM_LABELS))
    plt.figure(figsize=(9, 5))
    all_values: list[float] = []
    for algorithm, label in ALGORITHM_LABELS.items():
        x, y = values_for_cases(data, cases, engine, algorithm, "avg_time_ms")
        all_values.extend(y)
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Количество перемешиваний")
    plt.ylabel("Время, мс")
    is_log = apply_log_y(all_values, force=True)
    scale_note = "лог. шкала, " if is_log else ""
    plt.title(f"Время решения пятнашек, {engine.upper()} ({scale_note}общие точки с BFS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory(rows: list[dict[str, str]], filename: str) -> None:
    plt.figure(figsize=(9, 5))
    for engine, marker in [("python", "o"), ("cpp", "s")]:
        selected = ok_rows(rows, engine, "astar")
        x = [int(row["shuffles"]) for row in selected]
        y = [float(row["avg_peak_memory_kb"]) for row in selected]
        plt.plot(x, y, marker=marker, label=engine.upper())
    plt.xlabel("Количество перемешиваний")
    plt.ylabel("Алгоритмическая пиковая память, KiB")
    plt.title("Память алгоритма A* для пятнашек")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory_all_algorithms(rows: list[dict[str, str]], engine: str, filename: str) -> None:
    data = row_by_key(rows)
    cases = common_cases(rows, engine, tuple(ALGORITHM_LABELS))
    plt.figure(figsize=(9, 5))
    all_values: list[float] = []
    for algorithm, label in ALGORITHM_LABELS.items():
        x, y = values_for_cases(data, cases, engine, algorithm, "avg_peak_memory_kb")
        all_values.extend(y)
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Количество перемешиваний")
    plt.ylabel("Алгоритмическая пиковая память, KiB")
    plt.title(f"Память алгоритмов пятнашек, {engine.upper()} (лог. шкала)")
    if all_values:
        positive_values = [value for value in all_values if value > 0]
        if positive_values:
            plt.yscale("log")
            plt.ylim(bottom=min(positive_values) * 0.7, top=max(positive_values) * 1.5)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory_without_bfs(rows: list[dict[str, str]], engine: str, filename: str) -> None:
    plt.figure(figsize=(9, 5))
    all_values: list[float] = []
    for algorithm in DETAILED_ALGORITHMS:
        selected = ok_rows(rows, engine, algorithm)
        x = [int(row["shuffles"]) for row in selected]
        y = [float(row["avg_peak_memory_kb"]) for row in selected]
        all_values.extend(y)
        plt.plot(x, y, marker="o", linewidth=2, label=ALGORITHM_LABELS[algorithm])
    plt.xlabel("Количество перемешиваний")
    plt.ylabel("Алгоритмическая пиковая память, KiB")
    is_log = apply_log_y(all_values)
    scale_note = " (лог. шкала)" if is_log else ""
    plt.title(f"Память пятнашек без BFS, {engine.upper()}{scale_note}")
    if all_values and not is_log:
        plt.ylim(bottom=0, top=max(all_values) * 1.12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory_overhead_percent(rows: list[dict[str, str]], engine: str, filename: str) -> None:
    data = row_by_key(rows)
    cases = common_cases(rows, engine, DETAILED_ALGORITHMS)
    algorithms = ("astar", "backjumping")
    x_positions = list(range(len(cases)))
    width = 0.34

    plt.figure(figsize=(10, 5.5))
    for index, algorithm in enumerate(algorithms):
        values: list[float] = []
        for case in cases:
            baseline = float(data[(case, engine, "ida_star")]["avg_peak_memory_kb"])
            current = float(data[(case, engine, algorithm)]["avg_peak_memory_kb"])
            values.append(percent_increase(current, baseline))
        offsets = [pos + (index - 0.5) * width for pos in x_positions]
        bars = plt.bar(offsets, values, width=width, label=ALGORITHM_LABELS[algorithm])
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
    plt.xticks(x_positions, [str(case_sort_key(case)) for case in cases])
    plt.xlabel("Количество перемешиваний")
    plt.ylabel("Прирост памяти относительно IDA*, %")
    plt.title(f"Относительная память A* и Backjumping, {engine.upper()}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_memory_source_breakdown(rows: list[dict[str, str]], engine: str, filename: str) -> None:
    data = row_by_key(rows)
    cases = common_cases(rows, engine, DETAILED_ALGORITHMS)
    case = cases[-1]
    ida = float(data[(case, engine, "ida_star")]["avg_peak_memory_kb"])
    astar = float(data[(case, engine, "astar")]["avg_peak_memory_kb"])
    backjumping = float(data[(case, engine, "backjumping")]["avg_peak_memory_kb"])

    astar_extra = max(0.0, astar - ida)
    backjumping_extra = max(0.0, backjumping - ida)
    algorithms = ["IDA*", "A*", "Backjumping"]
    components = {
        "База IDA*": [ida, ida, ida],
        "Очередь и таблицы A*": [0.0, astar_extra, 0.0],
        "Кэш тупиков Backjumping": [0.0, 0.0, backjumping_extra],
    }

    x_positions = list(range(len(algorithms)))
    bottoms = [0.0 for _ in algorithms]
    plt.figure(figsize=(9, 5.5))
    for label, values in components.items():
        plt.bar(x_positions, values, bottom=bottoms, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for x, total in zip(x_positions, bottoms):
        plt.text(x, total, f"{total:.1f} KiB", ha="center", va="bottom", fontsize=9)

    plt.xticks(x_positions, algorithms)
    plt.ylabel("Алгоритмическая пиковая память, KiB")
    plt.title(f"Источники памяти пятнашек, {engine.upper()}, {case_sort_key(case)} перемешиваний")
    if bottoms:
        plt.ylim(bottom=0, top=max(bottoms) * 1.18)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def plot_speedup(rows: list[dict[str, str]], filename: str) -> None:
    by_key = row_by_key(rows)
    cases = sorted(
        set(common_cases(rows, "python", tuple(ALGORITHM_LABELS))).intersection(
            common_cases(rows, "cpp", tuple(ALGORITHM_LABELS))
        ),
        key=case_sort_key,
    )

    plt.figure(figsize=(9, 5))
    for algorithm, label in ALGORITHM_LABELS.items():
        points: list[tuple[int, float]] = []
        for case in cases:
            python_row = by_key.get((case, "python", algorithm))
            cpp_row = by_key.get((case, "cpp", algorithm))
            if python_row is None or cpp_row is None:
                continue
            speedup = float(python_row["avg_time_ms"]) / float(cpp_row["avg_time_ms"])
            points.append((int(python_row["shuffles"]), speedup))
        if points:
            x, y = zip(*points)
            plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Количество перемешиваний")
    plt.ylabel("Ускорение C++ относительно Python")
    plt.title("Сравнение скорости Python и C++ для пятнашек (общие точки с BFS)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=180)
    plt.close()


def main() -> None:
    rows = load_rows()
    output_paths = [
        RESULT_DIR / "puzzle15_time_python.png",
        RESULT_DIR / "puzzle15_time_cpp.png",
        RESULT_DIR / "puzzle15_memory_astar.png",
        RESULT_DIR / "puzzle15_memory_python_all_algorithms.png",
        RESULT_DIR / "puzzle15_memory_cpp_all_algorithms.png",
        RESULT_DIR / "puzzle15_memory_python_without_bfs.png",
        RESULT_DIR / "puzzle15_memory_cpp_without_bfs.png",
        RESULT_DIR / "puzzle15_memory_overhead_percent_python.png",
        RESULT_DIR / "puzzle15_memory_overhead_percent_cpp.png",
        RESULT_DIR / "puzzle15_memory_source_breakdown_python.png",
        RESULT_DIR / "puzzle15_memory_source_breakdown_cpp.png",
        RESULT_DIR / "puzzle15_speedup_cpp_vs_python.png",
    ]

    plot_time(rows, "python", "puzzle15_time_python.png")
    plot_time(rows, "cpp", "puzzle15_time_cpp.png")
    plot_memory(rows, "puzzle15_memory_astar.png")
    plot_memory_all_algorithms(rows, "python", "puzzle15_memory_python_all_algorithms.png")
    plot_memory_all_algorithms(rows, "cpp", "puzzle15_memory_cpp_all_algorithms.png")
    plot_memory_without_bfs(rows, "python", "puzzle15_memory_python_without_bfs.png")
    plot_memory_without_bfs(rows, "cpp", "puzzle15_memory_cpp_without_bfs.png")
    plot_memory_overhead_percent(rows, "python", "puzzle15_memory_overhead_percent_python.png")
    plot_memory_overhead_percent(rows, "cpp", "puzzle15_memory_overhead_percent_cpp.png")
    plot_memory_source_breakdown(rows, "python", "puzzle15_memory_source_breakdown_python.png")
    plot_memory_source_breakdown(rows, "cpp", "puzzle15_memory_source_breakdown_cpp.png")
    plot_speedup(rows, "puzzle15_speedup_cpp_vs_python.png")

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
