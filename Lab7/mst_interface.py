from __future__ import annotations

import csv
import re
import subprocess
import threading
from pathlib import Path
from tkinter import StringVar, Tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


ROOT_DIR = Path(__file__).resolve().parent
GRAPH_DIR = ROOT_DIR / "generated_graphs"
RESULTS_DIR = ROOT_DIR / "ui_results"
BENCHMARK_PATH = ROOT_DIR / "cpp" / "benchmark_mst.exe"
GRAPH_PATTERN = re.compile(r"n(?P<size>\d+)_p(?P<density>\d+)\.json$")

ALGORITHMS = {
    "kruskal": {"label": "Крускал", "color": "#1f77b4"},
    "prim_binary_heap": {"label": "Прим + бин. куча", "color": "#d62728"},
    "prim_fibonacci_heap": {"label": "Прим + куча Фибоначчи", "color": "#2ca02c"},
    "boruvka": {"label": "Борувка", "color": "#ff7f0e"},
}


def get_available_sizes(graph_dir: Path = GRAPH_DIR) -> list[int]:
    sizes = set()
    for path in graph_dir.glob("n*_p*.json"):
        match = GRAPH_PATTERN.match(path.name)
        if match:
            sizes.add(int(match.group("size")))
    return sorted(sizes)


def get_graphs_for_size(size: int, graph_dir: Path = GRAPH_DIR) -> list[tuple[int, Path]]:
    graphs: list[tuple[int, Path]] = []
    for path in graph_dir.glob(f"n{size}_p*.json"):
        match = GRAPH_PATTERN.match(path.name)
        if match:
            graphs.append((int(match.group("density")), path))
    return sorted(graphs, key=lambda item: item[0])


def ensure_benchmark_built() -> None:
    source_paths = [
        ROOT_DIR / "cpp" / "benchmark_mst.cpp",
        ROOT_DIR / "cpp" / "kruskal.cpp",
        ROOT_DIR / "cpp" / "prim_binary_heap.cpp",
        ROOT_DIR / "cpp" / "prim_fibonacci_heap.cpp",
        ROOT_DIR / "cpp" / "boruvka.cpp",
        ROOT_DIR / "cpp" / "algorithms.h",
        ROOT_DIR / "cpp" / "graph_io.h",
        ROOT_DIR / "cpp" / "mst_types.h",
        ROOT_DIR / "cpp" / "dsu.h",
    ]
    if BENCHMARK_PATH.exists():
        exe_mtime = BENCHMARK_PATH.stat().st_mtime
        if all(path.stat().st_mtime <= exe_mtime for path in source_paths):
            return

    command = [
        "g++",
        "-O2",
        "-std=c++17",
        "-Icpp",
        "cpp/benchmark_mst.cpp",
        "cpp/kruskal.cpp",
        "cpp/prim_binary_heap.cpp",
        "cpp/prim_fibonacci_heap.cpp",
        "cpp/boruvka.cpp",
        "-o",
        str(BENCHMARK_PATH),
    ]
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def run_benchmark_for_size(size: int, repeats: int, output_path: Path) -> None:
    ensure_benchmark_built()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [str(BENCHMARK_PATH), str(GRAPH_DIR), str(repeats), str(output_path), str(size)],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "benchmark_mst.exe failed"
        raise RuntimeError(message)


def load_rows_from_csv(csv_path: Path, size: int) -> list[dict[str, int | float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file was not created: {csv_path}")

    rows_by_density: dict[int, dict[str, int | float]] = {}
    with csv_path.open(encoding="utf-8", newline="") as file:
        for raw_row in csv.DictReader(file):
            if int(raw_row["n"]) != size:
                continue
            match = GRAPH_PATTERN.match(raw_row["graph"])
            if not match:
                continue
            density = int(match.group("density"))
            row = rows_by_density.setdefault(
                density,
                {
                    "density": density,
                    "graph_edges": int(raw_row["m"]),
                    "mst_weight": int(raw_row["mst_weight"]),
                    "mst_edges": int(raw_row["mst_edges"]),
                },
            )
            if int(row["mst_weight"]) != int(raw_row["mst_weight"]):
                raise ValueError(f"MST weight mismatch for density {density}")
            if int(row["mst_edges"]) != int(raw_row["mst_edges"]):
                raise ValueError(f"MST edge count mismatch for density {density}")
            row[f"{raw_row['algorithm']}_time_ms"] = float(raw_row["avg_time_ms"])

    if not rows_by_density:
        raise FileNotFoundError(f"Graphs for n={size} were not found")

    return [rows_by_density[density] for density in sorted(rows_by_density)]


class MstInterface:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Lab7 MST Interface")
        self.root.geometry("1400x900")

        self.size_var = StringVar()
        self.repeats_var = StringVar(value="5")
        self.status_var = StringVar(value="Готово к запуску")

        self._build_layout()
        self._populate_sizes()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(6, weight=1)

        ttk.Label(controls, text="Размер графа:").grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.size_box = ttk.Combobox(controls, textvariable=self.size_var, state="readonly", width=12)
        self.size_box.grid(row=0, column=1, padx=(0, 16), sticky="w")

        ttk.Label(controls, text="Повторы:").grid(row=0, column=2, padx=(0, 8), sticky="w")
        self.repeats_box = ttk.Combobox(
            controls,
            textvariable=self.repeats_var,
            state="readonly",
            values=("1", "3", "5", "10"),
            width=8,
        )
        self.repeats_box.grid(row=0, column=3, padx=(0, 16), sticky="w")

        self.run_button = ttk.Button(controls, text="Запустить алгоритмы", command=self.start_run)
        self.run_button.grid(row=0, column=4, padx=(0, 16), sticky="w")

        self.status_label = ttk.Label(controls, textvariable=self.status_var)
        self.status_label.grid(row=0, column=6, sticky="e")

        chart_frame = ttk.LabelFrame(self.root, text="График времени", padding=12)
        chart_frame.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        chart_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_xlabel("Плотность, %")
        self.axis.set_ylabel("Время, мс")
        self.axis.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.LabelFrame(self.root, text="Результаты по плотностям", padding=12)
        table_frame.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = (
            "density",
            "graph_edges",
            "mst_weight",
            "mst_edges",
            "kruskal_time",
            "prim_binary_time",
            "prim_fibonacci_time",
            "boruvka_time",
        )
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)
        headings = {
            "density": "Плотность, %",
            "graph_edges": "Рёбер в графе",
            "mst_weight": "Длина MST",
            "mst_edges": "Рёбер в MST",
            "kruskal_time": "Крускал, мс",
            "prim_binary_time": "Прим+бин., мс",
            "prim_fibonacci_time": "Прим+Фиб., мс",
            "boruvka_time": "Борувка, мс",
        }
        widths = {
            "density": 100,
            "graph_edges": 120,
            "mst_weight": 110,
            "mst_edges": 110,
            "kruskal_time": 120,
            "prim_binary_time": 130,
            "prim_fibonacci_time": 130,
            "boruvka_time": 120,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _populate_sizes(self) -> None:
        sizes = [str(size) for size in get_available_sizes()]
        self.size_box["values"] = sizes
        if sizes:
            self.size_var.set(sizes[0])

    def start_run(self) -> None:
        if not self.size_var.get():
            messagebox.showerror("Ошибка", "Выбери размер графа.")
            return

        self.run_button.config(state="disabled")
        self.status_var.set("Подготовка запуска...")
        worker = threading.Thread(target=self._run_selected_size, daemon=True)
        worker.start()

    def _run_selected_size(self) -> None:
        size = int(self.size_var.get())
        repeats = int(self.repeats_var.get())
        output_path = RESULTS_DIR / f"n{size}_r{repeats}.csv"

        try:
            self.root.after(0, self.status_var.set, f"Запуск benchmark_mst.exe для n={size}...")
            run_benchmark_for_size(size=size, repeats=repeats, output_path=output_path)
            rows = load_rows_from_csv(output_path, size)
        except Exception as error:  # pragma: no cover
            self.root.after(0, self._handle_error, str(error))
            return

        self.root.after(0, self._update_ui, size, rows, output_path)

    def _handle_error(self, message: str) -> None:
        self.run_button.config(state="normal")
        self.status_var.set("Ошибка")
        messagebox.showerror("Ошибка запуска", message)

    def _update_ui(self, size: int, rows: list[dict[str, int | float]], output_path: Path) -> None:
        self._draw_chart(size, rows)
        self._fill_table(rows)
        self.run_button.config(state="normal")
        self.status_var.set(f"Готово: n={size}, CSV: {output_path.name}")

    def _draw_chart(self, size: int, rows: list[dict[str, int | float]]) -> None:
        self.axis.clear()
        x_values = [int(row["density"]) for row in rows]

        for key, meta in ALGORITHMS.items():
            y_values = [float(row[f"{key}_time_ms"]) for row in rows]
            self.axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                color=meta["color"],
                label=meta["label"],
            )

        self.axis.set_title(f"Время алгоритмов для графов с n={size}")
        self.axis.set_xlabel("Плотность, %")
        self.axis.set_ylabel("Время, мс")
        self.axis.grid(True, alpha=0.3)
        self.axis.legend()
        self.canvas.draw()

    def _fill_table(self, rows: list[dict[str, int | float]]) -> None:
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert(
                "",
                "end",
                values=(
                    int(row["density"]),
                    int(row["graph_edges"]),
                    int(row["mst_weight"]),
                    int(row["mst_edges"]),
                    f"{float(row['kruskal_time_ms']):.3f}",
                    f"{float(row['prim_binary_heap_time_ms']):.3f}",
                    f"{float(row['prim_fibonacci_heap_time_ms']):.3f}",
                    f"{float(row['boruvka_time_ms']):.3f}",
                ),
            )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    MstInterface().run()
