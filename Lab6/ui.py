from __future__ import annotations

import math
import secrets
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MIN_TEMPERATURE = 0.1
DEFAULT_INITIAL_TEMPERATURE = 100.0
DEFAULT_ITERATIONS_PER_TEMPERATURE = 50
DEFAULT_ALPHA = 0.95
DEFAULT_ANT_ITERATIONS = 100
DEFAULT_PHEROMONE_IMPORTANCE = 1.0
DEFAULT_DISTANCE_IMPORTANCE = 2.0
DEFAULT_PHEROMONE_DEPOSIT = 100.0
DEFAULT_EVAPORATION_INTENSITY = 3.0
DEFAULT_ANT_COUNT = ""
SEED_UPPER_BOUND = 2**32

FIGURE1_FILE_NAME = "figure1.edgelist"
CANVAS_WIDTH = 380
CANVAS_HEIGHT = 460

FIGURE1_LAYOUT: dict[str, tuple[float, float]] = {
    "a": (0.56, 0.08),
    "b": (0.10, 0.26),
    "c": (0.10, 0.72),
    "g": (0.44, 0.48),
    "d": (0.58, 0.92),
    "f": (0.90, 0.48),
}

ANT_MODE_LABELS = {
    "Ant colony": "ant_colony_basic",
    "Elitist ant colony": "ant_colony_elitist",
}

ALGORITHM_TITLES = {
    "simulated_annealing": "Simulated annealing",
    "ant_colony_basic": "Ant colony",
    "ant_colony_elitist": "Elitist ant colony",
}


def _load_algorithm_api():
    try:
        try:
            from .annealing import load_tsp_graph, run_simulated_annealing
            from .ant_colony import run_basic_ant_colony, run_elitist_ant_colony
        except ImportError:
            from annealing import load_tsp_graph, run_simulated_annealing
            from ant_colony import run_basic_ant_colony, run_elitist_ant_colony
    except Exception as exc:  # pragma: no cover - handled in UI at runtime
        raise RuntimeError(
            "Algorithm modules are unavailable. Make sure Lab6/annealing.py and "
            "Lab6/ant_colony.py exist and export the expected functions."
        ) from exc

    return load_tsp_graph, run_simulated_annealing, run_basic_ant_colony, run_elitist_ant_colony


def _discover_graph_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []

    files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".stp", ".edgelist"}
    ]
    return sorted(files, key=lambda path: (path.suffix.lower() != ".edgelist", path.name.lower()))


def _generate_seed() -> int:
    return secrets.randbelow(SEED_UPPER_BOUND)


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _format_route(route: Any) -> str:
    if not route:
        return "n/a"
    return " -> ".join(str(node) for node in route)


def _extract_graph_name(graph: Any, fallback: str) -> str:
    if isinstance(graph, dict):
        for key in ("name", "graph_name", "label"):
            value = graph.get(key)
            if value:
                return str(value)

    for attr in ("name", "graph_name", "label"):
        value = getattr(graph, attr, None)
        if value:
            return str(value)

    return fallback


def _extract_result_value(result: dict[str, Any], key: str, fallback: Any = None) -> Any:
    return result.get(key, fallback)


def _supports_figure1_visualization(graph: Any) -> bool:
    source_path = getattr(graph, "source_path", None)
    if source_path is None:
        return False
    return Path(source_path).name.lower() == FIGURE1_FILE_NAME


def _lookup_edge_weight(graph: Any, source_label: str, target_label: str) -> float | None:
    node_labels = getattr(graph, "node_labels", None)
    distance_matrix = getattr(graph, "distance_matrix", None)
    if node_labels is None or distance_matrix is None:
        return None

    label_to_index = {str(label): idx for idx, label in enumerate(node_labels)}
    source_index = label_to_index.get(source_label)
    target_index = label_to_index.get(target_label)
    if source_index is None or target_index is None:
        return None

    return float(distance_matrix[source_index, target_index])


def _trim_segment(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    node_radius: float,
    arrow_padding: float,
) -> tuple[float, float, float, float]:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return x1, y1, x2, y2

    ux = dx / length
    uy = dy / length
    return (
        x1 + ux * node_radius,
        y1 + uy * node_radius,
        x2 - ux * (node_radius + arrow_padding),
        y2 - uy * (node_radius + arrow_padding),
    )


def _edge_label_position(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    label_offset: float,
    sign: int,
) -> tuple[float, float]:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return x1, y1

    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    nx = -dy / length
    ny = dx / length
    return mx + nx * label_offset * sign, my + ny * label_offset * sign


class AnnealingUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Lab6 - Parallel TSP Heuristics")
        self.root.geometry("1180x760")
        self.root.minsize(1020, 700)

        self._queue: Queue[tuple[str, Any]] = Queue()
        self._running = False
        self._pending_tasks: set[str] = set()
        self._had_errors = False
        self._last_visualization_graph: Any | None = None
        self._last_visualization_result: dict[str, Any] | None = None
        self._last_visualization_title = "Best cycle"
        self._visualization_message = "Run figure1.edgelist to visualize the best cycle."

        self.graph_files = _discover_graph_files()
        if not self.graph_files:
            self.graph_files = [DATA_DIR / FIGURE1_FILE_NAME]

        self.graph_var = tk.StringVar(value=self.graph_files[0].name)
        self.ant_mode_var = tk.StringVar(value=next(iter(ANT_MODE_LABELS)))
        self.initial_temperature_var = tk.StringVar(value=str(DEFAULT_INITIAL_TEMPERATURE))
        self.iterations_var = tk.StringVar(value=str(DEFAULT_ITERATIONS_PER_TEMPERATURE))
        self.cooling_var = tk.StringVar(value="geometric")
        self.alpha_var = tk.StringVar(value=str(DEFAULT_ALPHA))
        self.ant_iterations_var = tk.StringVar(value=str(DEFAULT_ANT_ITERATIONS))
        self.ant_count_var = tk.StringVar(value=DEFAULT_ANT_COUNT)
        self.pheromone_importance_var = tk.StringVar(value=str(DEFAULT_PHEROMONE_IMPORTANCE))
        self.distance_importance_var = tk.StringVar(value=str(DEFAULT_DISTANCE_IMPORTANCE))
        self.pheromone_deposit_var = tk.StringVar(value=str(DEFAULT_PHEROMONE_DEPOSIT))
        self.evaporation_intensity_var = tk.StringVar(value=str(DEFAULT_EVAPORATION_INTENSITY))
        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()
        self._update_cooling_controls()
        self.root.after(100, self._poll_queue)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        title = ttk.Label(container, text="Parallel TSP Heuristics", font=("Segoe UI", 15, "bold"))
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            container,
            text=(
                "Run simulated annealing and the selected ant colony mode in parallel. "
                "Each algorithm reports its result as soon as it finishes."
            ),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 10))

        top_controls = ttk.Frame(container)
        top_controls.grid(row=2, column=0, sticky="ew")
        top_controls.columnconfigure(0, weight=1)
        top_controls.columnconfigure(1, weight=1)
        top_controls.columnconfigure(2, weight=1)

        common_controls = ttk.LabelFrame(top_controls, text="Common Parameters", padding=12)
        common_controls.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        common_controls.columnconfigure(1, weight=1)

        self._row = 0
        self._add_combo(common_controls, "Graph", self.graph_var, [path.name for path in self.graph_files])
        self._add_combo(common_controls, "Ant colony mode", self.ant_mode_var, list(ANT_MODE_LABELS))

        button_row = ttk.Frame(common_controls)
        button_row.grid(row=self._row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.run_button = ttk.Button(button_row, text="Run Both", command=self._on_run)
        self.run_button.pack(side="left")
        ttk.Label(button_row, textvariable=self.status_var).pack(side="left", padx=12)

        self.sa_controls = ttk.LabelFrame(
            top_controls,
            text="Simulated Annealing Parameters",
            padding=12,
        )
        self.sa_controls.grid(row=0, column=1, sticky="nsew", padx=6)
        self.sa_controls.columnconfigure(1, weight=1)

        self._row = 0
        self._add_entry(self.sa_controls, "Initial temperature", self.initial_temperature_var)
        self._add_entry(self.sa_controls, "Iterations per temperature", self.iterations_var)
        self._add_combo(self.sa_controls, "Cooling mode", self.cooling_var, ["geometric", "cauchy"])
        self.alpha_entry = self._add_entry(self.sa_controls, "Alpha (geometric)", self.alpha_var)
        self._add_readonly(self.sa_controls, "Minimum temperature", f"{MIN_TEMPERATURE}")
        self.cooling_var.trace_add("write", self._on_cooling_mode_changed)

        self.ant_controls = ttk.LabelFrame(top_controls, text="Ant Colony Parameters", padding=12)
        self.ant_controls.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.ant_controls.columnconfigure(1, weight=1)

        self._row = 0
        self._add_entry(self.ant_controls, "Iterations", self.ant_iterations_var)
        self._add_entry(self.ant_controls, "Ant count (blank = |V|)", self.ant_count_var)
        self._add_entry(self.ant_controls, "Pheromone importance", self.pheromone_importance_var)
        self._add_entry(self.ant_controls, "Distance importance", self.distance_importance_var)
        self._add_entry(self.ant_controls, "Pheromone deposit", self.pheromone_deposit_var)
        self._add_entry(self.ant_controls, "Evaporation intensity (0..10)", self.evaporation_intensity_var)

        self.results_frame = ttk.Frame(container)
        self.results_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        self.results_frame.columnconfigure(0, weight=4)
        self.results_frame.columnconfigure(1, weight=4)
        self.results_frame.columnconfigure(2, weight=5)
        self.results_frame.rowconfigure(0, weight=1)

        sa_text_group = ttk.LabelFrame(self.results_frame, text="Simulated Annealing Result", padding=12)
        sa_text_group.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.sa_output_text = self._create_output_text(sa_text_group)

        ant_text_group = ttk.LabelFrame(self.results_frame, text="Ant Colony Result", padding=12)
        ant_text_group.grid(row=0, column=1, sticky="nsew", padx=6)
        self.ant_output_text = self._create_output_text(ant_text_group)

        visual_group = ttk.LabelFrame(self.results_frame, text="Best Cycle", padding=12)
        visual_group.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        self.path_canvas = tk.Canvas(
            visual_group,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            background="white",
            highlightthickness=1,
            highlightbackground="#c9c9c9",
        )
        self.path_canvas.pack(fill="both", expand=True)
        self.path_canvas.bind("<Configure>", self._on_canvas_resize)

        self._write_output(
            self.sa_output_text,
            "Simulated annealing is ready.\nPress Run Both to start the parallel comparison.",
        )
        self._write_output(
            self.ant_output_text,
            "Ant colony is ready.\nPress Run Both to start the parallel comparison.",
        )
        self._clear_visualization("Run figure1.edgelist to visualize the best cycle.")

    def _add_entry(self, parent: ttk.Labelframe, label: str, variable: tk.StringVar) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=self._row, column=0, sticky="w", padx=(0, 10), pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=18)
        entry.grid(row=self._row, column=1, sticky="ew", pady=4)
        self._row += 1
        return entry

    def _add_combo(
        self,
        parent: ttk.Labelframe,
        label: str,
        variable: tk.StringVar,
        values: list[str],
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=self._row, column=0, sticky="w", padx=(0, 10), pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18)
        combo.grid(row=self._row, column=1, sticky="ew", pady=4)
        self._row += 1
        return combo

    def _add_readonly(self, parent: ttk.Labelframe, label: str, value: str) -> None:
        ttk.Label(parent, text=label).grid(row=self._row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Label(parent, text=value).grid(row=self._row, column=1, sticky="ew", pady=4)
        self._row += 1

    def _create_output_text(self, parent: ttk.Labelframe) -> tk.Text:
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill="both", expand=True)

        text_widget = tk.Text(text_frame, wrap="word", height=16, width=34, undo=False)
        y_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=y_scroll.set)
        text_widget.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")
        return text_widget

    def _selected_ant_mode_key(self) -> str:
        return ANT_MODE_LABELS.get(self.ant_mode_var.get().strip(), "ant_colony_basic")

    def _on_cooling_mode_changed(self, *_args: object) -> None:
        self._update_cooling_controls()

    def _update_cooling_controls(self) -> None:
        cooling_mode = self.cooling_var.get().strip().lower()
        self.alpha_entry.configure(state="normal" if cooling_mode == "geometric" else "disabled")

    def _validate_inputs(self) -> dict[str, Any]:
        graph_name = self.graph_var.get().strip()
        graph_map = {path.name: path for path in self.graph_files}
        graph_path = graph_map.get(graph_name)
        if graph_path is None:
            raise ValueError("Select a graph from the list.")

        params: dict[str, Any] = {
            "graph_name": graph_name,
            "graph_path": graph_path,
            "ant_mode_key": self._selected_ant_mode_key(),
        }

        try:
            initial_temperature = float(self.initial_temperature_var.get().strip())
        except ValueError as exc:
            raise ValueError("Initial temperature must be a number.") from exc
        if initial_temperature <= MIN_TEMPERATURE:
            raise ValueError(f"Initial temperature must be greater than {MIN_TEMPERATURE}.")

        try:
            iterations_per_temperature = int(self.iterations_var.get().strip())
        except ValueError as exc:
            raise ValueError("Iterations per temperature must be an integer.") from exc
        if iterations_per_temperature <= 0:
            raise ValueError("Iterations per temperature must be positive.")

        cooling_mode = self.cooling_var.get().strip().lower()
        if cooling_mode not in {"geometric", "cauchy"}:
            raise ValueError("Cooling mode must be geometric or cauchy.")

        alpha = DEFAULT_ALPHA
        if cooling_mode == "geometric":
            try:
                alpha = float(self.alpha_var.get().strip())
            except ValueError as exc:
                raise ValueError("Alpha must be a number.") from exc
            if not (0.0 < alpha < 1.0):
                raise ValueError("Alpha must be between 0 and 1.")

        try:
            ant_iterations = int(self.ant_iterations_var.get().strip())
        except ValueError as exc:
            raise ValueError("Ant colony iterations must be an integer.") from exc
        if ant_iterations <= 0:
            raise ValueError("Ant colony iterations must be positive.")

        ant_count_text = self.ant_count_var.get().strip()
        ant_count: int | None = None
        if ant_count_text:
            try:
                ant_count = int(ant_count_text)
            except ValueError as exc:
                raise ValueError("Ant count must be an integer.") from exc
            if ant_count <= 0:
                raise ValueError("Ant count must be positive.")

        try:
            pheromone_importance = float(self.pheromone_importance_var.get().strip())
        except ValueError as exc:
            raise ValueError("Pheromone importance must be a number.") from exc
        if pheromone_importance < 0:
            raise ValueError("Pheromone importance must be >= 0.")

        try:
            distance_importance = float(self.distance_importance_var.get().strip())
        except ValueError as exc:
            raise ValueError("Distance importance must be a number.") from exc
        if distance_importance < 0:
            raise ValueError("Distance importance must be >= 0.")

        try:
            pheromone_deposit = float(self.pheromone_deposit_var.get().strip())
        except ValueError as exc:
            raise ValueError("Pheromone deposit must be a number.") from exc
        if pheromone_deposit <= 0:
            raise ValueError("Pheromone deposit must be > 0.")

        try:
            evaporation_intensity = float(self.evaporation_intensity_var.get().strip())
        except ValueError as exc:
            raise ValueError("Evaporation intensity must be a number.") from exc
        if not (0 <= evaporation_intensity <= 10):
            raise ValueError("Evaporation intensity must be between 0 and 10.")

        params.update(
            initial_temperature=initial_temperature,
            iterations_per_temperature=iterations_per_temperature,
            cooling_mode=cooling_mode,
            alpha=alpha,
            ant_iterations=ant_iterations,
            ant_count=ant_count,
            pheromone_importance=pheromone_importance,
            distance_importance=distance_importance,
            pheromone_deposit=pheromone_deposit,
            evaporation_intensity=evaporation_intensity,
        )
        return params

    def _on_run(self) -> None:
        if self._running:
            return

        try:
            params = self._validate_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.root)
            return

        params["annealing_seed"] = _generate_seed()
        params["ant_seed"] = _generate_seed()
        ant_mode_key = params["ant_mode_key"]

        self._pending_tasks = {"simulated_annealing", ant_mode_key}
        self._had_errors = False
        self._set_running(True)
        self.status_var.set(
            f"Running simulated annealing and {ALGORITHM_TITLES[ant_mode_key]} on "
            f"{params['graph_name']}..."
        )
        self._write_output(
            self.sa_output_text,
            "Pending run...\n"
            "Algorithm: Simulated annealing\n"
            f"Seed used: {params['annealing_seed']}",
        )
        self._write_output(
            self.ant_output_text,
            "Pending run...\n"
            f"Algorithm: {ALGORITHM_TITLES[ant_mode_key]}\n"
            f"Seed used: {params['ant_seed']}",
        )
        self._clear_visualization("Waiting for the first completed result on figure1.edgelist.")

        threading.Thread(target=self._annealing_worker, args=(params,), daemon=True).start()
        threading.Thread(target=self._ant_worker, args=(params,), daemon=True).start()

    def _annealing_worker(self, params: dict[str, Any]) -> None:
        try:
            (
                load_tsp_graph,
                run_simulated_annealing,
                _run_basic_ant_colony,
                _run_elitist_ant_colony,
            ) = _load_algorithm_api()
            graph = load_tsp_graph(params["graph_path"])
            graph_name = _extract_graph_name(graph, params["graph_name"])
            result = run_simulated_annealing(
                graph,
                initial_temperature=params["initial_temperature"],
                iterations_per_temperature=params["iterations_per_temperature"],
                cooling_mode=params["cooling_mode"],
                alpha=params["alpha"],
                min_temperature=MIN_TEMPERATURE,
                seed=params["annealing_seed"],
            )
            self._queue.put(
                (
                    "task_success",
                    ("simulated_annealing", graph, graph_name, result, params["annealing_seed"]),
                )
            )
        except Exception:
            self._queue.put(("task_error", ("simulated_annealing", traceback.format_exc())))

    def _ant_worker(self, params: dict[str, Any]) -> None:
        ant_mode_key = params["ant_mode_key"]

        try:
            (
                load_tsp_graph,
                _run_simulated_annealing,
                run_basic_ant_colony,
                run_elitist_ant_colony,
            ) = _load_algorithm_api()
            graph = load_tsp_graph(params["graph_path"])
            graph_name = _extract_graph_name(graph, params["graph_name"])

            if ant_mode_key == "ant_colony_basic":
                result = run_basic_ant_colony(
                    graph,
                    iterations=params["ant_iterations"],
                    ant_count=params["ant_count"],
                    pheromone_importance=params["pheromone_importance"],
                    distance_importance=params["distance_importance"],
                    pheromone_deposit=params["pheromone_deposit"],
                    evaporation_intensity=params["evaporation_intensity"],
                    seed=params["ant_seed"],
                )
            else:
                result = run_elitist_ant_colony(
                    graph,
                    iterations=params["ant_iterations"],
                    ant_count=params["ant_count"],
                    pheromone_importance=params["pheromone_importance"],
                    distance_importance=params["distance_importance"],
                    pheromone_deposit=params["pheromone_deposit"],
                    evaporation_intensity=params["evaporation_intensity"],
                    seed=params["ant_seed"],
                )

            self._queue.put(
                ("task_success", (ant_mode_key, graph, graph_name, result, params["ant_seed"]))
            )
        except Exception:
            self._queue.put(("task_error", (ant_mode_key, traceback.format_exc())))

    def _poll_queue(self) -> None:
        try:
            kind, payload = self._queue.get_nowait()
        except Empty:
            self.root.after(100, self._poll_queue)
            return

        if kind == "task_success":
            algorithm_key, graph, graph_name, result, seed_used = payload
            self._show_result(algorithm_key, graph_name, result, seed_used)
            self._render_visualization(graph, result, ALGORITHM_TITLES[algorithm_key])
            self._finish_task(algorithm_key)
        else:
            algorithm_key, error_text = payload
            self._had_errors = True
            self._show_task_error(algorithm_key, error_text)
            self._finish_task(algorithm_key)
            messagebox.showerror(
                "Run failed",
                f"{ALGORITHM_TITLES[algorithm_key]} failed. See its result panel for details.",
                parent=self.root,
            )

        self.root.after(100, self._poll_queue)

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.run_button.configure(state="disabled" if running else "normal")

    def _finish_task(self, algorithm_key: str) -> None:
        self._pending_tasks.discard(algorithm_key)
        if self._pending_tasks:
            waiting_for = ", ".join(ALGORITHM_TITLES[key] for key in sorted(self._pending_tasks))
            self.status_var.set(f"Waiting for: {waiting_for}")
            return

        self.status_var.set("Finished with errors." if self._had_errors else "Finished.")
        self._set_running(False)

    def _show_task_error(self, algorithm_key: str, error_text: str) -> None:
        self._write_output(
            self._output_widget_for_task(algorithm_key),
            f"Algorithm: {ALGORITHM_TITLES[algorithm_key]}\nRun failed.\n\n{error_text}",
        )

    def _output_widget_for_task(self, algorithm_key: str) -> tk.Text:
        if algorithm_key == "simulated_annealing":
            return self.sa_output_text
        return self.ant_output_text

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._last_visualization_graph is not None and self._last_visualization_result is not None:
            self._draw_figure1_cycle(self._last_visualization_graph, self._last_visualization_result)
        else:
            self._draw_visualization_message(self._visualization_message)

    def _show_result(
        self,
        algorithm_key: str,
        graph_name: str,
        result: dict[str, Any],
        seed_used: int,
    ) -> None:
        if algorithm_key == "simulated_annealing":
            summary_lines = [
                f"Algorithm: {ALGORITHM_TITLES[algorithm_key]}",
                f"Graph: {graph_name}",
                f"Cooling mode: {result.get('cooling_mode', 'n/a')}",
                f"Seed used: {seed_used}",
                f"Initial route: {_format_route(_extract_result_value(result, 'initial_route_labels'))}",
                f"Initial length: {_format_number(_extract_result_value(result, 'initial_length'))}",
                f"Best route: {_format_route(_extract_result_value(result, 'best_route_labels'))}",
                f"Best length: {_format_number(_extract_result_value(result, 'best_length'))}",
                f"Evaluations: {_format_number(_extract_result_value(result, 'evaluations'))}",
                f"Temperature steps: {_format_number(_extract_result_value(result, 'temperature_steps'))}",
                f"Accepted moves: {_format_number(_extract_result_value(result, 'accepted_moves'))}",
                f"Improving moves: {_format_number(_extract_result_value(result, 'improving_moves'))}",
                f"Elapsed seconds: {_format_number(_extract_result_value(result, 'elapsed_seconds'))}",
            ]
        else:
            summary_lines = [
                f"Algorithm: {ALGORITHM_TITLES[algorithm_key]}",
                f"Graph: {graph_name}",
                f"Colony mode: {_format_number(_extract_result_value(result, 'mode'))}",
                f"Seed used: {seed_used}",
                f"Best route: {_format_route(_extract_result_value(result, 'best_route_labels'))}",
                f"Best length: {_format_number(_extract_result_value(result, 'best_length'))}",
                f"Iterations: {_format_number(_extract_result_value(result, 'iterations'))}",
                f"Best iteration: {_format_number(_extract_result_value(result, 'best_iteration'))}",
                f"Ant count: {_format_number(_extract_result_value(result, 'ant_count'))}",
                f"Successful tours: {_format_number(_extract_result_value(result, 'successful_tours'))}",
                f"Failed tours: {_format_number(_extract_result_value(result, 'failed_tours'))}",
                f"Evaluations: {_format_number(_extract_result_value(result, 'evaluations'))}",
                f"Pheromone importance: {_format_number(_extract_result_value(result, 'pheromone_importance'))}",
                f"Distance importance: {_format_number(_extract_result_value(result, 'distance_importance'))}",
                f"Pheromone deposit: {_format_number(_extract_result_value(result, 'pheromone_deposit'))}",
                f"Evaporation intensity: {_format_number(_extract_result_value(result, 'evaporation_intensity'))}",
                f"Elapsed seconds: {_format_number(_extract_result_value(result, 'elapsed_seconds'))}",
            ]
            elite_ants = _extract_result_value(result, "elite_ants", 0)
            if elite_ants:
                summary_lines.insert(11, f"Elite ants: {_format_number(elite_ants)}")

        self._write_output(self._output_widget_for_task(algorithm_key), "\n".join(summary_lines))

    def _clear_visualization(self, message: str) -> None:
        self._last_visualization_graph = None
        self._last_visualization_result = None
        self._last_visualization_title = "Best cycle"
        self._visualization_message = message
        self._draw_visualization_message(message)

    def _draw_visualization_message(self, message: str) -> None:
        width, height = self._current_canvas_size()
        self.path_canvas.delete("all")
        self.path_canvas.create_text(
            width / 2,
            height / 2,
            text=message,
            width=max(120, width - 40),
            justify="center",
            fill="#5a5a5a",
            font=("Segoe UI", 11),
        )

    def _render_visualization(self, graph: Any, result: dict[str, Any], title: str) -> None:
        if not _supports_figure1_visualization(graph):
            self._clear_visualization("Visualization is available only for figure1.edgelist.")
            return

        route = _extract_result_value(result, "best_route_labels")
        if not isinstance(route, list) or len(route) < 2:
            self._clear_visualization("No best cycle is available for visualization.")
            return

        route_labels = [str(label) for label in route]
        if any(label not in FIGURE1_LAYOUT for label in route_labels):
            self._clear_visualization("The best cycle cannot be drawn with the fixed figure1 layout.")
            return

        self._last_visualization_graph = graph
        self._last_visualization_result = result
        self._last_visualization_title = title
        self._draw_figure1_cycle(graph, result)

    def _draw_figure1_cycle(self, graph: Any, result: dict[str, Any]) -> None:
        route = _extract_result_value(result, "best_route_labels")
        if not isinstance(route, list) or len(route) < 2:
            self._draw_visualization_message("No best cycle is available for visualization.")
            return

        route_labels = [str(label) for label in route]
        if any(label not in FIGURE1_LAYOUT for label in route_labels):
            self._draw_visualization_message("The best cycle cannot be drawn with the fixed figure1 layout.")
            return

        width, height = self._current_canvas_size()
        positions = self._scaled_figure1_layout(width, height)
        node_radius = self._node_radius(width, height)
        arrow_padding = max(8, int(node_radius * 0.35))
        label_offset = max(14, int(node_radius * 0.75))

        self.path_canvas.delete("all")
        self.path_canvas.create_text(
            width / 2,
            max(18, int(height * 0.05)),
            text=f"{self._last_visualization_title} best cycle for figure1.edgelist",
            fill="#303030",
            font=("Segoe UI", 12, "bold"),
        )

        for edge_index, (source_label, target_label) in enumerate(zip(route_labels, route_labels[1:])):
            self._draw_cycle_edge(
                graph,
                positions,
                source_label,
                target_label,
                edge_index,
                node_radius=node_radius,
                arrow_padding=arrow_padding,
                label_offset=label_offset,
            )

        for label, (x_coord, y_coord) in positions.items():
            self._draw_node(label, x_coord, y_coord, node_radius=node_radius)

    def _current_canvas_size(self) -> tuple[int, int]:
        width = self.path_canvas.winfo_width()
        height = self.path_canvas.winfo_height()
        if width <= 1:
            width = CANVAS_WIDTH
        if height <= 1:
            height = CANVAS_HEIGHT
        return width, height

    def _scaled_figure1_layout(self, width: int, height: int) -> dict[str, tuple[float, float]]:
        left_padding = max(36, int(width * 0.08))
        right_padding = max(36, int(width * 0.08))
        top_padding = max(56, int(height * 0.12))
        bottom_padding = max(36, int(height * 0.08))
        usable_width = max(120, width - left_padding - right_padding)
        usable_height = max(140, height - top_padding - bottom_padding)

        return {
            label: (
                left_padding + normalized_x * usable_width,
                top_padding + normalized_y * usable_height,
            )
            for label, (normalized_x, normalized_y) in FIGURE1_LAYOUT.items()
        }

    def _node_radius(self, width: int, height: int) -> int:
        return max(18, min(28, int(min(width, height) * 0.045)))

    def _draw_cycle_edge(
        self,
        graph: Any,
        positions: dict[str, tuple[float, float]],
        source_label: str,
        target_label: str,
        edge_index: int,
        *,
        node_radius: int,
        arrow_padding: int,
        label_offset: int,
    ) -> None:
        x1, y1 = positions[source_label]
        x2, y2 = positions[target_label]
        sx, sy, ex, ey = _trim_segment(
            x1,
            y1,
            x2,
            y2,
            node_radius=node_radius,
            arrow_padding=arrow_padding,
        )

        self.path_canvas.create_line(
            sx,
            sy,
            ex,
            ey,
            width=3,
            fill="#cc5500",
            arrow=tk.LAST,
            arrowshape=(16, 18, 6),
        )

        weight = _lookup_edge_weight(graph, source_label, target_label)
        if weight is not None and math.isfinite(weight):
            label_x, label_y = _edge_label_position(
                sx,
                sy,
                ex,
                ey,
                label_offset=label_offset,
                sign=1 if edge_index % 2 == 0 else -1,
            )
            self._draw_edge_weight(label_x, label_y, _format_number(weight))

    def _draw_edge_weight(self, x_coord: float, y_coord: float, text: str) -> None:
        text_id = self.path_canvas.create_text(
            x_coord,
            y_coord,
            text=text,
            fill="#1f1f1f",
            font=("Segoe UI", 10, "bold"),
        )
        bbox = self.path_canvas.bbox(text_id)
        if bbox is None:
            return

        background_id = self.path_canvas.create_rectangle(
            bbox[0] - 4,
            bbox[1] - 2,
            bbox[2] + 4,
            bbox[3] + 2,
            fill="white",
            outline="",
        )
        self.path_canvas.tag_raise(text_id, background_id)

    def _draw_node(self, label: str, x_coord: float, y_coord: float, *, node_radius: int) -> None:
        self.path_canvas.create_oval(
            x_coord - node_radius,
            y_coord - node_radius,
            x_coord + node_radius,
            y_coord + node_radius,
            fill="#ffffff",
            outline="#222222",
            width=2,
        )
        self.path_canvas.create_text(
            x_coord,
            y_coord,
            text=label,
            fill="#111111",
            font=("Segoe UI", 12, "bold"),
        )

    def _write_output(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text.rstrip() + "\n")
        widget.configure(state="disabled")


def run_interface(root: tk.Tk | None = None) -> None:
    root = root or tk.Tk()
    AnnealingUI(root)
    root.mainloop()


def main() -> None:
    run_interface()


if __name__ == "__main__":
    main()
