from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Any

from grid_path import ALGORITHMS as GRID_ALGORITHMS
from grid_path import GridPathProblem, export_paths as export_grid_paths, solve_problem
from models import SearchResult
from puzzle15 import ALGORITHMS as PUZZLE_ALGORITHMS
from puzzle15 import GOAL_STATE, generate_random_board, is_solvable, solve_board
from runtime import measure_search_run


Coord = tuple[int, int]
_ALGORITHM_LABELS = {**GRID_ALGORITHMS, **PUZZLE_ALGORITHMS}
GRID_ALL_PATHS_ALGORITHM = "connectivity"


def _fastest_successful(results: Iterable[SearchResult[Any]]) -> SearchResult[Any] | None:
    return min(
        (result for result in results if result.found),
        key=lambda result: result.metrics.elapsed_ms,
        default=None,
    )


def _format_metrics(
    result: SearchResult[Any],
    detail: str = "",
    labels: dict[str, str] | None = None,
) -> str:
    metrics = result.metrics
    label_map = labels or _ALGORITHM_LABELS
    lines = [
        f"Алгоритм: {label_map.get(result.algorithm, result.algorithm)}",
        f"Результат: {'решение найдено' if result.found else 'решение не найдено'}",
        f"Время: {metrics.elapsed_ms:.2f} мс",
        f"Раскрыто узлов: {metrics.nodes_expanded}",
        f"Сгенерировано состояний: {metrics.states_generated}",
        f"Отсечено состояний: {metrics.pruned_states}",
        f"Макс. глубина: {metrics.max_depth}",
        f"Макс. фронтир: {metrics.max_frontier}",
        f"Алгоритмическая память: {metrics.peak_memory_kb:.1f} KiB",
    ]
    if detail:
        lines.append(detail)
    if result.message:
        lines.append(f"Сообщение: {result.message}")
    return "\n".join(lines)


def _extract_grid_path(solution: Any) -> list[Coord]:
    if solution is None:
        return []
    if isinstance(solution, dict):
        candidate = solution.get("path")
        if isinstance(candidate, list):
            return [tuple(point) for point in candidate]
        return []
    if isinstance(solution, list):
        return [tuple(point) for point in solution]
    return []


def _extract_grid_paths(solution: Any) -> list[list[Coord]]:
    if solution is None:
        return []
    if isinstance(solution, dict):
        candidate = solution.get("paths")
        if isinstance(candidate, list):
            return [[tuple(point) for point in path] for path in candidate]
    path = _extract_grid_path(solution)
    return [path] if path else []


def _grid_solution_exhaustive(solution: Any) -> bool:
    if isinstance(solution, dict):
        return bool(solution.get("exhaustive", False))
    return False


def _format_grid_detail(result: SearchResult[Any]) -> str:
    if not result.found:
        return ""
    paths = _extract_grid_paths(result.solution)
    path = paths[0] if paths else []
    path_count = len(paths)
    if isinstance(result.solution, dict):
        path_count = int(result.solution.get("path_count", path_count))
    detail = [
        f"Длина пути: {max(len(path) - 1, 0)}",
        f"Найдено путей: {path_count}",
        f"Полный перебор: {'да' if _grid_solution_exhaustive(result.solution) else 'нет'}",
    ]
    return "\n".join(detail)


def _extract_moves(solution: Any) -> list[str]:
    if solution is None:
        return []
    if isinstance(solution, dict):
        candidate = solution.get("moves")
        if isinstance(candidate, list):
            return [str(move) for move in candidate]
        return []
    if isinstance(solution, list) and all(isinstance(item, str) for item in solution):
        return [str(item) for item in solution]
    return []


def _extract_states(solution: Any) -> list[tuple[int, ...]]:
    if solution is None:
        return []
    if isinstance(solution, dict):
        candidate = solution.get("states")
        if isinstance(candidate, list):
            return [tuple(state) for state in candidate]
        candidate = solution.get("path")
        if isinstance(candidate, list):
            return [tuple(state) for state in candidate]
    if isinstance(solution, list) and all(isinstance(item, tuple) for item in solution):
        return [tuple(item) for item in solution]
    return []


def _is_adjacent(index_a: int, index_b: int) -> bool:
    row_a, col_a = divmod(index_a, 4)
    row_b, col_b = divmod(index_b, 4)
    return abs(row_a - row_b) + abs(col_a - col_b) == 1


def _move_blank(state: tuple[int, ...], direction: str) -> tuple[int, ...] | None:
    blank = state.index(0)
    row, col = divmod(blank, 4)
    normalized = direction.strip().lower()
    offset_by_direction = {
        "u": 4,
        "up": 4,
        "вверх": 4,
        "d": -4,
        "down": -4,
        "вниз": -4,
        "l": 1,
        "left": 1,
        "влево": 1,
        "r": -1,
        "right": -1,
        "вправо": -1,
    }
    if normalized in {"u", "up", "вверх"} and row == 3:
        return None
    if normalized in {"d", "down", "вниз"} and row == 0:
        return None
    if normalized in {"l", "left", "влево"} and col == 3:
        return None
    if normalized in {"r", "right", "вправо"} and col == 0:
        return None
    if normalized not in offset_by_direction:
        return None
    swap_index = blank + offset_by_direction[normalized]
    board = list(state)
    board[blank], board[swap_index] = board[swap_index], board[blank]
    return tuple(board)


def _states_from_moves(
    start_state: tuple[int, ...],
    moves: Sequence[str],
) -> list[tuple[int, ...]]:
    states = [start_state]
    current = start_state
    for move in moves:
        next_state = _move_blank(current, move)
        if next_state is None:
            break
        states.append(next_state)
        current = next_state
    return states


@dataclass(slots=True)
class StoredResult:
    result: SearchResult[Any]
    context: Any


class BackgroundRunner:
    def __init__(self, root: tk.Tk, status_var: tk.StringVar) -> None:
        self.root = root
        self.status_var = status_var
        self._queue: queue.Queue[tuple[str, str, Any, Callable[[Any], None] | None]] = (
            queue.Queue()
        )
        self._busy = False
        self.root.after(100, self._poll)

    @property
    def busy(self) -> bool:
        return self._busy

    def start(
        self,
        label: str,
        work: Callable[[], Any],
        callback: Callable[[Any], None] | None = None,
    ) -> bool:
        if self._busy:
            messagebox.showinfo("Задача выполняется", "Дождитесь завершения текущего расчёта.")
            return False
        self._busy = True
        self.status_var.set(label)

        def target() -> None:
            try:
                payload = work()
                self._queue.put(("ok", label, payload, callback))
            except Exception as exc:  # pragma: no cover - handled at runtime
                self._queue.put(("error", label, (exc, traceback.format_exc()), callback))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return True

    def _poll(self) -> None:
        try:
            while True:
                kind, label, payload, callback = self._queue.get_nowait()
                self._busy = False
                if kind == "ok":
                    self.status_var.set(f"{label}: готово")
                    if callback is not None:
                        callback(payload)
                else:
                    exc, trace = payload
                    self.status_var.set(f"{label}: ошибка")
                    messagebox.showerror("Ошибка вычисления", f"{exc}\n\n{trace}")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll)


class GridPathTab(ttk.Frame):
    def __init__(self, master: ttk.Notebook, runner: BackgroundRunner) -> None:
        super().__init__(master, padding=12)
        self.runner = runner
        self.rows_var = tk.IntVar(value=5)
        self.cols_var = tk.IntVar(value=5)
        self.click_mode = tk.StringVar(value="start")
        self.algorithm_vars = {
            algorithm: tk.BooleanVar(value=True)
            for algorithm in GRID_ALGORITHMS
        }

        self.rows = 5
        self.cols = 5
        self.start: Coord = (0, 0)
        self.end: Coord = (4, 4)
        self.blocked: set[Coord] = set()
        self.current_path: list[Coord] = []
        self.result_rows: dict[str, StoredResult] = {}

        self._build_layout()
        self._draw_grid()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(self, text="Параметры и управление", padding=12)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        ttk.Label(controls, text="Размер сетки").grid(row=0, column=0, sticky="w")
        size_frame = ttk.Frame(controls)
        size_frame.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        ttk.Spinbox(size_frame, from_=2, to=8, width=5, textvariable=self.rows_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(size_frame, text="x").grid(row=0, column=1, padx=4)
        ttk.Spinbox(size_frame, from_=2, to=8, width=5, textvariable=self.cols_var).grid(
            row=0, column=2, sticky="w"
        )

        ttk.Button(controls, text="Применить размер", command=self._resize_grid).grid(
            row=2, column=0, sticky="ew", pady=2
        )
        ttk.Button(controls, text="Очистить путь", command=self._clear_path).grid(
            row=3, column=0, sticky="ew", pady=2
        )
        ttk.Button(controls, text="Снять блоки", command=self._clear_blocks).grid(
            row=4, column=0, sticky="ew", pady=(2, 12)
        )

        ttk.Label(controls, text="Режим левого клика").grid(row=5, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Старт", value="start", variable=self.click_mode).grid(
            row=6, column=0, sticky="w"
        )
        ttk.Radiobutton(controls, text="Финиш", value="end", variable=self.click_mode).grid(
            row=7, column=0, sticky="w"
        )
        ttk.Label(
            controls,
            text="Правый клик переключает блокировку клетки.",
            wraplength=250,
        ).grid(row=8, column=0, sticky="w", pady=(4, 12))

        ttk.Label(controls, text="Алгоритмы для сравнения").grid(row=9, column=0, sticky="w")
        algorithm_frame = ttk.Frame(controls)
        algorithm_frame.grid(row=10, column=0, sticky="ew", pady=(4, 10))
        for row_index, (algorithm, label) in enumerate(GRID_ALGORITHMS.items()):
            ttk.Checkbutton(
                algorithm_frame,
                text=label,
                variable=self.algorithm_vars[algorithm],
            ).grid(row=row_index, column=0, sticky="w")

        ttk.Button(controls, text="Запустить выбранные алгоритмы", command=self._compare_all).grid(
            row=11, column=0, sticky="ew", pady=2
        )
        ttk.Button(controls, text="Найти все пути", command=self._solve_all_paths).grid(
            row=12, column=0, sticky="ew", pady=(10, 2)
        )
        ttk.Button(controls, text="Экспорт путей", command=self._export_paths).grid(
            row=13, column=0, sticky="ew", pady=2
        )
        ttk.Label(
            controls,
            text=f"Полный перебор выполняется через {GRID_ALGORITHMS[GRID_ALL_PATHS_ALGORITHM]}.",
            wraplength=250,
        ).grid(
            row=14, column=0, sticky="w", pady=(8, 0)
        )

        board_panel = ttk.Frame(self)
        board_panel.grid(row=0, column=1, sticky="nsew")
        board_panel.columnconfigure(0, weight=1)
        board_panel.rowconfigure(0, weight=3)
        board_panel.rowconfigure(1, weight=2)

        self.canvas = tk.Canvas(board_panel, width=560, height=560, bg="#f5efe3", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        results_frame = ttk.LabelFrame(board_panel, text="Результаты", padding=10)
        results_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("algorithm", "found", "length", "paths", "time", "nodes", "pruned", "memory"),
            show="headings",
            height=6,
        )
        headings = {
            "algorithm": "Алгоритм",
            "found": "Найдено",
            "length": "Длина",
            "paths": "Пути",
            "time": "мс",
            "nodes": "Узлы",
            "pruned": "Отсечения",
            "memory": "Алг. память, KiB",
        }
        widths = {
            "algorithm": 180,
            "found": 80,
            "length": 70,
            "paths": 80,
            "time": 90,
            "nodes": 100,
            "pruned": 100,
            "memory": 125,
        }
        for key, title in headings.items():
            self.results_tree.heading(key, text=title)
            self.results_tree.column(key, width=widths[key], anchor="center", stretch=key == "algorithm")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self.details_text = tk.Text(results_frame, height=9, wrap="word")
        self.details_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.details_text.configure(state="disabled")

    def _resize_grid(self) -> None:
        self.rows = self.rows_var.get()
        self.cols = self.cols_var.get()
        self.start = (0, 0)
        self.end = (self.rows - 1, self.cols - 1)
        self.blocked.clear()
        self.current_path.clear()
        self._clear_results()
        self._draw_grid()

    def _clear_path(self) -> None:
        self.current_path.clear()
        self._draw_grid()

    def _clear_blocks(self) -> None:
        self.blocked.clear()
        self.current_path.clear()
        self._draw_grid()

    def _clear_results(self) -> None:
        self.result_rows.clear()
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._set_details("")

    def _build_problem(self) -> GridPathProblem:
        return GridPathProblem(
            rows=self.rows,
            cols=self.cols,
            start=self.start,
            end=self.end,
            blocked=frozenset(self.blocked),
        )

    def _selected_algorithms(self) -> list[str]:
        return [
            algorithm
            for algorithm, selected in self.algorithm_vars.items()
            if selected.get()
        ]

    def _solve_all_paths(self) -> None:
        problem = self._build_problem()
        algorithm = GRID_ALL_PATHS_ALGORITHM
        self.current_path.clear()
        self._clear_results()

        def work() -> SearchResult[Any]:
            return measure_search_run(lambda: solve_problem(problem, algorithm, max_solutions=0))

        def done(result: SearchResult[Any]) -> None:
            self._store_results(problem, [result])
            if result.found:
                self.current_path = _extract_grid_path(result.solution)
            self._draw_grid()
            self._set_details(_format_metrics(result, _format_grid_detail(result), GRID_ALGORITHMS))

        self.runner.start("Полный перебор путей задачи 1", work, done)

    def _compare_all(self) -> None:
        problem = self._build_problem()
        algorithms = self._selected_algorithms()
        if not algorithms:
            messagebox.showwarning("Алгоритмы не выбраны", "Отметьте хотя бы один алгоритм для запуска.")
            return
        self.current_path.clear()
        self._clear_results()

        def work() -> list[SearchResult[Any]]:
            results: list[SearchResult[Any]] = []
            for algorithm in algorithms:
                results.append(
                    measure_search_run(lambda algorithm=algorithm: solve_problem(problem, algorithm))
                )
            return results

        def done(results: list[SearchResult[Any]]) -> None:
            self._store_results(problem, results)
            successful = _fastest_successful(results)
            if successful is not None:
                self.current_path = _extract_grid_path(successful.solution)
            self._draw_grid()
            summary = "\n\n".join(
                _format_metrics(
                    result,
                    _format_grid_detail(result),
                    GRID_ALGORITHMS,
                )
                for result in results
            )
            self._set_details(summary)

        self.runner.start("Сравнение алгоритмов задачи 1", work, done)

    def _export_paths(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Экспорт путей",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")),
        )
        if not filename:
            return

        problem = self._build_problem()
        algorithm = GRID_ALL_PATHS_ALGORITHM

        def work() -> SearchResult[Any]:
            return measure_search_run(lambda: export_grid_paths(problem, algorithm, filename))

        def done(result: SearchResult[Any]) -> None:
            self._store_results(problem, [result])
            if result.found:
                self.current_path = _extract_grid_path(result.solution)
            self._draw_grid()
            if result.found:
                exported_count = 0
                if isinstance(result.solution, dict):
                    exported_count = int(result.solution.get("path_count", 0))
                messagebox.showinfo("Экспорт путей", f"Экспортировано путей: {exported_count}.")
            else:
                messagebox.showwarning("Экспорт путей", result.message or "Не удалось экспортировать пути.")
            self._set_details(_format_metrics(result, _format_grid_detail(result), GRID_ALGORITHMS))

        self.runner.start("Экспорт путей задачи 1", work, done)

    def _store_results(self, problem: GridPathProblem, results: Iterable[SearchResult[Any]]) -> None:
        self.result_rows.clear()
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for result in results:
            path = _extract_grid_path(result.solution)
            paths = _extract_grid_paths(result.solution)
            path_count = len(paths)
            if isinstance(result.solution, dict):
                path_count = int(result.solution.get("path_count", path_count))
            row_id = self.results_tree.insert(
                "",
                "end",
                values=(
                    GRID_ALGORITHMS.get(result.algorithm, result.algorithm),
                    "Да" if result.found else "Нет",
                    max(len(path) - 1, 0) if result.found else "-",
                    path_count if result.found else "-",
                    f"{result.metrics.elapsed_ms:.2f}",
                    result.metrics.nodes_expanded,
                    result.metrics.pruned_states,
                    f"{result.metrics.peak_memory_kb:.1f}",
                ),
            )
            self.result_rows[row_id] = StoredResult(result=result, context=problem)

    def _on_tree_select(self, _event: tk.Event[Any]) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        stored = self.result_rows.get(selection[0])
        if stored is None:
            return
        path = _extract_grid_path(stored.result.solution)
        self.current_path = path
        self._draw_grid()
        self._set_details(
            _format_metrics(stored.result, _format_grid_detail(stored.result), GRID_ALGORITHMS)
        )

    def _on_left_click(self, event: tk.Event[Any]) -> None:
        cell = self._canvas_to_cell(event.x, event.y)
        if cell is None:
            return
        if self.click_mode.get() == "start":
            if cell != self.end and cell not in self.blocked:
                self.start = cell
        else:
            if cell != self.start and cell not in self.blocked:
                self.end = cell
        self.current_path.clear()
        self._draw_grid()

    def _on_right_click(self, event: tk.Event[Any]) -> None:
        cell = self._canvas_to_cell(event.x, event.y)
        if cell is None or cell in {self.start, self.end}:
            return
        if cell in self.blocked:
            self.blocked.remove(cell)
        else:
            self.blocked.add(cell)
        self.current_path.clear()
        self._draw_grid()

    def _canvas_to_cell(self, x: int, y: int) -> Coord | None:
        width = int(self.canvas.winfo_width() or 560)
        height = int(self.canvas.winfo_height() or 560)
        padding = 20
        cell_size = min((width - 2 * padding) / self.cols, (height - 2 * padding) / self.rows)
        left = (width - cell_size * self.cols) / 2
        top = (height - cell_size * self.rows) / 2
        col = int((x - left) // cell_size)
        row = int((y - top) // cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    def _draw_grid(self) -> None:
        self.canvas.delete("all")
        width = int(self.canvas.winfo_width() or 560)
        height = int(self.canvas.winfo_height() or 560)
        padding = 20
        cell_size = min((width - 2 * padding) / self.cols, (height - 2 * padding) / self.rows)
        left = (width - cell_size * self.cols) / 2
        top = (height - cell_size * self.rows) / 2
        path_index = {coord: index for index, coord in enumerate(self.current_path)}

        for row in range(self.rows):
            for col in range(self.cols):
                x1 = left + col * cell_size
                y1 = top + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cell = (row, col)
                fill = "#fffdf7"
                outline = "#8a816e"
                if cell in self.blocked:
                    fill = "#3a3f45"
                    outline = "#1f2327"
                elif cell == self.start:
                    fill = "#8ecf9d"
                elif cell == self.end:
                    fill = "#f39a8f"
                elif cell in path_index:
                    fill = "#f5d46f"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=2)
                if cell in self.blocked:
                    self.canvas.create_line(x1 + 8, y1 + 8, x2 - 8, y2 - 8, fill="white", width=2)
                    self.canvas.create_line(x2 - 8, y1 + 8, x1 + 8, y2 - 8, fill="white", width=2)
                if cell in path_index:
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=str(path_index[cell]),
                        font=("Segoe UI", max(int(cell_size * 0.18), 9), "bold"),
                        fill="#3d2f11",
                    )

        if len(self.current_path) >= 2:
            points: list[float] = []
            for row, col in self.current_path:
                x = left + col * cell_size + cell_size / 2
                y = top + row * cell_size + cell_size / 2
                points.extend((x, y))
            self.canvas.create_line(
                *points,
                fill="#2d6a4f",
                width=max(cell_size * 0.12, 3),
                smooth=True,
            )

    def _set_details(self, text: str) -> None:
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", text)
        self.details_text.configure(state="disabled")


class PuzzleTab(ttk.Frame):
    def __init__(self, master: ttk.Notebook, runner: BackgroundRunner) -> None:
        super().__init__(master, padding=12)
        self.runner = runner
        self.mode_var = tk.StringVar(value="play")
        self.shuffle_var = tk.IntVar(value=20)
        self.board: tuple[int, ...] = GOAL_STATE
        self.drag_index: int | None = None
        self.result_rows: dict[str, StoredResult] = {}
        self.algorithm_vars = {
            algorithm: tk.BooleanVar(value=True)
            for algorithm in PUZZLE_ALGORITHMS
        }
        self.solution_states: list[tuple[int, ...]] = []
        self.animation_job: str | None = None
        self.base_board_for_results: tuple[int, ...] = self.board
        self.solvable_var = tk.StringVar()

        self._build_layout()
        self._refresh_board()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(self, text="Параметры и управление", padding=12)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        ttk.Label(controls, text="Режим").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Играть", value="play", variable=self.mode_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Radiobutton(controls, text="Редактировать", value="edit", variable=self.mode_var).grid(
            row=2, column=0, sticky="w"
        )

        ttk.Label(controls, text="Случайное перемешивание").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Spinbox(controls, from_=5, to=80, textvariable=self.shuffle_var, width=10).grid(
            row=4, column=0, sticky="w", pady=(4, 8)
        )
        ttk.Button(controls, text="Случайная доска", command=self._randomize).grid(
            row=5, column=0, sticky="ew", pady=2
        )
        ttk.Button(controls, text="Сброс к цели", command=self._reset_goal).grid(
            row=6, column=0, sticky="ew", pady=2
        )

        ttk.Label(controls, textvariable=self.solvable_var, wraplength=250).grid(
            row=7, column=0, sticky="w", pady=(8, 12)
        )

        ttk.Label(controls, text="Алгоритмы для запуска").grid(row=8, column=0, sticky="w")
        algorithm_frame = ttk.Frame(controls)
        algorithm_frame.grid(row=9, column=0, sticky="ew", pady=(4, 10))
        for row_index, (algorithm, label) in enumerate(PUZZLE_ALGORITHMS.items()):
            ttk.Checkbutton(
                algorithm_frame,
                text=label,
                variable=self.algorithm_vars[algorithm],
            ).grid(row=row_index, column=0, sticky="w")

        ttk.Button(controls, text="Запустить выбранные алгоритмы", command=self._compare_all).grid(
            row=10, column=0, sticky="ew", pady=2
        )
        ttk.Button(controls, text="Показать решение", command=self._play_solution).grid(
            row=11, column=0, sticky="ew", pady=(12, 2)
        )
        ttk.Button(controls, text="Остановить анимацию", command=self._stop_animation).grid(
            row=12, column=0, sticky="ew", pady=2
        )

        board_panel = ttk.Frame(self)
        board_panel.grid(row=0, column=1, sticky="nsew")
        board_panel.columnconfigure(0, weight=1)
        board_panel.rowconfigure(0, weight=3)
        board_panel.rowconfigure(1, weight=2)

        self.canvas = tk.Canvas(board_panel, width=560, height=560, bg="#f5efe3", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        results_frame = ttk.LabelFrame(board_panel, text="Результаты", padding=10)
        results_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("algorithm", "found", "moves", "time", "nodes", "frontier", "memory"),
            show="headings",
            height=6,
        )
        headings = {
            "algorithm": "Алгоритм",
            "found": "Найдено",
            "moves": "Ходы",
            "time": "мс",
            "nodes": "Узлы",
            "frontier": "Фронтир",
            "memory": "Алг. память, KiB",
        }
        widths = {
            "algorithm": 180,
            "found": 80,
            "moves": 80,
            "time": 90,
            "nodes": 110,
            "frontier": 100,
            "memory": 125,
        }
        for key, title in headings.items():
            self.results_tree.heading(key, text=title)
            self.results_tree.column(key, width=widths[key], anchor="center", stretch=key == "algorithm")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self.details_text = tk.Text(results_frame, height=9, wrap="word")
        self.details_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.details_text.configure(state="disabled")

    def _randomize(self) -> None:
        self._stop_animation()
        self.board = generate_random_board(shuffles=self.shuffle_var.get())
        self._refresh_board()

    def _reset_goal(self) -> None:
        self._stop_animation()
        self.board = GOAL_STATE
        self._refresh_board()

    def _refresh_board(self) -> None:
        self._draw_board()
        status = "решаема" if is_solvable(self.board) else "не решаема"
        self.solvable_var.set(f"Текущая доска: {status}. Редактирование — перетаскивание, игра — ход соседней костяшкой.")

    def _selected_algorithms(self) -> list[str]:
        return [
            algorithm
            for algorithm, selected in self.algorithm_vars.items()
            if selected.get()
        ]

    def _compare_all(self) -> None:
        if not is_solvable(self.board):
            messagebox.showwarning("Нерешаемая доска", "Текущая конфигурация не имеет решения.")
            return
        board = self.board
        algorithms = self._selected_algorithms()
        if not algorithms:
            messagebox.showwarning("Алгоритмы не выбраны", "Отметьте хотя бы один алгоритм для запуска.")
            return
        self.base_board_for_results = board
        self.solution_states.clear()
        self._clear_results()

        def work() -> list[SearchResult[Any]]:
            results: list[SearchResult[Any]] = []
            for algorithm in algorithms:
                results.append(
                    measure_search_run(lambda algorithm=algorithm: solve_board(board, algorithm))
                )
            return results

        def done(results: list[SearchResult[Any]]) -> None:
            self._store_results(board, results)
            successful = _fastest_successful(results)
            if successful is not None:
                self.solution_states = self._result_to_states(board, successful)
            summary = "\n\n".join(
                _format_metrics(
                    result,
                    f"Число ходов: {len(_extract_moves(result.solution))}" if result.found else "",
                    PUZZLE_ALGORITHMS,
                )
                for result in results
            )
            self._set_details(summary)

        self.runner.start("Сравнение алгоритмов задачи 2", work, done)

    def _clear_results(self) -> None:
        self.result_rows.clear()
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._set_details("")

    def _store_results(
        self,
        board: tuple[int, ...],
        results: Iterable[SearchResult[Any]],
    ) -> None:
        self.result_rows.clear()
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for result in results:
            moves = _extract_moves(result.solution)
            row_id = self.results_tree.insert(
                "",
                "end",
                values=(
                    PUZZLE_ALGORITHMS.get(result.algorithm, result.algorithm),
                    "Да" if result.found else "Нет",
                    len(moves) if moves else "-",
                    f"{result.metrics.elapsed_ms:.2f}",
                    result.metrics.nodes_expanded,
                    result.metrics.max_frontier,
                    f"{result.metrics.peak_memory_kb:.1f}",
                ),
            )
            self.result_rows[row_id] = StoredResult(result=result, context=board)

    def _on_tree_select(self, _event: tk.Event[Any]) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        stored = self.result_rows.get(selection[0])
        if stored is None:
            return
        self.solution_states = self._result_to_states(stored.context, stored.result)
        moves = _extract_moves(stored.result.solution)
        detail = f"Число ходов: {len(moves)}" if moves else ""
        self._set_details(_format_metrics(stored.result, detail, PUZZLE_ALGORITHMS))

    def _result_to_states(
        self,
        board: tuple[int, ...],
        result: SearchResult[Any],
    ) -> list[tuple[int, ...]]:
        states = _extract_states(result.solution)
        if states:
            return states
        moves = _extract_moves(result.solution)
        if moves:
            return _states_from_moves(board, moves)
        return []

    def _play_solution(self) -> None:
        if not self.solution_states:
            return
        self._stop_animation()
        self.board = self.solution_states[0]
        self._draw_board()

        def step(index: int) -> None:
            if index >= len(self.solution_states):
                self.animation_job = None
                return
            self.board = self.solution_states[index]
            self._draw_board()
            self.animation_job = self.after(250, step, index + 1)

        step(0)

    def _stop_animation(self) -> None:
        if self.animation_job is not None:
            self.after_cancel(self.animation_job)
            self.animation_job = None

    def _on_press(self, event: tk.Event[Any]) -> None:
        self.drag_index = self._canvas_to_index(event.x, event.y)

    def _on_release(self, event: tk.Event[Any]) -> None:
        if self.drag_index is None:
            return
        target = self._canvas_to_index(event.x, event.y)
        if target is None:
            self.drag_index = None
            return
        self._stop_animation()
        if self.mode_var.get() == "edit":
            self.board = self._swap_indices(self.board, self.drag_index, target)
        else:
            blank = self.board.index(0)
            if target == blank and _is_adjacent(self.drag_index, blank):
                self.board = self._swap_indices(self.board, self.drag_index, blank)
            elif self.drag_index == target and _is_adjacent(self.drag_index, blank):
                self.board = self._swap_indices(self.board, self.drag_index, blank)
        self.drag_index = None
        self._refresh_board()

    def _swap_indices(
        self,
        state: tuple[int, ...],
        index_a: int,
        index_b: int,
    ) -> tuple[int, ...]:
        if index_a == index_b:
            return state
        board = list(state)
        board[index_a], board[index_b] = board[index_b], board[index_a]
        return tuple(board)

    def _canvas_to_index(self, x: int, y: int) -> int | None:
        width = int(self.canvas.winfo_width() or 560)
        height = int(self.canvas.winfo_height() or 560)
        padding = 20
        cell_size = min((width - 2 * padding) / 4, (height - 2 * padding) / 4)
        left = (width - cell_size * 4) / 2
        top = (height - cell_size * 4) / 2
        col = int((x - left) // cell_size)
        row = int((y - top) // cell_size)
        if 0 <= row < 4 and 0 <= col < 4:
            return row * 4 + col
        return None

    def _draw_board(self) -> None:
        self.canvas.delete("all")
        width = int(self.canvas.winfo_width() or 560)
        height = int(self.canvas.winfo_height() or 560)
        padding = 20
        cell_size = min((width - 2 * padding) / 4, (height - 2 * padding) / 4)
        left = (width - cell_size * 4) / 2
        top = (height - cell_size * 4) / 2

        for index, value in enumerate(self.board):
            row, col = divmod(index, 4)
            x1 = left + col * cell_size
            y1 = top + row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#fffdf7", outline="#8a816e", width=2)
            if value == 0:
                self.canvas.create_rectangle(
                    x1 + 8,
                    y1 + 8,
                    x2 - 8,
                    y2 - 8,
                    outline="#c1b8a3",
                    dash=(6, 3),
                )
                continue
            fill = "#d56f3e" if index != GOAL_STATE.index(value) else "#3f7d58"
            self.canvas.create_rectangle(
                x1 + 6,
                y1 + 6,
                x2 - 6,
                y2 - 6,
                fill=fill,
                outline="",
            )
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=str(value),
                font=("Segoe UI", max(int(cell_size * 0.3), 14), "bold"),
                fill="white",
            )

    def _set_details(self, text: str) -> None:
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", text)
        self.details_text.configure(state="disabled")


class Lab8App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Lab 8 - Полный перебор, эвристики и отсечение")
        self.root.geometry("1360x920")
        self.root.minsize(1180, 820)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.status_var = tk.StringVar(value="Готово")
        self.runner = BackgroundRunner(self.root, self.status_var)

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(container)
        notebook.grid(row=0, column=0, sticky="nsew")
        notebook.add(GridPathTab(notebook, self.runner), text="Задача 1: путь по сетке")
        notebook.add(PuzzleTab(notebook, self.runner), text="Задача 2: пятнашки")

        status = ttk.Label(
            container,
            textvariable=self.status_var,
            anchor="w",
            padding=(4, 8, 4, 0),
        )
        status.grid(row=1, column=0, sticky="ew")

    def run(self) -> None:
        self.root.mainloop()
