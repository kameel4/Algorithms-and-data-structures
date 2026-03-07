import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

try:
    from .ga import run_ga, run_ga_bitwise
    from .history_files import save_run_histories
    from .pso import run_pso
except ImportError:
    from ga import run_ga, run_ga_bitwise
    from history_files import save_run_histories
    from pso import run_pso


def build_background(objective_function, bounds=(-515, 515), grid_n=300):
    low, high = bounds
    xs = np.linspace(low, high, grid_n)
    ys = np.linspace(low, high, grid_n)
    x_grid, y_grid = np.meshgrid(xs, ys)
    z_grid = objective_function(x_grid, y_grid)
    return x_grid, y_grid, z_grid


def visualize_comparison(
    ga_result,
    pso_result,
    objective_function,
    bounds=(-515, 515),
    interval=120,
):
    _, _, z_grid = build_background(
        objective_function=objective_function,
        bounds=bounds,
        grid_n=320,
    )
    low, high = bounds

    ga_hist_pos = ga_result["history_positions"]
    pso_hist_pos = pso_result["history_positions"]
    ga_hist_best = ga_result["history_best"]
    pso_hist_best = pso_result["history_best"]
    ga_hist_best_point = ga_result["history_best_point"]
    pso_hist_best_point = pso_result["history_best_point"]
    ga_hist_mean = ga_result["history_mean"]
    pso_hist_mean = pso_result["history_mean"]

    n_frames = max(len(ga_hist_pos), len(pso_hist_pos))

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

    ax_ga = fig.add_subplot(gs[0, 0])
    ax_pso = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[1, :])

    for ax, title in [(ax_ga, ga_result["name"]), (ax_pso, pso_result["name"])]:
        ax.imshow(
            z_grid,
            extent=(low, high, low, high),
            origin="lower",
            aspect="auto",
            alpha=0.95,
        )
        levels = np.linspace(np.nanmin(z_grid), np.nanpercentile(z_grid, 15), 8)
        ax.contour(
            np.linspace(low, high, z_grid.shape[1]),
            np.linspace(low, high, z_grid.shape[0]),
            z_grid,
            levels=levels,
            linewidths=0.4,
            alpha=0.6,
        )
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

    true_min = np.array([515, 404.2319])
    for ax in [ax_ga, ax_pso]:
        ax.scatter(true_min[0], true_min[1], marker="*", s=180, label="Reference min")
        ax.legend(loc="upper left", fontsize=9)

    ga_scatter = ax_ga.scatter([], [], s=18)
    pso_scatter = ax_pso.scatter([], [], s=18)
    ga_best_scatter = ax_ga.scatter([], [], s=80, marker="x")
    pso_best_scatter = ax_pso.scatter([], [], s=80, marker="x")

    ax_curve.set_title("Mean fitness evolution")
    ax_curve.set_xlabel("Iteration / generation")
    ax_curve.set_ylabel("f(x, y)")
    ax_curve.grid(True, alpha=0.3)

    line_ga, = ax_curve.plot([], [], label="GA mean")
    line_pso, = ax_curve.plot([], [], label="PSO mean")
    ax_curve.legend()

    all_mean = np.array(ga_hist_mean + pso_hist_mean, dtype=float)
    y_min = np.min(all_mean) - 20
    y_max = np.max(all_mean) + 20
    ax_curve.set_xlim(0, n_frames - 1)
    ax_curve.set_ylim(y_min, y_max)

    txt = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=11)

    def init():
        ga_scatter.set_offsets(np.empty((0, 2)))
        pso_scatter.set_offsets(np.empty((0, 2)))
        ga_best_scatter.set_offsets(np.empty((0, 2)))
        pso_best_scatter.set_offsets(np.empty((0, 2)))
        line_ga.set_data([], [])
        line_pso.set_data([], [])
        txt.set_text("")
        return ga_scatter, pso_scatter, ga_best_scatter, pso_best_scatter, line_ga, line_pso, txt

    def update(frame):
        ga_i = min(frame, len(ga_hist_pos) - 1)
        pso_i = min(frame, len(pso_hist_pos) - 1)

        ga_pos = ga_hist_pos[ga_i]
        pso_pos = pso_hist_pos[pso_i]

        ga_scatter.set_offsets(ga_pos)
        pso_scatter.set_offsets(pso_pos)
        ga_best_scatter.set_offsets(ga_hist_best_point[ga_i][None, :])
        pso_best_scatter.set_offsets(pso_hist_best_point[pso_i][None, :])

        line_ga.set_data(np.arange(ga_i + 1), ga_hist_mean[:ga_i + 1])
        line_pso.set_data(np.arange(pso_i + 1), pso_hist_mean[:pso_i + 1])

        txt.set_text(
            f"Frame {frame + 1}/{n_frames} | "
            f"GA best = {ga_hist_best[ga_i]:.4f} at ({ga_hist_best_point[ga_i][0]:.2f}, {ga_hist_best_point[ga_i][1]:.2f}) | "
            f"PSO best = {pso_hist_best[pso_i]:.4f} at ({pso_hist_best_point[pso_i][0]:.2f}, {pso_hist_best_point[pso_i][1]:.2f})"
        )

        return ga_scatter, pso_scatter, ga_best_scatter, pso_best_scatter, line_ga, line_pso, txt

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=interval,
        blit=False,
        repeat=False,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()
    return anim


def _parse_int(value, field_name, min_value=None, max_value=None):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name}: integer expected.") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name}: must be >= {min_value}.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{field_name}: must be <= {max_value}.")
    return parsed


def _parse_float(value, field_name, min_value=None, max_value=None):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name}: float expected.") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name}: must be >= {min_value}.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{field_name}: must be <= {max_value}.")
    return parsed


def _add_labeled_entry(frame, row, label, default_value, width=12):
    ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    var = tk.StringVar(value=str(default_value))
    entry = ttk.Entry(frame, textvariable=var, width=width)
    entry.grid(row=row, column=1, sticky="w", pady=2)
    return var, entry


def run_interface(
    fitness,
    objective_function,
    bounds=(-515, 515),
    interval=12,
):
    default_low, default_high = bounds
    last_result = {"value": None}

    root = tk.Tk()
    root.title("Lab5 optimization controls")
    root.geometry("420x820")

    panel = ttk.Frame(root, padding=12)
    panel.pack(anchor="nw", fill="both", expand=True)

    row = 0
    ttk.Label(panel, text="Common", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    low_var, _ = _add_labeled_entry(panel, row, "Bounds low", default_low)
    row += 1
    high_var, _ = _add_labeled_entry(panel, row, "Bounds high", default_high)
    row += 1
    interval_var, _ = _add_labeled_entry(panel, row, "Anim interval (ms)", interval)
    row += 1

    ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    ttk.Label(panel, text="GA", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    ga_mode_var = tk.StringVar(value="arithmetic")
    ttk.Label(panel, text="GA mode").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    mode_frame = ttk.Frame(panel)
    mode_frame.grid(row=row, column=1, sticky="w", pady=2)
    ttk.Radiobutton(mode_frame, text="Arithmetic", value="arithmetic", variable=ga_mode_var).pack(anchor="w")
    ttk.Radiobutton(mode_frame, text="Bitwise (Gray)", value="bitwise", variable=ga_mode_var).pack(anchor="w")
    row += 1

    ga_pop_size_var, _ = _add_labeled_entry(panel, row, "GA pop_size", 512)
    row += 1
    ga_generations_var, _ = _add_labeled_entry(panel, row, "GA generations", 100)
    row += 1
    ga_crossover_var, _ = _add_labeled_entry(panel, row, "GA crossover_prob", 0.9)
    row += 1
    ga_mutation_var, _ = _add_labeled_entry(panel, row, "GA mutation_prob", 0.30)
    row += 1
    ga_elite_var, _ = _add_labeled_entry(panel, row, "GA elite_size", 2)
    row += 1
    ga_tournament_var, _ = _add_labeled_entry(panel, row, "GA tournament_k", 5)
    row += 1
    ga_sigma_var, ga_sigma_entry = _add_labeled_entry(panel, row, "GA sigma0", 35.0)
    row += 1
    ga_seed_var, _ = _add_labeled_entry(panel, row, "GA seed", 123)
    row += 1
    ga_bit_width_var, ga_bit_width_entry = _add_labeled_entry(panel, row, "GA bit width (B)", 16)
    row += 1

    ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
    row += 1

    ttk.Label(panel, text="PSO", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    pso_mode_var = tk.StringVar(value="constriction")
    ttk.Label(panel, text="PSO mode").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    pso_mode_frame = ttk.Frame(panel)
    pso_mode_frame.grid(row=row, column=1, sticky="w", pady=2)
    ttk.Radiobutton(
        pso_mode_frame,
        text="With constriction",
        value="constriction",
        variable=pso_mode_var,
    ).pack(anchor="w")
    ttk.Radiobutton(
        pso_mode_frame,
        text="Standard (no constriction)",
        value="standard",
        variable=pso_mode_var,
    ).pack(anchor="w")
    row += 1

    pso_swarm_size_var, _ = _add_labeled_entry(panel, row, "PSO swarm_size", 512)
    row += 1
    pso_iterations_var, _ = _add_labeled_entry(panel, row, "PSO iterations", 100)
    row += 1
    pso_c1_var, _ = _add_labeled_entry(panel, row, "PSO c1", 2.05)
    row += 1
    pso_c2_var, _ = _add_labeled_entry(panel, row, "PSO c2", 2.3)
    row += 1
    pso_vmax_var, _ = _add_labeled_entry(panel, row, "PSO vmax_ratio", 0.26)
    row += 1
    pso_seed_var, _ = _add_labeled_entry(panel, row, "PSO seed", 123)
    row += 1

    status_var = tk.StringVar(value="Ready.")
    status_label = ttk.Label(panel, textvariable=status_var, foreground="#2f4f4f")
    status_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
    row += 1

    def update_ga_mode_widgets(*_):
        if ga_mode_var.get() == "bitwise":
            ga_bit_width_entry.configure(state="normal")
            ga_sigma_entry.configure(state="disabled")
        else:
            ga_bit_width_entry.configure(state="disabled")
            ga_sigma_entry.configure(state="normal")

    ga_mode_var.trace_add("write", update_ga_mode_widgets)
    update_ga_mode_widgets()

    def on_run_click():
        run_button.configure(state="disabled")
        root.update_idletasks()
        status_var.set("Running simulation...")

        try:
            low = _parse_float(low_var.get(), "Bounds low")
            high = _parse_float(high_var.get(), "Bounds high")
            if low >= high:
                raise ValueError("Bounds low must be < bounds high.")
            bounds_value = (low, high)

            interval_value = _parse_int(interval_var.get(), "Anim interval (ms)", min_value=1)

            ga_pop_size = _parse_int(ga_pop_size_var.get(), "GA pop_size", min_value=2)
            ga_generations = _parse_int(ga_generations_var.get(), "GA generations", min_value=1)
            ga_crossover_prob = _parse_float(ga_crossover_var.get(), "GA crossover_prob", min_value=0.0, max_value=1.0)
            ga_mutation_prob = _parse_float(ga_mutation_var.get(), "GA mutation_prob", min_value=0.0, max_value=1.0)
            ga_elite_size = _parse_int(ga_elite_var.get(), "GA elite_size", min_value=0)
            ga_tournament_k = _parse_int(ga_tournament_var.get(), "GA tournament_k", min_value=1)
            ga_seed = _parse_int(ga_seed_var.get(), "GA seed")

            if ga_elite_size > ga_pop_size:
                raise ValueError("GA elite_size must be <= GA pop_size.")

            ga_mode = ga_mode_var.get()
            if ga_mode == "bitwise":
                ga_bit_width = _parse_int(ga_bit_width_var.get(), "GA bit width (B)", min_value=2, max_value=32)
                ga_result = run_ga_bitwise(
                    fitness=fitness,
                    bounds=bounds_value,
                    pop_size=ga_pop_size,
                    generations=ga_generations,
                    crossover_prob=ga_crossover_prob,
                    mutation_prob=ga_mutation_prob,
                    elite_size=ga_elite_size,
                    tournament_k=ga_tournament_k,
                    bit_width=ga_bit_width,
                    seed=ga_seed,
                )
            else:
                ga_sigma0 = _parse_float(ga_sigma_var.get(), "GA sigma0", min_value=0.0)
                ga_result = run_ga(
                    fitness=fitness,
                    bounds=bounds_value,
                    pop_size=ga_pop_size,
                    generations=ga_generations,
                    crossover_prob=ga_crossover_prob,
                    mutation_prob=ga_mutation_prob,
                    elite_size=ga_elite_size,
                    tournament_k=ga_tournament_k,
                    sigma0=ga_sigma0,
                    seed=ga_seed,
                )

            pso_swarm_size = _parse_int(pso_swarm_size_var.get(), "PSO swarm_size", min_value=2)
            pso_iterations = _parse_int(pso_iterations_var.get(), "PSO iterations", min_value=1)
            pso_c1 = _parse_float(pso_c1_var.get(), "PSO c1")
            pso_c2 = _parse_float(pso_c2_var.get(), "PSO c2")
            pso_vmax_ratio = _parse_float(pso_vmax_var.get(), "PSO vmax_ratio", min_value=0.0)
            pso_seed = _parse_int(pso_seed_var.get(), "PSO seed")
            pso_mode = pso_mode_var.get()

            pso_result = run_pso(
                fitness=fitness,
                bounds=bounds_value,
                swarm_size=pso_swarm_size,
                iterations=pso_iterations,
                c1=pso_c1,
                c2=pso_c2,
                vmax_ratio=pso_vmax_ratio,
                use_constriction=(pso_mode == "constriction"),
                seed=pso_seed,
            )

            print("=== Results ===")
            print(f"GA : f = {ga_result['best_value']:.6f}, point = {ga_result['best_point']}")
            print(f"PSO: f = {pso_result['best_value']:.6f}, point = {pso_result['best_point']}")

            saved_paths = save_run_histories(
                [ga_result, pso_result],
                bounds=bounds_value,
            )
            for saved_path in saved_paths:
                print(f"History saved: {saved_path}")

            visualize_comparison(
                ga_result=ga_result,
                pso_result=pso_result,
                objective_function=objective_function,
                bounds=bounds_value,
                interval=interval_value,
            )

            last_result["value"] = (ga_result, pso_result)
            status_var.set("Finished. History files saved.")
        except Exception as exc:  # noqa: BLE001
            status_var.set("Failed. Check inputs.")
            messagebox.showerror("Run error", str(exc))
        finally:
            run_button.configure(state="normal")

    run_button = ttk.Button(panel, text="Run simulation", command=on_run_click)
    run_button.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))

    root.mainloop()
    return last_result["value"]
