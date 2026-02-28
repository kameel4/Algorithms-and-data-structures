import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------
#  Целевая функция (Eggholder)
# ---------------------------
def f_xy(x, y):
    """
    x, y can be scalars or numpy arrays of the same shape.
    """
    return -(y + 47.0) * np.sin(np.sqrt(np.abs(x / 2.0 + (y + 47.0)))) \
           - x * np.sin(np.sqrt(np.abs(x - (y + 47.0))))

def fitness(points):
    """
    points: array shape (N, 2)
    returns: array shape (N,)
    """
    return f_xy(points[:, 0], points[:, 1])

# ---------------------------
#  Генетический алгоритм (GA)
# ---------------------------
def tournament_selection(pop, fit, k=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, len(pop), size=k)
    best_idx = idx[np.argmin(fit[idx])]
    return pop[best_idx].copy()

def arithmetic_crossover(parent1, parent2, rng=None):
    """
    Реализованный кроссинговер (арифметический, real-coded).
    Возвращает двух потомков.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = rng.random(2)  # отдельный коэффициент для x и y
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

def mutate(ind, sigma, bounds, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mutated = ind + rng.normal(0.0, sigma, size=2)
    mutated = np.clip(mutated, bounds[0], bounds[1])
    return mutated

def run_ga(
    bounds=(-515, 515),
    pop_size=80,
    generations=120,
    crossover_prob=0.9,
    mutation_prob=0.25,
    elite_size=2,
    tournament_k=3,
    sigma0=30.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    low, high = bounds

    pop = rng.uniform(low, high, size=(pop_size, 2))
    fit = fitness(pop)

    history_positions = [pop.copy()]
    history_best = [fit.min()]
    history_best_point = [pop[np.argmin(fit)].copy()]
    history_mean = [fit.mean()]

    for gen in range(generations):
        # Элитизм
        elite_idx = np.argsort(fit)[:elite_size]
        new_pop = [pop[i].copy() for i in elite_idx]

        # Постепенно уменьшаем силу мутации
        sigma = sigma0 * (1.0 - gen / max(1, generations - 1))
        sigma = max(sigma, 1.0)

        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fit, k=tournament_k, rng=rng)
            p2 = tournament_selection(pop, fit, k=tournament_k, rng=rng)

            if rng.random() < crossover_prob:
                c1, c2 = arithmetic_crossover(p1, p2, rng=rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if rng.random() < mutation_prob:
                c1 = mutate(c1, sigma=sigma, bounds=bounds, rng=rng)
            if rng.random() < mutation_prob:
                c2 = mutate(c2, sigma=sigma, bounds=bounds, rng=rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop, dtype=float)
        pop = np.clip(pop, low, high)
        fit = fitness(pop)

        history_positions.append(pop.copy())
        history_best.append(fit.min())
        history_best_point.append(pop[np.argmin(fit)].copy())
        history_mean.append(fit.mean())

    best_idx = np.argmin(fit)
    result = {
        "best_point": pop[best_idx].copy(),
        "best_value": float(fit[best_idx]),
        "history_positions": history_positions,
        "history_best": history_best,
        "history_best_point": history_best_point,
        "history_mean": history_mean,
        "name": "GA (с кроссинговером)"
    }
    return result

# ---------------------------
#  Роевой алгоритм (PSO)
# ---------------------------
def run_pso(
    bounds=(-515, 515),
    swarm_size=80,
    iterations=120,
    c1=2.05,
    c2=2.05,
    vmax_ratio=0.2,
    seed=7,
):
    """
    PSO с коэффициентом сжатия (Clerc constriction factor).
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    span = high - low

    # Инициализация
    x = rng.uniform(low, high, size=(swarm_size, 2))
    v = rng.uniform(-span * 0.05, span * 0.05, size=(swarm_size, 2))

    pbest = x.copy()
    pbest_val = fitness(pbest)
    g_idx = np.argmin(pbest_val)
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    # Коэффициент сжатия chi
    phi = c1 + c2
    if phi <= 4.0:
        raise ValueError("Для коэффициента сжатия нужно c1 + c2 > 4.")
    chi = 2.0 / abs(2.0 - phi - np.sqrt(phi**2 - 4.0 * phi))

    vmax = vmax_ratio * span

    history_positions = [x.copy()]
    history_best = [gbest_val]
    history_best_point = [gbest.copy()]
    history_mean = [pbest_val.mean()]

    for _ in range(iterations):
        r1 = rng.random(size=(swarm_size, 2))
        r2 = rng.random(size=(swarm_size, 2))

        cognitive = c1 * r1 * (pbest - x)
        social = c2 * r2 * (gbest - x)

        # PSO с коэффициентом сжатия
        v = chi * (v + cognitive + social)

        # Ограничение скорости
        v = np.clip(v, -vmax, vmax)

        x = x + v

        # Отражение от границ (простая обработка)
        for d in range(2):
            low_mask = x[:, d] < low
            high_mask = x[:, d] > high

            x[low_mask, d] = low
            x[high_mask, d] = high

            v[low_mask | high_mask, d] *= -0.5  # затухающее отражение

        vals = fitness(x)

        improve_mask = vals < pbest_val
        pbest[improve_mask] = x[improve_mask]
        pbest_val[improve_mask] = vals[improve_mask]

        g_idx = np.argmin(pbest_val)
        if pbest_val[g_idx] < gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest = pbest[g_idx].copy()

        history_positions.append(x.copy())
        history_best.append(gbest_val)
        history_best_point.append(gbest.copy())
        history_mean.append(vals.mean())

    result = {
        "best_point": gbest,
        "best_value": gbest_val,
        "history_positions": history_positions,
        "history_best": history_best,
        "history_best_point": history_best_point,
        "history_mean": history_mean,
        "name": "PSO (с коэффициентом сжатия)"
    }
    return result

# ---------------------------
#  Визуализация
# ---------------------------
def build_background(bounds=(-515, 515), grid_n=300):
    low, high = bounds
    xs = np.linspace(low, high, grid_n)
    ys = np.linspace(low, high, grid_n)
    X, Y = np.meshgrid(xs, ys)
    Z = f_xy(X, Y)
    return X, Y, Z

def visualize_comparison(ga_result, pso_result, bounds=(-515, 515), interval=120):
    # Фон функции (контур/тепловая карта)
    _, _, Z = build_background(bounds=bounds, grid_n=320)
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

    # Одинаковый фон на обеих панелях
    for ax, title in [(ax_ga, ga_result["name"]), (ax_pso, pso_result["name"])]:
        ax.imshow(
            Z,
            extent=(low, high, low, high),
            origin="lower",
            aspect="auto",
            alpha=0.95,
        )
        # Несколько изолиний для формы
        levels = np.linspace(np.nanmin(Z), np.nanpercentile(Z, 15), 8)
        ax.contour(
            np.linspace(low, high, Z.shape[1]),
            np.linspace(low, high, Z.shape[0]),
            Z,
            levels=levels,
            linewidths=0.4,
            alpha=0.6
        )
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

    # Известный глобальный минимум Eggholder (для ориентира)
    true_min = np.array([515, 404.2319])
    for ax in [ax_ga, ax_pso]:
        ax.scatter(true_min[0], true_min[1], marker="*", s=180, label="Ориентир min")
        ax.legend(loc="upper left", fontsize=9)

    # Точки особей / частиц
    ga_scatter = ax_ga.scatter([], [], s=18)
    pso_scatter = ax_pso.scatter([], [], s=18)

    # Лучшая точка
    ga_best_scatter = ax_ga.scatter([], [], s=80, marker="x")
    pso_best_scatter = ax_pso.scatter([], [], s=80, marker="x")

    # График среднего значения
    ax_curve.set_title("Эволюция среднего значения функции")
    ax_curve.set_xlabel("Итерация / поколение")
    ax_curve.set_ylabel("f(x, y)")
    ax_curve.grid(True, alpha=0.3)

    line_ga, = ax_curve.plot([], [], label="GA mean")
    line_pso, = ax_curve.plot([], [], label="PSO mean")
    ax_curve.legend()

    # Установим разумные границы по Y заранее
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
            f"Кадр {frame + 1}/{n_frames} | "
            f"GA best = {ga_hist_best[ga_i]:.4f} в ({ga_hist_best_point[ga_i][0]:.2f}, {ga_hist_best_point[ga_i][1]:.2f}) | "
            f"PSO best = {pso_hist_best[pso_i]:.4f} в ({pso_hist_best_point[pso_i][0]:.2f}, {pso_hist_best_point[pso_i][1]:.2f})"
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
    return anim  # чтобы объект не был удалён сборщиком мусора

def main():
    bounds = (-515, 515)

    ga_result = run_ga(
        bounds=bounds,
        pop_size=512,
        generations=100,
        crossover_prob=0.9,
        mutation_prob=0.30,
        elite_size=2,
        tournament_k=6,
        sigma0=35.0,
        seed=123,
    )

    pso_result = run_pso(
        bounds=bounds,
        swarm_size=512,
        iterations=100,
        c1=2.05,
        c2=2.3,  # c1 + c2 = 4.1 -> chi ~ 0.7298
        vmax_ratio=0.26,
        seed=123,
    )

    print("=== Результаты ===")
    print(f"GA : f = {ga_result['best_value']:.6f}, point = {ga_result['best_point']}")
    print(f"PSO: f = {pso_result['best_value']:.6f}, point = {pso_result['best_point']}")
    print("Ожидаемый глобальный минимум Eggholder на [-512,512]^2:")
    print("x ≈ 512, y ≈ 404.2319, f ≈ -959.6407")

    visualize_comparison(ga_result, pso_result, bounds=bounds, interval=100)

if __name__ == "__main__":
    main()
