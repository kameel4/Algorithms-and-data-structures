import numpy as np

try:
    from .initialization import make_initial_points
except ImportError:
    from initialization import make_initial_points


def _clip_velocity_norm(velocity, vmax):
    if vmax <= 0:
        return np.zeros_like(velocity)

    norms = np.linalg.norm(velocity, axis=1, keepdims=True)
    mask = norms[:, 0] > vmax
    if np.any(mask):
        velocity = velocity.copy()
        velocity[mask] *= vmax / norms[mask]
    return velocity


def run_pso(
    fitness,
    bounds=(-515, 515),
    swarm_size=80,
    iterations=120,
    c1=2.05,
    c2=2.05,
    vmax_ratio=0.2,
    use_constriction=True,
    seed=7,
    init_mode="grid",
):
    rng = np.random.default_rng(seed)
    low, high = bounds
    span = high - low

    x = make_initial_points(swarm_size, bounds=bounds, mode=init_mode, rng=rng)
    v = rng.uniform(-span * 0.05, span * 0.05, size=(swarm_size, 2))

    pbest = x.copy()
    pbest_val = fitness(pbest)
    g_idx = np.argmin(pbest_val)
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    phi = c1 + c2
    if use_constriction:
        if phi <= 4.0:
            raise ValueError("For constriction mode, c1 + c2 must be > 4.")
        chi = 2.0 / abs(2.0 - phi - np.sqrt(phi**2 - 4.0 * phi))
    else:
        chi = 1.0

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
        v = chi * (v + cognitive + social)
        v = _clip_velocity_norm(v, vmax)
        x = x + v

        for d in range(2):
            low_mask = x[:, d] < low
            high_mask = x[:, d] > high

            x[low_mask, d] = low
            x[high_mask, d] = high
            v[low_mask | high_mask, d] *= -0.5

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
        "name": "PSO (with constriction)" if use_constriction else "PSO (standard, no constriction)",
    }
    return result
