import numpy as np

try:
    from .initialization import make_initial_points
except ImportError:
    from initialization import make_initial_points

_DEFAULT_BOUNDS = (-512, 512)


def _clip_velocity_norm(velocity, vmax):
    if vmax <= 0:
        return np.zeros_like(velocity)

    norms = np.linalg.norm(velocity, axis=1, keepdims=True)
    mask = norms[:, 0] > vmax
    if np.any(mask):
        velocity = velocity.copy()
        velocity[mask] *= vmax / norms[mask]
    return velocity


def _schedule(start, end, progress):
    return start + (end - start) * progress


def _refine_best_point(point, value, fitness, bounds, step_radius):
    if step_radius <= 0:
        return np.asarray(point, dtype=float).copy(), float(value)

    best_point = np.asarray(point, dtype=float).copy()
    best_value = float(value)
    directions = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=float,
    )

    step = float(step_radius)
    low, high = bounds
    while step >= 1e-4:
        candidates = np.clip(best_point[None, :] + step * directions, low, high)
        candidate_values = fitness(candidates)
        candidate_idx = int(np.argmin(candidate_values))
        candidate_value = float(candidate_values[candidate_idx])
        if candidate_value < best_value:
            best_value = candidate_value
            best_point = candidates[candidate_idx].copy()
        else:
            step *= 0.5

    return best_point, best_value


def run_pso(
    fitness,
    bounds=_DEFAULT_BOUNDS,
    swarm_size=80,
    iterations=120,
    c1=2.05,
    c2=2.05,
    vmax_ratio=0.2,
    use_constriction=True,
    seed=7,
    init_mode="grid",
    inertia_start=0.92,
    inertia_end=0.25,
    adaptive_coefficients=True,
    vmax_end_scale=0.01,
    repair_outliers=True,
    repair_start=0.82,
    repair_stagnation=18,
    repair_radius_ratio=0.02,
    repair_sigma_ratio=0.008,
    local_refine=True,
    local_refine_radius_ratio=0.005,
    reseed_particles=True,
    reseed_start=0.25,
    reseed_end=0.72,
    reseed_stagnation=28,
    reseed_fraction=0.06,
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
    stagnation_steps = np.zeros(swarm_size, dtype=int)
    gbest_stagnation = 0

    history_positions = [x.copy()]
    history_best = [gbest_val]
    history_best_point = [gbest.copy()]
    history_mean = [pbest_val.mean()]

    for step in range(iterations):
        progress = step / max(1, iterations - 1)
        inertia = _schedule(inertia_start, inertia_end, progress)
        c1_now = c1 * (0.1 + 0.9 * ((1.0 - progress) ** 1.3)) if adaptive_coefficients else c1
        c2_now = c2 * (0.75 + 0.25 * progress) if adaptive_coefficients else c2
        vmax_now = vmax * (vmax_end_scale + (1.0 - vmax_end_scale) * ((1.0 - progress) ** 1.5))

        r1 = rng.random(size=(swarm_size, 2))
        r2 = rng.random(size=(swarm_size, 2))

        cognitive = c1_now * r1 * (pbest - x)
        social = c2_now * r2 * (gbest - x)
        v = chi * (inertia * v + cognitive + social)
        v = _clip_velocity_norm(v, vmax_now)
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
        stagnation_steps[improve_mask] = 0
        stagnation_steps[~improve_mask] += 1

        repair_mask = None
        if repair_outliers and progress >= repair_start:
            distance_to_gbest = np.linalg.norm(x - gbest, axis=1)
            allowed_radius = max(5.0, span * repair_radius_ratio * (1.0 - progress + 0.15))
            repair_mask = (distance_to_gbest > allowed_radius) & (stagnation_steps >= repair_stagnation)

        if repair_mask is not None and np.any(repair_mask):
            sigma = max(0.1, span * repair_sigma_ratio * ((1.0 - progress) ** 2))
            repaired = gbest + rng.normal(0.0, sigma, size=(int(np.sum(repair_mask)), 2))
            repaired = np.clip(repaired, low, high)

            x[repair_mask] = repaired
            v[repair_mask] = 0.0
            repaired_vals = fitness(repaired)
            vals[repair_mask] = repaired_vals
            pbest[repair_mask] = repaired
            pbest_val[repair_mask] = repaired_vals
            stagnation_steps[repair_mask] = 0

        g_idx = np.argmin(pbest_val)
        if pbest_val[g_idx] < gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest = pbest[g_idx].copy()
            gbest_stagnation = 0
        else:
            gbest_stagnation += 1

        if local_refine:
            refine_radius = span * local_refine_radius_ratio * ((1.0 - progress) ** 1.5)
            refined_point, refined_value = _refine_best_point(
                gbest,
                gbest_val,
                fitness=fitness,
                bounds=bounds,
                step_radius=refine_radius,
            )
            if refined_value < gbest_val:
                gbest = refined_point
                gbest_val = refined_value
                gbest_stagnation = 0

        if (
            reseed_particles
            and reseed_start <= progress < reseed_end
            and gbest_stagnation >= reseed_stagnation
        ):
            reseed_count = max(2, int(np.ceil(swarm_size * reseed_fraction)))
            ranking = np.lexsort((-stagnation_steps, pbest_val))
            reseed_idx = ranking[-reseed_count:]
            reseeded_points = make_initial_points(reseed_count, bounds=bounds, mode="random", rng=rng)
            reseeded_velocities = rng.uniform(-span * 0.05, span * 0.05, size=(reseed_count, 2))
            reseeded_values = fitness(reseeded_points)

            x[reseed_idx] = reseeded_points
            v[reseed_idx] = reseeded_velocities
            pbest[reseed_idx] = reseeded_points
            pbest_val[reseed_idx] = reseeded_values
            vals[reseed_idx] = reseeded_values
            stagnation_steps[reseed_idx] = 0
            gbest_stagnation = 0

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
        "name": "PSO (adaptive constriction)" if use_constriction else "PSO (adaptive standard)",
    }
    return result
