import numpy as np

try:
    from .initialization import make_initial_points
except ImportError:
    from initialization import make_initial_points

_DEFAULT_BIT_WIDTH = 16
_DEFAULT_BOUNDS = (-512, 512)


def _bit_max(bit_width):
    return (1 << bit_width) - 1


def _resolve_uint_dtype(bit_width):
    if bit_width <= 16:
        return np.uint16
    if bit_width <= 32:
        return np.uint32
    raise ValueError("bit_width must be in [2, 32].")


def tournament_selection(pop, fit, k=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, len(pop), size=k)
    best_idx = idx[np.argmin(fit[idx])]
    return pop[best_idx].copy()


def arithmetic_crossover(parent1, parent2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    alpha = rng.random(2)
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def mutate(ind, sigma, bounds, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mutated = ind + rng.normal(0.0, sigma, size=2)
    mutated = np.clip(mutated, bounds[0], bounds[1])
    return mutated


def binary_to_gray(binary_values):
    values = np.asarray(binary_values)
    if values.dtype.kind != "u":
        values = values.astype(np.uint64)
    return values ^ (values >> 1)


def gray_to_binary(gray_values, bit_width=_DEFAULT_BIT_WIDTH):
    values = np.asarray(gray_values)
    if values.dtype.kind != "u":
        values = values.astype(_resolve_uint_dtype(bit_width))

    binary_values = values.copy()
    shift = 1
    while shift < bit_width:
        binary_values ^= (binary_values >> shift)
        shift <<= 1
    return binary_values


def binary_to_gray_uint16(binary_values):
    values = np.asarray(binary_values, dtype=np.uint16)
    return binary_to_gray(values).astype(np.uint16, copy=False)


def gray_to_binary_uint16(gray_values):
    values = np.asarray(gray_values, dtype=np.uint16)
    return gray_to_binary(values, bit_width=16).astype(np.uint16, copy=False)


def encode_points_to_gray(points, bounds=_DEFAULT_BOUNDS, bit_width=_DEFAULT_BIT_WIDTH):
    if bit_width < 2 or bit_width > 32:
        raise ValueError("bit_width must be in [2, 32].")

    low, high = bounds
    dtype = _resolve_uint_dtype(bit_width)
    bit_max = _bit_max(bit_width)

    points = np.asarray(points, dtype=float)
    clipped = np.clip(points, low, high)

    scale = bit_max / (high - low)
    binary_values = np.rint((clipped - low) * scale).astype(dtype)
    return binary_to_gray(binary_values).astype(dtype, copy=False)


def decode_gray_population(gray_population, bounds=_DEFAULT_BOUNDS, bit_width=_DEFAULT_BIT_WIDTH):
    if bit_width < 2 or bit_width > 32:
        raise ValueError("bit_width must be in [2, 32].")

    low, high = bounds
    bit_max = _bit_max(bit_width)
    gray_population = np.asarray(gray_population, dtype=_resolve_uint_dtype(bit_width))

    binary_values = gray_to_binary(gray_population, bit_width=bit_width)
    scale = (high - low) / bit_max
    return low + binary_values.astype(np.float64) * scale


def midpoint_bitwise_crossover(parent1, parent2, bit_width=_DEFAULT_BIT_WIDTH):
    if bit_width < 2 or bit_width > 32:
        raise ValueError("bit_width must be in [2, 32].")

    p1 = np.asarray(parent1)
    p2 = np.asarray(parent2)
    dtype = p1.dtype

    bit_max = _bit_max(bit_width)
    mid_bits = bit_width // 2
    low_mask = (1 << mid_bits) - 1
    high_mask = bit_max ^ low_mask

    low_mask = dtype.type(low_mask)
    high_mask = dtype.type(high_mask)

    child1 = (p1 & high_mask) | (p2 & low_mask)
    child2 = (p2 & high_mask) | (p1 & low_mask)
    return child1.astype(dtype, copy=False), child2.astype(dtype, copy=False)


def adaptive_bitwise_crossover(parent1, parent2, bit_width=_DEFAULT_BIT_WIDTH, rng=None):
    if bit_width < 2 or bit_width > 32:
        raise ValueError("bit_width must be in [2, 32].")

    if rng is None:
        rng = np.random.default_rng()

    p1 = np.asarray(parent1)
    p2 = np.asarray(parent2)
    dtype = p1.dtype
    bit_max = _bit_max(bit_width)

    child1 = p1.copy()
    child2 = p2.copy()
    for d in range(2):
        cut = int(rng.integers(1, bit_width))
        low_mask = (1 << cut) - 1
        high_mask = bit_max ^ low_mask
        low_mask = dtype.type(low_mask)
        high_mask = dtype.type(high_mask)
        child1[d] = (p1[d] & high_mask) | (p2[d] & low_mask)
        child2[d] = (p2[d] & high_mask) | (p1[d] & low_mask)

    return child1.astype(dtype, copy=False), child2.astype(dtype, copy=False)


def _mutation_bit_window(bit_width, progress):
    min_window = min(bit_width, 4)
    span = bit_width - min_window
    if span <= 0:
        return bit_width
    scaled = (1.0 - progress) ** 1.6
    return int(min_window + np.ceil(span * scaled))


def bit_flip_mutation(ind, bit_width=_DEFAULT_BIT_WIDTH, rng=None, progress=0.0):
    if bit_width < 2 or bit_width > 32:
        raise ValueError("bit_width must be in [2, 32].")

    if rng is None:
        rng = np.random.default_rng()

    mutated = ind.copy()
    active_bits = _mutation_bit_window(bit_width, progress)
    extra_flip_prob = 0.35 * max(0.0, 1.0 - progress)

    for d in range(2):
        flip_count = 1 + int(active_bits > 1 and rng.random() < extra_flip_prob)
        flip_count = min(flip_count, active_bits)
        bit_indices = np.atleast_1d(rng.choice(active_bits, size=flip_count, replace=False))

        mask = 0
        for bit_idx in bit_indices:
            mask ^= 1 << int(bit_idx)
        mutated[d] = mutated.dtype.type(mutated[d] ^ mutated.dtype.type(mask))

    return mutated


def _refine_best_gray_individual(individual, fitness, bounds, bit_width):
    candidate = np.asarray(individual).copy()
    point = decode_gray_population(candidate[None, :], bounds=bounds, bit_width=bit_width)[0]
    best_value = float(fitness(point[None, :])[0])

    improved = True
    while improved:
        improved = False
        for d in range(2):
            for bit_idx in range(bit_width):
                trial = candidate.copy()
                trial[d] = trial.dtype.type(trial[d] ^ trial.dtype.type(1 << bit_idx))
                trial_point = decode_gray_population(trial[None, :], bounds=bounds, bit_width=bit_width)[0]
                trial_value = float(fitness(trial_point[None, :])[0])
                if trial_value < best_value:
                    candidate = trial
                    point = trial_point
                    best_value = trial_value
                    improved = True

    return candidate, point, best_value


def run_ga(
    fitness,
    bounds=_DEFAULT_BOUNDS,
    pop_size=80,
    generations=120,
    crossover_prob=0.9,
    mutation_prob=0.25,
    elite_size=2,
    tournament_k=3,
    sigma0=30.0,
    seed=42,
    init_mode="grid",
):
    rng = np.random.default_rng(seed)
    low, high = bounds

    pop = make_initial_points(pop_size, bounds=bounds, mode=init_mode, rng=rng)
    fit = fitness(pop)

    history_positions = [pop.copy()]
    history_best = [fit.min()]
    history_best_point = [pop[np.argmin(fit)].copy()]
    history_mean = [fit.mean()]

    for gen in range(generations):
        elite_idx = np.argsort(fit)[:elite_size]
        new_pop = [pop[i].copy() for i in elite_idx]

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
        "name": "GA (arithmetic crossover)",
    }
    return result


def run_ga_bitwise(
    fitness,
    bounds=_DEFAULT_BOUNDS,
    pop_size=80,
    generations=120,
    crossover_prob=0.9,
    mutation_prob=0.25,
    elite_size=2,
    tournament_k=3,
    bit_width=_DEFAULT_BIT_WIDTH,
    seed=42,
    init_mode="grid",
):
    if bit_width < 2 or bit_width > 32:
        raise ValueError("bit_width must be in [2, 32].")

    rng = np.random.default_rng(seed)
    dtype = _resolve_uint_dtype(bit_width)

    points = make_initial_points(pop_size, bounds=bounds, mode=init_mode, rng=rng)
    pop = encode_points_to_gray(points, bounds=bounds, bit_width=bit_width).astype(dtype, copy=False)
    fit = fitness(points)

    history_positions = [points.copy()]
    history_best = [fit.min()]
    history_best_point = [points[np.argmin(fit)].copy()]
    history_mean = [fit.mean()]

    for gen in range(generations):
        progress = gen / max(1, generations - 1)
        elite_idx = np.argsort(fit)[:elite_size]
        new_pop = [pop[i].copy() for i in elite_idx]
        current_mutation_prob = mutation_prob * (0.15 + 0.85 * ((1.0 - progress) ** 1.25))

        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fit, k=tournament_k, rng=rng)
            p2 = tournament_selection(pop, fit, k=tournament_k, rng=rng)

            if rng.random() < crossover_prob:
                c1, c2 = adaptive_bitwise_crossover(p1, p2, bit_width=bit_width, rng=rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if rng.random() < current_mutation_prob:
                c1 = bit_flip_mutation(c1, bit_width=bit_width, rng=rng, progress=progress)
            if rng.random() < current_mutation_prob:
                c2 = bit_flip_mutation(c2, bit_width=bit_width, rng=rng, progress=progress)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop, dtype=dtype)
        points = decode_gray_population(pop, bounds=bounds, bit_width=bit_width)
        fit = fitness(points)

        best_idx = int(np.argmin(fit))
        refined_bits, refined_point, refined_value = _refine_best_gray_individual(
            pop[best_idx],
            fitness=fitness,
            bounds=bounds,
            bit_width=bit_width,
        )
        if refined_value < fit[best_idx]:
            pop[best_idx] = refined_bits
            points[best_idx] = refined_point
            fit[best_idx] = refined_value

        if progress >= 0.75:
            collapse_count = min(pop_size // 16, max(1, pop_size // 64))
            sorted_idx = np.argsort(fit)
            worst_idx = sorted_idx[-collapse_count:]
            elite_pool = sorted_idx[:max(1, min(elite_size, collapse_count))]
            for target_offset, target_idx in enumerate(worst_idx):
                source_idx = elite_pool[target_offset % len(elite_pool)]
                pop[target_idx] = pop[source_idx].copy()
            points = decode_gray_population(pop, bounds=bounds, bit_width=bit_width)
            fit = fitness(points)

        history_positions.append(points.copy())
        history_best.append(fit.min())
        history_best_point.append(points[np.argmin(fit)].copy())
        history_mean.append(fit.mean())

    best_idx = np.argmin(fit)
    result = {
        "best_point": points[best_idx].copy(),
        "best_value": float(fit[best_idx]),
        "history_positions": history_positions,
        "history_best": history_best,
        "history_best_point": history_best_point,
        "history_mean": history_mean,
        "name": f"GA (bitwise, Gray, B={bit_width}, adaptive)",
    }
    return result
