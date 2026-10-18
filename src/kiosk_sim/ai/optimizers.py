"""Reproducible black-box search on the four-weight probability simplex."""
import math
import random

from ..recommenders import FIXED_WEIGHTS
from ..simulator import simulate


def _sample(rng):
    raw = [rng.gammavariate(1., 1.) for _ in range(4)]
    return tuple(x / sum(raw) for x in raw)


def _mutate(parent, rng, scale=.12):
    raw = [max(0., x + rng.gauss(0., scale)) for x in parent]
    if sum(raw) < 1e-12:
        return _sample(rng)
    return tuple(x / sum(raw) for x in raw)


def _kernel(a, b, length=.32):
    return math.exp(-sum((x - y) ** 2 for x, y in zip(a, b)) / (2 * length ** 2))


def _factor(matrix):
    """Small positive definite linear system, with a diagonal noise term."""
    n = len(matrix)
    lower = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            residual = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(max(residual, 1e-12)) if i == j else residual / lower[j][j]
    return lower


def _solve(lower, vector):
    n = len(vector)
    forward = [0.] * n
    for i in range(n):
        forward[i] = (vector[i] - sum(lower[i][j] * forward[j] for j in range(i))) / lower[i][i]
    backward = [0.] * n
    for i in range(n - 1, -1, -1):
        backward[i] = (forward[i] - sum(lower[j][i] * backward[j] for j in range(i + 1, n))) / lower[i][i]
    return backward


def _bayesian_candidate(observations, rng):
    points = [row['weights'] for row in observations]
    values = [(row['customer'], row['store']) for row in observations]
    centers = [sum(row[i] for row in values) / len(values) for i in range(2)]
    preference = rng.random()  # varied tradeoff instead of one fixed utility
    matrix = [[_kernel(a, b) + (.0001 if i == j else 0.)
               for j, b in enumerate(points)] for i, a in enumerate(points)]
    lower = _factor(matrix)
    alphas = [_solve(lower, [v[i] - centers[i] for v in values]) for i in range(2)]
    frontier = _pareto(observations)
    pool = [_sample(rng) for _ in range(128)] + [
        _mutate(rng.choice(frontier)['weights'], rng) for _ in range(64)]
    def acquisition(candidate):
        similarities = [_kernel(candidate, point) for point in points]
        means = [centers[i] + sum(k * a for k, a in zip(similarities, alphas[i]))
                 for i in range(2)]
        variance = max(0., 1. - sum(k * x for k, x in zip(similarities,
                                                          _solve(lower, similarities))))
        return (preference * means[0] + (1 - preference) * means[1]
                + .08 * math.sqrt(variance))  # randomized UCB scalarization
    return max(pool, key=acquisition)


def _dominates(a, b):
    return (a['customer'] >= b['customer'] - 1e-12 and
            a['store'] >= b['store'] - 1e-12 and
            (a['customer'] > b['customer'] + 1e-12 or
             a['store'] > b['store'] + 1e-12))


def _pareto(records):
    return [row for row in records if not any(_dominates(other, row)
                                               for other in records if other is not row)]


def _balanced_choice(frontier):
    """Closest to the observed ideal in normalized customer/store coordinates."""
    lower = [min(x[axis] for x in frontier) for axis in ('customer', 'store')]
    upper = [max(x[axis] for x in frontier) for axis in ('customer', 'store')]
    def distance(row):
        return sum(((upper[i] - row[axis]) / max(upper[i] - lower[i], 1e-9)) ** 2
                   for i, axis in enumerate(('customer', 'store')))
    return min(frontier, key=lambda row: (distance(row), row['trial']))


def train_optimizer(scenarios, *, method, budget=48, seed=42, **simulation_options):
    """Search the customer/store Pareto frontier using only training days."""
    if method not in ('bayesian', 'genetic') or budget < 2 or not scenarios:
        raise ValueError('method must be bayesian or genetic; budget >= 2 and training days required')
    rng = random.Random(seed)
    trials = []
    for iteration in range(budget):
        if iteration == 0:
            candidate = FIXED_WEIGHTS
        elif method == 'bayesian':
            candidate = _sample(rng) if iteration < min(6, budget) else _bayesian_candidate(trials, rng)
        elif iteration < min(8, budget) or rng.random() < .15:
            candidate = _sample(rng)
        else:
            frontier = _pareto(trials)
            parents = rng.choices(frontier, k=2)
            ratio = rng.random()
            child = tuple(ratio * a + (1 - ratio) * b
                          for a, b in zip(parents[0]['weights'], parents[1]['weights']))
            candidate = _mutate(child, rng)
        results = [simulate(people, shop, 'gemini', candidate, day_seed,
                            **simulation_options) for people, shop, day_seed in scenarios]
        customer = sum(r.satisfaction for r in results) / len(results)
        store = sum(r.store_satisfaction for r in results) / len(results)
        value = sum(r.score for r in results) / len(results)
        trials.append({'trial': iteration + 1, 'weights': candidate,
                       'customer': customer, 'store': store, 'mean_score': value,
                       'daily_scores': [r.score for r in results],
                       'daily_customer': [r.satisfaction for r in results],
                       'daily_store': [r.store_satisfaction for r in results]})
    frontier = _pareto(trials)
    selected = _balanced_choice(frontier)
    return selected['weights'], {'optimizer': method, 'candidate_evaluations': budget,
                    'training_days_per_candidate': len(scenarios),
                    'objectives': ['customer_satisfaction', 'store_satisfaction'],
                    'selection_rule': 'minimum normalized squared distance to the observed two-objective ideal',
                    'selected_trial': selected['trial'],
                    'selected_training_score': selected['mean_score'],
                    'pareto_frontier': frontier, 'trials': trials,
                    'seed': seed}
