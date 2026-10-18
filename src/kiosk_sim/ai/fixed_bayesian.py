"""Incremental Bayesian optimisation for one store-wide fixed weight vector."""
import math
import random

from ..recommenders import FIXED_WEIGHTS


def _sample(rng):
    raw = [rng.gammavariate(1., 1.) for _ in range(4)]
    return tuple(value / sum(raw) for value in raw)


def _mutate(weights, rng, scale=.12):
    raw = [max(.001, value + rng.gauss(0., scale)) for value in weights]
    return tuple(value / sum(raw) for value in raw)


def _kernel(left, right, length=.32):
    return math.exp(-sum((a - b) ** 2 for a, b in zip(left, right)) / (2 * length ** 2))


def _factor(matrix):
    lower = [[0.] * len(matrix) for _ in matrix]
    for row in range(len(matrix)):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(lower[row][k] * lower[column][k] for k in range(column))
            lower[row][column] = math.sqrt(max(residual, 1e-12)) if row == column else residual / lower[column][column]
    return lower


def _solve(lower, vector):
    forward = [0.] * len(vector)
    for row in range(len(vector)):
        forward[row] = (vector[row] - sum(lower[row][k] * forward[k] for k in range(row))) / lower[row][row]
    result = [0.] * len(vector)
    for row in range(len(vector) - 1, -1, -1):
        result[row] = (forward[row] - sum(lower[k][row] * result[k] for k in range(row + 1, len(vector)))) / lower[row][row]
    return result


class BayesianFixedOptimizer:
    """Incremental Bayesian linear optimisation using Thompson sampling.

    A compact posterior over a polynomial weight feature map is updated once for
    each synthetic day. It never refits from the full history.
    """
    def __init__(self, a_inv=None, b=None, seed=42, trial=0, last_weights=None, ridge=1.):
        self.size = 15
        self.a_inv = a_inv or [[(1. / ridge if row == column else 0.)
                                for column in range(self.size)] for row in range(self.size)]
        self.b = list(b or [0.] * self.size)
        self.seed, self.trial, self.last_weights, self.ridge = seed, trial, tuple(last_weights or FIXED_WEIGHTS), ridge

    def features(self, weights):
        return (1., *weights, *(value * value for value in weights),
                *(weights[left] * weights[right] for left in range(4) for right in range(left + 1, 4)))

    @property
    def theta(self):
        return [sum(row[column] * self.b[column] for column in range(self.size)) for row in self.a_inv]

    def _score(self, weights, theta):
        return sum(feature * coefficient for feature, coefficient in zip(self.features(weights), theta))

    def _cholesky(self):
        lower = [[0.] * self.size for _ in range(self.size)]
        for row in range(self.size):
            for column in range(row + 1):
                residual = self.a_inv[row][column] - sum(lower[row][k] * lower[column][k] for k in range(column))
                lower[row][column] = math.sqrt(max(residual, 1e-12)) if row == column else residual / lower[column][column]
        return lower

    def propose(self):
        rng = random.Random(self.seed + self.trial * 13007)
        if self.trial == 0:
            self.trial += 1
            return FIXED_WEIGHTS
        lower, noise, mean = self._cholesky(), [rng.gauss(0., 1.) for _ in range(self.size)], self.theta
        sampled = [mean[row] + .18 * sum(lower[row][column] * noise[column] for column in range(row + 1))
                   for row in range(self.size)]
        pool = [_sample(rng) for _ in range(192)] + [_mutate(self.last_weights, rng) for _ in range(96)]
        self.trial += 1
        return max(pool, key=lambda weights: self._score(weights, sampled))

    def observe(self, weights, reward):
        features = self.features(weights)
        projected = [sum(row[column] * features[column] for column in range(self.size)) for row in self.a_inv]
        denominator = 1. + sum(value * other for value, other in zip(features, projected))
        self.a_inv = [[self.a_inv[row][column] - projected[row] * projected[column] / denominator
                       for column in range(self.size)] for row in range(self.size)]
        self.b = [value + reward * feature for value, feature in zip(self.b, features)]
        self.last_weights = tuple(weights)

    @property
    def best_weights(self):
        rng = random.Random(self.seed + 99_991 + self.trial)
        pool = [_sample(rng) for _ in range(384)] + [self.last_weights]
        return max(pool, key=lambda weights: self._score(weights, self.theta))

    def as_dict(self):
        return {"algorithm": "incremental_bayesian_thompson_sampling", "a_inv": self.a_inv, "b": self.b,
                "seed": self.seed, "trial": self.trial, "last_weights": list(self.last_weights),
                "ridge": self.ridge, "best_weights": list(self.best_weights)}

    @classmethod
    def from_dict(cls, data):
        if "a_inv" not in data:
            return cls(seed=data.get("seed", 42), trial=data.get("trial", 0),
                       last_weights=data.get("best_weights", FIXED_WEIGHTS))
        return cls(**{key: data[key] for key in ("a_inv", "b", "seed", "trial", "last_weights", "ridge")})
