"""Context-dependent recommendation weights trained on simulated outcomes.

Features are available before the recommendation. Parameters are selected using
training days only; evaluation days are never used to fit the policy.
"""
import math
import random
from dataclasses import dataclass

from ..recommenders import FIXED_WEIGHTS, affinity, predicted_wait
from ..simulator import simulate
from .optimizers import _pareto, _balanced_choice


FEATURES = ('taste_1', 'taste_2', 'taste_3', 'price_sensitivity',
            'wait_sensitivity', 'dessert_visit', 'stock_pressure',
            'cook_congestion', 'average_menu_margin', 'average_expiry_risk')


def context_features(customer, store, arrival):
    stock_pressure = 1. - min(1., sum(store.stock.values()) /
                              max(1., 40. * len(store.stock)))
    congestion = max(0., min(1., (min(store.cook_free_at) - arrival) / 30.))
    margin = sum(m.price - m.cost for m in store.menus) / len(store.menus)
    expiry = sum(m.expiry_risk for m in store.menus) / len(store.menus)
    return (customer.taste[0] - .5, customer.taste[1] - .5, customer.taste[2] - .5,
            customer.price_sensitivity - .7, customer.wait_sensitivity - .6,
            1. if customer.purpose == 'dessert' else -1. if customer.purpose == 'meal' else 0.,
            stock_pressure - .25, congestion - .25,
            margin / 10. - .5, expiry - .5)


@dataclass(frozen=True)
class ContextualPolicy:
    bias: tuple[float, float, float, float]
    coefficients: tuple[tuple[float, ...], ...]

    def __call__(self, customer, store, arrival):
        features = context_features(customer, store, arrival)
        logits = [self.bias[i] + sum(a * x for a, x in zip(self.coefficients[i], features))
                  for i in range(4)]
        peak = max(logits)
        raw = [math.exp(x - peak) for x in logits]
        return tuple(x / sum(raw) for x in raw)

    @property
    def base_weights(self):
        peak = max(self.bias)
        raw = [math.exp(x - peak) for x in self.bias]
        return tuple(x / sum(raw) for x in raw)

    def as_dict(self):
        return {'bias': self.bias, 'coefficients': self.coefficients,
                'features': FEATURES, 'reference_weights': self.base_weights}


class OnlineContextualBandit:
    """LinUCB: online linear reward estimation plus uncertainty-aware choice.

    The inverse ridge matrix is updated with Sherman-Morrison after every
    observed recommendation. Earlier days are never replayed for training.
    """
    def __init__(self, a_inv=None, b=None, updates=0, reward_mean=.5,
                 alpha=.55, ridge=1., decision_count=0, **_legacy):
        size = 1 + len(FEATURES) + 4 + len(FEATURES) * 4
        self.a_inv = a_inv or [[(1. / ridge if row == column else 0.)
                                for column in range(size)] for row in range(size)]
        self.b = list(b or [0.] * size)
        self.updates = updates
        self.reward_mean = reward_mean
        self.alpha = alpha
        self.ridge = ridge
        self.decision_count = decision_count

    def _components(self, customer, store, arrival, menu):
        return (affinity(customer, menu),
                menu.expiry_risk * min(1., store.stock[menu.ingredient] / 40.),
                (menu.price - menu.cost) / 15.,
                -min(1., predicted_wait(store, menu, arrival) / 30.))

    def action_features(self, customer, store, arrival, menu):
        context = context_features(customer, store, arrival)
        components = self._components(customer, store, arrival, menu)
        return (1., *context, *components,
                *(value * component for value in context for component in components))

    def prediction(self, features):
        return sum(weight * value for weight, value in zip(self.theta, features))

    @property
    def theta(self):
        return [sum(row[column] * self.b[column] for column in range(len(self.b)))
                for row in self.a_inv]

    def uncertainty(self, features):
        projected = [sum(row[column] * features[column] for column in range(len(features)))
                     for row in self.a_inv]
        return math.sqrt(max(0., sum(value * other for value, other in zip(features, projected))))

    def choose(self, customer, store, arrival, options):
        features = [self.action_features(customer, store, arrival, menu) for menu in options]
        self.decision_count += 1
        return options[max(range(len(options)), key=lambda index:
                           self.prediction(features[index]) + self.alpha * self.uncertainty(features[index]))]

    def observe(self, features, reward):
        # A <- A + xx' and A^-1 is updated without re-reading prior data.
        projected = [sum(row[column] * features[column] for column in range(len(features)))
                     for row in self.a_inv]
        denominator = 1. + sum(value * other for value, other in zip(features, projected))
        self.a_inv = [[self.a_inv[row][column] - projected[row] * projected[column] / denominator
                       for column in range(len(features))] for row in range(len(features))]
        self.b = [value + reward * feature for value, feature in zip(self.b, features)]
        self.updates += 1
        self.reward_mean += (reward - self.reward_mean) / self.updates

    def __call__(self, customer, store, arrival):
        """Expose interpretable four-component weights for the run record."""
        context = context_features(customer, store, arrival)
        base = 1 + len(FEATURES)
        interaction = base + 4
        theta = self.theta
        logits = [theta[base + component] + sum(
            theta[interaction + feature * 4 + component] * value
            for feature, value in enumerate(context)) for component in range(4)]
        peak = max(logits)
        raw = [math.exp(value - peak) for value in logits]
        return tuple(value / sum(raw) for value in raw)

    @property
    def base_weights(self):
        base = 1 + len(FEATURES)
        logits = self.theta[base:base + 4]
        peak = max(logits)
        raw = [math.exp(value - peak) for value in logits]
        return tuple(value / sum(raw) for value in raw)

    def as_dict(self):
        return {"policy_type": "linucb_contextual_bandit", "a_inv": self.a_inv, "b": self.b,
                "updates": self.updates, "reward_mean": self.reward_mean, "alpha": self.alpha,
                "ridge": self.ridge, "decision_count": self.decision_count,
                "features": FEATURES}

    @classmethod
    def from_dict(cls, data):
        if "a_inv" not in data:
            # Old epsilon-greedy records are readable but begin with a fresh
            # LinUCB uncertainty matrix when continued in a new run.
            return cls(updates=data.get("updates", 0), reward_mean=data.get("reward_mean", .5),
                       decision_count=data.get("decision_count", 0))
        return cls(**{key: data[key] for key in ("a_inv", "b", "updates", "reward_mean", "alpha",
                                                  "ridge", "decision_count")})

    def frozen(self):
        data = self.as_dict()
        return self.from_dict(data)




def _initial():
    return ContextualPolicy((0.,) * 4, ((0.,) * len(FEATURES),) * 4)


def _mutate(parent, rng):
    bias = list(parent.bias)
    slopes = [list(row) for row in parent.coefficients]
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(4)
        if rng.random() < .22:
            bias[i] = max(-2., min(2., bias[i] + rng.gauss(0., .35)))
        else:
            j = rng.randrange(len(FEATURES))
            slopes[i][j] = max(-2., min(2., slopes[i][j] + rng.gauss(0., .55)))
    return ContextualPolicy(tuple(bias), tuple(tuple(row) for row in slopes))


def train_contextual(scenarios, *, budget=96, seed=42, **simulation_options):
    if not scenarios or budget < 2:
        raise ValueError('Need training days and at least two contextual policy evaluations')
    rng = random.Random(seed)
    policies = []
    trials = []
    for step in range(budget):
        if step == 0:
            policy = _initial()
        else:
            parent = rng.choice(_pareto(trials))
            policy = _mutate(policies[parent['trial'] - 1], rng)
            if step < 8 or rng.random() < .10:
                policy = _mutate(_initial(), rng)
        outcomes = [simulate(people, shop, 'gemini', policy, event_seed,
                             **simulation_options) for people, shop, event_seed in scenarios]
        policies.append(policy)
        trials.append({'trial': step + 1, 'weights': policy.base_weights,
                       'coefficients': policy.coefficients, 'bias': policy.bias,
                       'customer': sum(r.satisfaction for r in outcomes) / len(outcomes),
                       'store': sum(r.store_satisfaction for r in outcomes) / len(outcomes),
                       'mean_score': sum(r.score for r in outcomes) / len(outcomes),
                       'daily_customer': [r.satisfaction for r in outcomes],
                       'daily_store': [r.store_satisfaction for r in outcomes],
                       'daily_scores': [r.score for r in outcomes]})
    frontier = _pareto(trials)
    selected = _balanced_choice(frontier)
    return policies[selected['trial'] - 1], {
        'optimizer': 'contextual', 'candidate_evaluations': budget,
        'training_days_per_candidate': len(scenarios), 'features': FEATURES,
        'objectives': ['customer_satisfaction', 'store_satisfaction'],
        'selection_rule': 'minimum normalized squared distance to observed two-objective ideal',
        'selected_trial': selected['trial'], 'pareto_frontier': frontier,
        'trials': trials, 'seed': seed}
