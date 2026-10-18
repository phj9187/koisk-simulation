"""New policies: one global weight vector vs context-dependent weight vectors.

Both optimize the SAME complete-day objective with the SAME paired random search.
No customer-level reward and no LinUCB/Thompson learning is used here.
"""
from dataclasses import dataclass
import math
import random
from kiosk_sim.recommenders import affinity, predicted_wait

CONTEXT_NAMES = ('bias', 'taste_1', 'taste_2', 'taste_3', 'price_sensitivity',
                 'wait_sensitivity', 'dessert_visit', 'stock_pressure', 'congestion')
COMPONENTS = ('affinity', 'waste_risk', 'margin', 'negative_wait')


def context(customer, store, arrival):
    return (1., *(x - .5 for x in customer.taste), customer.price_sensitivity - .7,
            customer.wait_sensitivity - .6,
            1. if customer.purpose == 'dessert' else -1. if customer.purpose == 'meal' else 0.,
            1. - min(1., sum(store.stock.values()) / max(1., 40 * len(store.stock))),
            min(1., max(0., min(store.cook_free_at) - arrival) / 30.))


@dataclass
class WeightPolicy:
    kind: str
    parameters: list[list[float]]

    @classmethod
    def initial(cls, kind):
        if kind not in ('fixed', 'contextual'): raise ValueError('unknown policy')
        return cls(kind, [[0.] * (1 if kind == 'fixed' else len(CONTEXT_NAMES)) for _ in range(4)])

    def __call__(self, customer, store, arrival):
        features = (1.,) if self.kind == 'fixed' else context(customer, store, arrival)
        logits = [sum(a * b for a, b in zip(row, features)) for row in self.parameters]
        peak = max(logits)
        values = [math.exp(value - peak) for value in logits]
        return tuple(value / sum(values) for value in values)

    def perturb(self, seed, sigma):
        rng = random.Random(seed)
        # Normalize expected parameter-space step length across model dimensions.
        dimension = sum(len(row) for row in self.parameters)
        step = sigma / math.sqrt(dimension)
        return WeightPolicy(self.kind, [[max(-8., min(8., value + rng.gauss(0., step))) for value in row] for row in self.parameters])

    def state(self):
        return {'kind': self.kind, 'parameters': self.parameters}


def objective(result, visitors, coefficients):
    return (coefficients['customer'] * result.satisfaction + coefficients['store'] * result.store_satisfaction
            + coefficients['served'] * result.served / max(1, visitors)
            - coefficients['wait'] * result.avg_total_wait / (result.avg_total_wait + 25.))
