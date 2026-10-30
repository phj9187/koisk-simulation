"""Outcome utilities: no policy parameters or recommendation scores are inputs.

Coefficients are declared research assumptions, not measured satisfaction.
"""
import math

CUSTOMER_METRICS = ('taste_fit_per_visitor', 'affordability_per_visitor', 'wait_component_per_visitor')
STORE_METRICS = ('margin_component', 'waste_reduction_component')

def affordability(price, budget):
    return max(0., min(1., 1. - price / max(budget, 1e-9)))

def wait_utility(minutes, scale):
    return scale / (scale + max(0., minutes))

def ingredient_risks(store):
    return {k: sum(m.expiry_risk for m in store.menus if m.ingredient == k) /
            max(1, sum(m.ingredient == k for m in store.menus)) for k in store.stock}

def score_outcomes(components, scenario, lam):
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError('invalid lambda')
    values = [components[k] for k in CUSTOMER_METRICS + STORE_METRICS]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('outcome components must be finite and in [0,1]')
    for key, size in (('customer', 3), ('store', 2)):
        weights = scenario[key]
        if len(weights) != size or any(not math.isfinite(v) or v < 0 for v in weights) or not math.isclose(sum(weights), 1.):
            raise ValueError('invalid satisfaction coefficients')
    customer = sum(a * components[k] for a, k in zip(scenario['customer'], CUSTOMER_METRICS))
    fatigue=components.get('choice_fatigue_per_visitor',0.)
    if not math.isfinite(fatigue) or fatigue<0:raise ValueError('invalid choice fatigue')
    customer=max(0.,customer-fatigue)
    store = sum(a * components[k] for a, k in zip(scenario['store'], STORE_METRICS))
    return {'customer': customer, 'store': store, 'reward': lam * customer + (1-lam) * store}
