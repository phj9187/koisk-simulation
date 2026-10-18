"""Business-day event simulation. All policies receive the same exogenous draws."""
import copy
import math
import random
from itertools import accumulate
from .models import Customer, Menu, Store
from .recommenders import affinity, available, predicted_wait, rank, FIXED_WEIGHTS
from .evaluation import Result, score
from .arrival import assumed_purpose_shape, arrival_profile


def _choice_value(c: Customer, m: Menu, wait: float) -> float:
    return 2.5 * affinity(c, m) - c.price_sensitivity * m.price / 20. - c.wait_sensitivity * wait / 30.


def _decision_minutes(option_count: int, accepted: bool, suggested: bool) -> float:
    """Explicit scenario assumption: fewer comparisons when a suggestion is accepted."""
    if accepted:
        return .8 + .10 * option_count
    if suggested:
        return 1.3 + .45 * option_count
    return 1.5 + .60 * option_count


def simulate(customers: list[Customer], original: Store, policy: str,
             weights: tuple[float, ...] = FIXED_WEIGHTS, seed: int = 0,
             hours: float = 8., hourly_weights: tuple[float, ...] | None = None,
             purpose_profiles=None, population_strength: float = .3,
             start_hour: int = 11, feedback: list | None = None,
             decision_multiplier: float = 1., decision_penalty: float = 1.,
             acceptance_multiplier: float = 1.) -> Result:
    if (decision_multiplier < 0 or decision_penalty < 0 or acceptance_multiplier < 0 or
        not all(map(math.isfinite, (decision_multiplier, decision_penalty, acceptance_multiplier)))):
        raise ValueError("decision settings must be finite and nonnegative")
    store = copy.deepcopy(original)
    rng = random.Random(seed)
    sales = {m.name: 0 for m in store.menus}
    # External arrivals and individual choice shocks are fixed by seed across policies.
    if hourly_weights is not None and (len(hourly_weights) != int(hours) or
       any(x < 0 for x in hourly_weights) or sum(hourly_weights) <= 0):
        raise ValueError("hourly_weights must contain one nonnegative weight per hour")
    events = []
    for customer in customers:
        purpose = (purpose_profiles or {}).get(customer.purpose)
        if purpose is None:
            purpose = assumed_purpose_shape(customer.purpose, start_hour, int(hours))
        profile = arrival_profile(hourly_weights, purpose, population_strength=population_strength)
        cumulative = list(accumulate(profile))
        draw = rng.random()
        hour = next(i for i, ceiling in enumerate(cumulative) if draw <= ceiling)
        arrival = (hour + rng.random()) * 60
        shocks = {m.name: rng.gammavariate(1., 1.) for m in store.menus}
        accept_draw = rng.random()
        events.append((arrival, customer, shocks, accept_draw))
    events.sort(key=lambda event: event[0])
    total_satisfaction = profit = wait_total = decision_total = queue_total = 0.
    kiosk_free_at = 0.
    served = accepted = recommended = stockouts = 0
    for arrival, customer, shocks, accept_draw in events:
        choice_start = max(arrival, kiosk_free_at)
        options = available(store, customer)
        if not options:
            # This includes budget and allergy restrictions, not only stockouts.
            stockouts += 1
            continue
        suggestion = rank(customer, store, choice_start, policy, weights, sales)
        decision_context = None
        if feedback is not None and callable(weights):
            from .ai.contextual import context_features
            decision_context = {
                "customer_taste": customer.taste, "customer_purpose": customer.purpose,
                "price_sensitivity": customer.price_sensitivity,
                "wait_sensitivity": customer.wait_sensitivity,
                "budget": customer.budget, "openness": customer.openness,
                "stock_before": dict(store.stock), "cooks": store.cooks,
                "features_before": context_features(customer, store, choice_start),
                "applied_weights": weights(customer, store, choice_start),
                "available_menus": [m.name for m in options]
            }
        # Private choice depends on preference, price, wait, and a common random shock.
        own = max(options, key=lambda m: _choice_value(customer, m, predicted_wait(store, m, choice_start))
                  + .16 * math.log(max(1e-9, shocks[m.name])))
        picked = own
        accepted_this = False
        if suggestion is not None:
            recommended += 1
            gain = _choice_value(customer, suggestion, predicted_wait(store, suggestion, choice_start)) - _choice_value(customer, own, predicted_wait(store, own, choice_start))
            accept_prob = min(1., acceptance_multiplier * customer.openness /
                              (1. + math.exp(-3. * (gain + .25))))
            if accept_draw < accept_prob:
                picked = suggestion
                accepted += 1
                accepted_this = True
        queue = choice_start - arrival
        decision = decision_multiplier * _decision_minutes(len(options), accepted_this, suggestion is not None)
        order_at = choice_start + decision
        kiosk_free_at = order_at
        free_idx = min(range(store.cooks), key=lambda k: store.cook_free_at[k])
        finish = max(order_at, store.cook_free_at[free_idx]) + picked.cook_minutes
        wait = finish - order_at
        store.cook_free_at[free_idx] = finish
        store.stock[picked.ingredient] -= picked.units
        sales[picked.name] += 1
        served += 1
        wait_total += wait
        decision_total += decision
        queue_total += queue
        profit += picked.price - picked.cost
        # Evaluation is independent of recommendation weights and bounded in [0, 1].
        customer_score = max(0., min(1., affinity(customer, picked) + .12 -
                                   .12 * customer.price_sensitivity * picked.price / 20. -
                                   .18 * customer.wait_sensitivity * wait / (wait + 20.) -
                                   .18 * decision_penalty * customer.wait_sensitivity * (queue + decision) /
                                   (queue + decision + 5.)))
        total_satisfaction += customer_score
        if feedback is not None and suggestion is not None:
            # Generated by the simulator after recommendation; never use the ranking score as feedback.
            store_score = max(0., min(1., .75 * (picked.price - picked.cost) / 9.
                                     + .25 * picked.expiry_risk))
            feedback.append({"purpose": customer.purpose, "store_purpose": store.purpose,
                             "recommended": suggestion.name, "chosen": picked.name,
                             "accepted": picked.name == suggestion.name,
                             "taste_fit": round(affinity(customer, picked), 3),
                             "customer_satisfaction": round(customer_score, 3),
                             "store_satisfaction": round(store_score, 3),
                             "decision_minutes": round(decision, 2),
                             "kiosk_queue_minutes": round(queue, 2),
                             "wait_minutes": round(wait, 2),
                             **({"bandit_features": weights.action_features(customer, store, choice_start, suggestion)}
                                if hasattr(weights, "action_features") else {}),
                             **({"decision_context": decision_context} if decision_context else {})})
    waste = sum(m.expiry_risk * store.stock[m.ingredient] / max(1, sum(x.ingredient == m.ingredient for x in store.menus))
                for m in store.menus)
    store_satisfaction = (.75 * min(1., profit / (max(1, len(customers)) * 9.))
                          + .25 * (1. - min(1., waste / 100.)))
    result = Result(total_satisfaction / max(1, served), profit,
                    wait_total / max(1, served), served,
                    accepted / max(1, recommended), stockouts, waste,
                    avg_decision=decision_total / max(1, served),
                    avg_kiosk_queue=queue_total / max(1, served),
                    avg_total_wait=(wait_total + decision_total + queue_total) / max(1, served),
                    store_satisfaction=store_satisfaction)
    result.score = score(result, len(customers))
    return result
