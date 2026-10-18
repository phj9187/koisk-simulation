"""Paired complete-day random search, with isolated validation and evaluation."""
import copy
import random
import time
from dataclasses import asdict
from kiosk_sim.generators import customers
from kiosk_sim.lhs_study import LHSStudy
from kiosk_sim.simulator import simulate
from kiosk_cloud.policy import WeightPolicy, objective, COMPONENTS


def run_store_day(task):
    config, case, day = task
    case = copy.deepcopy(case)
    case.records = []
    cycle, phase, offset = config.stage(day)
    if offset == 1 or not hasattr(case, 'policies'):
        case.policies = {name: WeightPolicy.initial(name) for name in config.models}
        case.optimizer_steps = {name: 0 for name in config.models}
    local = LHSStudy.__new__(LHSStudy)
    local.config = config.legacy()
    rng = random.Random(config.seed + day * 1009 + case.store_id * 17)
    visitors = (int(round(float(case.features['visitor_mean']))) if config.source_mode == 'empirical_donors_2025'
                else rng.randint(local.config.visitor_min, local.config.visitor_max))
    if config.visitors_per_day is not None: visitors = config.visitors_per_day
    people = customers(visitors, rng, case.template) if visitors else []
    seed = config.seed + day * 10_000 + case.store_id
    store = local._daily_store(case, day)
    observed_hours = case.features.get('opening_hours')
    hours = int(observed_hours) if observed_hours and 1 <= observed_hours <= 24 else 8
    logs, inventory = [], {}
    for model in config.models:
        started = time.perf_counter()
        incumbent = case.policies[model]
        feedback = []
        result = simulate(people, store, 'fixed', incumbent, seed=seed, hours=hours,
                          feedback=feedback if config.detailed_logs else None)
        before = objective(result, visitors, config.objective)
        accepted = False
        initial_score = None
        if phase == 'train':
            candidate = incumbent.perturb(config.seed + day * 9_999_991 + case.store_id * 97, config.search_sigma)
            candidate_feedback = []
            alternative = simulate(people, store, 'fixed', candidate, seed=seed, hours=hours,
                                   feedback=candidate_feedback if config.detailed_logs else None)
            after = objective(alternative, visitors, config.objective)
            if after > before:
                case.policies[model] = candidate
                result, feedback = alternative, candidate_feedback
                accepted = True
            case.optimizer_steps[model] += 1
            evaluations = 2
        else:
            # Held-out initial-model baseline uses the exact same customer/date stream.
            initial_result = simulate(people, store, 'fixed', WeightPolicy.initial(model), seed=seed, hours=hours)
            initial_score = objective(initial_result, visitors, config.objective)
            evaluations = 2
        values = asdict(result)
        score = objective(result, visitors, config.objective)
        # Score is the configured common objective, distinct from legacy Result.score.
        row = dict(day=day, cycle=cycle, phase=phase, store_id=case.store_id, model=model, visitors=visitors,
                   score=score, initial_score=initial_score, incumbent_train_score=before if phase == 'train' else None,
                   customer_satisfaction=result.satisfaction, store_satisfaction=result.store_satisfaction,
                   profit=result.profit, served=result.served, wait_minutes=result.avg_total_wait,
                   acceptance_rate=result.accept_rate, waste=result.waste, stockouts=result.stockouts,
                   training_observations=2 if phase == 'train' else 0, optimizer_steps=case.optimizer_steps[model],
                   simulator_evaluations=evaluations, accepted_candidate=accepted,
                   elapsed_seconds=time.perf_counter() - started)
        # Weights at a representative fixed reference context; not a whole policy summary.
        policy = case.policies[model]
        reference = [sum(policy.parameters[i][j] * (1. if j == 0 else 0.) for j in range(len(policy.parameters[i]))) for i in range(4)]
        import math
        weights = [math.exp(v - max(reference)) for v in reference]
        for key, weight in zip(COMPONENTS, weights): row['weight_' + key] = weight / sum(weights)
        case.records.append(row)
        if config.detailed_logs:
            for index, event in enumerate(feedback):
                logs.append(dict(day=day, cycle=cycle, phase=phase, store_id=case.store_id, model=model,
                                 event_index=index, payload=event))
        inventory[model] = {'initial_stock': dict(store.stock), 'waste': result.waste, 'daily_reset': True,
                            'terminal_stock_exposed_by_legacy': False}
    return case, logs, inventory
