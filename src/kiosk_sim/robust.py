"""Scenario and preference-grid comparisons; not probabilities of real-world success."""
from itertools import product
import random
from statistics import median

from .simulator import simulate

# Each grid is an explicit scenario assumption, not a fitted confidence interval.
ASSUMPTIONS = tuple(product((.5, 1., 2.), (.5, 1., 2.)))
# All nonnegative combinations of five stakeholder weights in increments of .25.
PREFERENCES = tuple(tuple(x / 4 for x in parts)
                    for parts in product(range(5), repeat=5) if sum(parts) == 4)
AXES = ("고객 만족도", "고객당 이익", "남은 재고 위험 감소", "응대율", "시간 효율")


def differences(candidate, baseline, customers):
    """Paired proportional changes; same five axes for all policies."""
    return (candidate.satisfaction - baseline.satisfaction,
            (candidate.profit - baseline.profit) / max(1., abs(candidate.profit), abs(baseline.profit)),
            (baseline.waste - candidate.waste) / max(1., abs(candidate.waste), abs(baseline.waste)),
            (candidate.served - baseline.served) / customers,
            (baseline.avg_total_wait - candidate.avg_total_wait) /
            max(1., baseline.avg_total_wait, candidate.avg_total_wait))


def comparison(pairs):
    """Pairs are (candidate, baseline, customer_count); ties remain ties."""
    if not pairs:
        raise ValueError("Need at least one paired simulation")
    wins = losses = ties = dominating = dominated = 0
    margins = []
    for candidate, baseline, count in pairs:
        delta = differences(candidate, baseline, count)
        dominating += all(v >= -1e-10 for v in delta) and any(v > 1e-10 for v in delta)
        dominated += all(v <= 1e-10 for v in delta) and any(v < -1e-10 for v in delta)
        for weights in PREFERENCES:
            margin = sum(w * v for w, v in zip(weights, delta))
            margins.append(margin)
            if margin > 1e-10:
                wins += 1
            elif margin < -1e-10:
                losses += 1
            else:
                ties += 1
    size = len(margins)
    return {"win_share": wins / size, "tie_share": ties / size,
            "loss_share": losses / size, "worst_margin": min(margins),
            "median_margin": median(margins),
            "pareto_better_days": dominating, "pareto_worse_days": dominated,
            "paired_runs": len(pairs), "preference_vectors": len(PREFERENCES)}


def evaluate_robust(scenarios, policies, *, assumptions=ASSUMPTIONS, **simulation_options):
    """Compare policies to no recommendation and fixed weights on held-out days."""
    outcomes = {key: [] for key in policies}
    for people, shop, seed in scenarios:
        for decision_multiplier, decision_penalty in assumptions:
            params = dict(simulation_options, decision_multiplier=decision_multiplier,
                          decision_penalty=decision_penalty)
            for policy, weights in policies.items():
                outcomes[policy].append(simulate(people, shop, policy, weights, seed, **params))
    return {policy: {baseline: comparison([(a, b, len(scenarios[i // len(assumptions)][0]))
                                            for i, (a, b) in enumerate(zip(outcomes[policy], outcomes[baseline]))])
                     for baseline in ("none", "fixed") if baseline != policy}
            for policy in policies if policy != "none"}


def random_reference_comparison(scenarios, learned, *, seed=42, count=12,
                                assumptions=ASSUMPTIONS, **simulation_options):
    """Context only: unseen random weight vectors, not an optimum or a training target."""
    rng = random.Random(seed)
    references = []
    for _ in range(count):
        raw = [rng.gammavariate(1., 1.) for _ in range(4)]
        references.append(tuple(value / sum(raw) for value in raw))
    learned_runs = []
    for people, shop, event_seed in scenarios:
        for multiplier, penalty in assumptions:
            params = dict(simulation_options, decision_multiplier=multiplier, decision_penalty=penalty)
            learned_runs.append((people, shop, event_seed, params,
                                 simulate(people, shop, "gemini", learned, event_seed, **params)))
    outcomes = []
    for weights in references:
        report = comparison([(learned_result,
                              simulate(people, shop, "fixed", weights, event_seed, **params), len(people))
                             for people, shop, event_seed, params, learned_result in learned_runs])
        outcomes.append(report)
    return {"reference_count": count,
            "better_than_references": sum(r["win_share"] > r["loss_share"] for r in outcomes),
            "tied_with_references": sum(r["win_share"] == r["loss_share"] for r in outcomes),
            "worse_than_references": sum(r["win_share"] < r["loss_share"] for r in outcomes),
            "seed": seed}
