"""Paired held-out synthetic evaluation; intervals describe scenario variation only."""
import random
from statistics import mean

from .dashboard import METRICS
from .recommenders import FIXED_WEIGHTS
from .simulator import simulate
from itertools import product


def _reference_vectors(seed, count):
    rng = random.Random(seed)
    vectors = [FIXED_WEIGHTS]
    for _ in range(count - 1):
        raw = [rng.gammavariate(1., 1.) for _ in range(4)]
        vectors.append(tuple(x / sum(raw) for x in raw))
    return vectors, rng


def _percent_summary(days, rng, draws):
    baseline = mean(pair[1] for pair in days)
    ai_mean = mean(pair[0] for pair in days)
    if baseline <= 0:
        raise ValueError("Reference average score must be positive for a percentage")
    values = sorted(100. * (mean(p[0] for p in selection) /
                            mean(p[1] for p in selection) - 1.)
                    for selection in (rng.choices(days, k=len(days)) for _ in range(draws)))
    return {"reference_mean_score": baseline, "ai_mean_score": ai_mean,
            "relative_gain_percent": 100. * (ai_mean / baseline - 1.),
            "lower_95_percent": values[int(.025 * draws)],
            "upper_95_percent": values[min(draws - 1, int(.975 * draws))]}


def weight_distribution_benchmark(scenarios, learned, *, seed=42, count=32,
                                  minimum_gain_percent=1., draws=2000,
                                  **simulation_options):
    """Frozen reference weight distribution, then paired held-out day evaluation.

    Uniform simplex sampling is an explicit assumption about unknown weights,
    not a distribution of actual restaurants or a claim about the optimum.
    """
    if not scenarios or count < 2 or minimum_gain_percent < 0:
        raise ValueError("Need test days, >=2 references and nonnegative minimum gain")
    vectors, rng = _reference_vectors(seed, count)
    days = []
    per_vector = [[] for _ in vectors]
    for people, shop, event_seed in scenarios:
        ai = simulate(people, shop, "gemini", learned, event_seed, **simulation_options).score
        scores = [simulate(people, shop, "fixed", weights, event_seed,
                           **simulation_options).score for weights in vectors]
        for j, value in enumerate(scores):
            per_vector[j].append(value)
        days.append((ai, mean(scores)))
    # Percentage is a ratio of means, not an average of noisy per-day percentages.
    summary = _percent_summary(days, rng, draws)
    ai_mean = summary["ai_mean_score"]
    reference_means = [mean(values) for values in per_vector]
    return {"reference_count": count, "test_days": len(days),
            **summary, "minimum_gain_percent": minimum_gain_percent,
            "meets_threshold": summary["lower_95_percent"] > minimum_gain_percent,
            "above_reference_count": sum(ai_mean > value + 1e-10 for value in reference_means),
            "equal_reference_count": sum(abs(ai_mean - value) <= 1e-10 for value in reference_means),
            "reference_mean_min": min(reference_means),
            "reference_mean_max": max(reference_means),
            "reference_seed": seed,
            "method": "Fixed equal weights + seeded uniform-simplex random weights; paired held-out days; day bootstrap"}


def sensitivity_check(scenarios, learned, *, seed=42, count=32, draws=1000,
                      minimum_gain_percent=1., **simulation_options):
    """Evaluate a frozen learned policy under explicitly hypothetical model changes."""
    vectors, _ = _reference_vectors(seed, count)
    # These are scenario assumptions, not estimated uncertainty ranges.
    levels = ((.5, 1., 1.5), (.5, 1., 2.))
    score_ratios = ((.55, .25), (.45, .35), (.35, .45))
    conditions = []
    for acceptance, choice_time in product(*levels):
        settings = dict(simulation_options, acceptance_multiplier=acceptance,
                        decision_multiplier=simulation_options.get("decision_multiplier", 1.) * choice_time)
        results = []
        for people, shop, event_seed in scenarios:
            ai = simulate(people, shop, "gemini", learned, event_seed, **settings)
            refs = [simulate(people, shop, "fixed", weights, event_seed, **settings)
                    for weights in vectors]
            results.append((len(people), ai, refs))
        for customer_weight, store_weight in score_ratios:
            def value(result, customer_count):
                return (customer_weight * result.satisfaction + store_weight * result.store_satisfaction
                        + .10 * result.served / customer_count
                        - .10 * result.avg_total_wait / (result.avg_total_wait + 25.))
            pairs = [(value(ai, size), mean(value(ref, size) for ref in refs))
                     for size, ai, refs in results]
            summary = _percent_summary(pairs, random.Random(seed + len(conditions) + 2026), draws)
            conditions.append({"acceptance_multiplier": acceptance,
                               "choice_time_multiplier": choice_time,
                               "customer_weight": customer_weight, "store_weight": store_weight,
                               **summary,
                               "meets_threshold": summary["lower_95_percent"] > minimum_gain_percent})
    return {"reference_count": count, "days": len(scenarios),
            "minimum_gain_percent": minimum_gain_percent,
            "conditions": conditions,
            "positive_average_count": sum(c["relative_gain_percent"] > 0 for c in conditions),
            "meets_threshold_count": sum(c["meets_threshold"] for c in conditions),
            "condition_count": len(conditions),
            "note": "27 researcher-chosen combinations, not a real-world probability or simultaneous confidence interval"}


def paired_validation(rows, *, seed=42, draws=2000):
    days = {}
    for row in rows:
        days.setdefault(row["run"], {})[row["policy"]] = row
    if not days or any(set(day) != {"none", "popular", "customer", "fixed", "gemini"}
                       for day in days.values()):
        raise ValueError("Every test day must contain all five policies")
    rng = random.Random(seed)
    output = {"days": len(days), "comparisons": {}}
    for name, base in (("gemini_vs_fixed", "fixed"),
                       ("fixed_vs_none", "none"), ("gemini_vs_none", "none")):
        report = {}
        for key in METRICS:
            values = [float(day["gemini" if name.startswith("gemini") else "fixed"][key]) -
                      float(day[base][key]) for day in days.values()]
            # Keep the whole simulated day as the resampling unit; customers within
            # a day share queues, stock, and a store and are not independent.
            resampled = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
            report[key] = {"mean_delta": mean(values), "lower_95": resampled[int(.025 * draws)],
                           "upper_95": resampled[min(draws - 1, int(.975 * draws))],
                           "better_days": sum((v > 1e-10 if METRICS[key][1] == "높을수록 좋음"
                                               else v < -1e-10) for v in values),
                           "equal_days": sum(abs(v) <= 1e-10 for v in values)}
        output["comparisons"][name] = report
    return output
