"""Iterative Gemini proposals driven by simulated customer and shop feedback."""
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request

from ..recommenders import FIXED_WEIGHTS
from ..simulator import simulate
from ..robust import ASSUMPTIONS, comparison


def validate_weights(values):
    if not isinstance(values, list) or len(values) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or
        not math.isfinite(v) or v < 0 for v in values
    ) or not .98 <= sum(values) <= 1.02:
        raise ValueError("Gemini must return four nonnegative weights summing to 1")
    total = sum(values)
    return tuple(v / total for v in values)


def request_weights(current, outcome, feedback, *, api_key, model, experiment_history=None):
    prompt = {
        "task": "Propose three DISTINCT candidate sets of four recommendation weights based on simulated per-customer feedback."
                " The order is taste_and_visit_purpose, expiry_risk, store_margin, wait_penalty."
                " Balance customer satisfaction and store satisfaction."
                " For each candidate return four numbers >=0 summing to 1. Include exploration"
                " far from current weights as well as one incremental proposal. These are hypothetical customers."
                " Do not infer unobserved counterfactual outcomes.",
        "current_weights": current,
        "day_result": {"customer_satisfaction": outcome.satisfaction,
                       "profit": outcome.profit, "wait": outcome.avg_wait,
                       "waste": outcome.waste, "kiosk_choice_minutes": outcome.avg_decision,
                       "kiosk_queue_minutes": outcome.avg_kiosk_queue,
                       "store_satisfaction_assumption": outcome.store_satisfaction},
        "customer_feedback": feedback[:40],
        "experiment_history": [{key: value for key, value in trial.items()
                                if key != "detailed_runs"} for trial in (experiment_history or [])],
        "instruction": "Use the measured scores and tradeoffs of ALL previous candidates to infer which directions improve the objective. Compare accepted and rejected trials. Propose one local refinement, one alternative supported by the observations, and one exploration candidate. Do not claim these simulations are real customers. Explain your reasoning briefly in the rationale field.",
    }
    body = {"contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": {"type": "OBJECT", "properties": {
                                     "candidates": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                                         "weights": {"type": "ARRAY", "items": {"type": "NUMBER"}},
                                         "rationale": {"type": "STRING"}},
                                         "required": ["weights"]}}},
                                     "required": ["candidates"]}}}
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + urllib.parse.quote(model, safe="-") + ":generateContent"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "x-goog-api-key": api_key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            data = json.load(response)
        parsed = json.loads("".join(part.get("text", "") for part in data["candidates"][0]["content"]["parts"]))
        candidates = parsed["candidates"]
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
            raise ValueError("Gemini must propose one to three candidates")
        return [{"weights": validate_weights(item["weights"]),
                 "rationale": str(item.get("rationale", ""))[:500]} for item in candidates]
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read(8192)).get("error", {}).get("message", "")
        except (ValueError, UnicodeDecodeError):
            message = ""
        # Google error messages sometimes include model names, not credentials.
        safe_message = message[:280].replace(api_key, "[redacted]") if message else "Check model and access tier."
        raise RuntimeError(f"Gemini API HTTP {exc.code} ({model}): {safe_message}") from exc
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Gemini request or response failed: {exc}") from exc


def train_gemini(scenarios, *, api_key=None, model="gemini-3.5-flash-lite", rounds=4,
                 propose=request_weights, **simulation_options):
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing; configure your own Gemini API key before running")
    if rounds < 1 or not scenarios:
        raise ValueError("Need at least one training round and training scenario")
    current = FIXED_WEIGHTS
    history = []
    experiments = []
    # Evaluate proposals on a training day under several hypothetical conditions.
    for step in range(rounds):
        people, shop, seed = scenarios[step % len(scenarios)]
        feedback = []
        before = simulate(people, shop, "gemini", current, seed, feedback=feedback,
                          **simulation_options)
        proposed = propose(current, before, feedback, api_key=api_key, model=model,
                           experiment_history=[{key: value for key, value in trial.items()
                                                if key != "detailed_runs"} for trial in experiments])
        # Keep a single-vector proposer usable in offline tests.
        if len(proposed) == 4 and all(isinstance(v, (int, float)) for v in proposed):
            proposed = [proposed]
        if not 1 <= len(proposed) <= 3:
            raise ValueError("Need one to three Gemini proposals")
        rationales = [str(item.get("rationale", ""))[:500] if isinstance(item, dict) else ""
                      for item in proposed]
        candidates = [validate_weights(list(item["weights"] if isinstance(item, dict) else item))
                      for item in proposed]
        # Compare across several plausible decision-time assumptions instead of
        # accepting one researcher's single weighted score as the truth.
        evaluation_days = [scenarios[(step * 6 + offset) % len(scenarios)]
                           for offset in range(min(6, len(scenarios)))]
        params_list = [dict(simulation_options, decision_multiplier=multiplier,
                            decision_penalty=penalty)
                       for multiplier, penalty in (ASSUMPTIONS[0], ASSUMPTIONS[4], ASSUMPTIONS[-1])]
        baselines = [(day, params, simulate(day[0], day[1], "gemini", current, day[2], **params))
                     for day in evaluation_days for params in params_list]
        scored = []
        round_records = []
        for candidate, rationale in zip(candidates, rationales):
            pairs = [(simulate(day[0], day[1], "gemini", candidate, day[2], **params), old, len(day[0]))
                     for day, params, old in baselines]
            robust_result = comparison(pairs)
            scored.append((candidate, robust_result))
            record = {"round": step + 1, "weights": candidate, "rationale": rationale,
                      "mean_score": sum(new.score for new, _, _ in pairs) / len(pairs),
                      "mean_baseline_score": sum(old.score for _, old, _ in pairs) / len(pairs),
                      "mean_customer_satisfaction": sum(new.satisfaction for new, _, _ in pairs) / len(pairs),
                      "mean_store_satisfaction": sum(new.store_satisfaction for new, _, _ in pairs) / len(pairs),
                      "mean_profit": sum(new.profit for new, _, _ in pairs) / len(pairs),
                      "mean_wait_minutes": sum(new.avg_total_wait for new, _, _ in pairs) / len(pairs),
                      "mean_waste_risk": sum(new.waste for new, _, _ in pairs) / len(pairs),
                      "detailed_runs": [{"day": (step * 6 + i // len(params_list)) % len(scenarios) + 1,
                                       "assumption": i % len(params_list) + 1,
                                       "score": new.score, "baseline_score": old.score,
                                       "customer_satisfaction": new.satisfaction,
                                       "store_satisfaction": new.store_satisfaction,
                                       "profit": new.profit, "total_wait_minutes": new.avg_total_wait,
                                       "waste_risk": new.waste}
                                      for i, (new, old, _) in enumerate(pairs)],
                      **robust_result}
            round_records.append(record)
        proposed, robust = max(scored, key=lambda item:
                               (item[1]["win_share"] - item[1]["loss_share"], item[1]["median_margin"]))
        accepted = robust["win_share"] > robust["loss_share"]
        for record in round_records:
            record["accepted"] = accepted and tuple(record["weights"]) == proposed
        experiments.extend(round_records)
        history.append({"round": step + 1, "feedback_count": len(feedback),
                        "candidates": len(candidates),
                        "win_share": round(robust["win_share"], 4),
                        "loss_share": round(robust["loss_share"], 4),
                        "accepted": accepted, "weights": proposed})
        if accepted:
            current = proposed
    return current, {"model": model, "api_calls": rounds, "accepted_updates": sum(h["accepted"] for h in history),
                     "history": history, "experiments": experiments}
