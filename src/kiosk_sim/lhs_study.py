"""Synchronous multi-store LHS study engine for the desktop application.

All outcomes are synthetic.  The engine deliberately keeps one online model per
store so each store learns only from its own accumulated simulated days.
"""
from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .ai.contextual import OnlineContextualBandit
from .ai.fixed_bayesian import BayesianFixedOptimizer
from .generators import ALLERGENS, INGREDIENTS, customers
from .models import Menu, Store
from .recommenders import FIXED_WEIGHTS
from .simulator import simulate
from .visitor_distribution import sample_store_means, daily_visitors, distribution_metadata

MODEL_LABELS = {
    "no_recommendation": "추천 없음",
    "baseline_fixed": "일반 고정 가중치",
    "ai_fixed": "AI 고정 최적화",
    "contextual": "상황별 AI (LinUCB)",
}


@dataclass
class StudyConfig:
    store_count: int = 64
    visitor_min: int = 20
    visitor_max: int = 160
    seed: int = 20261002
    min_days: int = 20
    max_days: int = 160
    stability_window: int = 8
    tolerance: float = 0.003
    required_stable_rounds: int = 3
    lhs_strata: int = 8
    visitor_distribution: str = 'uniform_scenario'
    survey_path: str = ''


@dataclass
class StoreCase:
    store_id: int
    features: dict[str, float | str]
    template: Store
    fixed: BayesianFixedOptimizer
    contextual: OnlineContextualBandit
    records: list[dict] = field(default_factory=list)


def _lhs_values(count: int, low: float, high: float, strata: int, rng: random.Random) -> list[float]:
    """Latin hypercube values: each of the eight strata is used equally often."""
    values = []
    for index in range(count):
        bucket = index % strata
        values.append(low + (high - low) * (bucket + rng.random()) / strata)
    rng.shuffle(values)
    return values


def _make_store(store_id: int, values: dict[str, float | str], rng: random.Random) -> Store:
    purpose = str(values["purpose"])
    menu_count = int(round(float(values["menu_count"])))
    center = (float(values["taste_center_1"]), float(values["taste_center_2"]), float(values["taste_center_3"]))
    diversity = float(values["menu_diversity"])
    menus = []
    for menu_id in range(menu_count):
        price = round(max(5., rng.gauss(float(values["price_level"]), 1.2)), 2)
        meal_mean = .76 if purpose == "meal" else .24 if purpose == "dessert" else .50
        meal_fit = max(.05, min(.95, rng.gauss(meal_mean, .14)))
        menus.append(Menu(
            f"L{store_id:03d}_M{menu_id + 1}",
            tuple(max(0., min(1., rng.gauss(item, diversity))) for item in center),
            price, round(price * float(values["cost_ratio"]), 2),
            max(2., rng.gauss(float(values["cook_minutes"]), 1.1)),
            rng.choice(INGREDIENTS), rng.choice((1, 1, 2)), rng.choice(ALLERGENS),
            max(.05, min(1., rng.gauss(float(values["expiry_risk"]), .08))),
            meal_fit, max(.05, min(.95, rng.gauss(1. - meal_fit, .11))),
        ))
    stock = {ingredient: int(round(float(values["stock_level"]))) for ingredient in INGREDIENTS}
    return Store(menus, stock, int(round(float(values["cooks"]))), purpose)


def create_lhs_stores(config: StudyConfig) -> list[StoreCase]:
    if config.store_count < config.lhs_strata or config.store_count % config.lhs_strata:
        raise ValueError(f"가게 수는 LHS {config.lhs_strata}분할의 배수여야 합니다.")
    rng = random.Random(config.seed)
    specs = {
        "menu_count": (4, 12), "cooks": (1, 3), "price_level": (8, 20),
        "cost_ratio": (.35, .65), "cook_minutes": (5, 18), "expiry_risk": (.10, 1.00),
        "stock_level": (28, 70), "menu_diversity": (.08, .34),
        "taste_center_1": (.20, .80), "taste_center_2": (.20, .80), "taste_center_3": (.20, .80),
    }
    columns = {name: _lhs_values(config.store_count, low, high, config.lhs_strata, rng)
               for name, (low, high) in specs.items()}
    visitor_samples = sample_store_means(config.store_count, config.lhs_strata, rng)
    purposes = (["meal", "dessert", "mixed"] * ((config.store_count + 2) // 3))[:config.store_count]
    rng.shuffle(purposes)
    cases = []
    for index in range(config.store_count):
        features = {name: round(values[index], 4) for name, values in columns.items()}
        features["purpose"] = purposes[index]
        features['visitor_mean'], features['visitor_bin'] = visitor_samples[index]
        features['visitor_lhs_replicate'] = index // config.lhs_strata + 1
        cases.append(StoreCase(index + 1, features, _make_store(index + 1, features, rng),
                               BayesianFixedOptimizer(seed=config.seed + index * 37),
                               OnlineContextualBandit()))
    return cases


class LHSStudy:
    def __init__(self, config: StudyConfig):
        if config.visitor_min < 1 or config.visitor_max < config.visitor_min:
            raise ValueError("방문자 범위를 확인해 주세요.")
        self.config = config
        self.sampling_audit = {}
        if config.visitor_distribution == 'empirical_donors_2025':
            from .empirical_stores import empirical_cases
            self.cases, self.sampling_audit = empirical_cases(config)
        else:
            self.cases = create_lhs_stores(config)
        self.day = 0
        self.stop_requested = False
        self.stop_reason = ""
        self.stable_rounds = 0
        self.aggregate_history: list[dict] = []

    def request_stop(self):
        self.stop_requested = True

    def _daily_store(self, case: StoreCase, day: int) -> Store:
        rng = random.Random(self.config.seed * 100_003 + case.store_id * 911 + day)
        store = copy.deepcopy(case.template)
        store.stock = {name: max(1, int(round(amount * rng.uniform(.65, 1.10)))) for name, amount in store.stock.items()}
        store.cook_free_at = [0.] * store.cooks
        return store

    @staticmethod
    def _row(day, case, name, result, visitors, weights=None):
        return {"day": day, "store_id": case.store_id, "model": name, "model_label": MODEL_LABELS[name],
                "visitors": visitors, "score": result.score, "customer_satisfaction": result.satisfaction,
                "store_satisfaction": result.store_satisfaction, "profit": result.profit,
                "wait_minutes": result.avg_total_wait, "served": result.served,
                "acceptance_rate": result.accept_rate,
                "weights": "|".join(f"{value:.4f}" for value in weights) if weights else ""}

    def run_day(self) -> dict:
        """Run one synchronized date: every store finishes before the next date."""
        self.day += 1
        for case in self.cases:
            rng = random.Random(self.config.seed + self.day * 1009 + case.store_id * 17)
            visitors = (int(round(float(case.features['visitor_mean'])))
                        if self.config.visitor_distribution == 'empirical_donors_2025' else
                        daily_visitors(float(case.features['visitor_mean']), rng)
                        if self.config.visitor_distribution == 'survey_general_restaurants_2025'
                        else rng.randint(self.config.visitor_min, self.config.visitor_max))
            people = customers(visitors, rng, case.template) if visitors else []
            seed = self.config.seed + self.day * 10_000 + case.store_id
            store = self._daily_store(case, self.day)
            observed_hours = case.features.get('opening_hours')
            hours = int(observed_hours) if observed_hours and 1 <= observed_hours <= 24 else 8
            no = simulate(people, store, "none", seed=seed, hours=hours)
            base = simulate(people, store, "fixed", FIXED_WEIGHTS, seed=seed, hours=hours)
            fixed_weights = case.fixed.propose()
            fixed = simulate(people, store, "fixed", fixed_weights, seed=seed, hours=hours)
            feedback = []
            contextual = simulate(people, store, "fixed", case.contextual, seed=seed, feedback=feedback, hours=hours)
            case.fixed.observe(fixed_weights, fixed.score)
            for item in feedback:
                case.contextual.observe(item["bandit_features"], .6 * item["customer_satisfaction"] + .4 * item["store_satisfaction"])
            rows = [self._row(self.day, case, "no_recommendation", no, visitors),
                    self._row(self.day, case, "baseline_fixed", base, visitors, FIXED_WEIGHTS),
                    self._row(self.day, case, "ai_fixed", fixed, visitors, fixed_weights),
                    self._row(self.day, case, "contextual", contextual, visitors, case.contextual.base_weights)]
            case.records.extend(rows)
        summary = self._summarize_day()
        self.aggregate_history.append(summary)
        if self.day >= self.config.min_days and self._is_stable():
            self.stable_rounds += 1
        else:
            self.stable_rounds = 0
        if self.stop_requested:
            self.stop_reason = "사용자가 요청한 현재 날짜 완료"
        elif self.stable_rounds >= self.config.required_stable_rounds:
            self.stop_reason = "모든 모델의 두 만족도 평균이 안정화됨"
        elif self.day >= self.config.max_days:
            self.stop_reason = "안전 최대 날짜 도달"
        return summary

    def finished(self):
        return bool(self.stop_reason)

    def _summarize_day(self):
        rows = [row for case in self.cases for row in case.records if row["day"] == self.day]
        summary = {"day": self.day, "stable_rounds": self.stable_rounds}
        for model in MODEL_LABELS:
            selected = [row for row in rows if row["model"] == model]
            for key in ("score", "customer_satisfaction", "store_satisfaction"):
                weights = {c.store_id: c.features.get('analysis_weight', 1.) for c in self.cases}
                summary[f"{model}_{key}"] = sum(row[key]*weights[row['store_id']] for row in selected) / sum(weights[row['store_id']] for row in selected)
        return summary

    def _is_stable(self):
        recent = self.aggregate_history[-self.config.stability_window:]
        if len(recent) < self.config.stability_window:
            return False
        for model in MODEL_LABELS:
            for metric in ("customer_satisfaction", "store_satisfaction"):
                values = [entry[f"{model}_{metric}"] for entry in recent]
                if max(values) - min(values) > self.config.tolerance:
                    return False
        return True

    def analysis(self) -> dict:
        latest = self.aggregate_history[-1] if self.aggregate_history else {}
        if not latest:
            return {"text": "실행 대기 중"}
        ranking = sorted(MODEL_LABELS, key=lambda model: latest[f"{model}_score"], reverse=True)
        baseline = latest["no_recommendation_score"]
        return {"leader": ranking[0], "leader_label": MODEL_LABELS[ranking[0]],
                "leader_gain_pct": (latest[f"{ranking[0]}_score"] / baseline - 1) * 100 if baseline else 0,
                "stable_rounds": self.stable_rounds, "text":
                f"{self.day}일차 평균 종합점수 1위: {MODEL_LABELS[ranking[0]]} "
                f"(추천 없음 대비 {((latest[f'{ranking[0]}_score'] / baseline - 1) * 100) if baseline else 0:+.2f}%). "
                f"안정화 연속 판정: {self.stable_rounds}/{self.config.required_stable_rounds}.\n"
                "모든 수치는 가상 고객·가게 데이터로부터 나온 시뮬레이션 결과이며 실제 고객 효과를 뜻하지 않습니다."}

    def export(self, desktop: Path | None = None) -> Path:
        desktop = desktop or Path.home() / "Desktop"
        target = desktop / f"Kiosk_LHS_Results_{datetime.now():%Y%m%d_%H%M%S}"
        target.mkdir(parents=True, exist_ok=True)
        rows = [row for case in self.cases for row in case.records]
        with (target / "daily_store_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["day"])
            writer.writeheader(); writer.writerows(rows)
        with (target / "store_features.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            features = [{"store_id": case.store_id, **case.features} for case in self.cases]
            writer = csv.DictWriter(handle, fieldnames=list(features[0]))
            writer.writeheader(); writer.writerows(features)
        with (target / "daily_average.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.aggregate_history[0]) if self.aggregate_history else ['day'])
            writer.writeheader(); writer.writerows(self.aggregate_history)
        payload = {"config": asdict(self.config), "days_completed": self.day, "stop_reason": self.stop_reason,
                   "sampling_audit": getattr(self, 'sampling_audit', {}),
                   "lhs_method": "8-bin stratification for existing features; replicated 8-point probability-space LHS for visitor means",
                   "public_data_note": "Survey-based store means; within-bin, upper-tail and daily variation are modeling assumptions. Living population is not restaurant visits.",
                   "visitor_distribution": distribution_metadata(),
                   "other_feature_distributions": {
                       "continuous_store_features": "8-bin uniform stratification over assumed ranges, not fitted empirical distributions",
                       "integer_features": "rounded continuous draws; cook/menu counts are not empirically calibrated",
                       "purpose": "approximately equal meal/dessert/mixed allocation, assumed",
                       "menus": "clipped Gaussian variation and categorical draws, assumed",
                       "customers": "clipped Gaussian taste; uniform sensitivities/budget/openness; assumed categories",
                       "daily_stock": "uniform multiplicative factor 0.65 to 1.10, assumed"},
                   "models": MODEL_LABELS, "model_evaluation_counts": {name: self.day * len(self.cases) for name in MODEL_LABELS},
                   "stores": [{"store_id": case.store_id, "features": case.features,
                               "ai_fixed": case.fixed.as_dict(), "contextual": case.contextual.as_dict()} for case in self.cases]}
        if self.config.visitor_distribution == 'empirical_donors_2025':
            payload['lhs_method'] = self.sampling_audit['method']
            payload['public_data_note'] = self.sampling_audit['assumptions']
            payload['visitor_distribution'] = 'weighted empirical store means; fixed daily counts; no invented daily noise'
            payload['other_feature_distributions'] = {'observed': 'whole-row donor fields retained together',
                'used': 'visitor mean, industry purpose mapping, representative menu prices (KRW/1000), integer opening hours',
                'not_directly_used': 'seats,spend,total staffing,waste,expenses,channel shares retained as metadata; staffing bounds assumed cooks',
                'assumed': 'customer behavior,budget,taste,menu count,cost ratio,cooks allocation,cook times,expiry,stock,arrival shape; missing opening hours -> 8; missing menu price -> original assumed range',
                'warning': 'Observed inputs do not validate the satisfaction function or model effectiveness.'}
        (target / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (target / "README.txt").write_text(
            "LHS 다점포 키오스크 시뮬레이션 결과\n\n"
            "이 결과는 가상 데이터 기반입니다. 실제 고객 또는 실제 매장의 효과로 해석하면 안 됩니다.\n"
            "서울시 상권분석서비스의 추정 매출 건수는 거래 건수이며 개별 식당 방문자 수가 아닙니다.\n"
            "방문 고객 분포 모드에서는 2025 일반음식점 조사 구간 비중을 사용합니다. 구간 내 균등분포, 100명 이상 꼬리 분포, 날짜별 포아송 변동은 모형 가정입니다.\n"
            "기존 균등 모드에서 visitor_min/max는 사용자가 명시한 모형 가정입니다.\n"
            f"종료 사유: {self.stop_reason}\n", encoding="utf-8")
        if self.config.visitor_distribution == 'empirical_donors_2025':
            (target / 'README.txt').write_text('Empirical store inputs, synthetic outcomes.\n'
                'See study.json for source hash, exclusions, sampling method and unvalidated assumptions.\n'
                'Whole-row donor sampling preserves observed tuples, not exact population correlations in a finite sample.\n'
                'Daily visitor counts are fixed; zero counts are retained. No claim of measured daily variability.\n'
                'Stability of synthetic averages is not proof of optimality or real-world effectiveness.\n', encoding='utf-8')
        return target
