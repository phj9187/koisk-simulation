"""Declared, optional behavioral and operating hypotheses, not fitted estimates."""
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
import copy
import math
import random


@dataclass
class Realism:
    budget_mode: str = 'legacy'
    budget_multiplier_low: float = .75
    budget_multiplier_high: float = 1.8
    inventory_mode: str = 'legacy'
    expected_purchase_rate: float = .7
    inventory_cover_days: float = 1.1
    start_date: str = '2026-01-05'
    weekday_multipliers: list = field(default_factory=lambda: [1.] * 7)
    monthly_profile_path: str = ''
    recommendation_count: int = 1
    acceptance_multiplier: float = 1.
    decision_multiplier: float = 1.
    decision_penalty: float = 0.
    choice_noise: float = .16
    consideration_mean: float | None = None
    salience_position: float = 0.
    salience_category: float = 0.
    salience_popularity: dict = field(default_factory=dict)
    taste_noise: float = 0.
    regular_taste_noise: float | None = None
    ai_taste_noise: float = 0.
    overload_penalty: float = 0.
    decision_rule: str = 'legacy'
    decision_base: float = .8
    decision_per_item: float = .25
    wait_visibility: str = 'full'
    wait_noise: float = 0.
    trust_learning: bool = False
    trust_rate: float = .1
    patience_minutes: float = 0.
    cook_time_cv: float = 0.
    demand_cv: float = 0.
    seats: int = 0
    dine_in_fraction: float = .6
    table_minutes: float = 25.
    repeat_fraction: float = 0.
    repeat_openness_bonus: float = .1
    inventory_horizon_days: int = 1
    shelf_life_days: int = 3
    reorder_target: float = 1.
    delivery_fill_rate: float = 1.
    holding_cost_per_unit: float = 0.
    disposal_cost_per_unit: float = 0.
    discount: float = .99
    population_path: str = ''
    population_area: str = ''
    population_strength: float = .3

    def validate(self):
        date.fromisoformat(self.start_date)
        if self.budget_mode not in ('legacy','price_relative'):raise ValueError('invalid budget_mode')
        if self.inventory_mode not in ('legacy','demand'):raise ValueError('invalid inventory_mode')
        if not 0<self.budget_multiplier_low<=self.budget_multiplier_high or not math.isfinite(self.budget_multiplier_high):raise ValueError('invalid budget range')
        if not 0<=self.expected_purchase_rate<=1:raise ValueError('invalid purchase rate')
        if not math.isfinite(self.inventory_cover_days) or self.inventory_cover_days<=0:raise ValueError('invalid inventory cover')
        if self.decision_rule not in ('legacy','inspected'):raise ValueError('invalid decision_rule')
        if self.wait_visibility not in ('full','menu_only','visible_queue'):raise ValueError('invalid wait_visibility')
        if type(self.trust_learning) is not bool:raise ValueError('invalid trust_learning')
        for key in ('consideration_mean','regular_taste_noise'):
            value=getattr(self,key)
            if value is not None and (not math.isfinite(value) or value<0):raise ValueError('invalid '+key)
        for key in ('taste_noise','ai_taste_noise','overload_penalty','decision_base','decision_per_item','wait_noise'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<0:raise ValueError('invalid '+key)
        if not 0<=self.trust_rate<=1:raise ValueError('invalid trust_rate')
        if any(not math.isfinite(v) for v in (self.salience_position,self.salience_category)):raise ValueError('invalid salience')
        if not isinstance(self.salience_popularity,dict) or any(not math.isfinite(v) for v in self.salience_popularity.values()):raise ValueError('invalid popularity')
        for key in ('recommendation_count','inventory_horizon_days','shelf_life_days'):
            value=getattr(self,key)
            if type(value) is not int or not 1 <= value <= (3 if key=='recommendation_count' else 30):
                raise ValueError('invalid '+key)
        if type(self.seats) is not int or self.seats<0:raise ValueError('invalid seats')
        for key in ('acceptance_multiplier','decision_penalty','choice_noise','patience_minutes','cook_time_cv','demand_cv','table_minutes','repeat_openness_bonus',
                    'reorder_target','holding_cost_per_unit','disposal_cost_per_unit'):
            value=getattr(self,key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('invalid '+key)
        if not math.isfinite(self.decision_multiplier) or self.decision_multiplier<=0:raise ValueError('invalid decision_multiplier')
        for key in ('delivery_fill_rate','population_strength','discount','dine_in_fraction','repeat_fraction'):
            if not math.isfinite(getattr(self,key)) or not 0<=getattr(self,key)<=1:raise ValueError('invalid '+key)
        if len(self.weekday_multipliers)!=7 or any(not math.isfinite(x) or x<0 for x in self.weekday_multipliers) or sum(self.weekday_multipliers)<=0:
            raise ValueError('seven nonnegative weekday multipliers required')
        return self

    def to_dict(self):return asdict(self)

    def calendar(self, episode, horizon_day=0):
        return date.fromisoformat(self.start_date)+timedelta(days=(episode-1)*self.inventory_horizon_days+horizon_day)


def options(config):return Realism(**config.realism).validate()


def monthly_multiplier(settings,calendar):
    if not settings.monthly_profile_path:return 1.
    import csv
    from pathlib import Path
    with Path(settings.monthly_profile_path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    values={int(row['month']):float(row['multiplier']) for row in rows}
    if len(rows)!=12 or set(values)!=set(range(1,13)) or any(not math.isfinite(v) or v<=0 for v in values.values()):
        raise ValueError('monthly scenario requires 12 unique positive month multipliers')
    return values[calendar.month]


def ingredient_costs(store):
    return {k:sum(m.cost/max(1,m.units) for m in store.menus if m.ingredient==k)/max(1,sum(m.ingredient==k for m in store.menus)) for k in store.stock}


class Inventory:
    """FIFO daily cohorts; replenishment arrives next morning, with exogenous fill risk.

    A finite multiday episode is the statistical unit. Inventory is reset only at
    episode boundaries, identically in all arms, and carried inside the episode.
    """
    def __init__(self, store, settings):
        self.settings=settings;self.base=dict(store.stock);self.costs=ingredient_costs(store)
        self.cohorts={k:[(v,0)] for k,v in store.stock.items()}
        self.pending={k:0 for k in self.base}

    def morning(self, store, seed):
        from dataclasses import replace
        shop=copy.deepcopy(store);rng=random.Random(seed);purchase=0.
        for k in self.base:
            delivered=sum(rng.random()<self.settings.delivery_fill_rate for _ in range(self.pending[k]))
            if delivered:self.cohorts[k].append((delivered,0))
            purchase+=delivered*self.costs[k]
        shop.stock={k:sum(v for v,age in rows) for k,rows in self.cohorts.items()}
        urgency={k:max((age/self.settings.shelf_life_days for quantity,age in rows if quantity),default=0.) for k,rows in self.cohorts.items()}
        shop.menus=[replace(m,expiry_risk=max(m.expiry_risk,urgency.get(m.ingredient,0.))) for m in shop.menus]
        shop.cook_free_at=[0.]*shop.cooks
        return shop,purchase

    def close(self, initial, terminal):
        waste=holding=0.;discarded=0
        for k,rows in self.cohorts.items():
            used=initial[k]-terminal[k];kept=[]
            for quantity,age in rows:
                consumed=min(quantity,used);used-=consumed;quantity-=consumed
                if age+1>=self.settings.shelf_life_days:
                    discarded+=quantity;waste+=quantity*(self.costs[k]+self.settings.disposal_cost_per_unit)
                elif quantity:kept.append((quantity,age+1))
            self.cohorts[k]=kept
            left=sum(v for v,age in kept);holding+=left*self.settings.holding_cost_per_unit
            self.pending[k]=max(0,math.ceil(self.base[k]*self.settings.reorder_target)-left)
        return {'expired_units':discarded,'inventory_loss':waste,'holding_cost':holding,
                'ordered_units':sum(self.pending.values()),'carried_units':sum(v for rows in self.cohorts.values() for v,age in rows)}
