from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml


@dataclass
class CloudConfig:
    seed: int = 20261006
    store_count: int = 64
    lhs_strata: int = 8
    source_mode: str = 'empirical_donors_2025'
    survey_path: str = ''
    visitors_per_day: int | None = None
    models: list[str] = field(default_factory=lambda: ['fixed', 'contextual'])
    train_days: int = 40
    validation_days: int = 5
    evaluation_days: int = 10
    cycles: int | None = None
    checkpoint_every: int = 5
    workers: int = 4
    detailed_logs: bool = False
    plots: bool = True
    search_sigma: float = 0.5
    objective: dict = field(default_factory=lambda: {'customer': .45, 'store': .35, 'served': .10, 'wait': .10})

    def validate(self):
        for key in ('store_count', 'lhs_strata', 'checkpoint_every', 'workers', 'train_days', 'validation_days', 'evaluation_days'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.workers > 4: raise ValueError('workers must be at most 4')
        if type(self.seed) is not int or self.seed < 0: raise ValueError('seed must be a nonnegative integer')
        for key in ('visitors_per_day', 'cycles'):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < (0 if key == 'visitors_per_day' else 1)):
                raise ValueError(f'invalid {key}')
        for key in ('detailed_logs', 'plots'):
            if type(getattr(self, key)) is not bool: raise ValueError(f'{key} must be boolean')
        if not self.models or len(self.models) != len(set(self.models)) or not set(self.models) <= {'fixed', 'contextual'}:
            raise ValueError('select fixed and/or contextual')
        if not math.isfinite(self.search_sigma) or self.search_sigma <= 0: raise ValueError('invalid search_sigma')
        if set(self.objective) != {'customer', 'store', 'served', 'wait'} or any(not math.isfinite(v) or v < 0 for v in self.objective.values()) or not any(self.objective.values()):
            raise ValueError('invalid objective coefficients')
        if self.source_mode not in ('empirical_donors_2025', 'uniform_scenario'): raise ValueError('unsupported source mode')
        if self.source_mode == 'empirical_donors_2025' and not Path(self.survey_path).is_file():
            raise ValueError('verified KREI survey file required; synthetic fallback is never automatic')
        if self.source_mode == 'uniform_scenario' and (self.store_count < self.lhs_strata or self.store_count % self.lhs_strata):
            raise ValueError('synthetic store count must be a multiple of lhs_strata')
        return self

    def legacy(self):
        # Reuse only the empirical store generator and environmental assumptions.
        from kiosk_sim.lhs_study import StudyConfig
        return StudyConfig(store_count=self.store_count, lhs_strata=self.lhs_strata, seed=self.seed,
                           visitor_distribution=self.source_mode, survey_path=self.survey_path)

    def stage(self, day):
        size = self.train_days + self.validation_days + self.evaluation_days
        cycle, offset = divmod(day - 1, size)
        phase = 'train' if offset < self.train_days else 'validation' if offset < self.train_days + self.validation_days else 'evaluation'
        return cycle + 1, phase, offset + 1

    def complete(self, day):
        return self.cycles is not None and day >= self.cycles * (self.train_days + self.validation_days + self.evaluation_days)

    def as_dict(self): return asdict(self)

    @classmethod
    def load(cls, path): return cls(**yaml.safe_load(Path(path).read_text(encoding='utf-8'))).validate()
