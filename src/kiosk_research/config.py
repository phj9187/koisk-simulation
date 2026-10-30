from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import yaml
from kiosk_research.design import screening_design,interior_validation_design

SCENARIOS = {
    'balanced': {'customer': [.45, .25, .30], 'store': [.75, .25]},
    'time_sensitive': {'customer': [.25, .15, .60], 'store': [.75, .25]},
    'waste_sensitive': {'customer': [.45, .25, .30], 'store': [.40, .60]},
}

@dataclass
class Config:
    algorithm: str = 'random_search'
    decision_time_mode: str = 'legacy'
    lambda_search: str = 'fixed'
    coarse_train_days: int | None = None
    global_probe_every: int = 5
    seed: int = 42
    stores: int = 8
    source: str = 'synthetic'
    survey_path: str = ''
    visitors: int | None = 30
    visitor_variation: str = 'fixed'
    sales_profile_path: str = ''
    sales_quarter: str = ''
    arrival_start_hour: int = 11
    train_days: int = 10
    validation_days: int = 2
    evaluation_days: int = 3
    repetitions: int | None = 1
    train_lambdas: int = 3
    validation_lambdas: list = field(default_factory=lambda: [0., .25, .5, .75, 1.])
    evaluation_lambdas: list = field(default_factory=lambda: [i/8 for i in range(9)])
    scenarios: dict = field(default_factory=screening_design)
    models: list = field(default_factory=lambda: ['fixed', 'contextual'])
    sigma: float = .5
    checkpoint_every: int = 2
    workers: int = 1
    raw_logs: bool = False
    plots: bool = True
    wait_scale: float = 25.
    checkpoint_keep: int = 2
    auto_cleanup: bool = True
    cross_evaluation: dict = field(default_factory=interior_validation_design)
    continual: bool = False
    early_stopping: bool = False
    convergence_window: int = 100
    convergence_tolerance: float = .01
    convergence_probe_every: int = 5
    realism: dict = field(default_factory=dict)
    learning: dict = field(default_factory=dict)
    diagnostics: bool = False
    repetition_seeds: bool = True

    def validate(self, require_source=True):
        from kiosk_research.realism import options
        settings=options(self)
        from kiosk_research.actor_critic import learning_options
        learning_options(self.learning)
        if settings.population_path and require_source and not Path(settings.population_path).is_file():raise ValueError('population file not found')
        if settings.monthly_profile_path and require_source:
            from kiosk_research.realism import monthly_multiplier
            monthly_multiplier(settings,settings.calendar(1))
        for key in ('diagnostics','repetition_seeds'):
            if type(getattr(self,key)) is not bool:raise ValueError('invalid '+key)
        if settings.inventory_horizon_days>1 and self.algorithm!='actor_critic':raise ValueError('multiday inventory requires actor_critic')
        if self.algorithm not in ('random_search','actor_critic'):raise ValueError('unknown algorithm')
        if self.decision_time_mode not in ('legacy','equal'):raise ValueError('unknown decision time mode')
        if self.lambda_search not in ('fixed','pareto_two_stage'):raise ValueError('unknown lambda search')
        if type(self.global_probe_every) is not int or self.global_probe_every<1:raise ValueError('invalid global probe interval')
        if self.lambda_search=='pareto_two_stage':
            if self.algorithm!='actor_critic' or self.train_days<2 or self.train_lambdas!=3:raise ValueError('two-stage search needs neural training, at least two train dates and three lambdas')
            if self.coarse_train_days is not None and (type(self.coarse_train_days) is not int or not 1<=self.coarse_train_days<self.train_days):raise ValueError('invalid coarse training dates')
            if not set(self.models)&{'fixed','contextual'}:raise ValueError('search needs a learned model')
        if 'no_recommendation' in self.models and self.decision_time_mode!='equal':raise ValueError('no-recommendation comparison requires equal decision times')
        for key in ('stores','train_days','validation_days','evaluation_days','checkpoint_every','workers','train_lambdas','checkpoint_keep'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f'{key} must be a positive integer')
        if self.checkpoint_keep<2: raise ValueError('at least two recovery checkpoints required')
        if self.workers>4 or self.train_lambdas<2: raise ValueError('workers <= 4; train_lambdas >= 2')
        if type(self.seed) is not int or self.seed<0: raise ValueError('invalid seed')
        for key in ('visitors','repetitions'):
            v=getattr(self,key)
            if v is not None and (type(v) is not int or v<(0 if key=='visitors' else 1)): raise ValueError(f'invalid {key}')
        for key in ('sigma','wait_scale'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0: raise ValueError(f'invalid {key}')
        for key in ('raw_logs','plots','auto_cleanup','continual','early_stopping'):
            if type(getattr(self,key)) is not bool: raise ValueError(f'{key} must be boolean')
        if self.early_stopping and not self.continual:raise ValueError('early stopping requires continual learning')
        for key in ('convergence_window','convergence_probe_every'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError('invalid '+key)
        if self.convergence_window<100:raise ValueError('at least 100 training dates required')
        if self.convergence_probe_every>self.convergence_window//2:raise ValueError('probe interval too long')
        if not math.isfinite(self.convergence_tolerance) or not 0<self.convergence_tolerance<=.1:raise ValueError('invalid convergence tolerance')
        if self.source not in ('synthetic','empirical'): raise ValueError('unknown source')
        if self.visitor_variation not in ('fixed','poisson'):raise ValueError('unknown visitor variation')
        if type(self.arrival_start_hour) is not int or not 0<=self.arrival_start_hour<=23:raise ValueError('invalid arrival start hour')
        if self.sales_profile_path and require_source and not Path(self.sales_profile_path).is_file():raise ValueError('sales profile file not found')
        if self.source=='synthetic' and self.stores%8: raise ValueError('synthetic stores must be a multiple of 8')
        if self.source=='empirical' and require_source and not Path(self.survey_path).is_file():
            raise ValueError('verified survey file required; no synthetic fallback')
        if not self.models or len(self.models)!=len(set(self.models)) or not set(self.models)<={'fixed','contextual','no_recommendation'}: raise ValueError('invalid models')
        if self.early_stopping and not set(self.models)&{'fixed','contextual'}:raise ValueError('convergence needs a learned model')
        for grid in (self.validation_lambdas,self.evaluation_lambdas):
            if not grid or len(set(grid))!=len(grid) or any(not math.isfinite(x) or not 0<=x<=1 for x in grid): raise ValueError('invalid lambda grid')
        if not set(self.validation_lambdas)<=set(self.evaluation_lambdas): raise ValueError('final grid must include validation grid for checking selected policies')
        if not self.scenarios: raise ValueError('at least one scenario required')
        if set(self.cross_evaluation)&set(self.scenarios): raise ValueError("cross-evaluation names must be distinct")
        for name,s in {**self.scenarios,**self.cross_evaluation}.items():
            if not name or not name.replace('_','').isalnum() or set(s)!={'customer','store'}: raise ValueError('invalid scenario')
            for key,n in (('customer',3),('store',2)):
                values=s[key]
                if len(values)!=n or any(not math.isfinite(x) or x<0 for x in values) or not math.isclose(sum(values),1.): raise ValueError('scenario coefficients must sum to 1')
        return self

    def stage(self,day):
        size=self.cycle_days
        rep,offset=divmod(day-1,size)
        if self.lambda_search=='pareto_two_stage':
            stage=self.search_stage(day)
            return rep+1,'train' if stage in ('coarse','refine') else 'evaluation' if stage=='evaluation' else 'validation',offset+1
        phase='train' if offset<self.train_days else 'validation' if offset<self.train_days+self.validation_days else 'evaluation'
        return rep+1,phase,offset+1

    def complete(self,day):
        return self.repetitions is not None and day>=self.repetitions*self.cycle_days

    @property
    def cycle_days(self):
        return self.train_days+self.validation_days*(2 if self.lambda_search=='pareto_two_stage' else 1)+self.evaluation_days

    @property
    def coarse_days(self):return self.coarse_train_days or self.train_days//2

    @property
    def training_end_offset(self):return self.train_days+(self.validation_days if self.lambda_search=='pareto_two_stage' else 0)

    @property
    def validation_end_offset(self):return self.training_end_offset+self.validation_days

    @property
    def phase_boundaries(self):
        values=[self.training_end_offset,self.validation_end_offset]
        if self.lambda_search=='pareto_two_stage':values += [self.coarse_days,self.coarse_days+self.validation_days]
        return tuple(values)

    def search_stage(self,day):
        if self.lambda_search!='pareto_two_stage':return 'fixed'
        offset=(day-1)%self.cycle_days
        if offset<self.coarse_days:return 'coarse'
        if offset<self.coarse_days+self.validation_days:return 'selection'
        if offset<self.training_end_offset:return 'refine'
        if offset<self.validation_end_offset:return 'validation'
        return 'evaluation'

    def to_dict(self): return asdict(self)

    @classmethod
    def load(cls,path): return cls(**yaml.safe_load(Path(path).read_text(encoding='utf-8'))).validate()
