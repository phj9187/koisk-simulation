from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import yaml

SCENARIOS = {
    'balanced': {'customer': [.45, .25, .30], 'store': [.75, .25]},
    'time_sensitive': {'customer': [.25, .15, .60], 'store': [.75, .25]},
    'waste_sensitive': {'customer': [.45, .25, .30], 'store': [.40, .60]},
}

@dataclass
class Config:
    seed: int = 42
    stores: int = 8
    source: str = 'synthetic'
    survey_path: str = ''
    visitors: int | None = 30
    train_days: int = 10
    validation_days: int = 2
    evaluation_days: int = 3
    repetitions: int | None = 1
    train_lambdas: int = 3
    validation_lambdas: list = field(default_factory=lambda: [0., .25, .5, .75, 1.])
    evaluation_lambdas: list = field(default_factory=lambda: [i/8 for i in range(9)])
    scenarios: dict = field(default_factory=lambda: {k:{n:list(v) for n,v in s.items()} for k,s in SCENARIOS.items()})
    models: list = field(default_factory=lambda: ['fixed', 'contextual'])
    sigma: float = .5
    checkpoint_every: int = 2
    workers: int = 1
    raw_logs: bool = False
    plots: bool = True
    wait_scale: float = 25.

    def validate(self, require_source=True):
        for key in ('stores','train_days','validation_days','evaluation_days','checkpoint_every','workers','train_lambdas'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f'{key} must be a positive integer')
        if self.workers>4 or self.train_lambdas<2: raise ValueError('workers <= 4; train_lambdas >= 2')
        if type(self.seed) is not int or self.seed<0: raise ValueError('invalid seed')
        for key in ('visitors','repetitions'):
            v=getattr(self,key)
            if v is not None and (type(v) is not int or v<(0 if key=='visitors' else 1)): raise ValueError(f'invalid {key}')
        for key in ('sigma','wait_scale'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0: raise ValueError(f'invalid {key}')
        for key in ('raw_logs','plots'):
            if type(getattr(self,key)) is not bool: raise ValueError(f'{key} must be boolean')
        if self.source not in ('synthetic','empirical'): raise ValueError('unknown source')
        if self.source=='synthetic' and self.stores%8: raise ValueError('synthetic stores must be a multiple of 8')
        if self.source=='empirical' and require_source and not Path(self.survey_path).is_file():
            raise ValueError('verified survey file required; no synthetic fallback')
        if not self.models or len(self.models)!=len(set(self.models)) or not set(self.models)<={'fixed','contextual'}: raise ValueError('invalid models')
        for grid in (self.validation_lambdas,self.evaluation_lambdas):
            if not grid or len(set(grid))!=len(grid) or any(not math.isfinite(x) or not 0<=x<=1 for x in grid): raise ValueError('invalid lambda grid')
        if not set(self.validation_lambdas)<=set(self.evaluation_lambdas): raise ValueError('final grid must include validation grid for checking selected policies')
        if not self.scenarios: raise ValueError('at least one scenario required')
        for name,s in self.scenarios.items():
            if not name or not name.replace('_','').isalnum() or set(s)!={'customer','store'}: raise ValueError('invalid scenario')
            for key,n in (('customer',3),('store',2)):
                values=s[key]
                if len(values)!=n or any(not math.isfinite(x) or x<0 for x in values) or not math.isclose(sum(values),1.): raise ValueError('scenario coefficients must sum to 1')
        return self

    def stage(self,day):
        size=self.train_days+self.validation_days+self.evaluation_days
        rep,offset=divmod(day-1,size)
        phase='train' if offset<self.train_days else 'validation' if offset<self.train_days+self.validation_days else 'evaluation'
        return rep+1,phase,offset+1

    def complete(self,day):
        return self.repetitions is not None and day>=self.repetitions*(self.train_days+self.validation_days+self.evaluation_days)

    def to_dict(self): return asdict(self)

    @classmethod
    def load(cls,path): return cls(**yaml.safe_load(Path(path).read_text(encoding='utf-8'))).validate()
