from dataclasses import dataclass,field,asdict
from pathlib import Path
import hashlib,json,math
import numpy as np

CATEGORIES=('분식','중식','일식','양식')
POLICIES=('none','popular','customer_only','researcher_fixed','learned_fixed','learned_contextual')

def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(repr((seed,*parts)).encode()).digest()[:8],'big')
def rng_for(seed,*parts):return np.random.default_rng(seed_for(seed,*parts))
def data_file(name):return Path(__file__).parent/'data'/name

def softmax(x):
 x=np.asarray(x,dtype=float);v=np.exp(np.clip(x-x.max(),-60,0));return v/v.sum()

@dataclass(frozen=True)
class Dish:
 id:str
 name:str
 counter:int
 category:str
 price:float
 recipe:dict
 taste:tuple
 cook_minutes:float
 popularity:float
 vegetarian:bool
 allergens:tuple
 source_id:str
 observed_name:bool
 price_status:str

@dataclass
class Court:
 id:str
 dishes:list
 servers:list
 kiosks:int
 seats:int
 duration:float
 initial_stock:list
 ingredient_costs:dict
 provenance:dict

@dataclass
class Settings:
 seed:int=20261103
 menu_counts:list=field(default_factory=lambda:[12,32,64])
 attention_modes:list=field(default_factory=lambda:['bounded','full'])
 visitors:list=field(default_factory=lambda:[300,600])
 repetitions:int=5
 duration:float=180.
 kiosks:int=2
 servers_per_counter:int=2
 seats:int=200
 attention_limit:int=6
 recommendation_count:int=2
 decision_base:float=.4
 per_item_minutes:float=.04
 recommendation_overhead:float=.12
 payment_minutes:float=.25
 patience_minutes:float=15.
 taste_noise:float=.2
 choice_noise:float=.12
 openness:float=.65
 outside_utility:float=.15
 budget_low:float=4500.
 budget_high:float=11000.
 demand_cv:float=.15
 cook_cv:float=.25
 stock_cover:float=1.1
 expected_purchase_rate:float=.75
 perishable_fraction:float=.35
 wait_scale:float=20.
 customer_weights:list=field(default_factory=lambda:[.45,.25,.30])
 researcher_weights:list=field(default_factory=lambda:[.4,.15,.2,.15,.1])
 customer_tolerance:float=.01
 minimum_profit_gain:float=0.
 tradeoff:float=.5
 profit_scale:float=6000.
 constraint_penalty:float=2.
 generations:int=12
 population:int=8
 train_days:int=2
 validation_days:int=4
 evaluation_days:int=20
 validation_every:int=3
 initial_std:float=.8
 workers:int=2
 event_logs:bool=False
 def validate(self):
  for key in ('repetitions','kiosks','servers_per_counter','attention_limit','recommendation_count','generations','population','train_days','validation_days','evaluation_days','validation_every','workers'):
   if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError('invalid '+key)
  if self.population<3 or self.workers>8 or self.recommendation_count>3:raise ValueError('population >=3, workers <=8, top-k <=3')
  if not self.menu_counts or any(type(n) is not int or n<4 or n>128 or n%4 for n in self.menu_counts):raise ValueError('menu counts must be multiples of four, 4..128')
  if len(set(self.menu_counts))!=len(self.menu_counts) or len(set(self.visitors))!=len(self.visitors):raise ValueError('duplicate design levels')
  if not self.visitors or any(type(n) is not int or n<1 for n in self.visitors):raise ValueError('positive visitor means')
  if not self.attention_modes or not set(self.attention_modes)<=set(('bounded','full')) or len(set(self.attention_modes))!=len(self.attention_modes):raise ValueError('attention modes')
  for key in ('duration','patience_minutes','stock_cover','wait_scale','profit_scale','initial_std'):
   if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError('invalid '+key)
  for key in ('decision_base','per_item_minutes','recommendation_overhead','payment_minutes','taste_noise','choice_noise','demand_cv','cook_cv','constraint_penalty','customer_tolerance','minimum_profit_gain'):
   if not math.isfinite(getattr(self,key)) or getattr(self,key)<0:raise ValueError('invalid '+key)
  for key in ('openness','perishable_fraction','expected_purchase_rate','tradeoff'):
   if not math.isfinite(getattr(self,key)) or not 0<=getattr(self,key)<=1:raise ValueError('invalid '+key)
  if type(self.seats) is not int or self.seats<0:raise ValueError('seats')
  if not math.isfinite(self.outside_utility):raise ValueError('outside utility')
  if not 0<self.budget_low<=self.budget_high or not math.isfinite(self.budget_high):raise ValueError('budget')
  for values,n in ((self.customer_weights,3),(self.researcher_weights,5)):
   if len(values)!=n or any(not math.isfinite(v) or v<0 for v in values) or not math.isclose(sum(values),1.):raise ValueError('weights')
  if type(self.seed) is not int or self.seed<0 or type(self.event_logs) is not bool:raise ValueError('seed/log flag')
  return self
 def to_dict(self):return asdict(self)
 @classmethod
 def load(cls,path):
  import yaml
  return cls(**__import__('yaml').safe_load(Path(path).read_text(encoding='utf-8'))).validate()


def generate_court(settings,rep,menu_count,visitors):
 settings.validate();anchors=[a for a in json.loads(data_file('menu_anchors.json').read_text(encoding='utf-8')) if a['role']=='main']
 centers=((.65,.55,.45),(.45,.75,.55),(.4,.45,.65),(.25,.65,.45));dishes=[]
 for slot in range(menu_count//4):
  for counter,category in enumerate(CATEGORIES):
   pool=[a for a in anchors if a['category']==category];a=pool[slot%len(pool)]
   rng=rng_for(settings.seed,'court',rep,counter,slot)
   observed=slot<len(pool);price=a['price_won'] or float(rng.choice(np.arange(3500,7000,500)))
   if not observed:price=max(3500.,price+float(rng.choice([-500,0,500])))
   vegetarian=bool(rng.random()<.12)
   recipe={'staple':1,'vegetable':2 if vegetarian else 1,'protein':0 if vegetarian else 1}
   recipe={k:v for k,v in recipe.items() if v}
   taste=tuple(float(v) for v in np.clip(rng.normal(centers[counter],.16),0,1))
   name=a['name'] if observed else '가상 '+a['name']+' 변형'+str(slot//len(pool)+1)
   dishes.append(Dish('C%dM%02d'%(counter,slot),name,counter,category,float(price),recipe,taste,float(rng.uniform(1.5,4.5)),float(rng.lognormal(0,.4)),vegetarian,('wheat',) if counter in (0,1,3) and not vegetarian else (),a['source_id'],observed,a['price_status'] if observed else 'synthetic_variant'))
 # Capacity/preparation stays constant across menu-count levels, not proportional to the number of menus.
 portions=int(np.ceil(visitors/4*settings.expected_purchase_rate*settings.stock_cover))
 stock=[dict(staple=portions,protein=portions,vegetable=int(np.ceil(portions*1.2))) for _ in CATEGORIES]
 return Court('virtual-campus-%03d'%rep,dishes,[settings.servers_per_counter]*4,settings.kiosks,settings.seats,settings.duration,stock,dict(staple=600.,protein=1200.,vegetable=400.),dict(type='source-informed synthetic composite',sources=['yonsei_2026','sejong_2026','hanyang_erica','scnu_2022','khu_2026'],assumed=['counter placement','taste','recipe','cost','allergen simulation tags','cooking times','popularity','capacity','seats','arrivals','student behavior','stock'],not_a_real_university=True))
