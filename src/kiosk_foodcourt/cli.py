"""Resumable episodic policy search; final tests never participate in selection."""
import argparse, hashlib, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .models import Settings, POLICIES, generate_court, rng_for
from .environment import simulate

def atomic(path, data):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); os.replace(temp,path)

def signature(s):
 h=hashlib.sha256(json.dumps(s.to_dict(),sort_keys=True).encode())
 for p in sorted(Path(__file__).parent.rglob('*')):
  if p.suffix in ('.py','.json'): h.update(p.relative_to(Path(__file__).parent).as_posix().encode()); h.update(p.read_bytes().replace(b'\r\n',b'\n'))
 return h.hexdigest()

def average(rows):
 return {k:float(np.mean([r[k] for r in rows])) for k,v in rows[0].items() if isinstance(v,(int,float))}

def run_job(args):
 settings, root, task, deadline=args; s=Settings(**settings).validate(); rep,menus,attention,visitors=task
 folder=Path(root)/'jobs'/f'r{rep}_m{menus}_{attention}_v{visitors}'; folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return str(folder)
 lock=folder/'worker.lock'
 try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError: raise RuntimeError(f'Job already locked: {lock}. Confirm no worker is active before removing this lock.')
 os.close(fd)
 try:
  court=generate_court(s,rep,menus,visitors)
  def episode(policy,theta,day,split):return simulate(court,s,policy,theta,day,split,visitors,attention)[0]
  def evaluate(policy,theta,days,split):return average([episode(policy,theta,d,split) for d in days])
  checkpoint=folder/'checkpoint.json'
  state=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {'generation':0,'models':{},'curve':[]}
  for policy,size in [('learned_fixed',5),('learned_contextual',30)]:
   state['models'].setdefault(policy,dict(mean=[0.]*size,std=[s.initial_std]*size,best=[0.]*size,validation=None,calls=0))
  validation_reference=evaluate('customer_only',None,range(s.validation_days),'validation')
  guard_metrics=('customer','customer_low_budget','customer_high_wait_sensitivity')
  def score(row,base):return s.tradeoff*row['customer']+(1-s.tradeoff)*row['store']-s.constraint_penalty*sum(max(0.,base[k]-row[k]-s.customer_tolerance) for k in guard_metrics)
  def rank(row):
   worst=min(row[k]-validation_reference[k] for k in guard_metrics)
   feasible=worst>=-s.customer_tolerance
   return (feasible,row['store'] if feasible else worst,row['customer'])
  def shaped(policy,x):return np.array(x).reshape(5,6) if policy=='learned_contextual' else np.array(x)
  for g in range(state['generation'],s.generations):
   if time.monotonic()>deadline:return str(folder)
   days=list(range(g*s.train_days,(g+1)*s.train_days)); reference=evaluate('customer_only',None,days,'training')
   for policy,model in state['models'].items():
    rng=rng_for(s.seed,'optimizer',task,policy,g)
    candidates=np.array(model['mean'])+rng.normal(size=(s.population,len(model['mean'])))*np.array(model['std']); candidates[0]=model['best']
    rows=[evaluate(policy,shaped(policy,x),days,'training') for x in candidates]
    scores=np.array([score(row,reference) for row in rows]); elite=candidates[np.argsort(scores)[-max(2,s.population//4):]]
    model['mean']=(.3*np.array(model['mean'])+.7*elite.mean(axis=0)).tolist();model['std']=np.maximum(.1,.3*np.array(model['std'])+.7*elite.std(axis=0)).tolist();model['calls']+=s.population*s.train_days
    candidate=candidates[int(scores.argmax())]
    if (g+1)%s.validation_every==0 or g+1==s.generations:
     row=evaluate(policy,shaped(policy,candidate),range(s.validation_days),'validation')
     if model['validation'] is None or rank(row)>rank(model['validation']):model['best']=candidate.tolist();model['validation']=row
    else:
     # Incumbent remains validation-selected; search distribution still advances.
     pass
    state['curve'].append(dict(generation=g+1,policy=policy,training_score=float(scores.max()),training_calls=model['calls'],validation=model['validation']))
   state['generation']=g+1;atomic(checkpoint,state)
  results=[]
  for day in range(s.evaluation_days):
   for policy in POLICIES:
    theta=shaped(policy,state['models'][policy]['best']) if policy in state['models'] else None
    row=episode(policy,theta,day,'final_test');results.append(dict(policy=policy,day=day,**row))
  from dataclasses import replace
  stresses={'low_acceptance':(replace(s,openness=.15,recommendation_overhead=.4),court),'uncertain_taste':(replace(s,taste_noise=.45),court),'tight_capacity':(s,replace(court,servers=[1]*4,seats=100)),'low_waste':(replace(s,perishable_fraction=.05),court)}
  stress_rows=[]
  for name,(stress_settings,stress_court) in stresses.items():
   for day in range(min(5,s.evaluation_days)):
    for policy in POLICIES:
     theta=shaped(policy,state['models'][policy]['best']) if policy in state['models'] else None
     row=simulate(stress_court,stress_settings,policy,theta,day,'stress_'+name,visitors,attention)[0]
     stress_rows.append(dict(scenario=name,policy=policy,day=day,**row))
  atomic(folder/'court.json',__import__('dataclasses').asdict(court))
  atomic(folder/'result.json',dict(task=task,training=state,final_test=results,stress_test=stress_rows,scope='synthetic meal-period; contribution profit excludes labor and rent'))
  return str(folder)
 finally:lock.unlink(missing_ok=True)

def summarize(root,s):
 clusters={}
 for file in (Path(root)/'jobs').glob('*/result.json'):
  data=json.loads(file.read_text(encoding='utf-8'));rep,m,a,v=data['task']
  splits={'final_test':data['final_test']}
  for row in data.get('stress_test',[]):splits.setdefault(row['scenario'],[]).append(row)
  for split,rows in splits.items():
   for baseline in ('none','customer_only'):
    for policy in POLICIES:
     changes=[]
     for day in sorted(set(r['day'] for r in rows)):
      x=next(r for r in rows if r['day']==day and r['policy']==policy);b=next(r for r in rows if r['day']==day and r['policy']==baseline)
      changes.append([x[k]-b[k] for k in ('customer','contribution_profit','customer_low_budget','customer_high_wait_sensitivity')])
     clusters.setdefault((m,a,v,baseline,policy,split),[]).append(np.mean(changes,axis=0))
 output=[]
 for key,values in clusters.items():
  x=np.array(values);mean=x.mean(axis=0);n=len(x)
  interval=(1.96*x.std(axis=0,ddof=1)/np.sqrt(n)).tolist() if n>=5 else None
  output.append(dict(menu_count=key[0],attention=key[1],visitors=key[2],baseline=key[3],policy=key[4],scenario=key[5],independent_courts=n,customer_change=float(mean[0]),profit_change_won=float(mean[1]),low_budget_customer_change=float(mean[2]),high_wait_sensitivity_customer_change=float(mean[3]),joint_noninferiority_point=bool(mean[0]>=-s.customer_tolerance and mean[1]>s.minimum_profit_gain),subgroup_noninferiority_point=bool(min(mean[0],mean[2],mean[3])>=-s.customer_tolerance),strict_joint_point=bool(mean[0]>0 and mean[1]>0),descriptive_95_half_width=interval))
 atomic(Path(root)/'summary.json',output)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True);p.add_argument('--plan',action='store_true');p.add_argument('--max-jobs',type=int);p.add_argument('--time-budget-minutes',type=float,default=20)
 args=p.parse_args();s=Settings.load(args.config)
 if args.max_jobs is not None and args.max_jobs<1:raise ValueError('max-jobs must be positive')
 if not np.isfinite(args.time_budget_minutes) or args.time_budget_minutes<=0:raise ValueError('time budget must be positive')
 tasks=[(r,m,a,v) for r in range(s.repetitions) for m in s.menu_counts for a in s.attention_modes for v in s.visitors]
 estimate=len(tasks)*(2*s.generations*s.population*s.train_days+s.generations*s.train_days+2*((s.generations+s.validation_every-1)//s.validation_every)*s.validation_days+s.validation_days+len(POLICIES)*(s.evaluation_days+4*min(5,s.evaluation_days)))
 print(json.dumps(dict(tasks=len(tasks),estimated_meal_period_calls=estimate,workers=s.workers,training='persistent CEM, equal fixed/context candidate budgets',test='disjoint final_test seeds'),ensure_ascii=False))
 if args.plan:return
 root=Path(args.root);root.mkdir(parents=True,exist_ok=True);manifest=dict(signature=signature(s),settings=s.to_dict())
 if (root/'manifest.json').exists():
  if json.loads((root/'manifest.json').read_text())!=manifest:raise ValueError('Configuration or source changed: use a NEW experiment directory.')
 else:atomic(root/'manifest.json',manifest)
 pending=[t for t in tasks if not (root/'jobs'/f'r{t[0]}_m{t[1]}_{t[2]}_v{t[3]}'/'result.json').exists()]
 if args.max_jobs is not None:pending=pending[:args.max_jobs]
 deadline=time.monotonic()+args.time_budget_minutes*60
 payload=[(s.to_dict(),str(root),t,deadline) for t in pending]
 if s.workers==1:
  for item in payload:print(run_job(item),flush=True)
 else:
  with ProcessPoolExecutor(max_workers=min(s.workers,os.cpu_count() or 1)) as pool:
   for path in pool.map(run_job,payload):print(path,flush=True)
 summarize(root,s)
 print('Saved summary; unfinished jobs resume from their last completed generation.')

if __name__=='__main__':main()
