import argparse,json
from pathlib import Path
import pandas as pd
from kiosk_research.engine import Experiment
from kiosk_research.benchmarks import candidate_benchmark
from kiosk_research.policy import Policy
from kiosk_research.episodes import run_episode
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--experiment',required=True);p.add_argument('--out',required=True);p.add_argument('--candidates',type=int,default=32);p.add_argument('--samples',type=int,default=3)
a=p.parse_args();run=Experiment.resume(a.root,a.experiment);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=[];protocol=[]
for case in run.state['cases']:
 for name,scenario in run.config.scenarios.items():
  for lam in (0.,.5,1.):
   info,records=candidate_benchmark(run.config,case,scenario,lam,a.candidates,a.samples,a.samples)
   protocol.append(dict(store_id=case['id'],scenario=name,lambda_value=lam,**info));rows.extend(dict(scenario=name,**r) for r in records)
   for model in run.config.models:
    if model=='no_recommendation':continue
    policy=Policy.restore(run.state['stores'][str(case['id'])]['policies'][name+':'+model])
    for sample in range(1,a.samples+1):
     r=run_episode(run.config,case,policy,lam,scenario,sample,'candidate_heldout')
     rows.append(dict(scenario=name,store_id=case['id'],lambda_value=lam,sample=sample,arm='trained_'+model,customer=r['customer'],store=r['store'],reward=r['reward']))
 print('BENCHMARK_STORE',case['id'],flush=True)
pd.DataFrame(rows).to_csv(out/'heldout.csv',index=False);(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
