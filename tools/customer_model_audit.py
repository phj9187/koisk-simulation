"""Mechanism benchmarks, not globally optimal bounds or a profitability claim."""
import argparse
import copy
import json
from pathlib import Path
import pandas as pd
from kiosk_research.episodes import run_episode
from kiosk_research.linecharts import load_snapshot
from kiosk_research.policy import Policy
from kiosk_sim.recommenders import affinity,predicted_wait
from kiosk_research.evaluation import affordability,wait_utility


class ConstantPolicy:
    kind='fixed'
    def __init__(self,weights):self.values=weights
    def weights(self,*args):return self.values


class GreedyOracle(ConstantPolicy):
    """True immediate customer utility; no future queues, orders or random shocks."""
    def __init__(self):super().__init__((1.,0.,0.,0.,0.))
    def menu_score(self,c,store,menu,arrival,now,decision,scenario,wait_scale):
        return sum(a*b for a,b in zip(scenario['customer'],(affinity(c,menu),affordability(menu.price,c.budget),
                   wait_utility(now-arrival+decision+predicted_wait(store,menu,now+decision),wait_scale))))


def audit(run,out,stores=2,samples=2):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    settings=[('existing',{}),('consideration_only',{'consideration_mean':3.}),
              ('equal_information_noise',{'consideration_mean':3.,'taste_noise':.25,'ai_taste_noise':.25}),
              ('ai_more_precise_assumption',{'consideration_mean':3.,'taste_noise':.25,'ai_taste_noise':.1}),
              ('stopping_and_hidden_wait',{'consideration_mean':3.,'taste_noise':.25,'ai_taste_noise':.25,'decision_rule':'inspected','wait_visibility':'menu_only'}),
              ('choice_fatigue',{'consideration_mean':3.,'taste_noise':.25,'ai_taste_noise':.25,'decision_rule':'inspected','wait_visibility':'menu_only','overload_penalty':.02})]
    policies={'no_recommendation':Policy.initial('no_recommendation'),'uniform':ConstantPolicy((.2,)*5),'greedy_oracle':GreedyOracle()}
    for i,name in enumerate(('taste','risk','margin','wait','price')):policies['one_hot_'+name]=ConstantPolicy(tuple(float(j==i) for j in range(5)))
    rows=[];reasons=[]
    scenario=next(iter(run.config.scenarios.values()))
    for label,override in settings:
        config=copy.deepcopy(run.config);config.realism.update(override)
        for case in run.state['cases'][:stores]:
            for sample in range(1,samples+1):
                baseline=run_episode(config,case,policies['no_recommendation'],.5,scenario,sample,'customer_information_audit',True)
                for event in baseline['logs']:
                    if not event['served']:reasons.append({'condition':label,'store_id':case['id'],'sample':sample,**event})
                for name,policy in policies.items():
                    result=baseline if name=='no_recommendation' else run_episode(config,case,policy,.5,scenario,sample,'customer_information_audit')
                    row={'condition':label,'store_id':case['id'],'sample':sample,'policy':name,'lambda_value':.5,'visitors':result['visitors']}
                    for metric in ('customer','store','reward','profit','served_fraction','mean_total_wait','mean_inspected_menus','infeasible_customers','abandoned',
                                   'infeasible_rescue_budget','infeasible_rescue_stock','infeasible_rescue_allergy','infeasible_rescue_combined'):
                        row[metric]=result[metric];row['delta_baseline_'+metric]=result[metric]-baseline[metric]
                    rows.append(row)
        print('AUDIT_COMPLETED',label,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'mechanism_benchmarks.csv',index=False)
    (out/'nonpurchase_events.json').write_text(json.dumps(reasons,ensure_ascii=False),encoding='utf-8')
    summary=frame.groupby(['condition','policy'])[['delta_baseline_customer','delta_baseline_store','delta_baseline_reward','mean_inspected_menus']].mean()
    summary.to_csv(out/'mechanism_summary.csv')
    (out/'protocol.json').write_text(json.dumps({'source_experiment':run.id,'conditions':dict(settings),'stores':stores,'independent_fixed_samples':samples,
        'scenario':scenario,'scope':'paired fixed-sample mechanism check; oracle is greedy, not a global upper bound; no minimum improvement target; no learner fit on these samples'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--experiment',required=True);p.add_argument('--out',required=True)
    p.add_argument('--stores',type=int,default=2);p.add_argument('--samples',type=int,default=2);args=p.parse_args()
    audit(load_snapshot(args.root,args.experiment),args.out,args.stores,args.samples)
