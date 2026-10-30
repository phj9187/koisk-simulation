"""Optional shared-network comparison; new customers, same store templates."""
import argparse,copy,json,random
from pathlib import Path
import pandas as pd
from kiosk_research.config import Config
from kiosk_research.engine import Experiment
from kiosk_research.actor_critic import ActorCriticPolicy
from kiosk_research.policy import Policy
from kiosk_research.episodes import run_episode,seed_for

def compare(config,cases,scenario,train_samples=3,test_samples=3):
    shared=ActorCriticPolicy('contextual',scenario,seed_for(config.seed,'shared'),learning=config.learning)
    independent={c['id']:ActorCriticPolicy('contextual',scenario,seed_for(config.seed,'independent',c['id']),learning=config.learning) for c in cases}
    rng=random.Random(seed_for(config.seed,'shared_order'))
    for sample in range(1,train_samples+1):
        order=list(cases);rng.shuffle(order)
        for case in order:
            for policy in (shared,independent[case['id']]):
                policy.training=True
                for lam in (0.,.5,1.):run_episode(config,case,policy,lam,scenario,sample,'shared_train')
                policy.end_day()
    shared.training=False
    for policy in independent.values():policy.training=False
    rows=[]
    for case in cases:
        for sample in range(1,test_samples+1):
            for lam in (0.,.5,1.):
                for arm,policy in [('shared',shared),('independent',independent[case['id']]),('no_recommendation',Policy.initial('no_recommendation'))]:
                    r=run_episode(config,case,policy,lam,scenario,sample,'shared_heldout')
                    rows.append(dict(store_id=case['id'],sample=sample,lambda_value=lam,arm=arm,customer=r['customer'],store=r['store'],reward=r['reward']))
    return rows,dict(shared=shared.to_dict(),independent={str(k):p.to_dict() for k,p in independent.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True);p.add_argument('--samples',type=int,default=3);a=p.parse_args()
    if a.samples<1:raise ValueError('positive samples required')
    run=Experiment(Config.load(a.config),a.root);scenario=next(iter(run.config.scenarios.values()))
    rows,policies=compare(run.config,run.state['cases'],scenario,a.samples,a.samples)
    pd.DataFrame(rows).to_csv(run.directory/'shared_heldout.csv',index=False)
    (run.directory/'shared_policies.json').write_text(json.dumps(policies),encoding='utf-8')
    (run.directory/'shared_protocol.json').write_text(json.dumps(dict(experiment_id=run.id,training_samples=a.samples,test_samples=a.samples,scope='same stores, independent customer episodes; equal simulator calls for shared and independent contextual learners; no heldout selection; one seed, no confidence claim'),indent=2),encoding='utf-8')
    print(run.id,flush=True)
