import copy
import hashlib
import json
import random
import time
from kiosk_sim.generators import customers
from kiosk_sim.models import Menu,Store
from kiosk_research.policy import Policy,COMPONENTS
from kiosk_research.simulation import simulate

def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(repr((seed,*parts)).encode()).digest()[:8],'big')

def restore_store(v):return Store([Menu(**m) for m in v['menus']],dict(v['stock']),v['cooks'],v['purpose'],list(v['cook_free_at']))

def store_day(task):
    config,case,previous,day=task;rep,phase,offset=config.stage(day)
    template=restore_store(case['template']);store=copy.deepcopy(template)
    rng=random.Random(seed_for(config.seed,'stock',day,case['id']))
    store.stock={k:max(1,round(v*rng.uniform(.65,1.10))) for k,v in template.stock.items()};store.cook_free_at=[0.]*store.cooks
    visitors=config.visitors if config.visitors is not None else round(float(case['features']['visitor_mean']))
    people=customers(visitors,random.Random(seed_for(config.seed,'customers',day,case['id'])),template) if visitors else []
    event_seed=seed_for(config.seed,'events',day,case['id'])
    if phase=='train':
        rng=random.Random(seed_for(config.seed,'lambdas',day));lambdas=[0.,1.]+[rng.random() for _ in range(config.train_lambdas-2)]
    else:lambdas=config.validation_lambdas if phase=='validation' else config.evaluation_lambdas
    hours=case['features'].get('opening_hours');hours=int(hours) if hours and 1<=hours<=24 else 8
    policies={};steps={};rows=[];logs=[];inventory={};fresh=previous is None or previous['repetition']!=rep
    for scenario_name,scenario in sorted(config.scenarios.items()):
        for model in config.models:
            key=scenario_name+':'+model
            incumbent=Policy.initial(model) if fresh else Policy.restore(previous['policies'][key])
            count=0 if fresh else previous['optimizer_steps'][key]
            started=time.perf_counter()
            outcomes=[simulate(people,store,incumbent,lam,scenario,event_seed,config.wait_scale,hours,config.raw_logs) for lam in lambdas]
            accepted=False;initial=None;before=sum(r['reward'] for r in outcomes)/len(lambdas)
            if phase=='train' and visitors:
                candidate=incumbent.perturb(seed_for(config.seed,'proposal',day,case['id'],scenario_name),config.sigma)
                alternatives=[simulate(people,store,candidate,lam,scenario,event_seed,config.wait_scale,hours,config.raw_logs) for lam in lambdas]
                if sum(r['reward'] for r in alternatives)>sum(r['reward'] for r in outcomes):incumbent=candidate;outcomes=alternatives;accepted=True
                count+=1
            elif phase!='train':initial=[simulate(people,store,Policy.initial(model),lam,scenario,event_seed,config.wait_scale,hours,False) for lam in lambdas]
            elapsed=time.perf_counter()-started;policies[key]=incumbent.to_dict();steps[key]=count
            for index,(lam,result) in enumerate(zip(lambdas,outcomes)):
                row={k:v for k,v in result.items() if k not in ('logs','initial_stock','terminal_stock','mean_weights')}
                row.update(day=day,repetition=rep,phase=phase,store_id=case['id'],scenario=scenario_name,model=model,lambda_value=float(lam),
                           accepted_candidate=accepted,optimizer_steps=count,incumbent_train_reward=before if phase=='train' else None,
                           training_simulator_calls=2 if phase=='train' and visitors else 0,simulator_calls=2 if phase!='train' or visitors else 1,
                           elapsed_seconds=elapsed/len(lambdas),analysis_weight=case['features'].get('analysis_weight',1.),
                           initial_customer=None if initial is None else initial[index]['customer'],
                           initial_store=None if initial is None else initial[index]['store'],initial_reward=None if initial is None else initial[index]['reward'])
                for name,v in zip(COMPONENTS,result['mean_weights']):row['weight_'+name]=v
                rows.append(row);inventory[key+':'+str(lam)]={'initial':result['initial_stock'],'terminal':result['terminal_stock']}
                if config.raw_logs:logs.extend({'day':day,'store_id':case['id'],'scenario':scenario_name,'model':model,'lambda_value':lam,'payload_json':json.dumps(event,ensure_ascii=False)} for event in result['logs'])
    return {'id':case['id'],'day':day,'repetition':rep,'policies':policies,'optimizer_steps':steps,'rows':rows,'logs':logs,'inventory':inventory}
