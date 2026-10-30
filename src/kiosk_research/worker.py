import copy
import hashlib
import json
import random
import time
from kiosk_sim.generators import customers
from kiosk_sim.models import Menu,Store
from kiosk_research.policy import Policy,COMPONENTS
from kiosk_research.simulation import simulate
from kiosk_research.arrivals import visitor_count,start_hour,purpose_profiles
from kiosk_research.episodes import run_episode

def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(repr((seed,*parts)).encode()).digest()[:8],'big')

def restore_store(v):return Store([Menu(**m) for m in v['menus']],dict(v['stock']),v['cooks'],v['purpose'],list(v['cook_free_at']))

def store_day(task):
    config,case,previous,day=task[:4];plan=task[4] if len(task)>4 else None;rep,phase,offset=config.stage(day)
    # The daily policy sees only the store's mean, not the realized count, so no future information leaks in.
    expected=config.visitors if config.visitors is not None else round(float(case['features']['visitor_mean']))
    if phase=='train':
        rng=random.Random(seed_for(config.seed,'lambdas',day));lambdas=[0.,1.]+[rng.random() for _ in range(config.train_lambdas-2)]
    else:lambdas=config.validation_lambdas if phase=='validation' else config.evaluation_lambdas
    hours=case['features'].get('opening_hours');hours=int(hours) if hours and 1<=hours<=24 else 8
    policies={};steps={};rows=[];logs=[];inventory={};fresh=previous is None or (previous['repetition']!=rep and not config.continual)
    def episode(policy,lam,raw=False):return run_episode(config,case,policy,lam,scenario,day,raw=raw)
    for scenario_name,scenario in sorted(config.scenarios.items()):
        schedule='fixed_protocol'
        if config.lambda_search=='pareto_two_stage' and config.search_stage(day) in ('coarse','selection','refine'):
            from kiosk_research.lambda_search import training_grid
            lambdas,schedule=training_grid(config,day,scenario_name,plan)
        for model in config.models:
            key=scenario_name+':'+model
            baseline=model=='no_recommendation'
            if baseline:
                def initial_policy():return Policy.initial('no_recommendation')
            elif config.algorithm=='actor_critic':
                from kiosk_research.actor_critic import ActorCriticPolicy
                init_rep=rep if config.repetition_seeds and not config.continual else 1
                def initial_policy():return ActorCriticPolicy(model,scenario,seed_for(config.seed,'neural',case['id'],scenario_name,init_rep),expected,hours,config.learning)
            else:
                def initial_policy():return Policy.initial(model)
            incumbent=initial_policy() if fresh else Policy.restore(previous['policies'][key])
            neural=config.algorithm=='actor_critic' and not baseline
            if neural:incumbent.training=phase=='train'
            count=0 if fresh else previous['optimizer_steps'][key]
            started=time.perf_counter()
            candidates=incumbent.learning['fixed_candidates'] if neural and model=='fixed' and phase=='train' else 1
            outcomes=[]
            for lam in lambdas:
                start=len(incumbent.daily_batch) if neural else 0
                trials=[episode(incumbent,lam,config.raw_logs) for _ in range(candidates)]
                # Report the first sampled trajectory, never the best training result.
                outcomes.append(trials[0])
                if neural and incumbent.learning['paired_baseline'] and model=='fixed' and phase=='train':
                    batch=incumbent.daily_batch[start:];width=len(batch)//candidates
                    for j,item in enumerate(batch):
                        peers=[batch[k*width+j%width][2] for k in range(candidates) if k!=j//width]
                        incumbent.actor_baselines.append(sum(peers)/len(peers))
            visitors=outcomes[0]['visitors']
            accepted=False;initial=None;before=sum(r['reward'] for r in outcomes)/len(lambdas)
            if baseline:
                count=0
                if phase!='train':initial=outcomes
            elif neural:
                incumbent.end_day();count=incumbent.updates
                if phase!='train':initial=[episode(initial_policy(),lam) for lam in lambdas]
            elif phase=='train' and visitors:
                candidate=incumbent.perturb(seed_for(config.seed,'proposal',day,case['id'],scenario_name),config.sigma)
                alternatives=[episode(candidate,lam,config.raw_logs) for lam in lambdas]
                if sum(r['reward'] for r in alternatives)>sum(r['reward'] for r in outcomes):incumbent=candidate;outcomes=alternatives;accepted=True
                count+=1
            elif phase!='train':initial=[episode(Policy.initial(model),lam) for lam in lambdas]
            elapsed=time.perf_counter()-started;policies[key]=incumbent.to_dict();steps[key]=count
            for index,(lam,result) in enumerate(zip(lambdas,outcomes)):
                row={k:v for k,v in result.items() if k not in ('logs','initial_stock','terminal_stock','mean_weights')}
                row.update(day=day,repetition=rep,phase=phase,store_id=case['id'],scenario=scenario_name,model=model,lambda_value=float(lam),
                           search_stage=config.search_stage(day),lambda_schedule=schedule,
                           accepted_candidate=accepted,optimizer_steps=count,incumbent_train_reward=before if phase=='train' else None,
                           training_simulator_calls=0 if baseline else (candidates if neural else 2) if phase=='train' and visitors else 0,simulator_calls=1 if baseline else (candidates if neural else 2) if phase=='train' and visitors else (2 if phase!='train' else 1),
                           elapsed_seconds=elapsed/len(lambdas),analysis_weight=case['features'].get('analysis_weight',1.),
                           initial_customer=None if initial is None else initial[index]['customer'],
                           initial_store=None if initial is None else initial[index]['store'],initial_reward=None if initial is None else initial[index]['reward'])
                row['simulator_calls']*=result['horizon_days']
                row['training_simulator_calls']*=result['horizon_days']
                for name,v in zip(COMPONENTS,result['mean_weights']):row['weight_'+name]=None if baseline else v
                rows.append(row);inventory[key+':'+str(lam)]={'initial':result['initial_stock'],'terminal':result['terminal_stock']}
                if config.raw_logs:logs.extend({'day':day,'store_id':case['id'],'scenario':scenario_name,'model':model,'lambda_value':lam,'payload_json':json.dumps(event,ensure_ascii=False)} for event in result['logs'])
    diagnostics=[]
    if config.diagnostics and phase=='evaluation':
        from kiosk_research.diagnostics import diagnose
        for scenario_name,scenario in sorted(config.scenarios.items()):
            for model in config.models:
                if model=='no_recommendation':continue
                trained=Policy.restore(policies[scenario_name+':'+model])
                diagnostics.extend(diagnose(config,case,trained,scenario,day,scenario_name,rep))
    return {'id':case['id'],'day':day,'repetition':rep,'policies':policies,'optimizer_steps':steps,'rows':rows,'logs':logs,'inventory':inventory,'diagnostics':diagnostics}
