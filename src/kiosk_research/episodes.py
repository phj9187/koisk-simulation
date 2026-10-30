"""One shared environment builder for training, held-out probes and diagnostics."""
import copy
import hashlib
import random
import json
import numpy as np
from kiosk_sim.generators import customers
from kiosk_research.arrivals import visitor_count, purpose_profiles, start_hour, sales_day_multiplier
from kiosk_research.realism import options, Inventory, monthly_multiplier
from kiosk_research.simulation import simulate


def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(repr((seed,*parts)).encode()).digest()[:8],'big')


def run_episode(config,case,policy,lam,scenario,episode_id,namespace='experiment',raw=False):
    from kiosk_research.worker import restore_store
    settings=options(config);horizon=settings.inventory_horizon_days
    template=restore_store(case['template']);store=copy.deepcopy(template)
    stock_rng=random.Random(seed_for(config.seed,namespace,'stock',episode_id,case['id']))
    store.stock={k:max(0,round(v*stock_rng.uniform(.65,1.10))) for k,v in template.stock.items()}
    hours=case['features'].get('opening_hours');hours=int(hours) if hours and 1<=hours<=24 else 8
    inventory=Inventory(store,settings) if horizon>1 else None
    results=[];logs=[];trust_state={}
    for index in range(horizon):
        calendar=settings.calendar(episode_id,index)
        observed_factor,weekday_source=sales_day_multiplier(config,calendar)
        factor=settings.weekday_multipliers[calendar.weekday()]*observed_factor*monthly_multiplier(settings,calendar)
        expected=(config.visitors if config.visitors is not None else float(case['features']['visitor_mean']))*factor
        demand=random.Random(seed_for(config.seed,namespace,'demand',episode_id,index,case['id']))
        if settings.demand_cv:
            import math
            expected_realized=expected*demand.lognormvariate(-.5*math.log1p(settings.demand_cv**2),math.sqrt(math.log1p(settings.demand_cv**2)))
        else:expected_realized=expected
        local_config=copy.copy(config);local_config.visitors=expected_realized
        count=visitor_count(local_config,case,seed_for(config.seed,namespace,'visitors',episode_id,index,case['id']))
        people=customers(count,random.Random(seed_for(config.seed,namespace,'customers',episode_id,index,case['id'])),template) if count else []
        person_keys=['new:'+str(index)+':'+str(i) for i in range(count)]
        if settings.repeat_fraction:
            from dataclasses import replace
            repeat_count=round(count*settings.repeat_fraction)
            regulars=customers(repeat_count,random.Random(seed_for(config.seed,namespace,'regulars',case['id'])),template) if repeat_count else []
            people[:len(regulars)]=[replace(c,openness=min(1.,c.openness+settings.repeat_openness_bonus)) for c in regulars]
            person_keys[:len(regulars)]=['regular:'+str(i) for i in range(len(regulars))]
        purchase=0.
        if inventory:store,purchase=inventory.morning(template,seed_for(config.seed,namespace,'delivery',episode_id,index,case['id']))
        if hasattr(policy,'expected_visitors'):policy.expected_visitors=expected;policy.hours=hours
        profiles=purpose_profiles(config,hours,calendar)
        outcome=simulate(people,store,policy,lam,scenario,seed_for(config.seed,namespace,'events',episode_id,index,case['id']),
                         config.wait_scale,hours,raw,config.decision_time_mode,profiles,start_hour(config,hours),
                         settings,terminal=index==horizon-1,continuing=index>0,
                         close_inventory=inventory.close if inventory else None,reward_scale=1./horizon,
                         customer_keys=person_keys,trust_state=trust_state)
        outcome['purchase_cost']=purchase
        outcome['calendar_date']=calendar.isoformat()
        logs.extend(dict(event,horizon_day=index+1,calendar_date=calendar.isoformat()) for event in outcome['logs'])
        results.append(outcome)
    out={}
    totals={'visitors','served','accepted','abandoned','profit','gross_profit','expired_units','inventory_loss','holding_cost','purchase_cost','ordered_units'}
    totals.update(k for k in results[0] if k.startswith('infeasible_'))
    waits={'mean_total_wait','mean_kiosk_wait','mean_decision_time','mean_cook_wait','mean_cook_time','mean_seat_wait'}
    for key,value in results[0].items():
        if key in ('logs','initial_stock','terminal_stock','mean_weights','calendar_date','eligible'):continue
        if key in totals:out[key]=sum(r[key] for r in results)
        elif key in waits:out[key]=sum(r[key]*r['served'] for r in results)/max(1,sum(r['served'] for r in results))
        elif isinstance(value,(float,int,bool)):out[key]=float(np.mean([r[key] for r in results]))
    out.update(initial_stock=results[0]['initial_stock'],terminal_stock=results[-1]['terminal_stock'],
               mean_weights=np.mean([r['mean_weights'] for r in results],axis=0).tolist(),logs=logs,
               eligible=any(r['eligible'] for r in results),horizon_days=horizon,
               calendar_start=results[0]['calendar_date'],calendar_end=results[-1]['calendar_date'],
               opening_hours=hours,opening_hour=start_hour(config,hours),
               arrivals_source=('sales_counts' if config.sales_profile_path else 'assumed')+('_population_blend' if settings.population_path else ''),weekday_source=weekday_source,
               day_type='weekend' if settings.calendar(episode_id).weekday()>=5 else 'weekday')
    out['customer_component_samples']=json.dumps([[r[k] for k in ('taste_fit_per_visitor','affordability_per_visitor','wait_component_per_visitor','choice_fatigue_per_visitor')] for r in results])
    return out
