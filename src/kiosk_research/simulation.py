"""Explicit synthetic behavior and component outcomes, separate from recommendation weights."""
import copy
import math
import random
from itertools import accumulate
from kiosk_sim.recommenders import affinity,available,predicted_wait
from kiosk_sim.arrival import assumed_purpose_shape
from kiosk_research.policy import COMPONENTS,recommendation_features
from kiosk_research.evaluation import affordability,wait_utility,ingredient_risks,score_outcomes

def simulate(people,original,policy,lam,scenario,seed,wait_scale=25.,hours=8,raw=False,decision_time_mode='legacy',purpose_profiles=None,start_hour=11,
             realism=None,terminal=True,continuing=False,close_inventory=None,reward_scale=1.):
    from kiosk_research.realism import Realism
    settings=realism or Realism()
    if decision_time_mode not in ('legacy','equal'):raise ValueError('unknown decision time mode')
    store=copy.deepcopy(original);initial=dict(store.stock);rng=random.Random(seed);events=[]
    if hasattr(policy,'begin_episode'):
        policy.begin_episode(original,lam,scenario,len(people),hours,continuing=continuing)
    for customer_id,c in enumerate(people):
        profile=purpose_profiles[c.purpose] if purpose_profiles else assumed_purpose_shape(c.purpose,start_hour,int(hours));cdf=list(accumulate(profile))
        draw=rng.random()*cdf[-1];hour=next(i for i,v in enumerate(cdf) if draw<=v)
        arrival=(hour+rng.random())*60.;shocks={m.name:rng.gammavariate(1.,1.) for m in store.menus}
        # Separate per-customer streams preserve pairing when a menu is unavailable.
        local=random.Random(str((seed,'operations',customer_id)))
        patience=local.expovariate(1/settings.patience_minutes)/max(.1,c.wait_sensitivity) if settings.patience_minutes else float('inf')
        cook_factors={m.name:local.lognormvariate(-.5*math.log1p(settings.cook_time_cv**2),math.sqrt(math.log1p(settings.cook_time_cv**2))) if settings.cook_time_cv else 1. for m in store.menus}
        events.append((arrival,c,shocks,rng.random(),customer_id,patience,cook_factors,local.random()<settings.dine_in_fraction))
    events.sort(key=lambda e:e[0])
    served=accepted=abandoned=0;taste=price=waiting=profit=wait_total=0.;kiosk_free=0.;logs=[];weight_sum=[0.]*len(COMPONENTS)
    kiosk_total=decision_total=cook_queue_total=cook_total=seat_total=0.
    seat_free=[0.]*settings.seats
    max_margin=max((m.price-m.cost for m in store.menus),default=1.)
    risks=ingredient_risks(store)
    risk_scale=max((risks[m.ingredient]*m.units for m in store.menus),default=0.)
    initial_risk=sum(risks[k]*v for k,v in initial.items());previous_reward=0.
    def private(c,m,now):return 2.5*affinity(c,m)-c.price_sensitivity*m.price/20.-c.wait_sensitivity*predicted_wait(store,m,now)/30.
    for index,(arrival,c,shocks,u,customer_id,patience,cook_factors,dine_in) in enumerate(events):
        options=available(store,c)
        if not options:
            if raw:logs.append({'event':customer_id,'served':False,'arrival':arrival,'reason':'no_feasible_menu'})
            continue
        seat=min(range(settings.seats),key=lambda i:seat_free[i]) if settings.seats and dine_in else None
        seat_entry=max(arrival,seat_free[seat]) if seat is not None else arrival
        now=max(seat_entry,kiosk_free)
        if now-arrival>patience:
            abandoned+=1
            if raw:logs.append({'event':customer_id,'served':False,'arrival':arrival,'reason':'kiosk_wait_abandonment'})
            continue
        common_decision=(1.3+.45*len(options))*settings.decision_multiplier
        no_recommendation=policy.kind=='no_recommendation'
        w=(0.,)*len(COMPONENTS) if no_recommendation else policy.weights(c,store,now,lam)
        predicted_decision=common_decision if decision_time_mode=='equal' else .8+.10*len(options)
        suggestions=[] if no_recommendation else sorted(options,key=lambda m:sum(a*b for a,b in zip(w,recommendation_features(c,store,m,arrival,now,wait_scale,risks,risk_scale,max_margin,predicted_decision))),reverse=True)[:settings.recommendation_count]
        suggestion=max(suggestions,key=lambda m:private(c,m,now)+settings.choice_noise*math.log(max(1e-9,shocks[m.name]))) if suggestions else None
        own=max(options,key=lambda m:private(c,m,now)+settings.choice_noise*math.log(max(1e-9,shocks[m.name])))
        gain=0. if no_recommendation else private(c,suggestion,now)-private(c,own,now)
        probability=0. if no_recommendation else min(1.,settings.acceptance_multiplier*c.openness/(1.+math.exp(-max(-50.,min(50.,3.*(gain+.25))))))
        took=u<probability;picked=suggestion if took else own
        decision=common_decision if decision_time_mode=='equal' else ((.8+.10*len(options))*settings.decision_multiplier) if took else common_decision
        if not no_recommendation:decision+=settings.decision_penalty*len(suggestions)
        if now-arrival+decision+predicted_wait(store,picked,now+decision)>patience:
            abandoned+=1
            if hasattr(policy,'feedback'):policy.feedback(0.)
            if raw:logs.append({'event':customer_id,'served':False,'arrival':arrival,'recommended':suggestion.name if suggestion else None,'reason':'predicted_wait_abandonment'})
            continue
        order=now+decision;kiosk_free=order
        cook=min(range(store.cooks),key=lambda k:store.cook_free_at[k])
        cook_start=max(order,store.cook_free_at[cook]);cook_minutes=picked.cook_minutes*cook_factors[picked.name]
        finish=cook_start+cook_minutes
        store.cook_free_at[cook]=finish;store.stock[picked.ingredient]-=picked.units
        if seat is not None:seat_free[seat]=finish+settings.table_minutes
        wait=finish-arrival;fit=affinity(c,picked);affordable=affordability(picked.price,c.budget);wait_value=wait_utility(wait,wait_scale)
        taste+=fit;price+=affordable;waiting+=wait_value;profit+=picked.price-picked.cost;wait_total+=wait
        served+=1;accepted+=int(took)
        seat_total+=seat_entry-arrival;kiosk_total+=now-seat_entry;decision_total+=decision;cook_queue_total+=cook_start-order;cook_total+=cook_minutes
        if hasattr(policy,'feedback'):
            n=max(1,len(people))
            cumulative={'taste_fit_per_visitor':taste/n,'affordability_per_visitor':price/n,
                        'wait_component_per_visitor':waiting/n,
                        'margin_component':min(1.,max(0.,profit/max(1e-9,n*max_margin))),
                        'waste_reduction_component':max(0.,min(1.,1.-sum(risks[k]*v for k,v in store.stock.items())/initial_risk)) if initial_risk>0 else 0.}
            reward=score_outcomes(cumulative,scenario,lam)['reward']
            policy.feedback((reward-previous_reward)*reward_scale);previous_reward=reward
        for i,v in enumerate(w):weight_sum[i]+=v
        if raw:logs.append({'event':customer_id,'served':True,'arrival':arrival,'chosen':picked.name,'recommended':None if suggestion is None else suggestion.name,'recommendations':[m.name for m in suggestions],
                            'own_choice':own.name,'decision_minutes':decision,'acceptance_probability':probability,'menu_changed':picked.name!=own.name,
                            'accepted':took,'taste_fit':fit,'affordability':affordable,'wait_component':wait_value,
                            'total_wait':wait,'seat_wait':seat_entry-arrival,'kiosk_wait':now-seat_entry,'cook_wait':cook_start-order,'cook_minutes':cook_minutes,
                            'stock_after':dict(store.stock),'weights':list(w)})
    risk=risks
    initial_risk=sum(risk[k]*v for k,v in initial.items());leftover=sum(risk[k]*v for k,v in store.stock.items())
    n=len(people);components=[taste/max(1,n),price/max(1,n),waiting/max(1,n)]
    margin=min(1.,max(0.,profit/max(1e-9,n*max_margin))) if n else 0.
    waste_reduction=max(0.,min(1.,1.-leftover/initial_risk)) if n and initial_risk>0 else 0.
    inventory_metrics={}
    gross_profit=profit
    if close_inventory is not None:
        inventory_metrics=close_inventory(initial,dict(store.stock))
        profit-=inventory_metrics['inventory_loss']+inventory_metrics['holding_cost']
        # Signed inventory losses affect the bounded objective; raw net profit is also retained.
        margin=max(0.,min(1.,.5+.5*profit/max(1e-9,n*max_margin))) if n else 0.
        waste_reduction=max(0.,1.-inventory_metrics['expired_units']/max(1,sum(initial.values()))) if n else 0.
    outcomes=dict(zip(('taste_fit_per_visitor','affordability_per_visitor','wait_component_per_visitor','margin_component','waste_reduction_component'),[*components,margin,waste_reduction]))
    final_reward=score_outcomes(outcomes,scenario,lam)['reward']
    if hasattr(policy,'feedback') and close_inventory is not None and (getattr(policy,'action',None) is not None or getattr(policy,'pending',None) is not None):policy.feedback((final_reward-previous_reward)*reward_scale)
    if hasattr(policy,'end_episode'):policy.end_episode(final_reward*reward_scale,terminal=terminal,discount=settings.discount)
    return {**score_outcomes(outcomes,scenario,lam),'taste_fit_per_visitor':components[0],
            'affordability_per_visitor':components[1],'wait_component_per_visitor':components[2],'margin_component':margin,
            'waste_reduction_component':waste_reduction,'visitors':n,'served':served,'eligible':bool(n),'profit':profit,
            'mean_total_wait':wait_total/max(1,served),'leftover_units':sum(store.stock.values()),'weighted_leftover':leftover,
            'risk_reduction_applicable':bool(initial_risk>0),'served_fraction':served/max(1,n),
            'accepted':accepted,'abandoned':abandoned,'gross_profit':gross_profit,**inventory_metrics,
            'mean_kiosk_wait':kiosk_total/max(1,served),'mean_decision_time':decision_total/max(1,served),
            'mean_seat_wait':seat_total/max(1,served),
            'mean_cook_wait':cook_queue_total/max(1,served),'mean_cook_time':cook_total/max(1,served),
            'initial_stock':initial,'terminal_stock':dict(store.stock),
            'mean_weights':[v/max(1,served) for v in weight_sum],'logs':logs}
