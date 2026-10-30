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
             realism=None,terminal=True,continuing=False,close_inventory=None,reward_scale=1.,customer_keys=None,trust_state=None):
    from kiosk_research.realism import Realism
    from kiosk_research.customer_model import CustomerDecision,feasibility
    settings=realism or Realism()
    trust_state={} if trust_state is None else trust_state
    if customer_keys is not None and len(customer_keys)!=len(people):raise ValueError('customer key count mismatch')
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
    fatigue_total=inspected_total=0.
    infeasible=0;feasibility_failures={'stock':0,'budget':0,'allergy':0,'combined':0}
    seat_free=[0.]*settings.seats
    max_margin=max((m.price-m.cost for m in store.menus),default=1.)
    risks=ingredient_risks(store)
    risk_scale=max((risks[m.ingredient]*m.units for m in store.menus),default=0.)
    initial_risk=sum(risks[k]*v for k,v in initial.items());previous_reward=0.
    for index,(arrival,c,shocks,u,customer_id,patience,cook_factors,dine_in) in enumerate(events):
        options=available(store,c)
        if not options:
            infeasible+=1;detail=feasibility(store,c)
            # Count whether relaxing just one restriction restores any option.
            masks=detail['menu_failure_masks'];rescues=[]
            for label,bit in (('stock',1),('budget',2),('allergy',4)):
                if any(not(int(mask)&~bit) and number for mask,number in masks.items()):
                    feasibility_failures[label]+=1;rescues.append(label)
            if not rescues:feasibility_failures['combined']+=1
            if raw:logs.append({'event':customer_id,'served':False,'arrival':arrival,'reason':'no_feasible_menu','feasibility':detail,'single_constraint_rescues':rescues})
            continue
        seat=min(range(settings.seats),key=lambda i:seat_free[i]) if settings.seats and dine_in else None
        seat_entry=max(arrival,seat_free[seat]) if seat is not None else arrival
        now=max(seat_entry,kiosk_free)
        if now-arrival>patience:
            abandoned+=1
            if raw:logs.append({'event':customer_id,'served':False,'arrival':arrival,'reason':'kiosk_wait_abandonment'})
            continue
        common_decision=(1.3+.45*len(options))*settings.decision_multiplier
        person_key=customer_keys[customer_id] if customer_keys is not None else 'new:'+str(customer_id)
        regular=person_key.startswith('regular:')
        trust=trust_state.get(person_key,0.) if settings.trust_learning and regular else 0.
        customer_model=CustomerDecision(c,store,options,settings,seed,customer_id,shocks,now,
                                       visible_queue=max(0.,now-arrival)/max(.01,common_decision),regular=regular,trust=trust)
        no_recommendation=policy.kind=='no_recommendation'
        w=(0.,)*len(COMPONENTS) if no_recommendation else policy.weights(c,store,now,lam)
        predicted_decision=common_decision if decision_time_mode=='equal' else .8+.10*len(options)
        if settings.decision_rule=='inspected':predicted_decision=(settings.decision_base+settings.decision_per_item*(1+settings.recommendation_count))*settings.decision_multiplier
        def recommendation_score(menu):
            features=list(recommendation_features(c,store,menu,arrival,now,wait_scale,risks,risk_scale,max_margin,predicted_decision))
            features[0]=customer_model.perceived_taste(menu,ai=True)
            if hasattr(policy,'menu_score'):return policy.menu_score(c,store,menu,arrival,now,predicted_decision,scenario,wait_scale)
            return sum(a*b for a,b in zip(w,features))
        suggestions=[] if no_recommendation else sorted(options,key=recommendation_score,reverse=True)[:settings.recommendation_count]
        choice=customer_model.choose(suggestions,u);choice['suggestions']=suggestions
        own,suggestion,picked=choice['own'],choice['suggestion'],choice['picked']
        took,probability=choice['accepted'],choice['probability']
        decision=customer_model.decision_minutes(choice,no_recommendation,decision_time_mode)
        perceived_wait_at_choice=customer_model.perceived_wait(picked)
        if now-arrival+decision+customer_model.perceived_wait(picked,now+decision)>patience:
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
        fatigue_total+=customer_model.fatigue(choice);inspected_total+=choice['inspected_count']
        if settings.trust_learning and regular:trust_state[person_key]=customer_model.update_trust(choice,fit)
        seat_total+=seat_entry-arrival;kiosk_total+=now-seat_entry;decision_total+=decision;cook_queue_total+=cook_start-order;cook_total+=cook_minutes
        if hasattr(policy,'feedback'):
            n=max(1,len(people))
            cumulative={'taste_fit_per_visitor':taste/n,'affordability_per_visitor':price/n,'choice_fatigue_per_visitor':fatigue_total/n,
                        'wait_component_per_visitor':waiting/n,
                        'margin_component':min(1.,max(0.,profit/max(1e-9,n*max_margin))),
                        'waste_reduction_component':max(0.,min(1.,1.-sum(risks[k]*v for k,v in store.stock.items())/initial_risk)) if initial_risk>0 else 0.}
            reward=score_outcomes(cumulative,scenario,lam)['reward']
            policy.feedback((reward-previous_reward)*reward_scale);previous_reward=reward
        for i,v in enumerate(w):weight_sum[i]+=v
        if raw:logs.append({'event':customer_id,'served':True,'arrival':arrival,'chosen':picked.name,'recommended':None if suggestion is None else suggestion.name,'recommendations':[m.name for m in suggestions],
                            'own_choice':own.name,'decision_minutes':decision,'acceptance_probability':probability,'menu_changed':picked.name!=own.name,
                            'accepted':took,'taste_fit':fit,'affordability':affordable,'wait_component':wait_value,
                            'considered':[m.name for m in customer_model.considered],'inspected_count':choice['inspected_count'],
                            'acceptance_reference':choice['acceptance_reference'],'perceived_fit':customer_model.perceived_taste(picked),
                            'ai_predicted_fit':customer_model.perceived_taste(picked,ai=True),'perceived_wait':perceived_wait_at_choice,
                            'choice_fatigue':customer_model.fatigue(choice),'trust_before':trust,'trust_after':trust_state.get(person_key,trust),
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
    outcomes['choice_fatigue_per_visitor']=fatigue_total/max(1,n)
    final_reward=score_outcomes(outcomes,scenario,lam)['reward']
    if hasattr(policy,'feedback') and close_inventory is not None and (getattr(policy,'action',None) is not None or getattr(policy,'pending',None) is not None):policy.feedback((final_reward-previous_reward)*reward_scale)
    if hasattr(policy,'end_episode'):policy.end_episode(final_reward*reward_scale,terminal=terminal,discount=settings.discount)
    return {**score_outcomes(outcomes,scenario,lam),'taste_fit_per_visitor':components[0],
            'affordability_per_visitor':components[1],'wait_component_per_visitor':components[2],'margin_component':margin,
            'choice_fatigue_per_visitor':fatigue_total/max(1,n),'mean_inspected_menus':inspected_total/max(1,served),
            'waste_reduction_component':waste_reduction,'visitors':n,'served':served,'eligible':bool(n),'profit':profit,
            'mean_total_wait':wait_total/max(1,served),'leftover_units':sum(store.stock.values()),'weighted_leftover':leftover,
            'risk_reduction_applicable':bool(initial_risk>0),'served_fraction':served/max(1,n),
            'accepted':accepted,'abandoned':abandoned,'gross_profit':gross_profit,**inventory_metrics,
            'infeasible_customers':infeasible,**{'infeasible_rescue_'+k:v for k,v in feasibility_failures.items()},
            'mean_kiosk_wait':kiosk_total/max(1,served),'mean_decision_time':decision_total/max(1,served),
            'mean_seat_wait':seat_total/max(1,served),
            'mean_cook_wait':cook_queue_total/max(1,served),'mean_cook_time':cook_total/max(1,served),
            'initial_stock':initial,'terminal_stock':dict(store.stock),
            'mean_weights':[v/max(1,served) for v in weight_sum],'logs':logs}
