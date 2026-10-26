"""Explicit synthetic behavior and component outcomes, separate from recommendation weights."""
import copy
import math
import random
from itertools import accumulate
from kiosk_sim.recommenders import affinity,available,predicted_wait
from kiosk_sim.arrival import assumed_purpose_shape
from kiosk_research.policy import COMPONENTS,recommendation_features
from kiosk_research.evaluation import affordability,wait_utility,ingredient_risks,score_outcomes

def simulate(people,original,policy,lam,scenario,seed,wait_scale=25.,hours=8,raw=False):
    store=copy.deepcopy(original);initial=dict(store.stock);rng=random.Random(seed);events=[]
    for c in people:
        profile=assumed_purpose_shape(c.purpose,11,int(hours));cdf=list(accumulate(profile))
        draw=rng.random()*cdf[-1];hour=next(i for i,v in enumerate(cdf) if draw<=v)
        arrival=(hour+rng.random())*60.;shocks={m.name:rng.gammavariate(1.,1.) for m in store.menus}
        events.append((arrival,c,shocks,rng.random()))
    events.sort(key=lambda e:e[0])
    served=accepted=0;taste=price=waiting=profit=wait_total=0.;kiosk_free=0.;logs=[];weight_sum=[0.]*len(COMPONENTS)
    max_margin=max((m.price-m.cost for m in store.menus),default=1.)
    risks=ingredient_risks(store)
    risk_scale=max((risks[m.ingredient]*m.units for m in store.menus),default=0.)
    def private(c,m,now):return 2.5*affinity(c,m)-c.price_sensitivity*m.price/20.-c.wait_sensitivity*predicted_wait(store,m,now)/30.
    for index,(arrival,c,shocks,u) in enumerate(events):
        options=available(store,c)
        if not options:
            if raw:logs.append({'event':index,'served':False,'arrival':arrival,'reason':'no_feasible_menu'})
            continue
        now=max(arrival,kiosk_free);w=policy.weights(c,store,now,lam)
        suggestion=max(options,key=lambda m:sum(a*b for a,b in zip(w,recommendation_features(c,store,m,arrival,now,wait_scale,risks,risk_scale,max_margin,.8+.10*len(options)))))
        own=max(options,key=lambda m:private(c,m,now)+.16*math.log(max(1e-9,shocks[m.name])))
        gain=private(c,suggestion,now)-private(c,own,now)
        probability=c.openness/(1.+math.exp(-max(-50.,min(50.,3.*(gain+.25)))))
        took=u<probability;picked=suggestion if took else own
        decision=(.8+.10*len(options)) if took else (1.3+.45*len(options))
        order=now+decision;kiosk_free=order
        cook=min(range(store.cooks),key=lambda k:store.cook_free_at[k])
        finish=max(order,store.cook_free_at[cook])+picked.cook_minutes
        store.cook_free_at[cook]=finish;store.stock[picked.ingredient]-=picked.units
        wait=finish-arrival;fit=affinity(c,picked);affordable=affordability(picked.price,c.budget);wait_value=wait_utility(wait,wait_scale)
        taste+=fit;price+=affordable;waiting+=wait_value;profit+=picked.price-picked.cost;wait_total+=wait
        served+=1;accepted+=int(took)
        for i,v in enumerate(w):weight_sum[i]+=v
        if raw:logs.append({'event':index,'served':True,'arrival':arrival,'chosen':picked.name,'recommended':suggestion.name,
                            'accepted':took,'taste_fit':fit,'affordability':affordable,'wait_component':wait_value,
                            'total_wait':wait,'stock_after':dict(store.stock),'weights':list(w)})
    risk=risks
    initial_risk=sum(risk[k]*v for k,v in initial.items());leftover=sum(risk[k]*v for k,v in store.stock.items())
    n=len(people);components=[taste/max(1,n),price/max(1,n),waiting/max(1,n)]
    margin=min(1.,max(0.,profit/max(1e-9,n*max_margin))) if n else 0.
    waste_reduction=max(0.,min(1.,1.-leftover/initial_risk)) if n and initial_risk>0 else 0.
    outcomes=dict(zip(('taste_fit_per_visitor','affordability_per_visitor','wait_component_per_visitor','margin_component','waste_reduction_component'),[*components,margin,waste_reduction]))
    return {**score_outcomes(outcomes,scenario,lam),'taste_fit_per_visitor':components[0],
            'affordability_per_visitor':components[1],'wait_component_per_visitor':components[2],'margin_component':margin,
            'waste_reduction_component':waste_reduction,'visitors':n,'served':served,'eligible':bool(n),'profit':profit,
            'mean_total_wait':wait_total/max(1,served),'leftover_units':sum(store.stock.values()),'weighted_leftover':leftover,
            'risk_reduction_applicable':bool(initial_risk>0),'served_fraction':served/max(1,n),
            'accepted':accepted,'initial_stock':initial,'terminal_stock':dict(store.stock),
            'mean_weights':[v/max(1,served) for v in weight_sum],'logs':logs}
