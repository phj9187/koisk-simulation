"""Frozen, paired diagnostics; neither fit nor select models on diagnostic samples."""
import copy
import json
from kiosk_research.policy import Policy
from kiosk_research.episodes import run_episode, seed_for


class RulePolicy:
    kind='contextual'
    def weights(self,*args):return (.4,.15,.2,.15,.1)


def diagnose(config,case,trained,scenario,day,name,repetition):
    from kiosk_research.actor_critic import ActorCriticPolicy
    rep=repetition if config.repetition_seeds and not config.continual else 1
    seed=seed_for(config.seed,'neural',case['id'],name,rep)
    initial=ActorCriticPolicy(trained.kind,scenario,seed) if config.algorithm=='actor_critic' else Policy.initial(trained.kind)
    arms={'initial':initial,'trained':trained,'no_recommendation':Policy.initial('no_recommendation'),'rule':RulePolicy()}
    records=[]
    for lam in config.evaluation_lambdas:
        outcomes={arm:run_episode(config,case,copy.deepcopy(policy),lam,scenario,1,'diagnostics',True) for arm,policy in arms.items()}
        reference=outcomes['initial'];base=outcomes['no_recommendation']
        def events(result):return {(r['horizon_day'],r['event']):r for r in result['logs']}
        initial_events=events(reference)
        for arm,result in outcomes.items():
            paired=[(event,initial_events[key]) for key,event in events(result).items() if key in initial_events]
            comparable=[(a,b) for a,b in paired if a.get('served') and b.get('served')]
            changed=sum(a.get('chosen')!=b.get('chosen') for a,b in paired)
            rec_changed=sum(a.get('recommendations')!=b.get('recommendations') for a,b in comparable)
            row={'day':day,'repetition':repetition,'store_id':case['id'],'scenario':name,'model':trained.kind,'arm':arm,'lambda_value':lam,
                 'sample_protocol':'fixed independent diagnostic episode 1, shared exogenous seeds',
                 'recommendation_change_rate':rec_changed/max(1,len(comparable)),
                 'order_change_rate':changed/max(1,len(paired)),
                 'acceptance_rate':result['accepted']/max(1,result['served']),
                 'matched_customers':len(paired),'both_served_customers':len(comparable)}
            row['event_log_json']=json.dumps(result['logs'],ensure_ascii=False)
            for metric in ('customer','store','reward','profit','served_fraction','mean_total_wait','abandoned'):
                row[metric]=result[metric]
                row['delta_baseline_'+metric]=result[metric]-base[metric]
                row['delta_initial_'+metric]=result[metric]-reference[metric]
            records.append(row)
    return records


def sensitivity(config,case,policy,scenario,sample=1):
    """One factor at a time on fixed held-out samples; never used to retune weights."""
    controls=[('reference',{}),('acceptance_low',{'acceptance_multiplier':.5}),('acceptance_high',{'acceptance_multiplier':1.5}),
              ('decision_fast',{'decision_multiplier':.75}),('decision_slow',{'decision_multiplier':1.25}),
              ('recommendation_overhead',{'decision_penalty':.25}),('patience_15',{'patience_minutes':15.}),
              ('cook_cv_30',{'cook_time_cv':.3}),('demand_cv_30',{'demand_cv':.3})]
    rows=[]
    for label,override in controls:
        local=copy.deepcopy(config);local.realism.update(override)
        for lam in config.evaluation_lambdas:
            result=run_episode(local,case,copy.deepcopy(policy),lam,scenario,sample,'sensitivity')
            base=run_episode(local,case,Policy.initial('no_recommendation'),lam,scenario,sample,'sensitivity')
            row={'condition':label,'store_id':case['id'],'model':policy.kind,'lambda_value':lam}
            for metric in ('customer','store','reward','profit','served_fraction','mean_total_wait'):
                row[metric]=result[metric];row['delta_baseline_'+metric]=result[metric]-base[metric]
            rows.append(row)
    return rows
