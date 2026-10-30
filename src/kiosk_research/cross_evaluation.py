"""Frozen-policy utility reassessment. Final results are descriptive, never selection data."""
import pandas as pd
import json
from kiosk_research.evaluation import CUSTOMER_METRICS, STORE_METRICS


def cross_evaluate(points, criteria, tradeoff=.5):
    """Hold each trained policy AND its input lambda fixed; change only evaluator.

    A common evaluation tradeoff makes candidate regret comparable. Policy lambda
    remains a distinct column; changing it in the policy would require simulation.
    """
    if not 0 <= tradeoff <= 1:raise ValueError('invalid cross evaluation tradeoff')
    columns=['repetition','model','training_scenario','policy_lambda','evaluation_scenario',
             'evaluation_lambda','customer','store','reward','regret','rank','phase']
    rows=[]
    for name,weights in criteria.items():
        for _,p in points.iterrows():
            c=sum(a*float(p[k]) for a,k in zip(weights['customer'],CUSTOMER_METRICS))
            if 'customer_component_samples' in p:
                samples=json.loads(p.customer_component_samples)
                c=sum(row['weight']*max(0.,sum(a*b for a,b in zip(weights['customer'],row['values'][:3]))-row['values'][3]) for row in samples)
            else:c=max(0.,c-float(p.get('choice_fatigue_per_visitor',0.)))
            s=sum(a*float(p[k]) for a,k in zip(weights['store'],STORE_METRICS))
            rows.append({'repetition':p.repetition,'model':p.model,'training_scenario':p.scenario,
                         'policy_lambda':p.lambda_value,'evaluation_scenario':name,'evaluation_lambda':tradeoff,
                         'customer':c,'store':s,'reward':tradeoff*c+(1-tradeoff)*s,'phase':'evaluation'})
    result=pd.DataFrame(rows)
    if result.empty:return pd.DataFrame(columns=columns),pd.DataFrame(columns=['repetition','model','training_scenario','policy_lambda','mean_regret','worst_regret','top_fraction'])
    groups=result.groupby(['repetition','model','evaluation_scenario'])['reward']
    result['regret']=groups.transform('max')-result.reward
    result['rank']=groups.rank(method='min',ascending=False)
    result['top']=result['regret']<=1e-12
    robustness=result.groupby(['repetition','model','training_scenario','policy_lambda'],as_index=False).agg(
        mean_regret=('regret','mean'),worst_regret=('regret','max'),top_fraction=('top','mean'))
    return result[columns],robustness
