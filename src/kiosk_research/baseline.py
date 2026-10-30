"""Paired held-out effects versus unassisted customer choice."""
import numpy as np
import pandas as pd


def compare_baseline(frame,seed=42,independent=False):
    columns=['scenario','model','lambda_value','metric','ai_mean','baseline_mean','delta_ai_minus_baseline',
             'improvement','relative_improvement','paired_observations','repetitions_or_cycles','ci95_low','ci95_high','inference']
    if frame.empty or 'no_recommendation' not in set(frame.model):return pd.DataFrame(columns=columns)
    metrics=[k for k in ('customer','store','reward','profit','mean_total_wait','leftover_units','served_fraction',
                         'taste_fit_per_visitor','affordability_per_visitor','wait_component_per_visitor','margin_component','waste_reduction_component',
                         'abandoned','expired_units','inventory_loss','holding_cost','mean_kiosk_wait','mean_decision_time','mean_cook_wait','mean_cook_time','mean_seat_wait') if k in frame]
    keys=['day','repetition','store_id','scenario','lambda_value']
    evaluated=frame[frame.phase.eq('evaluation')&frame.eligible]
    base=evaluated[evaluated.model.eq('no_recommendation')][keys+metrics]
    if base.duplicated(keys).any():raise ValueError('duplicate baseline observations')
    rows=[];rng=np.random.default_rng(seed)
    for model in ('fixed','contextual'):
        ai=evaluated[evaluated.model.eq(model)]
        if ai.empty:continue
        paired=ai.merge(base,on=keys,suffixes=('_ai','_baseline'),how='left',validate='one_to_one',indicator=True)
        if not paired['_merge'].eq('both').all():raise ValueError('missing matched baseline')
        for (scenario,lam),group in paired.groupby(['scenario','lambda_value']):
            weights=group.analysis_weight.to_numpy(float) if 'analysis_weight' in group else np.ones(len(group))
            for metric in metrics:
                a=group[metric+'_ai'].to_numpy(float);b=group[metric+'_baseline'].to_numpy(float)
                sign=-1. if metric in ('mean_total_wait','leftover_units','abandoned','expired_units','inventory_loss','holding_cost','mean_kiosk_wait','mean_decision_time','mean_cook_wait','mean_cook_time','mean_seat_wait') else 1.
                improvement=sign*(a-b);ci=[None,None]
                # Days/stores inside a learning repetition are not independent replicas.
                differences=[]
                for rep in group.repetition.unique():
                    mask=group.repetition.to_numpy()==rep
                    differences.append(float(np.average(improvement[mask],weights=weights[mask])))
                if independent and len(differences)>=5:
                    ci=np.quantile(rng.choice(differences,(5000,len(differences)),replace=True).mean(axis=1),[.025,.975]).tolist()
                av=float(np.average(a,weights=weights));bv=float(np.average(b,weights=weights));gain=sign*(av-bv)
                rows.append(dict(zip(columns,[scenario,model,float(lam),metric,av,bv,av-bv,gain,
                    gain/abs(bv) if abs(bv)>1e-12 else None,len(group),len(differences),*ci,
                    'paired_repetition_bootstrap' if ci[0] is not None else 'descriptive_only'])))
    return pd.DataFrame(rows,columns=columns)
