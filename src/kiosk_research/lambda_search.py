"""Validation-only geometric Pareto knee search, with global coverage."""
import math
import numpy as np

COARSE=[0.,.5,1.]


def knee_grid(points):
    """C,S already share the declared [0,1] utility scale; do not fit a quadratic."""
    clean=[]
    for p in points:
        c,s,l=map(float,(p['customer'],p['store'],p['lambda_value']))
        if not all(math.isfinite(x) and 0<=x<=1 for x in (c,s,l)):raise ValueError('invalid Pareto point')
        clean.append({'customer':c,'store':s,'lambda_value':l,'model':p['model']})
    front=[p for p in clean if not any(q['customer']>=p['customer'] and q['store']>=p['store'] and (q['customer']>p['customer'] or q['store']>p['store']) for q in clean)]
    unique=[]
    for p in sorted(front,key=lambda p:(p['customer'],p['store'],abs(p['lambda_value']-.5),p['model'])):
        if not any(abs(p['customer']-q['customer'])<=1e-10 and abs(p['store']-q['store'])<=1e-10 for q in unique):unique.append(p)
    result={'grid':list(COARSE),'reason':'no_clear_knee_keep_global','knee_lambda':None,'front':front,'candidates':clean}
    if len(unique)<3:return result
    a=np.array([unique[0]['customer'],unique[0]['store']]);b=np.array([unique[-1]['customer'],unique[-1]['store']])
    direction=b-a;norm=float(np.linalg.norm(direction))
    if norm<=1e-10:return result
    normal=np.array([-direction[1],direction[0]])/norm
    scores=[float(np.dot(np.array([p['customer'],p['store']])-a,normal)) for p in unique]
    best=max(range(1,len(unique)-1),key=lambda i:(scores[i],-abs(unique[i]['lambda_value']-.5)))
    if scores[best]<=1e-10:return result
    lam=unique[best]['lambda_value'];lo=max(0.,lam-.25);hi=min(1.,lam+.25)
    # Retain a half-width interval even if the selected policy lambda is 0 or 1.
    if lam<=.25:lo,hi=0.,.5
    elif lam>=.75:lo,hi=.5,1.
    result.update(grid=[lo,(lo+hi)/2.,hi],reason='geometric_knee_candidate',knee_lambda=lam,knee_distance=scores[best])
    return result


def select_plan(frame,config,day):
    if frame.empty or not frame.phase.eq('validation').all() or not frame.search_stage.eq('selection').all():raise ValueError('selection requires coarse validation only')
    if frame.day.nunique()!=config.validation_days:raise ValueError('incomplete selection dates')
    cycle=config.stage(day)[0]
    start=(cycle-1)*config.cycle_days+config.coarse_days+1
    if sorted(map(int,frame.day.unique()))!=list(range(start,start+config.validation_days)):raise ValueError('wrong selection date range')
    expected=set(config.scenarios);choices={}
    eligible=frame[frame.eligible&frame.model.isin(['fixed','contextual'])]
    for scenario in sorted(expected):
        group=eligible[eligible.scenario.eq(scenario)]
        if group.empty:choices[scenario]={'grid':list(COARSE),'reason':'no_eligible_selection_data','knee_lambda':None,'front':[],'candidates':[]};continue
        # First average held-out dates per store, then equally weight stores.
        local=group.groupby(['store_id','model','lambda_value'],as_index=False)[['customer','store']].mean()
        points=local.groupby(['model','lambda_value'],as_index=False)[['customer','store']].mean().to_dict('records')
        choices[scenario]=knee_grid(points)
    return {'selected_on_day':day,'selection_days':sorted(map(int,frame.day.unique())),
            'scope':'per alpha/beta scenario, pooled AI candidates, shared grid across arms',
            'criterion':'outward perpendicular distance to extreme-point chord in fixed [0,1] C,S scale',
            'interpretation':'geometric candidate; not a unique preference optimum or global guarantee','choices':choices}


def training_grid(config,day,scenario,plan=None):
    stage=config.search_stage(day)
    if stage in ('coarse','selection'):return list(COARSE),'global_coarse'
    if stage!='refine':raise ValueError('not an adaptive training stage')
    if plan is None or scenario not in plan['choices']:raise ValueError('missing persisted search plan')
    step=(day-1)%config.cycle_days-(config.coarse_days+config.validation_days)+1
    if step%config.global_probe_every==0:return list(COARSE),'global_coverage'
    grid=plan['choices'][scenario]['grid']
    if len(grid)!=3 or len(set(grid))!=3 or any(not math.isfinite(v) or not 0<=v<=1 for v in grid):raise ValueError('invalid search grid')
    return list(grid),'refinement'
