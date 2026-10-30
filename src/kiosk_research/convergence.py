"""Functional stability on fixed probes, not a proof of global optimality."""
import copy
import hashlib
import io
import random
import numpy as np
from kiosk_sim.models import Customer
from kiosk_sim.generators import customers
from kiosk_research.policy import Policy
from kiosk_research.worker import restore_store,seed_for
from kiosk_research.simulation import simulate
from kiosk_research.episodes import run_episode
from kiosk_research.persistence import atomic_bytes,encoded,read_json,write_json


def training_dates(config,day):
    size=config.cycle_days
    full,offset=divmod(day,size)
    if config.lambda_search=='pareto_two_stage':
        return full*config.train_days+min(offset,config.coarse_days)+max(0,min(offset-config.coarse_days-config.validation_days,config.train_days-config.coarse_days))
    return full*config.train_days+min(offset,config.train_days)


def fixed_probes(run,include_scores=False):
    keys=[];weights=[];scores=[]
    for case in run.state['cases']:
        original=restore_store(case['template'])
        price=float(np.median([m.price for m in original.menus]))
        contexts=[]
        for i,(taste,ps,ws,purpose) in enumerate([
            ((.5,.5,.5),.7,.6,'mixed'),((.8,.2,.5),1.2,.3,'meal'),((.2,.8,.5),.3,1.2,'dessert')]):
            shop=copy.deepcopy(original)
            if i==1:shop.stock={k:round(v*.25) for k,v in shop.stock.items()}
            if i==2:shop.cook_free_at=[30.]*shop.cooks
            contexts.append((Customer(taste,ps,ws,None,.5,max(.01,price*(1.05+i*.5)),purpose),shop))
        for name,scenario in sorted(run.config.scenarios.items()):
            for model in run.config.models:
                if model=='no_recommendation':continue
                keys.append(str(case['id'])+':'+name+':'+model)
                policy=Policy.restore(run.state['stores'][str(case['id'])]['policies'][name+':'+model])
                if hasattr(policy,'expected_visitors'):
                    from kiosk_research.realism import options,monthly_multiplier
                    from kiosk_research.arrivals import sales_day_multiplier
                    settings=options(run.config);calendar=settings.calendar(1)
                    mean=run.config.visitors if run.config.visitors is not None else float(case['features']['visitor_mean'])
                    policy.expected_visitors=mean*settings.weekday_multipliers[calendar.weekday()]*sales_day_multiplier(run.config,calendar)[0]*monthly_multiplier(settings,calendar)
                    hours=case['features'].get('opening_hours')
                    policy.hours=int(hours) if hours and 1<=hours<=24 else 8
                weights.append([policy.weights(person,shop,0.,lam) for person,shop in contexts for lam in (0.,.25,.5,.75,1.)])
                if include_scores:
                    # Separate seed namespace: never the training, validation or final dates.
                    values=[]
                    for lam in (0.,.5,1.):
                        rewards=[]
                        for sample in range(3):
                            result=run_episode(run.config,case,policy,lam,scenario,sample+1,namespace='convergence')
                            rewards.append(result['reward'])
                        values.append(float(np.mean(rewards)))
                    scores.append(values)
    return keys,np.asarray(weights,dtype=float),np.asarray(scores,dtype=float)


def stability_metrics(weight_history,score_history):
    """Upper bound on every pair's TV distance across the entire window."""
    w=np.asarray(weight_history,dtype=float)
    s=np.asarray(score_history,dtype=float)
    if w.ndim!=4 or w.shape[-1]!=5 or s.ndim!=3 or w.shape[1]!=s.shape[1]:raise ValueError('bad probe shapes')
    if not np.isfinite(w).all() or not np.isfinite(s).all():raise ValueError('non-finite probe')
    if (w<0).any() or not np.allclose(w.sum(axis=-1),1.,atol=1e-8):raise ValueError('weights must be simplex vectors')
    if (s<0).any() or (s>1).any():raise ValueError('score outside unit range')
    drift=.5*np.ptp(w,axis=0).sum(axis=-1).max(axis=-1)
    score_range=np.ptp(s,axis=0).max(axis=-1)
    return drift,score_range


def observe(run):
    day=run.state['day'];count=training_dates(run.config,day)
    root=run.directory/'convergence';root.mkdir(exist_ok=True)
    identity=hashlib.sha256(encoded(run.identity)).hexdigest()
    period=run.config.convergence_probe_every
    with_scores=count==1 or count%period==0
    keys,weights,scores=fixed_probes(run,include_scores=with_scores)
    stream=io.BytesIO()
    np.savez_compressed(stream,weights=weights,scores=scores)
    payload=stream.getvalue();stem='day_%06d'%day
    atomic_bytes(root/(stem+'.npz'),payload)
    write_json(root/(stem+'.json'),{'day':day,'training_date':count,'identity':identity,'keys':keys,
        'sha256':hashlib.sha256(payload).hexdigest(),'has_scores':with_scores})
    window=run.config.convergence_window
    history=[]
    for path in sorted(root.glob('day_*.json'),reverse=True):
        meta=read_json(path)
        if meta['day']>day or meta['training_date']<count-window:continue
        if meta['identity']!=identity or meta['keys']!=keys:raise ValueError('foreign convergence snapshot')
        data=path.with_suffix('.npz').read_bytes()
        if hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError('corrupt convergence snapshot')
        with np.load(io.BytesIO(data),allow_pickle=False) as saved:
            history.append((meta,saved['weights'],saved['scores']))
        if len(history)>=window+1:break
    history.reverse()
    result={'day':day,'training_dates':count,'window_training_dates':window,
        'tolerance':run.config.convergence_tolerance,'converged':False,'status':'warming_up',
        'interpretation':'fixed-probe functional stability; not optimality or real-world accuracy'}
    result['probe_protocol']='shared environment builder; fixed seed/sample/calendar/inventory per probe, independent of training dates'
    counts=[h[0]['training_date'] for h in history]
    score_rows=[h for h in history if h[0]['has_scores']]
    enough=(counts==list(range(count-window,count+1)) and with_scores and score_rows
        and score_rows[0][0]['training_date']==count-window and score_rows[-1][0]['training_date']==count)
    if enough and run.config.visitors!=0:
        drift,score_range=stability_metrics([h[1] for h in history],[h[2] for h in score_rows])
        tolerance=run.config.convergence_tolerance
        flags=(drift<=tolerance)&(score_range<=tolerance)
        result.update(status='converged' if flags.all() else 'monitoring',converged=bool(flags.all()),
            max_weight_tv_bound=float(drift.max()),max_probe_score_range=float(score_range.max()),
            stable_policies=int(flags.sum()),total_policies=len(keys),
            policies={k:{'weight_tv_bound':float(a),'score_range':float(b),'stable':bool(ok)}
                for k,a,b,ok in zip(keys,drift,score_range,flags)})
        result['sensitivity']={str(t):{'stable_policies':int(((drift<=t)&(score_range<=t)).sum()),
            'all_stable':bool(((drift<=t)&(score_range<=t)).all())} for t in (.005,.01,.03)}
    write_json(root/'latest.json',result)
    write_json(root/'diagnostics'/('day_%06d.json'%day),result)
    return result
