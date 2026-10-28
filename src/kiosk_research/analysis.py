"""Two-objective evaluation; no scalar final ranking across satisfaction assumptions."""
from pathlib import Path
from kiosk_research.policy import COMPONENTS
from kiosk_research.evaluation import CUSTOMER_METRICS,STORE_METRICS
import numpy as np
import pandas as pd
from kiosk_research.persistence import write_json,read_json,atomic_bytes


def nondominated(points):
    p=np.asarray(points,dtype=float)
    if not len(p):return np.array([],dtype=bool)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('finite two-objective points required')
    return np.array([not np.any(np.all(p>=x,axis=1)&np.any(p>x,axis=1)) for x in p])


def hypervolume(points):
    p=np.asarray(points,dtype=float)
    if not len(p):return 0.
    if np.any(p<0) or np.any(p>1):raise ValueError('objectives must be in [0,1]')
    p=p[nondominated(p)];p=p[np.argsort(p[:,0])];area=previous=0.
    for c,s in p:area+=(c-previous)*s;previous=c
    return float(area)


def aggregate(frame,per_store=False,phase='evaluation'):
    keys=['repetition','scenario','model','lambda_value']+(['store_id'] if per_store else [])
    metrics=['customer','store','reward','profit','mean_total_wait','served','leftover_units','initial_customer','initial_store','initial_reward']+list(CUSTOMER_METRICS+STORE_METRICS)+['served_fraction','risk_reduction_applicable']+[k for k in frame if k.startswith('weight_')]
    rows=[]
    for group,v in frame[(frame.phase==phase)&frame.eligible].groupby(keys):
        row=dict(zip(keys,group));weights=v.analysis_weight.to_numpy(float)
        for key in metrics:
            valid=v[key].notna();row[key]=float(np.average(v.loc[valid,key],weights=weights[valid])) if valid.any() else None
        row.update(simulator_calls=int(v.simulator_calls.sum()),elapsed_seconds=float(v.elapsed_seconds.sum()));rows.append(row)
    out=pd.DataFrame(rows,columns=keys+metrics+['simulator_calls','elapsed_seconds'])
    out['phase']=phase
    out['pareto']=False
    if not out.empty:
        for _,g in out.groupby(['repetition','scenario','model']+(['store_id'] if per_store else [])):out.loc[g.index,'pareto']=nondominated(g[['customer','store']].to_numpy())
    return out


def compare(points,seed,per_store=False,independent=True):
    if points.empty:return []
    values=[]
    groups=['repetition','scenario','model']+(['store_id'] if per_store else [])
    for keys,g in points.groupby(groups):values.append(dict(zip(groups,keys))|{'hypervolume':hypervolume(g[['customer','store']])})
    table=pd.DataFrame(values)
    if per_store:
        table=table.groupby(['repetition','scenario','model'],as_index=False).hypervolume.mean()
    results=[]
    for scenario,v in table.groupby('scenario'):
        p=v.pivot(index='repetition',columns='model',values='hypervolume').dropna()
        if not {'fixed','contextual'}<=set(p):continue
        diff=(p.contextual-p.fixed).to_numpy();ci=None
        if len(diff)>=5 and independent:
            rng=np.random.default_rng(seed);ci=np.quantile(rng.choice(diff,(5000,len(diff)),replace=True).mean(axis=1),[.025,.975]).tolist()
        results.append({'scenario':scenario,'independent_repetitions':len(diff),'mean_hypervolume_contextual_minus_fixed':float(diff.mean()),
                        'ci95':ci,'differences':diff.tolist(),'status':'descriptive_only' if ci is None else 'paired_repetition_bootstrap',
                        'reference_point':[0.,0.],'scope':'equal-store mean of store-specific frontiers; conditional on sampled stores' if per_store else 'shared-lambda aggregate frontier; conditional on sampled stores; not a population or causal estimate'})
        if not independent:
            results[-1].update(independent_repetitions=0,dependent_cycles=len(diff),status='dependent_cycles_descriptive_only')
    return results


def recommend(points,min_customer=None,min_store=None):
    if min_customer is None and min_store is None:raise ValueError('an external operating constraint is required')
    for v in (min_customer,min_store):
        if v is not None and not 0<=v<=1:raise ValueError('constraint outside [0,1]')
    for column in ('scenario','repetition','store_id'):
        if column in points and points[column].nunique()>1:raise ValueError('select within one '+column)
    if 'phase' in points and not points.phase.eq('validation').all():raise ValueError('selection requires validation data')
    selected=points.copy()
    if min_customer is not None:selected=selected[selected.customer>=min_customer]
    if min_store is not None:selected=selected[selected.store>=min_store]
    if selected.empty:return {'status':'no_feasible_policy'}
    key='store' if min_customer is not None else 'customer'
    row=selected.sort_values([key,'lambda_value','model'],ascending=[False,True,True]).iloc[0]
    return {'status':'selected_on_validation','lambda':float(row.lambda_value),'customer':float(row.customer),'store':float(row.store),'model':row.model}


def store_effects(points,cases):
    """Exploratory grouped donor CV; standardized ridge associations, not causal."""
    names=['visitor_mean','cooks','menu_price','prep_minutes','stock_units'];features=[]
    for c in cases:
        t=c['template'];menus=t['menus']
        features.append({'store_id':c['id'],'donor':str(c['features'].get('source_row','synthetic-'+str(c['id']))),
                         'visitor_mean':float(c['features']['visitor_mean']),'cooks':t['cooks'],
                         'menu_price':np.mean([m['price'] for m in menus]),'prep_minutes':np.mean([m['cook_minutes'] for m in menus]),'stock_units':sum(t['stock'].values())})
    records=[]
    if points.empty:return pd.DataFrame(columns=['scenario','model','lambda_value','outcome','feature','coefficient','group_cv_mae'])
    for keys,g in points.groupby(['scenario','model','lambda_value']):
        v=g.groupby('store_id')[['customer','store']].mean().reset_index().merge(pd.DataFrame(features),on='store_id')
        if v.donor.nunique()<6:continue
        x=v[names].to_numpy(float)
        def fit(tx,ty,test):
            mean=tx.mean(axis=0);scale=tx.std(axis=0);scale[scale<1e-12]=1.
            design=np.column_stack([np.ones(len(tx)),(tx-mean)/scale]);penalty=np.diag([0.]+[1.]*len(names))
            beta=np.linalg.solve(design.T@design+penalty,design.T@ty)
            return np.column_stack([np.ones(len(test)),(test-mean)/scale])@beta,beta
        for outcome in ('customer','store'):
            y=v[outcome].to_numpy(float);prediction=np.zeros(len(v))
            for donor in v.donor.unique():
                mask=(v.donor==donor).to_numpy();prediction[mask]=fit(x[~mask],y[~mask],x[mask])[0]
            _,beta=fit(x,y,x)
            for name,coefficient in zip(['intercept']+names,beta):records.append(dict(zip(['scenario','model','lambda_value'],keys))|{'outcome':outcome,'feature':name,'coefficient':float(coefficient),'group_cv_mae':float(np.abs(y-prediction).mean()),'unique_donors':v.donor.nunique(),'interpretation':'association, not causal'})
    return pd.DataFrame(records)


def function_curves(experiment):
    from kiosk_research.policy import Policy
    from kiosk_research.worker import restore_store
    from kiosk_sim.models import Customer
    rows=[];c=Customer((.5,.5,.5),.7,.6,None,.5,20.,'mixed')
    for path in sorted((experiment.root/'models'/experiment.id).glob('repetition_*_trained.json')):
        state=read_json(path);rep=experiment.config.stage(state['day'])[0]
        for case in experiment.state['cases']:
            store=restore_store(case['template']);saved=state['stores'][str(case['id'])]
            for key,value in saved['policies'].items():
                scenario,model=key.split(':')
                if model=='no_recommendation':continue
                policy=Policy.restore(value)
                for lam in experiment.config.evaluation_lambdas:
                    weights=policy.weights(c,store,0.,lam)
                    row={'repetition':rep,'store_id':case['id'],'scenario':scenario,'model':model,'lambda_value':lam}
                    row.update(zip(COMPONENTS,weights));rows.append(row)
    return pd.DataFrame(rows)


def plots(frame,points,statistics,curves,target,identity,suffix=""):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    if not points.empty:
        for (scenario,model),g in points.groupby(['scenario','model']):
            mean=g.groupby('lambda_value')[['customer','store']].mean();axes[0,0].plot(mean.customer,mean.store,'o-',label=scenario+'/'+model)
    axes[0,0].set(xlabel='Customer',ylabel='Store',title='Final held-out lambda tradeoffs (mean points)')
    for ax,(metric,title) in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1]],[('customer','Customer satisfaction per visitor'),('store','Store satisfaction'),('mean_total_wait','Wait per served customer'),('profit','Daily profit')]):
        for (model,phase),v in frame[frame.eligible].groupby(['model','phase']):
            trend=v.groupby('day')[metric].mean();ax.plot(trend.index,trend.values,'.-',label=model+'/'+phase)
        ax.set(title=title,xlabel='Day')
    for item in statistics:axes[1,2].scatter(range(1,len(item['differences'])+1),item['differences'],label=item['scenario'])
    axes[1,2].axhline(0,color='gray',ls='--');axes[1,2].set(title='Repetition hypervolume differences',xlabel='Repetition',ylabel='Contextual minus fixed')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.has_data():ax.legend(fontsize=6)
    fig.suptitle('Synthetic research · '+str(frame.scenario.iloc[0])+' · descriptive panels · code '+str(identity['git_commit'])[:12]);fig.tight_layout();fig.savefig(target/('comparison'+suffix+'.png'),dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,4))
    for (model,phase),v in frame[frame.eligible].groupby(['model','phase']):
        trend=v.groupby('day').reward.mean();axes[0].plot(trend.index,trend.values,'.-',label=model+'/'+phase)
    axes[0].set(title='Mean sampled-lambda reward (descriptive)',xlabel='Day')
    for model,v in frame[frame.eligible].groupby('model'):
        if model=='no_recommendation':continue
        for key in [k for k in frame if k.startswith('weight_')]:
            trend=v.groupby('day')[key].mean();axes[1].plot(trend.index,trend.values,label=model+'/'+key[7:])
    axes[1].set(title='Actually applied mean weights',xlabel='Day')
    if not curves.empty:
        for model,v in curves.groupby('model'):
            for key in COMPONENTS:
                trend=v.groupby('lambda_value')[key].mean();axes[2].plot(trend.index,trend.values,label=model+'/'+key)
    axes[2].set(title='Learned function at reference context',xlabel='Lambda')
    for ax in axes:
        if ax.has_data():ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(target/('learning_weights'+suffix+'.png'),dpi=140);plt.close(fig)


def export(experiment,target_root=None):
    frame=experiment.export_metrics()
    if frame.empty:raise ValueError('no completed dates')
    target=Path(target_root or experiment.directory/'summary_exports')/f"day-{experiment.state['day']:06d}"
    completion={'identity':experiment.identity,'day':experiment.state['day']}
    if target.exists():
        if (target/'complete.json').exists() and read_json(target/'complete.json')==completion:return target
        raise FileExistsError('incomplete or foreign summary; choose another output root')
    target.mkdir(parents=True)
    size=experiment.config.train_days+experiment.config.validation_days+experiment.config.evaluation_days
    completed=list(range(1,experiment.state['day']//size+1))
    validation_complete=[r for r in range(1,experiment.config.stage(experiment.state['day'])[0]+1) if (r-1)*size+experiment.config.train_days+experiment.config.validation_days<=experiment.state['day']]
    final=frame[frame.repetition.isin(completed)];points=aggregate(final);stores=aggregate(final,True)
    from kiosk_research.baseline import compare_baseline
    baseline=compare_baseline(final,experiment.config.seed,independent=not experiment.config.continual)
    atomic_bytes(target/'baseline_comparison.csv',baseline.to_csv(index=False).encode('utf-8'))
    validation=aggregate(frame[frame.repetition.isin(validation_complete)],True,'validation');curves=function_curves(experiment)
    for name,table in [('summary.csv',points),('store_frontiers.csv',stores),('validation.csv',validation),('function_curves.csv',curves),('store_effects.csv',store_effects(stores,experiment.state['cases']))]:atomic_bytes(target/name,table.to_csv(index=False).encode('utf-8'))
    from kiosk_research.cross_evaluation import cross_evaluate
    cross,robustness=cross_evaluate(points,{**experiment.config.scenarios,**experiment.config.cross_evaluation})
    atomic_bytes(target/'cross_evaluation.csv',cross.to_csv(index=False).encode('utf-8'))
    atomic_bytes(target/'robustness.csv',robustness.to_csv(index=False).encode('utf-8'))
    independent=not experiment.config.continual
    statistics=compare(points,experiment.config.seed,independent=independent);write_json(target/'statistics.json',{'comparisons':statistics,'store_comparisons':compare(stores,experiment.config.seed,per_store=True,independent=independent),'completed_repetitions':completed if independent else [],'completed_cycles':completed if not independent else []})
    write_json(target/'metadata.json',{**experiment.metadata,'completed_day':experiment.state['day'],'completed_repetitions':completed if independent else [],'completed_cycles':completed if not independent else [],'config':experiment.config.to_dict()})
    if experiment.config.plots:
        from kiosk_research.linecharts import export_lines
        export_lines(experiment,target/'line_charts')
        for i,scenario in enumerate(experiment.config.scenarios):
            plots(frame[frame.scenario==scenario],points[points.scenario==scenario],
                  [s for s in statistics if s['scenario']==scenario],
                  curves[curves.scenario==scenario] if not curves.empty else curves,target,experiment.identity,
                  '' if i==0 else '_'+scenario)
    text=f"""# {experiment.id}: day {experiment.state['day']}

Synthetic outcomes, not measured human satisfaction.

- Code: {experiment.identity['git_commit']}
- Source SHA256: {experiment.identity['source_sha256']}
- Data: {experiment.directory}
- Completed independent repetitions: {len(completed)}; sampled stores held fixed.
- Each repetition restarts models. Training/validation/final evaluation use separate dates.
- Both models optimize the same lambda-dependent daily objective with equal simulator-call budgets.
- Model dimensions: fixed=15, contextual=135. Runtime and simulator calls are recorded.
- Initial uniform policies are evaluated on the same held-out dates; no improvement is assumed.
- Pareto/hypervolume comparison uses only complete final repetitions. Fixed reference (0,0).
- Reports both shared-lambda aggregate frontiers and equal-store mean store-specific hypervolumes; these answer different questions.
- Fewer than five repetitions: descriptive differences only; otherwise repetition-level paired bootstrap.
- Store effects are exploratory grouped-donor cross-validation associations, not causal effects.
- Zero-visitor stores remain in raw metrics but do not train or enter performance comparisons.
- No universal optimal lambda is asserted. Selection requires an external operating constraint and validation only.
- Different satisfaction scenarios are different assumed utility definitions, not directly comparable social scores.
- Cross evaluation freezes trained policies and their input lambda, changes utility coefficients, and uses common evaluation lambda=0.5.
- Regret is within the sampled candidate set per model/repetition; not global regret.
- Robustness fractions equally weight declared utility scenarios; they are not population probabilities.
- Cross evaluation of final dates is descriptive and must not be used to select a final policy.
- Weighted sums, finite search and model classes do not guarantee the whole/global Pareto frontier.
"""
    if not independent:
        text=text.replace(f'Completed independent repetitions: {len(completed)}',f'Completed dependent learning cycles: {len(completed)}')
        text=text.replace('Each repetition restarts models.','Models carry forward across cycles; models do not restart.')
        text=text.replace('Fewer than five repetitions: descriptive differences only; otherwise repetition-level paired bootstrap.','Dependent cycles: descriptive differences only; no independent-repetition confidence interval.')
    if experiment.config.algorithm=='actor_critic':
        text=text.replace('Model dimensions: fixed=15, contextual=135.','Model dimensions: both variants have Actor=773 and Critic=705 neural parameters (1,478 total).')
        text=text.replace('Initial uniform policies are evaluated','Initial uniform-mean neural policies are evaluated')
        text += '\nActor-Critic: contextual TD(0) customer updates; fixed daily batch updates. Gamma=1 within a finite day. Validation and evaluation use frozen mean policies. Training rows contain on-policy outcomes; use held-out dates for model comparison. Adam and private RNG states are checkpointed.\n'
    if 'no_recommendation' in experiment.config.models:
        text += '\nUnassisted customer-choice control: baseline_comparison.csv reports paired evaluation effects per scenario and lambda. All arms use the same decision-time formula. Baseline has no neural parameters, optimizer updates or recommendation weights. Independent-repetition confidence intervals require at least five independent repetitions; continual cycles remain descriptive.\n'
    atomic_bytes(target/'report.md',text.encode('utf-8'));write_json(target/'complete.json',completion)
    return target
