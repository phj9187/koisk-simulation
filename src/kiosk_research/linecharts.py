"""Read-only line charts for scores and learned recommendation functions."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from kiosk_research.config import Config
from kiosk_research.persistence import read_json,load_checkpoint,atomic_bytes,write_json
from kiosk_research.policy import Policy,COMPONENTS
from kiosk_research.convergence import fixed_probes


def load_snapshot(root,experiment_id):
    if not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in experiment_id):raise ValueError('invalid ID')
    root=Path(root).resolve();directory=root/'experiments'/experiment_id
    metadata=read_json(directory/'metadata.json');identity=metadata['identity']
    paths=sorted((root/'checkpoints'/experiment_id).glob('day_*.json'),reverse=True)
    if not paths:raise ValueError('no checkpoint')
    state=load_checkpoint(paths[0])
    if state['identity']!=identity:raise ValueError('checkpoint identity mismatch')
    frames=[]
    for part in state['parts']:
        path=directory/part['path']
        if not path.resolve().is_relative_to(directory.resolve()):raise ValueError('result path outside experiment')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=part['sha256']:raise ValueError('corrupt result shard')
        frames.append(pd.read_parquet(path))
    frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    return SimpleNamespace(root=root,id=experiment_id,directory=directory,metadata=metadata,
        identity=identity,state=state,config=Config(**identity['config']).validate(require_source=False),export_metrics=lambda:frame)


def export_lines(run,target=None):
    frame=run.export_metrics()
    if frame.empty:raise ValueError('no completed dates')
    target=Path(target or run.directory/'line_charts'/('day-%06d'%run.state['day']))
    target.mkdir(parents=True,exist_ok=True)
    metrics=['customer','store','reward']
    daily=frame.groupby(['day','phase','store_id','scenario','model'],as_index=False)[metrics].mean()
    # Do not mix changing policies across dependent cycles into one function.
    evaluated=frame[frame.phase.eq('evaluation')]
    points=evaluated[evaluated.repetition.eq(evaluated.repetition.max())] if not evaluated.empty else evaluated
    curve=points.groupby(['store_id','scenario','model','lambda_value'],as_index=False)[metrics].mean()
    curve['cycle']=int(points.repetition.max()) if not points.empty else 0
    curve['interpretation']='latest available frozen evaluation cycle; synthetic outcomes'
    rows=[]
    from kiosk_sim.models import Customer
    from kiosk_research.worker import restore_store
    customer=Customer((.5,.5,.5),.7,.6,None,.5,20.,'mixed')
    for case in run.state['cases']:
        shop=restore_store(case['template'])
        for key,value in run.state['stores'][str(case['id'])]['policies'].items():
            scenario,model=key.split(':');policy=Policy.restore(value)
            for lam in np.linspace(0,1,101):
                rows.append({'store_id':case['id'],'scenario':scenario,'model':model,'lambda_value':float(lam),
                    **dict(zip(COMPONENTS,policy.weights(customer,shop,0.,float(lam))))})
    weights=pd.DataFrame(rows)
    for name,data in [('evaluation_functions.csv',curve),('evaluation_learning_curves.csv',daily),('recommendation_weight_functions.csv',weights)]:
        atomic_bytes(target/name,data.to_csv(index=False).encode('utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chart_names=[]
    colors={'customer':'#2563eb','store':'#db2777','reward':'#059669'}
    for scenario in sorted(run.config.scenarios):
        fig,axes=plt.subplots(len(run.config.models),2,figsize=(13,4*len(run.config.models)),squeeze=False)
        for row,model in enumerate(run.config.models):
            scores=curve[(curve.scenario==scenario)&(curve.model==model)]
            if not scores.empty:
                scores=scores.groupby('lambda_value')[metrics].mean()
                for metric in metrics:axes[row,0].plot(scores.index,scores[metric],'.-',label=metric,color=colors[metric])
            trend=daily[(daily.scenario==scenario)&(daily.model==model)].groupby('day')[metrics].mean()
            for metric in metrics:axes[row,1].plot(trend.index,trend[metric],label=metric,color=colors[metric],linewidth=1)
            axes[row,0].set(title=model+' / latest frozen evaluation',xlabel='Customer-store balance lambda',ylabel='Score [0,1]',ylim=(0,1))
            axes[row,1].set(title=model+' / daily sampled-input mean (descriptive)',xlabel='Simulation day',ylabel='Score [0,1]',ylim=(0,1))
            for ax in axes[row]:ax.grid(alpha=.2);ax.legend()
        fig.suptitle(scenario+' | equal-store means | synthetic, not measured satisfaction')
        fig.tight_layout();name='evaluation_'+scenario+'.png';fig.savefig(target/name,dpi=140);plt.close(fig);chart_names.append(name)
    diagnostics=[read_json(p) for p in sorted((run.directory/'convergence'/'diagnostics').glob('day_*.json'))]
    stable=[{k:v for k,v in r.items() if k in ('day','training_dates','tolerance','max_weight_tv_bound','max_probe_score_range','stable_policies','total_policies')}
        for r in diagnostics if 'max_weight_tv_bound' in r]
    if stable:
        trace=pd.DataFrame(stable);atomic_bytes(target/'convergence_trace.csv',trace.to_csv(index=False).encode('utf-8'))
        fig,ax=plt.subplots(figsize=(11,4))
        ax.plot(trace.training_dates,trace.max_weight_tv_bound,label='Worst-policy weight TV bound')
        ax.plot(trace.training_dates,trace.max_probe_score_range,label='Worst-policy probe-score range')
        ax.axhline(run.config.convergence_tolerance,color='gray',ls='--',label='Stopping tolerance')
        ax.set(xlabel='Actual training dates (frozen dates excluded)',ylabel='Full-window change',title='100-training-date functional stability, not global optimality')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(target/'convergence.png',dpi=140);plt.close(fig);chart_names.append('convergence.png')
    data={'scores':curve.to_dict('records'),'weights':weights.to_dict('records'),'day':run.state['day'],
          'scenarios':sorted(run.config.scenarios),'models':run.config.models,
          'stores':[c['id'] for c in run.state['cases']]}
    template=Path(__file__).with_name('linecharts_template.html').read_text(encoding='utf-8')
    # JSON is data in a non-executable script node, escaped against closing tags.
    html=template.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    atomic_bytes(target/'index.html',html.encode('utf-8'))
    write_json(target/'complete.json',{'day':run.state['day'],'identity':run.identity,'charts':chart_names,
        'evaluation_last_day':None if points.empty else int(points.day.max()),
        'note':'evaluator alpha/beta are fixed assumptions; learned w(lambda,context) is a recommendation function'})
    return target


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--experiment',required=True)
    args=parser.parse_args();print(export_lines(load_snapshot(args.root,args.experiment)))
