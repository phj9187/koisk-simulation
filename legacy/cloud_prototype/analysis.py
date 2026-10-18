"""Aggregate metrics and paired uncertainty at independent-cycle level."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from kiosk_cloud.storage import atomic_write, json_write

METRICS = ('score', 'customer_satisfaction', 'store_satisfaction', 'profit', 'served', 'wait_minutes', 'waste')


def weighted_summary(frame, store_weights):
    records = []
    for (cycle, phase, model), rows in frame.groupby(['cycle', 'phase', 'model']):
        weights = rows.store_id.map(store_weights).to_numpy(dtype=float)
        row = dict(cycle=int(cycle), phase=phase, model=model)
        for key in METRICS:
            row[key] = float(np.average(rows[key], weights=weights))
        row['elapsed_seconds'] = float(rows.elapsed_seconds.sum())
        row['training_observations'] = int(rows.training_observations.sum())
        row['simulator_evaluations'] = int(rows.simulator_evaluations.sum())
        initial = rows.initial_score.notna()
        row['initial_score'] = float(np.average(rows.loc[initial, 'initial_score'], weights=weights[initial])) if initial.any() else None
        records.append(row)
    return pd.DataFrame(records)


def paired_comparison(summary, completed_cycles, seed=42):
    final = summary[(summary.phase == 'evaluation') & summary.cycle.isin(completed_cycles)]
    if final.empty: return {'status': 'no_completed_final_evaluation', 'independent_cycles': 0}
    paired = final.pivot(index='cycle', columns='model', values='score').dropna()
    if not {'fixed', 'contextual'} <= set(paired): return {'status': 'both_models_required', 'independent_cycles': 0}
    differences = (paired.contextual - paired.fixed).to_numpy()
    result = {'status': 'descriptive_only' if len(differences) < 5 else 'paired_cycle_bootstrap',
              'independent_cycles': len(differences), 'mean_contextual_minus_fixed': float(differences.mean()),
              'differences': differences.tolist(), 'ci95': None,
              'note': 'Conditional on the same sampled stores and synthetic environment; dates/stores are not treated as independent repetitions.'}
    if len(differences) >= 5:
        rng = np.random.default_rng(seed)
        means = rng.choice(differences, (5000, len(differences)), replace=True).mean(axis=1)
        result['ci95'] = np.quantile(means, [.025, .975]).tolist()
    return result


def plot_results(frame, target, store_weights):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    panels = [('customer_satisfaction','Customer satisfaction (served customers)'),
              ('score','Common objective (train / frozen eval)'),
              ('store_satisfaction','Store satisfaction'),
              ('profit','Daily profit'), ('wait_minutes','Total wait (minutes)'), ('waste','Weighted leftover proxy')]
    for ax, (metric,title) in zip(axes.flat,panels):
        for model,rows in frame.groupby('model'):
            trend = rows.assign(analysis_weight=rows.store_id.map(store_weights)).groupby('day').apply(lambda x: np.average(x[metric],weights=x.analysis_weight), include_groups=False)
            ax.plot(trend.index,trend.values,label=model)
        ax.set(title=title,xlabel='Synchronized day'); ax.grid(alpha=.2)
    ax=axes.flat[6]
    summary = frame[frame.phase=='evaluation'].groupby('model').customer_satisfaction.mean()
    if not summary.empty: ax.bar(summary.index,summary.values)
    ax.set_title('Unweighted descriptive held-out satisfaction')
    ax=axes.flat[7]
    for model,rows in frame.groupby('model'):
        for key in ('weight_affinity','weight_waste_risk','weight_margin','weight_negative_wait'):
            trend=rows.groupby('day')[key].mean()
            ax.plot(trend.index,trend.values,label=model+':'+key[7:])
    ax.set_title('Weights at zero-centered reference context');ax.legend(fontsize=6)
    ax=axes.flat[8]
    pivot=frame.pivot_table(index=['cycle','day','store_id'],columns='model',values='score')
    if {'fixed','contextual'} <= set(pivot):
        difference=(pivot.contextual-pivot.fixed).groupby('day').mean()
        ax.plot(difference.index,difference.values);ax.axhline(0,color='gray',ls='--')
    ax.set_title('Descriptive contextual − fixed objective')
    for a in axes.flat[:6]: a.legend(fontsize=8)
    fig.suptitle('Synthetic kiosk research · shared daily objective · phases shown in daily_metrics.parquet')
    fig.tight_layout()
    fig.savefig(target,dpi=140);plt.close(fig)


def export_summary(experiment, target_root):
    frame=experiment.export_data()
    if frame.empty: raise ValueError('no completed days to summarize')
    target=Path(target_root)/experiment.id
    # Unique snapshot name allows repeated exports without overwriting prior reports.
    target=target/f'day-{experiment.study.day:06d}'
    if target.exists():
        raise FileExistsError('summary snapshot already exists; preserve it or use a different target root')
    target.mkdir(parents=True)
    weights={c.store_id:c.features.get('analysis_weight',1.) for c in experiment.study.cases}
    summary=weighted_summary(frame,weights)
    atomic_write(target/'summary.csv',summary.to_csv(index=False).encode('utf-8'))
    size=experiment.config.train_days+experiment.config.validation_days+experiment.config.evaluation_days
    completed=list(range(1,experiment.study.day//size+1))
    comparison=paired_comparison(summary,completed,experiment.config.seed)
    json_write(target/'statistics.json',comparison)
    json_write(target/'metadata.json',{**experiment.metadata,'completed_day':experiment.study.day,'stop_reason':experiment.study.stop_reason,
                                     'completed_cycles':completed, 'config':experiment.config.as_dict()})
    if experiment.config.plots: plot_results(frame,target/'comparison.png',weights)
    text=f'''# {experiment.id} — day {experiment.study.day}

Synthetic outcomes; not a claim about actual customer satisfaction.

- Code commit: `{experiment.identity['git_commit']}`
- Source SHA256: `{experiment.identity['source_sha256']}`
- Drive data: `{experiment.directory}`
- Train: paired incumbent/candidate complete-day evaluations, two calls per model/store/day.
- Validation/final evaluation: frozen policies, separate dates, no optimizer updates. One trained-policy and one initial-policy call per model/day.
- Independent completed cycles: {len(completed)}. Each cycle starts models from scratch with new external dates; the sampled store population is held fixed.
- Customer satisfaction is conditional on served customers; served counts and zero-visitor stores are retained.
- Contextual policy has 36 parameters; fixed policy has 4. Equal simulator budgets do not imply equal compute time or parameter complexity.
- Only completed final-evaluation cycles enter uncertainty estimates. Fewer than five cycles: descriptive results only.
- Details: `statistics.json`, `summary.csv`, full per-day metrics on Drive.

These configurable objective coefficients are provisional assumptions, not a uniquely optimal social tradeoff.
'''
    atomic_write(target/'report.md',text.encode())
    # Independent-cycle distribution and CI, separate from day/store descriptive panels.
    if experiment.config.plots and comparison.get('differences'):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4));ax.hist(comparison['differences'],bins=min(12,len(comparison['differences'])))
        ax.axvline(0,color='gray',ls='--');ax.set(title='Independent-cycle paired final objective differences',xlabel='Contextual minus fixed',ylabel='Cycles')
        if comparison['ci95'] is not None:
            lo,hi=comparison['ci95'];ax.axvspan(lo,hi,alpha=.15,label='95% bootstrap CI of mean');ax.legend()
        fig.tight_layout();fig.savefig(target/'repetitions.png',dpi=140);plt.close(fig)
    return target
