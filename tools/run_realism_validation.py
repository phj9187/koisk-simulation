"""Run a new empirical pilot, export diagnostics and retain a reviewable audit."""
import argparse
from pathlib import Path
import json
from kiosk_research.config import Config
from kiosk_research.engine import Experiment
from kiosk_research.analysis import export
from kiosk_research.convergence import fixed_probes
from kiosk_research.model_board import ModelBoard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='work/realism_inputs/pilot_local.yaml')
    parser.add_argument('--root',default='work/realism-pilot');parser.add_argument('--resume')
    args=parser.parse_args()
    run=Experiment.resume(args.root,args.resume) if args.resume else Experiment(Config.load(args.config),args.root)
    print('NEW EXPERIMENT',run.id,flush=True)
    print('INPUT HASHES',json.dumps(run.identity['input_files'],ensure_ascii=True),flush=True)
    print('STORE MEANS',[c['features']['visitor_mean'] for c in run.state['cases']],flush=True)
    run.run(progress=lambda day,done,total:print('EPISODE',day,'STORE',done,'/',total,flush=True),
            day_callback=lambda r:print('COMPLETED',r.state['day'],r.config.search_stage(r.state['day']),flush=True))
    target=export(run)
    board=ModelBoard(run,None,display=False)
    _,weights,scores=fixed_probes(run,True)
    frame=run.records();daily=frame.drop_duplicates(['day','store_id'])
    audit={'experiment_id':run.id,'completed_episodes':run.state['day'],'calendar_days':int(frame.horizon_days.iloc[0])*run.state['day'],
           'source':run.config.source,'stores':len(run.state['cases']),'input_files':run.identity['input_files'],
           'visitors_min':int(daily.visitors.min()),'visitors_max':int(daily.visitors.max()),
           'visitor_series':{str(store):g.visitors.astype(int).tolist() for store,g in daily.groupby('store_id')},
           'opening_hours':sorted(frame.opening_hours.unique().tolist()),'arrival_sources':sorted(frame.arrivals_source.unique().tolist()),
           'weekday_sources':sorted(frame.weekday_source.unique().tolist()),'score_probe_shape':list(scores.shape),
           'weight_probe_shape':list(weights.shape),'probes_repeatable':None,
           'summary':str(target.resolve()),'dashboard':str((board.folder/'model_board.html').resolve()),
           'colab_browser_verified':False,'interpretation':'functional pilot only, not convergence or effectiveness proof'}
    _,again_w,again_s=fixed_probes(run,True)
    audit['probes_repeatable']=bool((weights==again_w).all() and (scores==again_s).all())
    path=run.directory/'realism_validation.json';path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VALIDATION',json.dumps(audit,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
