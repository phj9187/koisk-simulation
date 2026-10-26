"""Synchronous unattended execution for Colab. No hidden background service."""
from pathlib import Path
from datetime import datetime,timezone
from kiosk_research.config import Config
from kiosk_research.engine import Experiment
from kiosk_research.analysis import export
from kiosk_research.persistence import read_json,write_json


def status(root,experiment_id=None):
    if experiment_id:
        if not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in experiment_id):raise ValueError('invalid experiment ID')
        path=Path(root)/'experiments'/experiment_id/'progress.json'
    else:path=Path(root)/'latest_run.json'
    return read_json(path) if path.exists() else {'status':'not_started'}


def run_session(root,config=None,resume_id=None,day_budget=None):
    run=Experiment.resume(root,resume_id) if resume_id else Experiment(config or Config(),root)
    print('Experiment:',run.id,flush=True)
    print('Recovery commit:',run.identity['git_commit'],flush=True)
    print('Drive status:',run.directory/'progress.json',flush=True)
    def checkpoint(r):
        print('Day',r.state['day'],r.config.stage(r.state['day'])[1],flush=True)
    try:
        run.run(day_budget=day_budget,day_callback=checkpoint)
        summary=export(run)
        write_json(run.directory/'session_result.json',{'experiment_id':run.id,'summary':str(summary),
                   'stop_reason':run.state['stop_reason'],'updated_at_utc':datetime.now(timezone.utc).isoformat()})
        print('Summary:',summary,flush=True)
    except BaseException as exc:
        write_json(run.directory/'session_error.json',{'experiment_id':run.id,'error_type':type(exc).__name__,
                   'updated_at_utc':datetime.now(timezone.utc).isoformat(),'instruction':'resume with original commit and experiment ID; do not restart as new'})
        raise
    return run
