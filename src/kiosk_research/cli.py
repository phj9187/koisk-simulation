import argparse
from pathlib import Path
from kiosk_research.config import Config
from kiosk_research.engine import Experiment
from kiosk_research.analysis import export
from kiosk_research.persistence import atomic_bytes

def main():
    p=argparse.ArgumentParser(description='Lambda-conditioned Pareto research')
    p.add_argument('--config',default='config/research_smoke.yaml');p.add_argument('--root',default='runs');p.add_argument('--resume');p.add_argument('--stop');p.add_argument('--days',type=int)
    p.add_argument('--target-day',type=int)
    args=p.parse_args()
    if args.days is not None and args.days<1:p.error('--days must be positive')
    if args.stop:
        if not args.stop or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.stop):p.error('invalid ID')
        path=Path(args.root)/'experiments'/args.stop
        if not path.is_dir():p.error('unknown experiment')
        atomic_bytes(path/'STOP',b'stop\n');return
    run=Experiment.resume(args.root,args.resume) if args.resume else Experiment(Config.load(args.config),args.root)
    print('Experiment:',run.id,flush=True)
    run.run(day_budget=args.days,target_day=args.target_day,day_callback=lambda r:print('Day',r.state['day'],r.config.stage(r.state['day'])[1],flush=True))
    print('Summary:',export(run))

if __name__=='__main__':main()
