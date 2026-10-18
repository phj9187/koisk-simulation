import argparse
from kiosk_cloud.config import CloudConfig
from kiosk_cloud.runner import Experiment
from kiosk_cloud.analysis import export_summary


def main():
    parser=argparse.ArgumentParser(description='New paired-objective kiosk research runner')
    parser.add_argument('--config',default='config/default.yaml')
    parser.add_argument('--data-root',default='runs')
    parser.add_argument('--resume',metavar='EXPERIMENT_ID')
    parser.add_argument('--days',type=int,help='Pause after this many additional complete dates')
    parser.add_argument('--stop',metavar='EXPERIMENT_ID',help='Request stop in another active runtime/process')
    args=parser.parse_args()
    if args.days is not None and args.days<1: parser.error('--days must be positive')
    if args.stop:
        from pathlib import Path
        from kiosk_cloud.storage import atomic_write
        path=Path(args.data_root)/'experiments'/args.stop
        if not path.is_dir(): parser.error('unknown experiment')
        atomic_write(path/'STOP',b'stop\n');return
    run=Experiment.resume(args.data_root,args.resume) if args.resume else Experiment(CloudConfig.load(args.config),args.data_root)
    print('Experiment:',run.id,flush=True)
    run.run(day_budget=args.days,day_callback=lambda r: print('Completed day',r.study.day,'phase',r.config.stage(r.study.day)[1],flush=True))
    print('Summary:',export_summary(run,run.root/'experiments'/run.id/'summary_exports'))


if __name__=='__main__': main()
