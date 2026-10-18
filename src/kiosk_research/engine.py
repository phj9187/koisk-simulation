from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import multiprocessing
from pathlib import Path
import shutil
import uuid
import warnings
import pandas as pd
from kiosk_sim.lhs_study import LHSStudy,StudyConfig
from kiosk_cloud.provenance import runtime_info,source_hash,git_version
from kiosk_research.config import Config
from kiosk_research.persistence import layout,exclusive_lock,read_json,write_json,save_checkpoint,load_checkpoint,parquet,atomic_bytes
from kiosk_research.worker import store_day

class Experiment:
    def __init__(self,config,root,experiment_id=None,resume=False):
        self.config=config.validate(require_source=not resume);self.root=layout(root)
        self.id=experiment_id or 'EXP-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        if not self.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in self.id):raise ValueError('invalid experiment ID')
        self.directory=self.root/'experiments'/self.id;self.checkpoints=self.root/'checkpoints'/self.id
        self.runtime=runtime_info()
        self.identity={'runtime_versions':{'python':self.runtime['python'],'libraries':self.runtime['libraries']},'schema':2,'id':self.id,'experiment_id':self.id,'config':config.to_dict(),'source_sha256':source_hash(),'git_commit':git_version()['git_commit']}
        if resume:
            if not self.directory.is_dir():raise FileNotFoundError('experiment not found')
            with exclusive_lock(self.directory/'RUNNING.lock'):self._load()
        else:
            self.directory.mkdir(exist_ok=False);self.checkpoints.mkdir(exist_ok=False)
            with exclusive_lock(self.directory/'RUNNING.lock'):
                lc=StudyConfig(store_count=config.stores,lhs_strata=8,seed=config.seed,visitor_distribution='empirical_donors_2025' if config.source=='empirical' else 'uniform_scenario',survey_path=config.survey_path)
                original=LHSStudy(lc)
                cases=[{'id':c.store_id,'features':c.features,'template':asdict(c.template)} for c in original.cases]
                self.metadata={'identity':self.identity,'environment':self.runtime,'sampling_audit':original.sampling_audit,'data_path':str(self.directory),
                               'assumptions':'Synthetic customer behavior/satisfaction; no real-world effectiveness claim.',
                               'protocol':'lambda-conditioned weights; paired daily random search; frozen validation/evaluation; fresh independent repetitions',
                               'rng':'All exogenous and optimizer streams are deterministically regenerated from seed/day/store/purpose; no hidden global RNG state.'}
                self.state={'identity':self.identity,'day':0,'stop_reason':'','cases':cases,'stores':{},'parts':[],'raw_parts':[]}
                write_json(self.directory/'metadata.json',self.metadata);self._save()
        self._frontier=self._signature()

    @classmethod
    def resume(cls,root,experiment_id):
        if not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in experiment_id):raise ValueError('invalid experiment ID')
        meta=read_json(Path(root)/'experiments'/experiment_id/'metadata.json')
        return cls(Config(**meta['identity']['config']),root,experiment_id,True)

    def _signature(self):return sorted((str(p.relative_to(self.checkpoints)),p.stat().st_size,p.stat().st_mtime_ns) for p in self.checkpoints.rglob('*.json'))

    def _valid(self,path,check_parts=False):
        state=load_checkpoint(path)
        if state['identity']!=self.identity:raise ValueError('identity mismatch')
        if check_parts:
            for part in state['parts']+state['raw_parts']:
                p=self.directory/part['path']
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=part['sha256']:raise ValueError('missing/corrupt result shard '+part['path'])
        return state

    def _load(self):
        self.metadata=read_json(self.directory/'metadata.json')
        if self.metadata['identity']!=self.identity:raise ValueError('Code, commit or settings differ; restore original version or use a new experiment ID')
        for path in sorted(self.checkpoints.glob('day_*.json'),reverse=True):
            try:self.state=self._valid(path,True);return
            except (OSError,ValueError,KeyError) as exc:warnings.warn(f'Skipping invalid checkpoint {path.name}: {exc}')
        raise ValueError('No valid checkpoint. Data preserved; restore a backup rather than silently restarting.')

    def _save(self):
        save_checkpoint(self.checkpoints/f"day_{self.state['day']:06d}.json",self.state)
        valid=[]
        for p in sorted(self.checkpoints.glob('day_*.json')):
            try:self._valid(p);valid.append(p)
            except (OSError,ValueError,KeyError):pass
        for path in valid[:-2]:path.unlink()
        if len(valid)>=2:
            cutoff=int(valid[-2].stem.split('_')[1])
            for folder in self.checkpoints.glob('partial_*'):
                if int(folder.name.split('_')[1])<=cutoff:shutil.rmtree(folder)
        model_state={'day':self.state['day'],'stores':self.state['stores']}
        write_json(self.root/'models'/self.id/'latest.json',model_state)
        if self.state['day']>0:
            rep,phase,offset=self.config.stage(self.state['day'])
            if phase=='train' and offset==self.config.train_days:write_json(self.root/'models'/self.id/f'repetition_{rep:04d}_trained.json',model_state)
        write_json(self.directory/'progress.json',{'day':self.state['day'],'stop_reason':self.state['stop_reason']})

    def request_stop(self):atomic_bytes(self.directory/'STOP',b'stop after complete date\n')
    def records(self):return pd.concat([pd.read_parquet(self.directory/p['path']) for p in self.state['parts']],ignore_index=True) if self.state['parts'] else pd.DataFrame()

    def _next_day(self,progress=None):
        day=self.state['day']+1;folder=self.checkpoints/f'partial_{day:06d}';folder.mkdir(exist_ok=True)
        done={};tasks=[]
        for case in self.state['cases']:
            path=folder/f"store_{case['id']:04d}.json"
            try:
                saved=load_checkpoint(path)
                if saved['identity']!=self.identity or saved['result']['day']!=day or saved['result']['id']!=case['id']:raise ValueError('partial identity mismatch')
                done[case['id']]=saved['result']
            except (OSError,ValueError,KeyError):tasks.append((self.config,case,self.state['stores'].get(str(case['id'])),day))
        def commit(result):
            save_checkpoint(folder/f"store_{result['id']:04d}.json",{'identity':self.identity,'result':result});done[result['id']]=result
            if progress:progress(day,len(done),len(self.state['cases']))
        if self.config.workers==1:
            for task in tasks:commit(store_day(task))
        elif tasks:
            with ProcessPoolExecutor(max_workers=self.config.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(store_day,t) for t in tasks]
                for future in as_completed(futures):commit(future.result())
        rows=[];logs=[];stores={}
        for case in self.state['cases']:
            result=done[case['id']];rows.extend(result['rows']);logs.extend(result['logs'])
            stores[str(case['id'])]={k:v for k,v in result.items() if k not in ('rows','logs')}
        relative=f'detailed_results/daily_{day:06d}.parquet';digest=parquet(self.directory/relative,rows);raw=None
        if logs:
            raw_path=f'detailed_results/events_{day:06d}.parquet';raw={'path':raw_path,'sha256':parquet(self.directory/raw_path,logs)}
        self.state['parts'].append({'path':relative,'sha256':digest})
        if raw:self.state['raw_parts'].append(raw)
        self.state['stores']=stores;self.state['day']=day
        if (self.directory/'STOP').exists():self.state['stop_reason']='paused'
        elif self.config.complete(day):self.state['stop_reason']='completed'

    def run(self,day_budget=None,progress=None,day_callback=None):
        if day_budget is not None and (type(day_budget) is not int or day_budget<1):raise ValueError('invalid day budget')
        with exclusive_lock(self.directory/'RUNNING.lock'):
            if self._signature()!=self._frontier:raise RuntimeError('Disk state changed; reload this experiment')
            if self.state['stop_reason']=='completed':return
            if self.state['stop_reason']=='paused':
                (self.directory/'STOP').unlink(missing_ok=True);self.state['stop_reason']=''
            count=0
            try:
                while not self.state['stop_reason']:
                    self._next_day(progress);count+=1
                    if day_budget and count>=day_budget and not self.state['stop_reason']:self.state['stop_reason']='paused'
                    _,_,offset=self.config.stage(self.state['day'])
                    boundary=offset in (self.config.train_days,self.config.train_days+self.config.validation_days)
                    if self.state['day']%self.config.checkpoint_every==0 or self.state['stop_reason'] or boundary:self._save()
                    if day_callback:day_callback(self)
            except BaseException:
                self._save();raise
            finally:self._frontier=self._signature()

    def export_metrics(self):
        frame=self.records()
        if not frame.empty:parquet(self.directory/'daily_metrics.parquet',frame.to_dict(orient='records'))
        return frame
