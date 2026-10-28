import copy
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
from kiosk_research.persistence import unique_records

class Experiment:
    def __init__(self,config,root,experiment_id=None,resume=False):
        config=copy.deepcopy(config)
        self.config=config.validate(require_source=not resume);self.root=layout(root)
        self.id=experiment_id or 'EXP-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        if not self.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in self.id):raise ValueError('invalid experiment ID')
        self.directory=self.root/'experiments'/self.id;self.checkpoints=self.root/'checkpoints'/self.id
        self.runtime=runtime_info()
        if config.algorithm=='actor_critic':
            import torch
            self.runtime['libraries']['torch']=torch.__version__
        self.identity={'runtime_versions':{'python':self.runtime['python'],'libraries':self.runtime['libraries']},'schema':4,'id':self.id,'experiment_id':self.id,'config':config.to_dict(),'source_sha256':source_hash(),'git_commit':git_version()['git_commit']}
        if config.algorithm=='actor_critic':self.identity['schema']=5
        if 'no_recommendation' in config.models:self.identity['schema']=6
        if config.lambda_search=='pareto_two_stage':self.identity['schema']=7
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
                if config.continual:
                    self.metadata['protocol']='continual learning across dependent cycles; frozen validation/evaluation within each cycle; not independent repetitions'
                if config.algorithm=='actor_critic':
                    self.metadata['protocol']='Dirichlet actor + TD(0) critic, gamma=1 finite day; contextual customer updates; fixed daily batch update; frozen validation/evaluation; '+('dependent continual cycles' if config.continual else 'independent repetitions')
                    self.metadata['rng']='Seeded exogenous streams; private per-policy torch CPU RNG and Adam state saved in checkpoints.'
                if 'no_recommendation' in config.models:
                    self.metadata['protocol']+='; unassisted customer-choice control; equal decision-time formula in every arm; no automatic migration of prior experiments'
                if config.lambda_search=='pareto_two_stage':
                    self.metadata['protocol']+='; coarse training / frozen selection / refined training / frozen validation / held-out evaluation; validation-only shared Pareto-knee grid per scenario'
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
        self._valid(self.checkpoints/f"day_{self.state['day']:06d}.json",True)
        removed=[]
        keep=self.config.checkpoint_keep
        if self.config.auto_cleanup:
            valid=[]
            for p in sorted(self.checkpoints.glob('day_*.json')):
                try:self._valid(p,True);valid.append(p)
                except (OSError,ValueError,KeyError):pass
            # Persist and verify a cleanup plan before deleting redundant snapshots.
            old=valid[:-keep]
            folders=[]
            if len(valid)>=keep:
                cutoff=int(valid[-keep].stem.split('_')[1])
                folders=[p for p in self.checkpoints.glob('partial_*') if p.name[8:].isdigit() and int(p.name[8:])<=cutoff and not p.is_symlink()]
            plan={'day':self.state['day'],'checkpoints':[p.name for p in old], 'partial_directories':[p.name for p in folders]}
            write_json(self.directory/'cleanup_plan.json',plan)
            for path in old:path.unlink();removed.append(path.name)
            for folder in folders:shutil.rmtree(folder);removed.append(folder.name)
            # Delete only unreferenced byte-identical copies of committed data.
            references={p['path']:p['sha256'] for p in self.state['parts']+self.state['raw_parts']}
            protected=set(references)
            for checkpoint in valid[-keep:]:
                saved=self._valid(checkpoint,True)
                protected.update(p['path'] for p in saved['parts']+saved['raw_parts'])
            known=set(references.values())
            for path in (self.directory/'detailed_results').glob('*.parquet'):
                if path.is_symlink() or path.relative_to(self.directory).as_posix() in protected:continue
                if hashlib.sha256(path.read_bytes()).hexdigest() in known:
                    path.unlink();removed.append(path.relative_to(self.directory).as_posix())
            write_json(self.directory/'cleanup_last.json',{'day':self.state['day'],'removed':removed,'retained_checkpoints':[p.name for p in valid[-keep:]],'policy':'verified redundant snapshots and byte-identical unreferenced shards only'})
        model_state={'day':self.state['day'],'stores':self.state['stores']}
        write_json(self.root/'models'/self.id/'latest.json',model_state)
        if self.state['day']>0:
            rep,phase,offset=self.config.stage(self.state['day'])
            if phase=='train' and offset==self.config.training_end_offset:write_json(self.root/'models'/self.id/f'repetition_{rep:04d}_trained.json',model_state)
        self._progress(checkpoint_day=self.state['day'])

    def _progress(self,checkpoint_day=None,active_day=None,done=None,error=None):
        previous={}
        try:previous=read_json(self.directory/'progress.json')
        except (OSError,ValueError):pass
        day=self.state['day']
        total=None if self.config.repetitions is None else self.config.repetitions*self.config.cycle_days
        payload={'experiment_id':self.id,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
                 'git_commit':self.identity['git_commit'],'source_sha256':self.identity['source_sha256'],
                 'day':day,'checkpoint_day':checkpoint_day if checkpoint_day is not None else previous.get('checkpoint_day',0),
                 'active_day':active_day,'completed_stores':done,'total_stores':self.config.stores,
                 'total_days':total,'phase':self.config.stage(active_day or max(1,day))[1],
                 'search_stage':self.config.search_stage(active_day or max(1,day)),
                 'stop_reason':self.state['stop_reason'],'status':'error' if error else self.state['stop_reason'] or 'ready',
                 'error_type':error,'data_path':str(self.directory),
                 'learning_mode':'continual' if self.config.continual else 'independent_repetitions'}
        if active_day is not None:payload['status']='running'
        write_json(self.directory/'progress.json',payload)
        write_json(self.root/'latest_run.json',payload)

    def request_stop(self):atomic_bytes(self.directory/'STOP',b'stop after complete date\n')
    def records(self):return pd.concat([pd.read_parquet(self.directory/p['path']) for p in self.state['parts']],ignore_index=True) if self.state['parts'] else pd.DataFrame()

    def _next_day(self,progress=None,should_stop=None):
        day=self.state['day']+1
        rep,_,offset=self.config.stage(day)
        plan=self.state.get('lambda_search',{}).get(str(rep))
        if self.config.search_stage(day)=='refine' and plan is None:raise ValueError('missing persisted selection plan')
        self._progress(active_day=day,done=0)
        folder=self.checkpoints/f'partial_{day:06d}';folder.mkdir(exist_ok=True)
        done={};tasks=[]
        for case in self.state['cases']:
            path=folder/f"store_{case['id']:04d}.json"
            try:
                saved=load_checkpoint(path)
                if saved['identity']!=self.identity or saved['result']['day']!=day or saved['result']['id']!=case['id']:raise ValueError('partial identity mismatch')
                done[case['id']]=saved['result']
            except (OSError,ValueError,KeyError):
                task=(self.config,case,self.state['stores'].get(str(case['id'])),day)
                tasks.append(task+(plan,) if self.config.lambda_search=='pareto_two_stage' else task)
        def commit(result):
            save_checkpoint(folder/f"store_{result['id']:04d}.json",{'identity':self.identity,'result':result});done[result['id']]=result
            self._progress(active_day=day,done=len(done))
            if progress:progress(day,len(done),len(self.state['cases']))
        if self.config.workers==1:
            for task in tasks:
                if should_stop and should_stop():return False
                commit(store_day(task))
        elif tasks:
            with ProcessPoolExecutor(max_workers=self.config.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(store_day,t) for t in tasks]
                for future in as_completed(futures):commit(future.result())
        rows=[];logs=[];stores={}
        for case in self.state['cases']:
            result=done[case['id']];rows.extend(result['rows']);logs.extend(result['logs'])
            stores[str(case['id'])]={k:v for k,v in result.items() if k not in ('rows','logs')}
        rows=unique_records(pd.DataFrame(rows),['day','store_id','scenario','model','lambda_value']).to_dict(orient='records')
        relative=f'detailed_results/daily_{day:06d}.parquet';digest=parquet(self.directory/relative,rows);raw=None
        if logs:
            raw_path=f'detailed_results/events_{day:06d}.parquet';raw={'path':raw_path,'sha256':parquet(self.directory/raw_path,logs)}
        self.state['parts'].append({'path':relative,'sha256':digest})
        if raw:self.state['raw_parts'].append(raw)
        self.state['stores']=stores;self.state['day']=day
        if self.config.lambda_search=='pareto_two_stage' and offset==self.config.coarse_days+self.config.validation_days:
            from kiosk_research.lambda_search import select_plan
            validation=pd.concat([pd.read_parquet(self.directory/p['path']) for p in self.state['parts'][-self.config.validation_days:]],ignore_index=True)
            selected=select_plan(validation,self.config,day)
            self.state.setdefault('lambda_search',{})[str(rep)]=selected
            write_json(self.directory/'lambda_search'/f'cycle_{rep:04d}.json',selected)
        if self.config.early_stopping and self.config.stage(day)[1]=='train':
            from kiosk_research.convergence import observe
            result=observe(self)
            self.state['convergence']=result
            if result['converged'] and self.config.search_stage(day) in ('fixed','refine'):self.state['stop_reason']='converged'
        if (self.directory/'STOP').exists():self.state['stop_reason']='paused'
        elif self.config.complete(day):self.state['stop_reason']='completed'
        return True

    def run(self,day_budget=None,progress=None,day_callback=None,target_day=None,deadline_utc=None):
        deadline=None
        if deadline_utc is not None:
            deadline=datetime.fromisoformat(deadline_utc.replace('Z','+00:00'))
            if deadline.tzinfo is None:raise ValueError('deadline requires timezone')
            if self.config.workers!=1:raise ValueError('deadline mode requires one worker for safe store-boundary stop')
        def expired():return deadline is not None and datetime.now(timezone.utc)>=deadline
        if day_budget is not None and (type(day_budget) is not int or day_budget<1):raise ValueError('invalid day budget')
        if target_day is not None and (type(target_day) is not int or target_day<1):raise ValueError('invalid absolute target day')
        with exclusive_lock(self.directory/'RUNNING.lock'):
            if self._signature()!=self._frontier:raise RuntimeError('Disk state changed; reload this experiment')
            if self.state['stop_reason'] in ('completed','converged'):return
            if target_day is not None and self.state['day']>=target_day:
                self.state['stop_reason']='target_reached';self._save();self._frontier=self._signature();return
            if self.state['stop_reason'] in ('paused','target_reached','deadline_reached'):
                (self.directory/'STOP').unlink(missing_ok=True);self.state['stop_reason']=''
            count=0
            try:
                while not self.state['stop_reason']:
                    if expired():
                        self.state['stop_reason']='deadline_reached';self._save();break
                    finished=self._next_day(progress,expired if deadline else None)
                    if finished is False:
                        self.state['stop_reason']='deadline_reached';self._save();break
                    count+=1
                    if target_day is not None and self.state['day']>=target_day:self.state['stop_reason']='target_reached'
                    if day_budget and count>=day_budget and not self.state['stop_reason']:self.state['stop_reason']='paused'
                    _,_,offset=self.config.stage(self.state['day'])
                    boundary=offset in self.config.phase_boundaries
                    if self.state['day']%self.config.checkpoint_every==0 or self.state['stop_reason'] or boundary:self._save()
                    if day_callback:day_callback(self)
            except BaseException as exc:
                self._save();self._progress(error=type(exc).__name__);raise
            finally:self._frontier=self._signature()

    def export_metrics(self):
        frame=self.records()
        # Day shards are canonical; avoid a growing duplicate of the entire history.
        if not frame.empty:frame=unique_records(frame,['day','store_id','scenario','model','lambda_value'])
        return frame
