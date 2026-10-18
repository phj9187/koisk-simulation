"""Synchronized store-days, durable partial recovery and day-boundary pause."""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import uuid
import warnings
import pandas as pd
from kiosk_sim.durable_study import read_snapshot
from kiosk_sim.lhs_study import LHSStudy
from kiosk_cloud.config import CloudConfig
from kiosk_cloud.provenance import runtime_info, source_hash, git_version
from kiosk_cloud.storage import drive_layout, json_write, atomic_write, parquet_write, snapshot_write, writer_lock
from kiosk_cloud.worker import run_store_day


class Experiment:
    def __init__(self, config, data_root, experiment_id=None, resume=False):
        self.config = config.validate()
        self.root = drive_layout(data_root)
        self.id = experiment_id or ('EXP-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        if not self.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in self.id):
            raise ValueError('invalid experiment id')
        self.directory = self.root / 'experiments' / self.id
        self.checkpoints = self.root / 'checkpoints' / self.id
        self.identity = {'schema_version': 1, 'experiment_id': self.id, 'config': config.as_dict(),
                         'source_sha256': source_hash(), 'git_commit': git_version()['git_commit']}
        if resume:
            if not self.directory.is_dir(): raise FileNotFoundError('experiment does not exist')
            with writer_lock(self.directory / 'RUNNING.lock'):
                self._load()
        else:
            self.directory.mkdir(exist_ok=False)
            self.checkpoints.mkdir(exist_ok=False)
            with writer_lock(self.directory / 'RUNNING.lock'):
                self.study = LHSStudy(config.legacy())
                # Old optimizer objects are not part of this new learning protocol.
                for case in self.study.cases:
                    case.fixed = None; case.contextual = None
                self.inventory = {}
                self.study.stop_reason = ''
                self.metadata = {'identity': self.identity, 'environment': runtime_info(),
                                 'sampling_audit': self.study.sampling_audit,
                                 'data_path': str(self.directory),
                                 'evaluation_type': 'fresh independent cycles; train / frozen validation / frozen final evaluation',
                                 'reward_note': 'Both policies use the same configured whole-day objective and paired incumbent/candidate random search. Equal simulator-call budgets; model parameter counts differ.',
                                 'rng_note': 'Python Random streams and optimizer perturbations regenerated from seed/day/store. No persistent global RNG; exact policy parameters and optimizer step counts are checkpointed.',
                                 'inventory_note': 'Original simulator deep-copies stock for each policy and resets each day. Templates and next-day seed are sufficient at day boundaries; terminal item stock is not exposed.',
                                 'raw_log_scope': 'Recommendation events only; no_recommendation has no legacy feedback events.'}
                json_write(self.directory / 'metadata.json', self.metadata)
                self._save()
        self.loaded_day = self.study.day
        self._valid_days = self._disk_valid_days()
        self._partial_loaded = self._partial_signature()

    @classmethod
    def resume(cls, data_root, experiment_id):
        directory = Path(data_root) / 'experiments' / experiment_id
        meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        return cls(CloudConfig(**meta['identity']['config']), data_root, experiment_id, resume=True)

    def _load(self):
        self.metadata = json.loads((self.directory / 'metadata.json').read_text(encoding='utf-8'))
        if self.metadata['identity'] != self.identity:
            raise ValueError('Config, code or commit differs: use the original code/config or a new experiment ID')
        paths = sorted(self.checkpoints.glob('day_*.checkpoint'), reverse=True)
        for path in paths:
            try:
                value = read_snapshot(path)
            except (OSError, ValueError, EOFError) as exc:
                warnings.warn(f'Corrupt checkpoint skipped: {path.name}: {exc}')
                continue
            if value['identity'] != self.identity:
                raise ValueError('checkpoint identity mismatch')
            self.study, self.inventory = value['study'], value['inventory']
            return
        raise ValueError('No valid checkpoint; original data preserved. Restore a trusted backup instead of starting over.')

    def _disk_valid_days(self):
        days = []
        for path in self.checkpoints.glob('day_*.checkpoint'):
            try:
                value = read_snapshot(path)
                if value['identity'] == self.identity: days.append(value['study'].day)
            except (OSError, ValueError, EOFError): pass
        return set(days)

    def _save(self):
        snapshot_write(self.checkpoints / f'day_{self.study.day:06d}.checkpoint',
                       {'identity': self.identity, 'study': self.study, 'inventory': self.inventory})
        # Keep two complete generations; partial days are retained since the older generation.
        valid = sorted(self._disk_valid_days())
        if len(valid) > 2:
            for day in valid[:-2]:
                (self.checkpoints / f'day_{day:06d}.checkpoint').unlink()
        valid = sorted(self._disk_valid_days())
        if len(valid) >= 2:
            for path in self.checkpoints.glob('partial_*'):
                day = int(path.name.split('_')[1])
                if day <= valid[-2]:
                    import shutil
                    shutil.rmtree(path)
        self.loaded_day = self.study.day
        self._valid_days = self._disk_valid_days()

    def request_stop(self):
        atomic_write(self.directory / 'STOP', b'stop after the current complete day\n')

    def records(self):
        return pd.DataFrame([row for case in self.study.cases for row in case.records])

    def _summary(self):
        rows = [r for c in self.study.cases for r in c.records if r['day'] == self.study.day]
        weights = {c.store_id: c.features.get('analysis_weight', 1.) for c in self.study.cases}
        summary = {'day': self.study.day}
        for model in self.config.models:
            selected = [r for r in rows if r['model'] == model]
            for key in ('score', 'customer_satisfaction', 'store_satisfaction'):
                summary[f'{model}_{key}'] = sum(r[key] * weights[r['store_id']] for r in selected) / sum(weights[r['store_id']] for r in selected)
        return summary

    def _next_day(self, progress=None):
        day = self.study.day + 1
        folder = self.checkpoints / f'partial_{day:06d}'
        folder.mkdir(exist_ok=True)
        done, pending = {}, []
        for case in self.study.cases:
            path = folder / f'store_{case.store_id:04d}.checkpoint'
            if path.exists():
                try:
                    value = read_snapshot(path)
                    if value['identity'] != self.identity or value['day'] != day or value['result'][0].store_id != case.store_id:
                        raise ValueError('partial identity mismatch')
                    done[case.store_id] = value['result']
                except (OSError, ValueError, EOFError) as exc:
                    warnings.warn(f'Invalid partial {path.name}: {exc}; recomputing deterministic day')
                    pending.append(case)
            else: pending.append(case)
        def commit(result):
            case = result[0]
            snapshot_write(folder / f'store_{case.store_id:04d}.checkpoint',
                           {'identity': self.identity, 'day': day, 'result': result})
            done[case.store_id] = result
            if progress: progress(day, len(done), len(self.study.cases))
        tasks = []
        for case in pending:
            slim = copy.copy(case); slim.records = []
            tasks.append((self.config, slim, day))
        if self.config.workers == 1:
            for task in tasks: commit(run_store_day(task))
        elif tasks:
            with ProcessPoolExecutor(max_workers=self.config.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(run_store_day, task) for task in tasks]
                for future in as_completed(futures): commit(future.result())
        cases, logs = [], []
        inventory = {}
        for old in self.study.cases:
            case, events, state = done[old.store_id]
            # Write the potentially failing detail shard before advancing coordinator state.
            logs.extend(events)
            cases.append(case)
            inventory[case.store_id] = state
        if logs:
            detailed = pd.DataFrame([{**{k:v for k,v in event.items() if k != 'payload'}, 'payload_json': json.dumps(event['payload'], ensure_ascii=False)} for event in logs])
            parquet_write(self.directory / 'detailed_results' / f'day_{day:06d}.parquet', detailed)
        for old, case in zip(self.study.cases, cases):
            case.records = old.records + case.records
        self.study.cases, self.inventory = cases, inventory
        self.study.day = day
        self.study.aggregate_history.append(self._summary())
        if (self.directory / 'STOP').exists(): self.study.stop_reason = 'paused'
        elif self.config.complete(day): self.study.stop_reason = 'completed'
        return self.study.aggregate_history[-1]

    def run(self, progress=None, day_callback=None, day_budget=None):
        """Blocking worker entrypoint; notebook UI dispatches it in a background thread."""
        with writer_lock(self.directory / 'RUNNING.lock'):
            # Another process must never advance a stale coordinator.
            if self._disk_valid_days() != self._valid_days:
                raise RuntimeError('Disk checkpoint changed; reload experiment')
            if self.study.stop_reason == 'completed':
                return self.records()
            if self.study.stop_reason == 'paused':
                (self.directory / 'STOP').unlink(missing_ok=True)
                self.study.stop_reason = ''
            # Refuse partial work from another stale writer via a disk frontier fingerprint.
            now = self._partial_signature()
            if hasattr(self, '_partial_loaded') and now != self._partial_loaded:
                raise RuntimeError('Partial work changed; reload experiment')
            count = 0
            try:
                while not self.study.stop_reason:
                    # A stop before first day still completes the next full date, by design.
                    self._next_day(progress)
                    count += 1
                    if day_budget is not None and count >= day_budget and not self.study.stop_reason:
                        self.study.stop_reason = 'paused'
                    if self.study.day % self.config.checkpoint_every == 0 or self.study.stop_reason:
                        self._save()
                        self.export_data()
                    if day_callback: day_callback(self)
            except BaseException:
                # Only completed dates are saved; partial stores already have durable snapshots.
                self._save()
                raise
            finally:
                self._partial_loaded = self._partial_signature()
        return self.records()

    def _partial_signature(self):
        return sorted((str(p.relative_to(self.checkpoints)), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in self.checkpoints.glob('partial_*/*.checkpoint'))

    def export_data(self):
        frame = self.records()
        if not frame.empty:
            parquet_write(self.directory / 'daily_metrics.parquet', frame)
        models = [{'store_id': c.store_id, 'optimizer_steps': getattr(c, 'optimizer_steps', {}), 'policies': {name: policy.state() for name, policy in getattr(c, 'policies', {}).items()}} for c in self.study.cases]
        json_write(self.root / 'models' / self.id / 'latest.json', {'day': self.study.day, 'models': models})
        # Stable primary identity never changes with pause/resume.
        json_write(self.directory / 'progress.json', {'completed_day': self.study.day, 'stop_reason': self.study.stop_reason,
                                                     'checkpoint_day': self.loaded_day})
        return frame
