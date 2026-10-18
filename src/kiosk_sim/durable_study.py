"""Per-store write-ahead snapshots and immutable daily checkpoints (trusted files only)."""
import copy
import csv
import hashlib
import io
import json
import os
import multiprocessing
import pickle
import struct
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .lhs_study import LHSStudy, MODEL_LABELS

MAGIC = b'KIOSK3\0'


def write_snapshot(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    blob = MAGIC + hashlib.sha256(data).digest() + struct.pack('>Q', len(data)) + data
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_snapshot(path):
    blob = Path(path).read_bytes()
    offset = len(MAGIC)
    if blob[:offset] != MAGIC or len(blob) < offset + 40:
        raise ValueError('Invalid checkpoint header')
    digest, size, data = blob[offset:offset+32], struct.unpack('>Q', blob[offset+32:offset+40])[0], blob[offset+40:]
    if size != len(data) or hashlib.sha256(data).digest() != digest:
        raise ValueError('Incomplete or corrupt checkpoint')
    # Hash detects accidental corruption, NOT malicious pickle files.
    return pickle.loads(data)


def store_day(task):
    config, case, day = task
    local = LHSStudy.__new__(LHSStudy)
    local.config, local.cases = config, [copy.deepcopy(case)]
    local.day = day - 1
    local.stop_requested, local.stop_reason = False, ''
    local.stable_rounds, local.aggregate_history = 0, []
    local.run_day()
    return local.cases[0]


class DurableStudy:
    """One writer per experiment directory. Retains all completed day snapshots."""
    def __init__(self, config, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        package = Path(__file__).parent
        code_hash = hashlib.sha256(b''.join(p.relative_to(package).as_posix().encode() + p.read_bytes()
                                 for p in sorted(package.rglob('*.py')))).hexdigest()
        self.identity = {'config': asdict(config), 'code_sha256': code_hash}
        manifest = self.directory / 'experiment.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != self.identity:
                raise ValueError('Settings or code changed. Use a NEW run directory; do not mix experiments.')
        else:
            with manifest.open('x', encoding='utf-8') as f:
                json.dump(self.identity, f, indent=2)
        self.study = None
        for path in sorted(self.directory.glob('day_*.checkpoint'), reverse=True):
            try:
                envelope = read_snapshot(path)
                if envelope['identity'] != self.identity:
                    raise ValueError('Checkpoint belongs to another experiment')
                self.study = envelope['study']
                break
            except Exception as exc:
                warnings.warn(f'Skipping invalid checkpoint {path.name}: {exc}')
        if self.study is None:
            self.study = LHSStudy(config)
            self.save_day()

    def save_day(self):
        write_snapshot(self.directory / f'day_{self.study.day:06d}.checkpoint',
                       {'identity': self.identity, 'study': self.study})

    def next_day(self, workers=1, progress=None):
        if self.study.finished():
            return None
        lock = self.directory / 'RUNNING.lock'
        try:
            handle = lock.open('x')
        except FileExistsError:
            raise RuntimeError('Another writer or a stale RUNNING.lock exists. Confirm the old runtime stopped before removing it.')
        handle.close()
        try:
            # Refuse a stale in-memory runner after another writer advanced the run.
            newer = list(self.directory.glob('day_*.checkpoint'))
            for path in newer:
                if int(path.stem.split('_')[1]) <= self.study.day:
                    continue
                try:
                    saved = read_snapshot(path)
                except (OSError, ValueError, EOFError, pickle.UnpicklingError):
                    continue
                if saved['identity'] == self.identity:
                    raise RuntimeError('Disk state advanced. Reload the experiment before continuing.')
            day = self.study.day + 1
            folder = self.directory / f'partial_{day:06d}'
            folder.mkdir(exist_ok=True)
            done, pending = {}, []
            for case in self.study.cases:
                path = folder / f'store_{case.store_id:04d}.checkpoint'
                try:
                    saved = read_snapshot(path)
                    if saved['identity'] != self.identity or saved['day'] != day or saved['case'].store_id != case.store_id:
                        raise ValueError('Wrong store snapshot')
                    done[case.store_id] = saved['case']
                except (OSError, ValueError, EOFError, pickle.UnpicklingError):
                    pending.append(case)
            def commit(case):
                write_snapshot(folder / f'store_{case.store_id:04d}.checkpoint',
                               {'identity': self.identity, 'day': day, 'case': case})
                done[case.store_id] = case
                if progress:
                    progress(day, len(done), len(self.study.cases))
            tasks = [(self.study.config, c, day) for c in pending]
            if workers == 1:
                for task in tasks:
                    commit(store_day(task))
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures = [pool.submit(store_day, t) for t in tasks]
                    for future in as_completed(futures):
                        commit(future.result())
            candidate = copy.deepcopy(self.study)
            candidate.cases = [done[c.store_id] for c in self.study.cases]
            candidate.day = day
            candidate.aggregate_history.append(candidate._summarize_day())
            candidate.stable_rounds = candidate.stable_rounds + 1 if day >= candidate.config.min_days and candidate._is_stable() else 0
            candidate.aggregate_history[-1]['stable_rounds'] = candidate.stable_rounds
            if (self.directory / 'STOP').exists():
                candidate.stop_reason = 'Requested stop after complete day'
            elif candidate.stable_rounds >= candidate.config.required_stable_rounds:
                candidate.stop_reason = 'Synthetic average stability (not convergence proof)'
            elif day >= candidate.config.max_days:
                candidate.stop_reason = 'Maximum days reached'
            write_snapshot(self.directory / f'day_{day:06d}.checkpoint',
                           {'identity': self.identity, 'study': candidate})
            self.study = candidate
            # Human-readable per-day data; full model snapshots above are authoritative.
            rows = [r for c in candidate.cases for r in c.records if r['day'] == day]
            out = io.StringIO(newline='')
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            temp = folder / 'results.csv.tmp'
            temp.write_text(out.getvalue(), encoding='utf-8-sig')
            os.replace(temp, folder / 'results.csv')
            return candidate.aggregate_history[-1]
        finally:
            lock.unlink(missing_ok=True)
