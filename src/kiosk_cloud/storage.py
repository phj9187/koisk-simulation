"""Atomic, bounded-retry writes. Checkpoints remain trusted Python pickle files."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from contextlib import contextmanager
from kiosk_sim.durable_study import read_snapshot, write_snapshot


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, path)
            return
        except OSError:
            if attempt == 2: raise
            time.sleep(.1 * (attempt + 1))
        finally:
            Path(name).unlink(missing_ok=True)


def json_write(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode())


def parquet_write(path, frame):
    # Build parquet in runtime temp storage; one remote write per aggregate batch.
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / 'data.parquet'
        frame.to_parquet(output, index=False)
        atomic_write(path, output.read_bytes())


def snapshot_write(path, value):
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / 'state.checkpoint'
        write_snapshot(output, value)
        atomic_write(path, output.read_bytes())


@contextmanager
def writer_lock(path):
    path = Path(path)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('RUNNING.lock exists. Confirm the old runtime is stopped before removing a stale lock.') from exc
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump({'pid': os.getpid(), 'created_at': time.time()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def drive_layout(root):
    root = Path(root)
    for name in ('raw_data', 'experiments', 'checkpoints', 'models', 'backups'):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root
