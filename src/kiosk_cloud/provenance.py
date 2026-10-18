import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def source_hash():
    return hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in sorted((ROOT / 'src').rglob('*.py')))).hexdigest()


def git_version():
    result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True)
    commit = result.stdout.strip() if result.returncode == 0 else None
    # Untracked research files count as dirty; no fabricated commit on unborn repos.
    status = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'], capture_output=True, text=True)
    return {'git_commit': commit, 'git_dirty': bool(status.stdout.strip()) or status.returncode != 0}


def runtime_info():
    versions = {}
    for name in ('numpy', 'openpyxl', 'pandas', 'pyarrow', 'matplotlib', 'ipywidgets', 'PyYAML', 'requests'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    memory = {}
    if Path('/proc/meminfo').exists():
        memory = {k: v.strip() for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemTotal', 'MemAvailable')}
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        gpu_info = gpu.stdout.strip() if gpu.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired): gpu_info = 'unavailable'
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), cpu)
    return {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
            'platform': platform.platform(), 'cpu': cpu, 'cpu_count': os.cpu_count(), 'memory': memory,
            'gpu': gpu_info, 'libraries': versions, **git_version(), 'source_sha256': source_hash()}
