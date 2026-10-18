import hashlib
from pathlib import Path
import urllib.request
from kiosk_sim.empirical_stores import SOURCE_URL, SOURCE_SHA256
from kiosk_cloud.storage import atomic_write


def ensure_survey(path):
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            data = response.read(100_000_001)
        if len(data) > 100_000_000: raise ValueError('survey download unexpectedly large')
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('KREI source SHA256 mismatch. Do not substitute another dataset or disable verification.')
    if not path.exists(): atomic_write(path, data)
    return path
