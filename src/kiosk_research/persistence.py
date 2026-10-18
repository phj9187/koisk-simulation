import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from contextlib import contextmanager
import pandas as pd

def encoded(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode('utf-8')

def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,path);return
        except OSError:
            if attempt==2:raise
            time.sleep(.1*(attempt+1))
        finally:Path(tmp).unlink(missing_ok=True)

def write_json(path,value):atomic_bytes(path,encoded(value))
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_checkpoint(path,state):write_json(path,{'sha256':hashlib.sha256(encoded(state)).hexdigest(),'state':state})
def load_checkpoint(path):
    value=read_json(path)
    if hashlib.sha256(encoded(value['state'])).hexdigest()!=value['sha256']:raise ValueError('checkpoint checksum mismatch')
    return value['state']

def parquet(path,rows):
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)/'data.parquet'
        frame=pd.DataFrame(rows)
        frame=frame.reindex(sorted(frame.columns),axis=1)
        for key in ('initial_customer','initial_store','initial_reward','incumbent_train_reward'):
            if key in frame:frame[key]=pd.to_numeric(frame[key]).astype(float)
        frame.to_parquet(out,index=False)
        data=out.read_bytes();atomic_bytes(path,data)
        return hashlib.sha256(data).hexdigest()

@contextmanager
def exclusive_lock(path):
    path=Path(path)
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise RuntimeError('RUNNING.lock exists; confirm the old runtime stopped before removing it') from None
    try:
        with os.fdopen(fd,'w') as stream:json.dump({'pid':os.getpid(),'started':time.time()},stream)
        yield
    finally:path.unlink(missing_ok=True)

def layout(root):
    root=Path(root)
    for name in ('raw_data','experiments','checkpoints','models','backups'):(root/name).mkdir(parents=True,exist_ok=True)
    return root
