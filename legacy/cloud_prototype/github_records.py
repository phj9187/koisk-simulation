"""Publish only explicit small summary artifacts to a new results branch.

Token comes from caller (Colab Secrets), is sent only to api.github.com in a header,
and is never persisted, embedded in a URL, printed, or included in exceptions.
"""
import base64
import csv
import io
import json
from pathlib import Path
import re
import requests
from kiosk_cloud.storage import json_write

ALLOWED = {'metadata.json','summary.csv','statistics.json','comparison.png','repetitions.png','report.md'}


class GitHubError(RuntimeError): pass


def upload_summary(directory, repository, token=None, create_pr=False):
    directory=Path(directory)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository): raise ValueError('repository must be owner/name')
    if not token:
        return {'status':'skipped','reason':'No token; summary retained at '+str(directory)}
    metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    experiment_id=metadata['identity']['experiment_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',experiment_id): raise ValueError('invalid experiment ID')
    files=sorted(p for p in directory.iterdir() if p.name in ALLOWED and p.is_file())
    if any(p.is_symlink() for p in files): raise ValueError('summary files must not be symlinks')
    if sum(p.stat().st_size for p in files)>10_000_000: raise ValueError('summary exceeds 10 MB; raw data must stay on Drive')
    session=requests.Session();session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    def api(method,path,**kwargs):
        try: response=session.request(method,'https://api.github.com/repos/'+repository+path,timeout=30,allow_redirects=False,**kwargs)
        except requests.RequestException: raise GitHubError('GitHub network request failed; summary remains on Drive') from None
        if response.status_code not in (200,201):
            raise GitHubError(f'GitHub returned HTTP {response.status_code}; no success assumed. Check repository access, token permissions and whether the results branch already exists.')
        return response.json()
    try:
        info=api('GET','')
        from urllib.parse import quote
        base=info['default_branch']
        ref=api('GET','/git/ref/heads/'+quote(base,safe=''))
        commit=api('GET','/git/commits/'+ref['object']['sha'])
        branch=f'results/{experiment_id}-day-{metadata["completed_day"]:06d}'
        nodes=[]
        prefix=f'results/summaries/{experiment_id}/{directory.name}'
        for path in files:
            blob=api('POST','/git/blobs',json={'content':base64.b64encode(path.read_bytes()).decode(),'encoding':'base64'})
            nodes.append({'path':prefix+'/'+path.name,'mode':'100644','type':'blob','sha':blob['sha']})
        # Registry is append-only relative to the selected default-branch snapshot.
        # Concurrent results branches may need registry conflict resolution when merged.
        entries=[]
        registry_path='results/experiment_registry.csv'
        tree=api('GET','/git/trees/'+commit['tree']['sha']+'?recursive=1')
        if tree.get('truncated'): raise GitHubError('Repository tree truncated; refusing an incomplete registry update')
        existing=next((n for n in tree['tree'] if n['path']==registry_path),None)
        if existing:
            blob=api('GET','/git/blobs/'+existing['sha'])
            entries=list(csv.DictReader(io.StringIO(base64.b64decode(blob['content']).decode('utf-8-sig'))))
        key=(experiment_id,str(metadata['completed_day']))
        if any((r['experiment_id'],r['completed_day'])==key for r in entries): raise GitHubError('Experiment summary already registered; existing data preserved')
        entries.append({'experiment_id':experiment_id,'completed_day':str(metadata['completed_day']),
                        'git_commit':metadata['identity']['git_commit'] or 'UNCOMMITTED',
                        'source_sha256':metadata['identity']['source_sha256'],'summary_path':prefix,
                        'drive_path':metadata['data_path']})
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['experiment_id','completed_day','git_commit','source_sha256','summary_path','drive_path'])
        writer.writeheader();writer.writerows(entries)
        blob=api('POST','/git/blobs',json={'content':stream.getvalue(),'encoding':'utf-8'})
        nodes.append({'path':registry_path,'mode':'100644','type':'blob','sha':blob['sha']})
        tree=api('POST','/git/trees',json={'base_tree':commit['tree']['sha'],'tree':nodes})
        new=api('POST','/git/commits',json={'message':f'Record {experiment_id} day {metadata["completed_day"]}',
                                        'tree':tree['sha'],'parents':[commit['sha']]})
        api('POST','/git/refs',json={'ref':'refs/heads/'+branch,'sha':new['sha']})
        result={'status':'uploaded','branch':branch,'commit':new['sha'],
                'url':f'https://github.com/{repository}/tree/{quote(branch,safe="")}/{prefix}'}
        # Persist branch success before optional PR creation (which can fail separately).
        json_write(directory/'github_upload.json',result)
        if create_pr:
            pr=api('POST','/pulls',json={'title':f'Experiment {experiment_id} · day {metadata["completed_day"]}',
                                       'head':branch,'base':base,'body':'Adds synthetic research summary artifacts only. Raw data and checkpoints remain on Drive.'})
            result['pull_request']=pr['html_url'];json_write(directory/'github_upload.json',result)
        return result
    finally:
        session.headers.pop('Authorization',None);session.close()
