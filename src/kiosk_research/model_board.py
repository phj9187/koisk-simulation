"""Matched live model/control comparisons with bounded per-view histories."""
from collections import defaultdict,deque
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs,urlsplit
import pandas as pd
from kiosk_research.persistence import load_checkpoint,write_json,atomic_bytes

METRICS=('customer','store','reward','profit','mean_total_wait','leftover_units')
COMPONENTS=('affinity','waste_risk','margin','wait_utility','affordability')


def matched_points(frame):
    if frame.empty:return []
    keys=['day','store_id','scenario','lambda_value']
    base=frame[frame.model.eq('no_recommendation')]
    if base.duplicated(keys).any():raise ValueError('duplicate control points')
    if frame[frame.model.isin(['fixed','contextual'])].duplicated(keys+['model']).any():raise ValueError('duplicate model points')
    lookup={tuple(r[k] for k in keys):r for r in base.to_dict('records')}
    points=[]
    for row in frame[frame.model.isin(['fixed','contextual'])&frame.eligible].to_dict('records'):
        control=lookup.get(tuple(row[k] for k in keys))
        if control is None:raise ValueError('missing matched unassisted control')
        p={k:row[k] for k in keys}
        p.update(model=row['model'],phase=row['phase'],stage=row.get('search_stage','fixed'),cycle=int(row['repetition']),
                 weights=[row.get('weight_'+k) for k in COMPONENTS],served=int(row['served']),visitors=int(row['visitors']))
        for k in METRICS:p[k]=float(row[k]);p['baseline_'+k]=float(control[k])
        points.append(p)
    return points


def average_points(points):
    if not points:return None
    first=points[0]
    p={k:first[k] for k in ('day','phase','stage','cycle')}
    for k in METRICS:
        for prefix in ('','baseline_'):p[prefix+k]=sum(r[prefix+k] for r in points)/len(points)
    p['weights']=[sum(r['weights'][i] for r in points)/len(points) for i in range(5)]
    p['lambdas']=sorted(set(float(r['lambda_value']) for r in points))
    return p


class ModelBoard:
    def __init__(self,run,deadline,display=True,max_points=600):
        if 'no_recommendation' not in run.config.models:raise ValueError('model board requires unassisted control')
        self.run,self.deadline=run,deadline
        self.folder=run.directory/'live';self.folder.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.limit=max_points;self.series={};self.metadata={};self.latest={};self.eval_days={}
        self.provisional={};self.provisional_lambdas={};self.server=None;self.active_day=None;self.completed_stores=0;self.updated=datetime.now(timezone.utc).isoformat()
        for case in run.state['cases']:
            for scenario,coefficients in sorted(run.config.scenarios.items()):
                for model in run.config.models:
                    if model=='no_recommendation':continue
                    key=f"{case['id']}:{scenario}:{model}"
                    self.metadata[key]={'id':key,'store_id':case['id'],'scenario':scenario,'model':model,
                                        'alpha':coefficients['customer'],'beta':coefficients['store']}
                    self.series[key]=defaultdict(lambda:deque(maxlen=self.limit))
        # Resume incrementally from canonical day shards once, not on every refresh.
        for part in run.state['parts']:
            self.ingest(matched_points(pd.read_parquet(run.directory/part['path'])),persist=False)
        self.refresh()
        if display:self.start_server(show=True)

    def ingest(self,points,persist=True):
        if not points:return
        grouped=defaultdict(list)
        for p in points:grouped[f"{p['store_id']}:{p['scenario']}:{p['model']}"].append(p)
        with self.lock:
            for key,items in grouped.items():
                if key not in self.metadata:raise ValueError('unknown model view')
                day=items[0]['day'];aggregate=average_points(items)
                # A completed day replaces the prior provisional day, never duplicates it.
                for bucket in self.series[key].values():
                    if bucket and bucket[-1]['day']==day:bucket.pop()
                for p in items:self.series[key][str(float(p['lambda_value']))].append(p)
                self.series[key]['all'].append(aggregate)
                self.latest[key]=aggregate
                if aggregate['phase']=='evaluation':
                    if set(aggregate['lambdas'])!=set(self.run.config.evaluation_lambdas):raise ValueError('incomplete evaluation lambda grid')
                    self.eval_days.setdefault(key,{})[day]=aggregate
                    # Only the current and previous cycle are needed for live ranking.
                    for old in list(self.eval_days[key]):
                        if self.eval_days[key][old]['cycle']<aggregate['cycle']-1:del self.eval_days[key][old]
            if persist:
                day=int(points[0]['day'])
                write_json(self.folder/'cards'/f'points_{day:06d}.json',points)

    def ranking(self,key,lam='all'):
        days=self.eval_days.get(key,{}) if lam=='all' else {p['day']:p for p in self.series[key].get(lam,[]) if p['phase']=='evaluation'}
        by_cycle=defaultdict(list)
        for p in days.values():by_cycle[p['cycle']].append(p)
        complete=[c for c,v in by_cycle.items() if len(v)==self.run.config.evaluation_days]
        if not complete:return None
        cycle=max(complete);points=by_cycle[cycle]
        return {'cycle':cycle,'days':sorted(p['day'] for p in points),
                'ai':{k:sum(p[k] for p in points)/len(points) for k in METRICS},
                'baseline':{k:sum(p['baseline_'+k] for p in points)/len(points) for k in METRICS},
                **{k:sum(p[k]-p['baseline_'+k] for p in points)/len(points) for k in METRICS}}

    def snapshot(self):
        with self.lock:
            cards=[]
            for key,meta in self.metadata.items():
                history=list(self.series[key]['all'])
                rank=self.ranking(key)
                spark=[{k:p[k] for k in ('day','reward','baseline_reward')} for p in history[-40:]]
                cards.append({**meta,'latest':self.latest.get(key),'rank':rank,
                              'spark':spark,'provisional':self.provisional.get(key)})
            cards.sort(key=lambda x: (x['rank'] is None,-(x['rank']['reward'] if x['rank'] else 0.),x['id']))
            return {'experiment_id':self.run.id,'updated_at_utc':self.updated,
                    'deadline_utc':self.deadline,'completed_day':self.run.state['day'],
                    'active_day':self.active_day,'completed_stores':self.completed_stores,'total_stores':self.run.config.stores,
                    'search_stage':self.run.config.search_stage(max(1,self.run.state['day']+1)),
                    'stop_reason':self.run.state['stop_reason'],'history_limit':self.limit,'cards':cards,
                    'plans':self.run.state.get('lambda_search',{}).copy(),
                    'note':'synthetic utilities; rank uses completed frozen evaluation cycles only; alpha/beta define different assumed utilities'}

    def detail(self,key,lam='all'):
        with self.lock:
            if key not in self.metadata:raise KeyError(key)
            if lam!='all':
                try:lam=str(float(lam))
                except ValueError:raise KeyError(lam)
                if lam not in self.series[key]:lam='all'
            values=list(self.series[key].get(lam,[]))
            return {**self.metadata[key],'lambda':lam,'available_lambdas':sorted(float(k) for k in self.series[key] if k!='all'),
                    'series':values,'rank':self.ranking(key,lam),'provisional':self.provisional.get(key) if lam=='all' else self.provisional_lambdas.get(key,{}).get(lam),
                    'coefficients':self.run.config.scenarios[self.metadata[key]['scenario']],
                    'weight_names':COMPONENTS,'parameters':{'actor':773,'critic':705},
                    'history_limit':self.limit,'plans':self.run.state.get('lambda_search',{}).copy(),
                    'run':{'completed_day':self.run.state['day'],'stop_reason':self.run.state['stop_reason'],'deadline_utc':self.deadline,'updated_at_utc':self.updated}}

    def refresh(self,persist=True):
        self.updated=datetime.now(timezone.utc).isoformat()
        payload=self.snapshot()
        if persist:
            write_json(self.folder/'board_index.json',payload)
            template=Path(__file__).with_name('model_board.html').read_text(encoding='utf-8')
            # Embedded initial snapshot also makes a useful saved offline overview.
            initial=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
            atomic_bytes(self.folder/'model_board.html',template.replace('__INITIAL__',initial).encode('utf-8'))
        write_json(self.folder/'status.json',{k:v for k,v in payload.items() if k not in ('cards','plans')})

    def progress(self,day,done,total):
        # Partial results are shown but never counted in the frozen ranking.
        rows=[]
        for path in (self.run.checkpoints/f'partial_{day:06d}').glob('store_*.json'):
            saved=load_checkpoint(path)
            if saved['identity']!=self.run.identity:raise ValueError('foreign partial checkpoint')
            rows.extend(saved['result']['rows'])
        grouped=defaultdict(list)
        for p in matched_points(pd.DataFrame(rows)):
            grouped[f"{p['store_id']}:{p['scenario']}:{p['model']}"].append(p)
        with self.lock:
            self.active_day=day;self.completed_stores=done
            self.provisional={key:average_points(v) for key,v in grouped.items()}
            self.provisional_lambdas={key:{str(float(p['lambda_value'])):p for p in values} for key,values in grouped.items()}
        self.refresh(persist=False)

    def completed(self,run):
        frame=pd.read_parquet(run.directory/run.state['parts'][-1]['path'])
        self.ingest(matched_points(frame))
        with self.lock:self.provisional={};self.provisional_lambdas={};self.active_day=None;self.completed_stores=0
        self.refresh()

    def start_server(self,show=False):
        board=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                route=urlsplit(self.path);query=parse_qs(route.query)
                try:
                    if route.path in ('/','/index.html'):
                        body=(board.folder/'model_board.html').read_bytes();kind='text/html; charset=utf-8'
                    elif route.path=='/api/index':
                        body=json.dumps(board.snapshot(),ensure_ascii=False,allow_nan=False).encode();kind='application/json'
                    elif route.path=='/api/model':
                        value=board.detail(query.get('id',[''])[0],query.get('lambda',['all'])[0])
                        body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();kind='application/json'
                    else:self.send_error(404);return
                except (KeyError,ValueError):self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        if show:
            from google.colab import output
            output.serve_kernel_port_as_iframe(self.server.server_port,height=900)
        return self.server.server_port
