"""Incremental, read-only score dashboard and wall-clock bounded execution."""
from datetime import datetime,timezone
import html
import json
import pandas as pd
from kiosk_research.engine import Experiment
from kiosk_research.persistence import atomic_bytes,load_checkpoint,write_json

METRICS=('customer','store','reward')
KEYS=['day','phase','scenario','model']


def summarize(frame):
    if frame.empty:return pd.DataFrame(columns=KEYS+list(METRICS))
    frame=frame[frame.eligible]
    # Average lambdas within a store, then equally weight stores.
    local=frame.groupby(KEYS+['store_id'],as_index=False)[list(METRICS)].mean()
    return local.groupby(KEYS,as_index=False)[list(METRICS)].mean()


def chart(frame,scenario):
    source=frame[frame.scenario.eq(scenario)]
    xmax=max(2,int(source.day.max()) if not source.empty else 2)
    pieces=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1050 260" role="img" aria-label="평가점수 추이">']
    for col,metric in enumerate(METRICS):
        left=col*350+42;top=30;width=290;height=170
        label={'customer':'Customer score','store':'Store score','reward':'Common reward'}[metric]
        pieces.append(f'<text x="{left}" y="18" font-size="14">{label}</text>')
        for tick in (0,.25,.5,.75,1.):
            y=top+(1-tick)*height
            pieces.append(f'<path d="M{left},{y}h{width}" stroke="#ddd"/><text x="{left-30}" y="{y+4}" font-size="11">{tick:g}</text>')
        for model,color in [('fixed','#315be8'),('contextual','#e44b7a'),('no_recommendation','#168461')]:
            data=source[source.model.eq(model)].sort_values('day')
            coords=[(left+(float(r.day)-1)/max(1,xmax-1)*width,top+(1-float(getattr(r,metric)))*height) for r in data.itertuples()]
            if coords:
                points=' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)
                pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
                for (x,y),r in zip(coords,data.itertuples()):
                    phase=html.escape(str(r.phase))
                    pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>Day {r.day}, {phase}: {getattr(r,metric):.5f}</title></circle>')
        pieces.append(f'<text x="{left}" y="222" font-size="11">Day 1</text><text x="{left+width-48}" y="222" font-size="11">Day {xmax}</text>')
    pieces.append('<text x="42" y="248" fill="#315be8" font-size="13">fixed: 일별</text><text x="220" y="248" fill="#e44b7a" font-size="13">contextual: 개인별</text><text x="460" y="248" fill="#168461" font-size="13">no_recommendation: 추천 없음</text></svg>')
    return ''.join(pieces)


class LiveDashboard:
    def __init__(self,run,deadline_utc,display=True):
        self.run,self.deadline=run,deadline_utc
        self.folder=run.directory/'live';self.folder.mkdir(exist_ok=True)
        self.history=summarize(run.export_metrics())
        self.handle=None
        if display:
            from IPython.display import HTML,display as show
            self.handle=show(HTML('<p>실시간 평가 그래프 준비 중…</p>'),display_id=True)
        self.refresh()

    def progress(self,day,done,total):
        partial=[]
        for p in (self.run.checkpoints/f'partial_{day:06d}').glob('store_*.json'):
            saved=load_checkpoint(p)
            if saved['identity']!=self.run.identity:raise ValueError('foreign partial checkpoint')
            partial.extend(saved['result']['rows'])
        active=summarize(pd.DataFrame(partial))
        self.refresh(active,day,done,total)

    def completed(self,run):
        # Read the just-committed shard only; do not re-read all past results.
        frame=pd.read_parquet(run.directory/run.state['parts'][-1]['path'])
        previous=self.history[self.history.day.ne(run.state['day'])]
        latest=summarize(frame)
        self.history=latest if previous.empty else pd.concat([previous,latest],ignore_index=True)
        self.refresh()

    def refresh(self,active=None,day=None,done=None,total=None):
        previous=self.history[self.history.day.ne(day)]
        frame=self.history if active is None or active.empty else active if previous.empty else pd.concat([previous,active],ignore_index=True)
        updated=datetime.now(timezone.utc).isoformat()
        status={'experiment_id':self.run.id,'updated_at_utc':updated,'deadline_utc':self.deadline,
                'search_stage':self.run.config.search_stage(day or max(1,self.run.state['day'])),
                'completed_day':self.run.state['day'],'active_day':day,'completed_stores':done,'total_stores':total,
                'stop_reason':self.run.state['stop_reason'],'provisional_last_point':day is not None}
        title=f"{html.escape(self.run.id)} · 완료 {self.run.state['day']}일"
        if day is not None:title+=f' · {day}일 계산 중 {done}/{total}매장 (마지막 점 임시값)'
        if self.run.config.lambda_search=='pareto_two_stage':title+=' · '+status['search_stage']
        content=[f'<h3>{title}</h3><p>갱신 UTC: {updated} · 중지 UTC: {html.escape(self.deadline or "별도 시각 제한 없음")}</p>',
                 '<p>매장 완료마다 갱신. 고객 점수·매장 점수·공통 보상, 범위 0~1. λ 평균은 기술통계이며 학습·검증·평가 날짜의 입력은 다릅니다. 엄밀한 모델 비교에는 동결 평가 결과를 사용하세요. 점 위에 마우스를 올리면 날짜·단계를 확인할 수 있습니다.</p>']
        plans=self.run.state.get('lambda_search',{})
        if plans:
            latest=plans[max(plans,key=int)]
            summary={name:choice['grid'] for name,choice in latest['choices'].items()}
            content.append('<p>검증으로 선택한 2단계 λ: '+html.escape(json.dumps(summary,ensure_ascii=False))+'</p>')
        for i,scenario in enumerate(sorted(self.run.config.scenarios)):
            content.append(f'<details {"open" if i==0 else ""}><summary>{html.escape(scenario)}</summary>{chart(frame,scenario)}</details>')
        if self.run.state['stop_reason']:content.append('<p>중지 상태: '+html.escape(self.run.state['stop_reason'])+'</p>')
        body=''.join(content)
        page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>Kiosk live scores</title><style>body{font:15px sans-serif;margin:24px;color:#243047}svg{width:100%;max-width:1200px}summary{padding:12px;background:#eef2fa;cursor:pointer}details{margin:12px 0}</style>'+body
        atomic_bytes(self.folder/'index.html',page.encode('utf-8'))
        atomic_bytes(self.folder/'scores.csv',frame.to_csv(index=False).encode('utf-8'))
        write_json(self.folder/'status.json',status)
        if self.handle is not None:
            from IPython.display import HTML
            self.handle.update(HTML(body))


def run_live(root,config,deadline_utc,resume_id=None,target_day=None,display=True,cards=True):
    if config.algorithm!='actor_critic':raise ValueError('neural experiment required')
    run=Experiment.resume(root,resume_id) if resume_id else Experiment(config,root)
    if run.config.algorithm!='actor_critic':raise ValueError('cannot resume legacy random search here')
    write_json(run.directory/'execution_control.json',{'deadline_utc':deadline_utc,'target_day':target_day,'stop_boundary':'store; preserve partial day','git_commit':run.identity['git_commit']})
    print('Experiment:',run.id,'\nDeadline UTC:',deadline_utc,'\nLive dashboard:',run.directory/('live/model_board.html' if cards else 'live/index.html'),flush=True)
    if cards:
        from kiosk_research.model_board import ModelBoard
        dashboard=ModelBoard(run,deadline_utc,display)
    else:dashboard=LiveDashboard(run,deadline_utc,display)
    try:
        run.run(progress=dashboard.progress,day_callback=dashboard.completed,target_day=target_day,deadline_utc=deadline_utc)
    finally:
        pending=run.checkpoints/f"partial_{run.state['day']+1:06d}"
        count=len(list(pending.glob('store_*.json')))
        if count:dashboard.progress(run.state['day']+1,count,run.config.stores)
        else:dashboard.refresh()
    if run.state['day']:
        from kiosk_research.analysis import export
        summary=export(run)
        write_json(run.directory/'session_result.json',{'experiment_id':run.id,'summary':str(summary),'day':run.state['day'],'stop_reason':run.state['stop_reason'],'deadline_utc':deadline_utc})
    return run
