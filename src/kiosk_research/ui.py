"""Nonblocking Colab controls. Stop callbacks run while workers process the day."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import traceback
from kiosk_research.config import Config
from kiosk_research.design import screening_design
SCENARIOS=screening_design()
from kiosk_research.engine import Experiment
from kiosk_research.analysis import export,recommend
from kiosk_research.persistence import write_json

def controls(data_root,repository='phj9187/koisk-simulation'):
    import ipywidgets as w
    from IPython.display import display,Image
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except ImportError:pass
    source=w.Dropdown(options=[('가상 매장: 연결 검증용','synthetic'),('KREI 실측 행 표본','empirical')],description='매장 자료')
    survey=w.Text(value=str(Path(data_root)/'raw_data/source_2025.xlsx'),description='원자료 경로',layout=w.Layout(width='95%'))
    stores=w.BoundedIntText(value=8,min=1,max=512,description='매장 수')
    override=w.Checkbox(value=True,description='하루 방문자 수 직접 지정 (미선택: 매장별 조사 평균)')
    visitors=w.BoundedIntText(value=30,min=0,max=1000000,description='방문자 수')
    seed=w.BoundedIntText(value=42,min=0,max=2147483647,description='난수 시드')
    train=w.BoundedIntText(value=10,min=1,max=100000,description='학습 일수')
    validation=w.BoundedIntText(value=2,min=1,max=10000,description='검증 일수')
    evaluation=w.BoundedIntText(value=3,min=1,max=10000,description='평가 일수')
    repetitions=w.BoundedIntText(value=1,min=0,max=100000,description='반복 (0=계속)')
    workers=w.BoundedIntText(value=1,min=1,max=4,description='CPU 작업자')
    interval=w.BoundedIntText(value=2,min=1,max=1000,description='저장 주기')
    keep=w.BoundedIntText(value=2,min=2,max=20,description='복구 보관 수')
    cleanup=w.Checkbox(value=True,description='검증된 중복 자료 자동 정리')
    models=w.SelectMultiple(options=['fixed','contextual'],value=('fixed','contextual'),description='비교 모델')
    scenarios=w.SelectMultiple(options=list(SCENARIOS),value=tuple(SCENARIOS),description='만족도 가정')
    coefficients=w.Textarea(value=json.dumps(SCENARIOS,indent=2),description='가정 계수 JSON',layout=w.Layout(width='95%',height='170px'))
    logs=w.Checkbox(value=False,description='고객 상세 기록');plots=w.Checkbox(value=True,description='그래프 저장')
    existing=w.Text(description='재개 실험 ID',layout=w.Layout(width='90%'))
    start=w.Button(description='새 실험 시작',button_style='success');resume=w.Button(description='체크포인트 재개')
    stop=w.Button(description='오늘 완료 후 중지',button_style='warning');results=w.Button(description='결과 보기');upload=w.Button(description='GitHub 요약 업로드')
    constraint=w.BoundedFloatText(value=.5,min=0,max=1,step=.05,description='최소 고객 만족도');select=w.Button(description='검증 자료로 λ 추천')
    status=w.HTML('대기 중. 기본 설정은 소규모 가상 검증용입니다.');out=w.Output();pool=ThreadPoolExecutor(max_workers=1)
    state={'experiment':None,'future':None,'summary':None}
    fields=[source,survey,stores,override,visitors,seed,train,validation,evaluation,repetitions,workers,interval,keep,cleanup,models,scenarios,coefficients,logs,plots,existing]
    def busy():return state['future'] is not None and not state['future'].done()
    def launch(resuming):
        if busy():return
        try:
            if resuming:run=Experiment.resume(data_root,existing.value.strip())
            else:
                if source.value=='empirical':
                    from kiosk_cloud.data import ensure_survey
                    ensure_survey(survey.value)
                all_scenarios=json.loads(coefficients.value)
                c=Config(seed=seed.value,stores=stores.value,source=source.value,survey_path=survey.value,
                         visitors=visitors.value if override.value else None,train_days=train.value,validation_days=validation.value,
                         evaluation_days=evaluation.value,repetitions=repetitions.value or None,workers=workers.value,
                         checkpoint_every=interval.value,checkpoint_keep=keep.value,auto_cleanup=cleanup.value,models=list(models.value),scenarios={k:all_scenarios[k] for k in scenarios.value},raw_logs=logs.value,plots=plots.value)
                run=Experiment(c,data_root)
            state.update(experiment=run,summary=None);existing.value=run.id
            for field in fields:field.disabled=True
            start.disabled=resume.disabled=True;status.value='실행 중: '+run.id
            def execute():
                try:
                    run.run(progress=lambda day,done,total:setattr(status,'value',f'{run.id} · {day}일 · 저장 매장 {done}/{total}'),
                            day_callback=lambda r:setattr(status,'value',f"{r.id} · {r.state['day']}일 완료 · {r.config.stage(r.state['day'])[1]}"))
                    state['summary']=export(run)
                    status.value=f"{run.id} · {run.state['day']}일 · {run.state['stop_reason']} · 결과 저장 완료"
                except Exception:
                    status.value='오류 발생. 저장된 날짜/매장 부분 진행에서 재개하세요.'
                    with out:traceback.print_exc()
                finally:
                    for field in fields:field.disabled=False
                    start.disabled=resume.disabled=False
            state['future']=pool.submit(execute)
        except Exception:
            with out:traceback.print_exc()
    def halt(_):
        if state['experiment'] is not None:state['experiment'].request_stop();status.value='중지 요청됨. 현재 날짜의 모든 매장을 완료하고 저장합니다.'
    def show(_):
        if busy():
            with out:print('먼저 안전 중지를 요청하고 완료를 기다려주세요.')
            return
        if state['experiment'] is None:return
        with out:
            try:
                state['summary']=export(state['experiment']);print('요약:',state['summary'])
                import pandas as pd
                display(pd.read_csv(state['summary']/'summary.csv'))
                for name in ['comparison.png','learning_weights.png']+[f'{prefix}_{scenario}.png' for scenario in list(state['experiment'].config.scenarios)[1:] for prefix in ('comparison','learning_weights')]:
                    if (state['summary']/name).exists():display(Image(filename=str(state['summary']/name)))
            except Exception:traceback.print_exc()
    def publish(_):
        if busy():return
        show(None)
        if state['summary'] is None:return
        token=None
        with out:
            try:
                try:
                    from google.colab import userdata
                    token=userdata.get('KIOSK_GITHUB_TOKEN')
                except Exception:pass
                from kiosk_research.github import upload_summary
                print(upload_summary(state['summary'],repository,token))
            except Exception:traceback.print_exc()
            finally:token=None
    def choose(_):
        if busy():return
        show(None)
        if state['summary'] is None:return
        with out:
            try:
                import pandas as pd
                values=pd.read_csv(state['summary']/'validation.csv');choices=[]
                final=pd.read_csv(state['summary']/'store_frontiers.csv')
                for keys,g in values.groupby(['repetition','scenario','store_id']):
                    choice=recommend(g,min_customer=constraint.value)
                    entry=dict(zip(['repetition','scenario','store_id'],[v.item() if hasattr(v,'item') else v for v in keys]))|choice
                    if choice['status']=='selected_on_validation' and not final.empty:
                        checked=final[(final.repetition==keys[0])&(final.scenario==keys[1])&(final.store_id==keys[2])&(final.model==choice['model'])&(final.lambda_value==choice['lambda'])]
                        if len(checked):entry.update(final_customer=float(checked.iloc[0].customer),final_store=float(checked.iloc[0].store))
                    choices.append(entry)
                path=state['summary'].parent/f'recommendations-customer-{constraint.value:g}.json'
                write_json(path,{'criterion':'maximize store subject to minimum customer on validation only','minimum_customer':constraint.value,'choices':choices})
                print('조건부 추천:',path);display(pd.DataFrame(choices))
            except Exception:traceback.print_exc()
    start.on_click(lambda _:launch(False));resume.on_click(lambda _:launch(True));stop.on_click(halt);results.on_click(show);upload.on_click(publish);select.on_click(choose)
    display(w.VBox([w.HTML('<h3>λ 조건부 추천 함수 · 파레토 연구</h3><p>학습 → 동결 검증 → 동결 평가. 반복마다 새 모델로 시작합니다. CPU 실행부터 확인하세요.</p>'),source,survey,
                    w.HBox([stores,workers]),override,visitors,seed,models,scenarios,coefficients,w.HBox([train,validation,evaluation]),repetitions,interval,keep,cleanup,
                    w.HBox([logs,plots]),existing,w.HBox([start,resume,stop]),w.HBox([results,upload]),w.HBox([constraint,select]),status,out]))
    return state
