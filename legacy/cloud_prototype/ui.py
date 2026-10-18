"""Nonblocking notebook controls: worker thread leaves widget callbacks responsive."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import traceback
from kiosk_cloud.config import CloudConfig
from kiosk_cloud.runner import Experiment
from kiosk_cloud.analysis import export_summary
from kiosk_cloud.github_records import upload_summary


def show_controls(data_root,repository='phj9187/koisk-simulation'):
    import ipywidgets as w
    from IPython.display import display,Image,clear_output
    from google.colab import output
    output.enable_custom_widget_manager()
    source=w.Dropdown(options=[('KREI 실측 행 표본','empirical_donors_2025'),('가상 매장 (연결 시험용)','uniform_scenario')],description='매장 자료')
    survey=w.Text(value=str(Path(data_root)/'raw_data/source_2025.xlsx'),description='KREI 경로',layout=w.Layout(width='95%'))
    override=w.Checkbox(value=False,description='모든 가게의 하루 방문자 수 직접 지정')
    visitors=w.BoundedIntText(value=100,min=0,max=1000000,description='방문자 수')
    stores=w.BoundedIntText(value=64,min=1,max=512,description='매장 수')
    models=w.SelectMultiple(options=[('고정 가중치','fixed'),('상황별 가중치','contextual')],value=('fixed','contextual'),description='비교 모델')
    seed=w.BoundedIntText(value=20261006,min=0,max=2147483647,description='난수 시드')
    train=w.BoundedIntText(value=40,min=1,max=100000,description='학습 일수')
    validation=w.BoundedIntText(value=5,min=1,max=10000,description='검증 일수')
    evaluation=w.BoundedIntText(value=10,min=1,max=10000,description='평가 일수')
    cycles=w.BoundedIntText(value=0,min=0,max=100000,description='반복 수 (0=계속)')
    interval=w.BoundedIntText(value=5,min=1,max=1000,description='저장 주기')
    import os
    workers=w.BoundedIntText(value=min(4,os.cpu_count() or 1),min=1,max=4,description='CPU 작업자')
    logs=w.Checkbox(value=False,description='추천 이벤트 상세 기록')
    plots=w.Checkbox(value=True,description='그래프 저장')
    coefficients={name:w.FloatText(value=value,description=label) for name,label,value in [('customer','고객 계수',.45),('store','가게 계수',.35),('served','응대 계수',.10),('wait','대기 감점',.10)]}
    existing=w.Text(description='재개할 실험 ID',layout=w.Layout(width='80%'))
    start=w.Button(description='새 실험 시작',button_style='success');resume=w.Button(description='체크포인트 재개')
    stop=w.Button(description='오늘 완료 후 중지',button_style='warning');results=w.Button(description='결과 확인')
    publish=w.Button(description='GitHub 요약 업로드')
    status=w.HTML(value='대기 중 · CPU 실행 · 실제 만족도 검증이 아닌 가상 실험입니다.')
    out=w.Output();pool=ThreadPoolExecutor(max_workers=1)
    state={'run':None,'job':None,'summary':None}
    controls=[source,survey,override,visitors,stores,models,seed,train,validation,evaluation,cycles,interval,workers,logs,plots,*coefficients.values(),existing]
    def busy():return state['job'] is not None and not state['job'].done()
    def launch(resuming):
        if busy(): return
        try:
            if resuming:
                run=Experiment.resume(data_root,existing.value.strip())
            else:
                if source.value=='empirical_donors_2025':
                    from kiosk_cloud.data import ensure_survey
                    ensure_survey(survey.value)
                config=CloudConfig(seed=seed.value,store_count=stores.value,source_mode=source.value,survey_path=survey.value,
                                   visitors_per_day=visitors.value if override.value else None,models=list(models.value),
                                   train_days=train.value,validation_days=validation.value,evaluation_days=evaluation.value,
                                   cycles=cycles.value or None,checkpoint_every=interval.value,workers=workers.value,
                                   detailed_logs=logs.value,plots=plots.value,objective={k:v.value for k,v in coefficients.items()})
                run=Experiment(config,data_root)
            state.update(run=run,summary=None);existing.value=run.id
            for control in controls:control.disabled=True
            start.disabled=resume.disabled=True
            status.value='실행 중: '+run.id
            def execute():
                try:
                    run.run(progress=lambda day,done,total:setattr(status,'value',f'{run.id} · 날짜 {day} · 저장 완료 매장 {done}/{total}'),
                            day_callback=lambda r:setattr(status,'value',f'{r.id} · {r.study.day}일 완료 · {r.config.stage(r.study.day)[1]}'))
                    status.value=f'{run.id} · {run.study.day}일 완료 · {run.study.stop_reason}'
                except Exception:
                    status.value='오류 발생 · 저장된 체크포인트/매장 부분 진행에서 재개하세요.'
                    with out:traceback.print_exc()
                finally:
                    for control in controls:control.disabled=False
                    start.disabled=resume.disabled=False
            state['job']=pool.submit(execute)
        except Exception:
            with out:traceback.print_exc()
    def request_stop(_):
        if state['run'] is not None:
            state['run'].request_stop();status.value='중지 요청됨 · 모든 가게의 현재 날짜를 완료하고 저장합니다.'
    def show_results(_):
        if busy():
            with out:print('결과를 저장하려면 안전 중지를 요청한 뒤 완료를 기다리세요.')
            return
        if state['run'] is None:return
        with out:
            try:
                if state['summary'] is None:state['summary']=export_summary(state['run'],state['run'].directory/'summary_exports')
                target=state['summary'];print('요약:',target)
                import pandas as pd
                display(pd.read_csv(target/'summary.csv'))
                for name in ('comparison.png','repetitions.png'):
                    if (target/name).exists():display(Image(filename=str(target/name)))
            except Exception:traceback.print_exc()
    def publish_results(_):
        if busy():return
        show_results(None)
        if state['summary'] is None:return
        with out:
            token=None
            try:
                from google.colab import userdata
                token=userdata.get('KIOSK_GITHUB_TOKEN')
            except Exception:print('KIOSK_GITHUB_TOKEN 없음: Drive 요약은 유지하고 업로드를 건너뜁니다.')
            try:print(upload_summary(state['summary'],repository,token=token))
            except Exception:traceback.print_exc()
            finally:token=None
    start.on_click(lambda _:launch(False));resume.on_click(lambda _:launch(True));stop.on_click(request_stop)
    results.on_click(show_results);publish.on_click(publish_results)
    display(w.VBox([w.HTML('<h3>새 연구: 동일한 일일 목적·동일한 시뮬레이션 예산</h3><p>학습 → 동결 검증 → 동결 평가를 반복합니다. 각 반복은 새로운 초기 모델로 시작합니다. 기존 LinUCB/톰프슨 실험과 다릅니다.</p>'),source,survey,
                    w.HBox([stores,workers]),override,visitors,models,seed,w.HBox([train,validation,evaluation]),cycles,interval,
                    w.HBox([logs,plots]),w.HTML('공통 점수 계수 (잠정 가정; 두 모델에 동일 적용)'),w.HBox(list(coefficients.values())),existing,
                    w.HBox([start,resume,stop]),w.HBox([results,publish]),status,out]))
    return state
