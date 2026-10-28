"""Build explicit wall-clock bounded Actor-Critic notebook."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(revision='main'):
    n=json.loads((ROOT/'notebooks/run_actor_critic.ipynb').read_text(encoding='utf-8'))
    n['metadata']['colab']['name']='Kiosk_Actor_Critic_Live.ipynb'
    n['cells'][0]['source']=['# 모델별 실시간 비교 · 2026년 10월 29일 오후 6시까지\n\n파레토 2단계 탐색으로 추천 없음과 두 Actor–Critic AI를 비교합니다. 기본 continual 실행은 일수 제한 없이 오늘 18:00 KST까지 진행합니다. 세 방식의 결정시간 공식은 동일합니다. 기존 결과는 보존하고 새 실험 ID를 만듭니다. 결과·계획·체크포인트는 Drive에 저장합니다. 아래 실시간 카드에서 모델을 클릭하면 확대 그래프, α·β·λ, 실제 추천 가중치를 확인할 수 있습니다. 순위는 완료된 별도 동결 평가 구간의 미추천 대비 종합 효용 개선량입니다.\n']
    n['cells'][5]['source']=['## 3. 자동 2단계 탐색 · 실시간 평가 그래프\n\npilot: 8매장·1평가기준·5고객, 1일 넓은 학습/1일 구간 선택 검증/1일 집중 학습/1일 최종 검증/1일 평가(총 5일). research: 9평가기준, 10/5/10/5/5일(35일) × 5 독립 반복. continual: 같은 35일 주기 사이에 신경망을 이어 학습합니다. α·β는 각 기준 안에서 고정합니다. 두 AI의 후보를 함께 보고 기준별 공통 λ 구간을 정하며, 무릎점이 불명확하면 전체 범위를 유지합니다. 집중 학습의 매 5번째 날짜에는 0·0.5·1을 재확인합니다. 추천 없음 대비 개선량은 별도 최종 평가 날짜에서 계산합니다. 중지 시각은 선택 사항이며 별도의 일수 상한은 없습니다.\n']
    code='''from datetime import datetime,timezone
from kiosk_research.config import Config
from kiosk_research.live import run_live
PRESET = "continual" #@param ["pilot", "research", "continual"]
DEADLINE_KST = "2026-10-29T18:00:00+09:00" #@param {type:"string"}
RESUME_ID = "" #@param {type:"string"}
deadline = datetime.fromisoformat(DEADLINE_KST) if DEADLINE_KST.strip() else None
if deadline is not None and (deadline.tzinfo is None or deadline <= datetime.now(timezone.utc)):
    raise ValueError("미래의 시간대 포함 중지 시각을 입력하세요.")
config = Config.load(CODE_ROOT / "config" / ("actor_critic_"+PRESET+".yaml"))
if deadline is not None: config.early_stopping = False
config.auto_cleanup = False
assert 'no_recommendation' in config.models and config.decision_time_mode == 'equal'
assert config.lambda_search == 'pareto_two_stage'
run = run_live(DATA_ROOT, config, deadline.astimezone(timezone.utc).isoformat() if deadline else None,
               resume_id=RESUME_ID.strip() or None, target_day=None)
print("Saved:",run.id,run.state["day"],run.state["stop_reason"])
'''
    n['cells'][6]['source']=code.splitlines(keepends=True)
    for c in n['cells']:
        if c['cell_type']=='code':
            c['source']=[s.replace('REVISION = "main"',f'REVISION = "{revision}"') for s in c['source']]
            c['outputs']=[];c['execution_count']=None
            compile(''.join(c['source']),'live-notebook','exec')
    for c in n['cells']:
        if c['cell_type']=='markdown':
            c['source']=[s.replace('1000일 상한도 유지합니다.','일수 상한은 없습니다.').replace('정오 또는 1000일 중 먼저 도달하는 조건에서','정오에') for s in c['source']]
    target=ROOT/'notebooks/run_actor_critic_live.ipynb'
    target.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    return target


if __name__=='__main__':build()
