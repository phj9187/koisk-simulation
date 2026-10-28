"""Build explicit wall-clock bounded Actor-Critic notebook."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(revision='main'):
    n=json.loads((ROOT/'notebooks/run_actor_critic.ipynb').read_text(encoding='utf-8'))
    n['metadata']['colab']['name']='Kiosk_Actor_Critic_Live.ipynb'
    n['cells'][0]['source']=['# 추천 없음 · 하루 고정 AI · 고객별 AI 비교\n\n세 방식의 탐색·결정시간 공식을 동일하게 적용합니다. 매장 완료마다 평가 그래프가 갱신됩니다. 기존 실험과 섞지 않고 새 실험 ID를 만듭니다. 기본 pilot은 4일 점검 실행입니다. GitHub의 지정 커밋을 불러오며 결과·체크포인트는 Drive에 저장합니다.\n']
    n['cells'][5]['source']=['## 3. 세 방식 비교 · 실시간 평가 그래프\n\nPRESET은 pilot(8매장·1평가기준·5고객·2/1/1일), research(9평가기준 독립 반복), continual(20/5/5일 누적 학습)입니다. 추천 없음은 고객의 자기 선택만 사용하며 학습하지 않습니다. λ별 보상과 두 AI의 추천 없음 대비 개선량은 동결 평가 날짜에서 비교합니다. 선택적인 중지 시각은 미래 KST ISO 시각으로 입력하세요. 비우면 시각 제한 없이 선택한 실험 일정을 따릅니다. 별도의 일수 상한은 없습니다.\n']
    code='''from datetime import datetime,timezone
from kiosk_research.config import Config
from kiosk_research.live import run_live
PRESET = "pilot" #@param ["pilot", "research", "continual"]
DEADLINE_KST = "" #@param {type:"string"}
RESUME_ID = "" #@param {type:"string"}
deadline = datetime.fromisoformat(DEADLINE_KST) if DEADLINE_KST.strip() else None
if deadline is not None and (deadline.tzinfo is None or deadline <= datetime.now(timezone.utc)):
    raise ValueError("미래의 시간대 포함 중지 시각을 입력하세요.")
config = Config.load(CODE_ROOT / "config" / ("actor_critic_"+PRESET+".yaml"))
if deadline is not None: config.early_stopping = False
config.auto_cleanup = False
assert 'no_recommendation' in config.models and config.decision_time_mode == 'equal'
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
