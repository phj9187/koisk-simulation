"""Build explicit wall-clock bounded Actor-Critic notebook."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(revision='main'):
    n=json.loads((ROOT/'notebooks/run_actor_critic.ipynb').read_text(encoding='utf-8'))
    n['metadata']['colab']['name']='Kiosk_Actor_Critic_Live.ipynb'
    n['cells'][0]['source']=['# 액터–크리틱 실시간 시뮬레이션\n\n2026년 10월 28일 정오(한국 시각)까지 실행합니다. 매장 완료마다 아래 그래프가 갱신됩니다. 정오부터 새 매장 계산을 시작하지 않고 진행 중인 매장을 저장한 뒤 중지합니다. 1000일 상한도 유지합니다. Colab 런타임 자체가 종료되면 실행도 멈추므로 progress.json과 체크포인트로 확인하세요.\n']
    n['cells'][5]['source']=['## 3. 정오까지 실행 · 실시간 평가 그래프\n\n8매장·9평가기준·30방문자, 20일 학습/5일 동결 검증/5일 동결 평가를 이어갑니다. 이번 시간 제한 실험은 수렴 조기 중지를 끄고 정오 또는 1000일 중 먼저 도달하는 조건에서 중지합니다. 각 평가기준을 펼치면 고객·매장·공통 보상의 두 모델 추이를 볼 수 있습니다. 계산 중 마지막 점은 일부 매장만 반영한 임시값이며 확정 결과와 구분됩니다.\n']
    code='''from datetime import datetime,timezone
from kiosk_research.config import Config
from kiosk_research.live import run_live
DEADLINE_KST = "2026-10-28T12:00:00+09:00" #@param {type:"string"}
RESUME_ID = "" #@param {type:"string"}
TARGET_DAY = 1000 #@param {type:"integer"}
deadline = datetime.fromisoformat(DEADLINE_KST)
if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
    raise ValueError("미래의 시간대 포함 중지 시각을 입력하세요.")
config = Config.load(CODE_ROOT / "config/actor_critic_continual.yaml")
config.early_stopping = False
config.auto_cleanup = False
run = run_live(DATA_ROOT, config, deadline.astimezone(timezone.utc).isoformat(),
               resume_id=RESUME_ID.strip() or None, target_day=TARGET_DAY)
print("Saved:",run.id,run.state["day"],run.state["stop_reason"])
'''
    n['cells'][6]['source']=code.splitlines(keepends=True)
    for c in n['cells']:
        if c['cell_type']=='code':
            c['source']=[s.replace('REVISION = "main"',f'REVISION = "{revision}"') for s in c['source']]
            c['outputs']=[];c['execution_count']=None
            compile(''.join(c['source']),'live-notebook','exec')
    target=ROOT/'notebooks/run_actor_critic_live.ipynb'
    target.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    return target


if __name__=='__main__':build()
