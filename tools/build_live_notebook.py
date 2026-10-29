"""Build explicit wall-clock bounded Actor-Critic notebook."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BADGE='[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phj9187/koisk-simulation/blob/main/notebooks/run_actor_critic_live.ipynb)'

SYNC='''## 0. GitHub ↔ Colab 연동

이 노트북은 항상 위의 **Open in Colab** 배지로 여세요. 그러면 GitHub `main`의 최신 노트북이 열립니다. Drive에 있는 옛 사본(`Kiosk_Actor_Critic_Live.ipynb` 등)은 더 이상 쓰지 않습니다.

노트북을 고쳤다면 **파일 → GitHub에 사본 저장**을 누르고 저장소 `phj9187/koisk-simulation`, 브랜치 `main`, 경로 `notebooks/run_actor_critic_live.ipynb`를 그대로 선택해 저장하세요. 처음 한 번은 GitHub 권한 승인 창이 뜹니다. 코드(`src/`, `config/`)는 아래 1번 셀이 실행할 때마다 GitHub에서 새로 받아옵니다.
'''

DATA_MD='''## 2-1. 실제 자료 준비

KREI 원자료는 `AI-Kiosk-Data/raw_data/source_2025.xlsx`에 둡니다. 없으면 옛 위치 `MyDrive/Kiosk_LHS/source_2025.xlsx`에서 복사하고, 그것도 없으면 공식 주소에서 내려받습니다. 어느 경우든 SHA256이 맞아야 진행합니다.

시간대별 도착 패턴에는 서울시 상권분석서비스의 **추정매출-상권** CSV를 `AI-Kiosk-Data/raw_data/seoul_sales.csv`로 올리세요. CP949 인코딩도 그대로 읽습니다. `SALES_QUARTER`에 예컨대 `20251`을 넣으면 그 분기만 씁니다. 비워 두면 파일 전체를 합칩니다. 체크를 끄면 가정 곡선(점심·저녁 봉우리)을 씁니다.
'''

DATA_CODE='''import shutil
from kiosk_cloud.data import ensure_survey
USE_SALES_PROFILE = True #@param {type:"boolean"}
SALES_QUARTER = "" #@param {type:"string"}
RAW = DATA_ROOT / "raw_data"
RAW.mkdir(parents=True, exist_ok=True)
SURVEY = RAW / "source_2025.xlsx"
LEGACY_SURVEY = Path("/content/drive/MyDrive/Kiosk_LHS/source_2025.xlsx")
if not SURVEY.exists() and LEGACY_SURVEY.exists():
    shutil.copy2(LEGACY_SURVEY, SURVEY)
    print("옛 위치에서 복사:", LEGACY_SURVEY)
ensure_survey(SURVEY)
print("KREI 원자료 확인:", SURVEY)
SALES = RAW / "seoul_sales.csv"
if USE_SALES_PROFILE and not SALES.exists():
    raise FileNotFoundError("서울시 상권분석서비스 추정매출-상권 CSV를 " + str(SALES) + " 에 올리거나, USE_SALES_PROFILE 체크를 끄고 다시 실행하세요.")
print("시간대별 매출 자료:", SALES if USE_SALES_PROFILE else "사용 안 함(가정 곡선)")
'''

RUN_CODE='''import yaml
from datetime import datetime,timezone
from kiosk_research.config import Config
from kiosk_research.live import run_live
PRESET = "pilot" #@param ["pilot", "research", "continual"]
DEADLINE_KST = "" #@param {type:"string"}
RESUME_ID = "" #@param {type:"string"}
deadline = datetime.fromisoformat(DEADLINE_KST) if DEADLINE_KST.strip() else None
if deadline is not None and (deadline.tzinfo is None or deadline <= datetime.now(timezone.utc)):
    raise ValueError("미래의 시간대 포함 중지 시각을 입력하세요.")
config = Config(**yaml.safe_load((CODE_ROOT / "config" / ("actor_critic_"+PRESET+".yaml")).read_text(encoding="utf-8")))
config.survey_path = str(SURVEY)
config.sales_profile_path = str(SALES) if USE_SALES_PROFILE else ""
config.sales_quarter = SALES_QUARTER.strip() if USE_SALES_PROFILE else ""
config.validate()
if deadline is not None: config.early_stopping = False
config.auto_cleanup = False
assert 'no_recommendation' in config.models and config.decision_time_mode == 'equal'
assert config.lambda_search == 'pareto_two_stage'
run = run_live(DATA_ROOT, config, deadline.astimezone(timezone.utc).isoformat() if deadline else None,
               resume_id=RESUME_ID.strip() or None, target_day=None)
print("Saved:",run.id,run.state["day"],run.state["stop_reason"])
'''


def markdown(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}


def build(revision='main'):
    n=json.loads((ROOT/'notebooks/run_actor_critic.ipynb').read_text(encoding='utf-8'))
    n['metadata']['colab']['name']='Kiosk_Actor_Critic_Live.ipynb'
    n['cells'][0]['source']=[BADGE+'\n\n# 모델별 실시간 비교 · KREI 실제 매장 자료\n\n파레토 2단계 탐색으로 추천 없음과 두 Actor–Critic AI를 비교합니다. 매장은 KREI 2025 외식업 경영실태 원자료에서 LHS로 뽑은 표본이고, 매장별 평균 방문자 수를 중심으로 날짜마다 방문자 수가 포아송 분포로 달라집니다. 서울시 추정매출 자료를 연결하면 실제 시간대별 매출 패턴으로 손님이 도착합니다. 세 방식의 결정시간 공식은 동일합니다. 기존 결과는 보존하고 새 실험 ID를 만듭니다. 결과·계획·체크포인트는 Drive에 저장합니다. 아래 실시간 카드에서 모델을 클릭하면 확대 그래프, α·β·λ, 실제 추천 가중치를 확인할 수 있습니다. 순위는 완료된 별도 동결 평가 구간의 미추천 대비 종합 효용 개선량입니다.\n']
    n['cells'][5]['source']=['## 3. 자동 2단계 탐색 · 실시간 평가 그래프\n\npilot: KREI 실제 매장 8곳·1평가기준, 1일 넓은 학습/1일 구간 선택 검증/1일 집중 학습/1일 최종 검증/1일 평가(총 5일). research: 실제 매장 64곳(8구간 LHS × 8블록)·9평가기준, 10/5/10/5/5일(35일) × 5 독립 반복. continual: 같은 64매장으로 35일 주기 사이에 신경망을 이어 학습합니다. 방문자 수는 매장별 평균을 중심으로 날짜마다 달라지고, 하루 고정 AI에는 그날의 실제 방문자 수가 아니라 매장 평균만 줍니다. 64매장은 예전 가상 8매장·30명 설정보다 하루 계산량이 약 15배이므로 먼저 pilot으로 소요 시간을 확인하세요. α·β는 각 기준 안에서 고정합니다. 두 AI의 후보를 함께 보고 기준별 공통 λ 구간을 정하며, 무릎점이 불명확하면 전체 범위를 유지합니다. 집중 학습의 매 5번째 날짜에는 0·0.5·1을 재확인합니다. 추천 없음 대비 개선량은 별도 최종 평가 날짜에서 계산합니다. 중지 시각은 선택 사항이며 별도의 일수 상한은 없습니다. 중지 시각을 넣으면 작업자는 1개만 씁니다.\n']
    n['cells'][6]['source']=RUN_CODE.splitlines(keepends=True)
    n['cells'][5:5]=[markdown(DATA_MD),{'cell_type':'code','metadata':{},'source':DATA_CODE.splitlines(keepends=True),'outputs':[],'execution_count':None}]
    n['cells'][1:1]=[markdown(SYNC)]
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
