# AI 키오스크 · λ 조건부 파레토 연구

**새 기본 λ 탐색은 2단계입니다.** α·β 기준마다 0·0.5·1로 넓게 학습하고, 동결 검증의 파레토 무릎점 후보 주변 세 값으로 집중 학습합니다. 추천 없음 비교와 공통 결정시간을 유지합니다. pilot은 5일, 연구·누적 주기는 35일입니다. [정확한 일정·구간 선택·재개 규칙](docs/pareto_two_stage_ko.md).

**새 실험은 추천 없음·하루 고정 AI·고객별 AI의 3개 비교군을 사용합니다.** 탐색·결정시간을 동일한 공식으로 계산하고, 동결 평가 날짜에서 λ별 추천 없음 대비 개선량을 `baseline_comparison.csv`에 저장합니다. [함수 구조도·고객 선택식·λ 설명](docs/customer_choice_comparison_ko.md). 기존 603일 결과는 원래 가정으로 보존하며 새 비교 결과와 합치지 않습니다.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phj9187/koisk-simulation/blob/main/notebooks/run_actor_critic.ipynb)

**새 기본 진입점은 Actor–Critic 신경망 실험입니다.** 개인별 모델은 고객 결과마다, 일별 모델은 하루 결과로 학습합니다. 두 모델 모두 5개 추천 가중치를 출력하고 기존 평가함수를 보상으로 사용합니다. [구조·학습·저장 설명](docs/actor_critic_ko.md). 기존 `run_automatic.ipynb`, `screening_*.yaml` 및 1000일 실행은 무작위 탐색 기준 실험으로 보존됩니다.

**새 Colab 실행 노트북**에서 고객·가게 만족도의 여러 타협점을 탐색합니다. 고정 모델은 `w=f(λ)`, 상황별 모델은 `w=f(상황, λ)`를 매장·만족도 시나리오별로 학습합니다. 두 모델은 동일한 하루 목적과 동일한 시뮬레이션 호출 예산으로 비교합니다. 성능 향상이나 상황별 모델의 우월성을 미리 가정하지 않습니다.

## Colab에서 시작

1. 위 버튼으로 새 노트북을 엽니다. CPU 런타임부터 사용하세요.
2. 비공개 저장소라면 Colab Secrets에 `KIOSK_GITHUB_TOKEN`을 등록하고 이 노트북의 접근을 허용합니다. 코드 읽기에 해당 저장소 Contents 읽기, 결과 브랜치 업로드에는 쓰기 권한이 필요합니다. 토큰을 코드에 넣지 마세요.
3. **런타임 → 모두 실행**을 누르고 Drive 인증을 승인합니다. 기본 pilot 실험은 자동으로 실행·저장 후 종료합니다.
4. 다음 실행은 PRESET을 research로 바꾸거나, RESUME_ID와 원래 커밋을 지정해 재개합니다. [자동 실행·복구 안내](docs/automatic_colab_ko.md)를 참고하세요.
5. 버튼으로 설정하려면 [대화형 노트북](https://colab.research.google.com/github/phj9187/koisk-simulation/blob/main/notebooks/run_simulation.ipynb)을 사용합니다. KREI 자료는 별도로 원자료를 검증해야 하며 자동 실행 기본값은 가상 자료입니다.

Drive 루트는 `/content/drive/MyDrive/AI-Kiosk-Data`입니다. 원본 Colab과 기존 `Kiosk_LHS` 결과를 덮어쓰지 않습니다. 재개 시에는 실험 metadata의 정확한 코드 커밋을 사용하세요.

## 학습과 평가

- λ가 고객·가게 성과의 선호 비율입니다. 만족도 구성 가중치, 추천 가중치와 별개입니다.
- 새 두 정책은 λ별 추천 가중치를 Dirichlet Actor로 출력하고 Critic의 기대 보상과 실제 보상의 차이로 학습합니다. `actor_critic_*.yaml`을 사용합니다. 이전 paired random search는 `screening_*.yaml`에서만 유지합니다.
- 반복마다 초기 모델로 시작해 **학습 → 동결 검증 → 동결 최종 평가**를 수행합니다. 반복 수 0은 중지할 때까지 반복합니다.
- 결과는 고객·가게 성과 쌍, 근사 파레토 후보, λ별 가중치 함수, 하이퍼볼륨과 독립 반복 차이로 저장합니다.
- 만족도 시나리오는 명시적인 가정입니다. 실제 주관적 만족도를 검증한 결과가 아닙니다. 단일 최적 λ는 운영 제약 없이 추천하지 않습니다.

## 로컬 실행·검증

Python 3.11 이상(권장 3.12):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src python -m kiosk_research.cli --config config/actor_critic_pilot.yaml --root runs
```

하루 경계에서 중지: `--days 1`. 재개: `--resume 실험ID --root runs`. 다른 프로세스에서 중지 요청: `--stop 실험ID --root runs`. 신경망 pilot은 8개 가상 매장·평가기준 1개·하루 5명·학습 2일·검증 1일·평가 1일이며 실제 연구 성과로 해석하지 않습니다.

## 코드와 결과

- `src/kiosk_research/`: 새 정책, 시뮬레이션, 실험·복구, 분석, Colab UI
- `src/kiosk_sim/`: 업로드된 기존 노트북에서 추출한 원본 21개(해시 보존). 매장·고객 생성 등 환경 일부를 재사용합니다.
- `src/kiosk_cloud/`: 공용 저장·출처 기록·검증된 원자료 다운로드
- `notebooks/run_simulation.ipynb`: 새 진입점
- `notebooks/original/`: 기존 노트북(출력 제거)
- `legacy/`: 이전 웹앱과 중간 프로토타입 보존
- `docs/research_methodology_ko.md`: 가정, 수식, 공정 비교와 연구 한계
- `docs/colab_research_ko.md`: 인증, 저장, 중지·복구, GitHub 기록

원시 데이터·모델·체크포인트는 Drive에 저장합니다. 요약 업로드는 `results/실험ID-day-날짜` 새 브랜치를 만들고 기본 브랜치를 직접 변경하지 않습니다. 토큰이 없으면 실험은 계속 가능하며 Drive에 요약을 보존합니다.

실제 Colab 인증·Drive 동기화·GitHub 결과 쓰기는 사용자 계정에서 검증해야 합니다. Pro+도 무제한 실행을 보장하지 않습니다.


함수 및 실행 점검(schema 4): 추천은 가격 여유를 포함한 5개 가중치를 출력하며, 결과 평가는 독립 모듈에서 수행합니다. 이전 모델과 호환되지 않으므로 새 실험으로 시작하세요. [수정 근거와 점검 기록](docs/model_audit_20261026_ko.md) · [전체 구조](docs/simulation_structure_ko.md)


## 2026-10-26 자동 실행 업데이트

- 기본 9개 평가 기준과 학습에 쓰지 않는 3개 중간 평가 기준.
- 정책을 고정한 교차평가와 후보 내 후회·순위 분석.
- 사용자 인증 뒤 Colab에서 자동 실행하는 `notebooks/run_automatic.ipynb`.
- 체크포인트 기본 2개 보존, 체크섬 확인 후 중복 부분 스냅샷·동일 파일 정리.
- Drive 진행 상황 파일 `latest_run.json`, 실험별 `progress.json`.
- 독립 날짜·고유 데이터·다른 실험은 자동 삭제하지 않음.

새 버전은 구버전 체크포인트를 이어 쓰지 않습니다. 과거 실험은 원래 커밋으로 재개하세요.
