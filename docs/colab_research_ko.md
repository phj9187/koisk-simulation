# 실행·인증·저장·복구

## Colab 시작

README의 Open in Colab → 코드 불러오기 → Drive 마운트 인증 → 설정 패널 순서로 실행합니다. 비공개 저장소는 Colab Secrets의 KIOSK_GITHUB_TOKEN에 해당 저장소 Contents 읽기 권한을 부여합니다. 결과 새 브랜치 작성에는 Contents 쓰기가 필요합니다. 토큰을 소스/출력/URL에 넣지 않으며 채팅으로 보낼 필요도 없습니다.

CPU에서 실행하며 GPU는 현재 정책/시뮬레이터의 속도를 자동으로 높이지 않습니다. Pro+도 자원·런타임 지속성을 보장하지 않습니다. 초기에 8개 가상 매장으로 중지·재개를 검증한 뒤 연구 설정을 늘리세요. KREI 64매장 실제 실험은 config/research_empirical.yaml을 참고합니다. 조사 파일의 SHA256을 검증하며 다른 자료로 대체하지 않습니다.

매장 자료를 경험 표본으로 쓰려면 UI에서 empirical, 매장 수 64, '방문자 수 직접 지정' 해제를 선택합니다. 원자료가 기존 Kiosk_LHS/source_2025.xlsx에 있다면 그 경로를 지정할 수 있습니다. 직접 방문자 수를 지정하면 원래 매장 방문자 분포가 달라집니다.

## Drive 구조

`/content/drive/MyDrive/AI-Kiosk-Data/` 아래:

- raw_data/: 검증된 조사 원자료
- experiments/실험ID/: metadata.json, progress.json, daily_metrics.parquet, detailed_results/의 일별 Parquet과 선택적 고객 이벤트, summary_exports/의 작은 요약
- checkpoints/실험ID/: 체크섬 검증 JSON 두 세대와 매장별 부분 진행
- models/실험ID/: 최신 상태와 반복별 학습 완료 파라미터
- backups/: 사용자가 별도 백업을 보관할 위치 (외부 자동 백업을 보장하지 않음)

원본 노트북/기존 Drive 결과는 수정하지 않습니다. 메타데이터에 실행 커밋, dirty 상태, 소스 SHA256, 라이브러리·CPU/GPU/메모리, UTC 시작 시각, 설정을 기록합니다. 로컬 미커밋 실험은 git_commit=null로 명시합니다.

## 중지·재개

UI는 작업자 스레드에 실행을 맡겨 셀이 반환되고 중지 콜백에 응답합니다. 중지 버튼 또는 experiments/실험ID/STOP 파일로 요청하면 모든 매장의 현재 날짜를 마친 뒤 저장합니다. 작업 완료를 기다린 후 결과를 보거나 재개하세요. 강제 인터럽트가 작업자 스레드를 안전하게 즉시 종료한다고 가정하지 않습니다.

재개 ID를 입력하고 동일 코드 커밋으로 실행합니다. 매장별 저장 결과를 재사용하므로 이미 완료된 매장을 중복 학습하지 않습니다. 코드·설정·커밋이 바뀌면 거부합니다. 저장된 템플릿이 있으므로 재개만을 위해 원자료를 다시 다운로드할 필요는 없습니다.

체크포인트·일별 결과 샤드의 체크섬을 확인하고 손상 시 이전 유효 세대로 복구합니다. 모두 손상되면 새 실험으로 가장하지 않고 오류를 냅니다. JSON만 사용하며 새 체크포인트를 pickle로 역직렬화하지 않습니다. 일별 데이터를 배치 저장하고 매 고객마다 Drive에 쓰지 않습니다. 저장 주기 사이의 작업은 부분 결과에서 재계산될 수 있습니다. Drive 동기화 전 세션 종료나 저장소 오류로 일부 데이터가 손실될 수 있습니다.

한 실험은 한 런타임에서만 실행하세요. RUNNING.lock은 이전 실행이 완전히 끝났음을 확인한 뒤에만 제거합니다. Drive FUSE가 서로 다른 런타임에 분산 트랜잭션·잠금 원자성을 보장한다고 가정하지 않습니다. 무제한 반복에서도 일별 원시 데이터는 디스크로 저장하며 전체 고객 로그를 메모리에 누적하지 않습니다. 최종 결과를 한꺼번에 불러오는 export는 실험 크기에 따라 메모리를 사용합니다.

## 결과·업로드

결과 보기는 CSV/JSON/PNG/Markdown 요약을 만듭니다. 최종 반복이 미완료이면 최종 비교는 비어 있고 학습 진행을 표시합니다. λ 추천은 검증 자료와 사용자가 입력한 운영 제약만으로 선택하며 최종 평가 성과를 별도로 표시합니다.

GitHub 업로드는 새 results/실험ID-day-날짜 브랜치에 작은 요약만 저장합니다. 기본 브랜치 직접 변경, 원시 데이터·모델·체크포인트 업로드는 하지 않습니다. 합계 10 MB 상한이며 토큰이 없으면 건너뛰고 Drive에 남깁니다. 여러 브랜치를 병합할 때 registry CSV 충돌을 해결해야 할 수 있습니다. HTTP/인증 실패를 성공으로 보고하지 않습니다. 업로드가 성공한 브랜치는 github_upload.json에 기록됩니다.

## Cloud Codex 검증

```sh
python -m venv /workspace/kiosk-venv
/workspace/kiosk-venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src MPLCONFIGDIR=/tmp/kiosk-mpl /workspace/kiosk-venv/bin/python -m unittest discover -s tests -p 'test_pareto_research.py' -v
PYTHONPATH=src MPLCONFIGDIR=/tmp/kiosk-mpl /workspace/kiosk-venv/bin/python -m kiosk_research.cli --config config/research_smoke.yaml --root /tmp/kiosk-smoke
```

로컬 테스트는 Colab 로그인, 실제 Drive FUSE 버튼 응답/동기화, GitHub API 결과 업로드를 대신 검증하지 않습니다. 첫 Colab 실행에서 이 인증·연결을 확인해야 합니다.
