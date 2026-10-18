# 실행 구조

```mermaid
flowchart TD
  A[Cloud Codex 개발·검증] --> B[GitHub 소스 커밋]
  B --> C[새 Colab 노트북: 커밋 고정]
  C --> D[비동기 설정·실행 UI]
  D --> E[매장·만족도 시나리오별 고정/상황별 함수]
  E --> F[공통 λ 묶음으로 하루 후보 비교]
  F --> G[동결 검증·최종 평가]
  F --> H[Drive: 매장별 부분 진행·JSON 체크포인트]
  G --> I[고객·가게 성과 쌍과 파레토 분석]
  I --> J[Drive 원시 결과·모델·그래프]
  I --> K[GitHub 결과 전용 브랜치]
```

실제 실행은 Colab CPU에서 이루어집니다. Codex나 GitHub Actions가 Colab의 장시간 실행을 보장하거나 비공식 API로 세션을 시작하지 않습니다. 원본 매장·고객 생성은 src/kiosk_sim, 새 연구는 src/kiosk_research, 공용 helper는 src/kiosk_cloud입니다.

각 매장의 시뮬레이션 상태는 모델·만족도 시나리오·λ별로 독립적이며 같은 외생 난수를 공유합니다. 매일 모든 매장을 완료한 뒤 날짜를 진행합니다. 체크포인트는 JSON 상태와 일별 Parquet 해시 목록을 저장합니다. 재개는 동일한 코드·커밋·설정·Python/라이브러리 버전을 요구합니다.
