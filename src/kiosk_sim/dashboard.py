"""Self-contained browser dashboard for one simulation run."""
import json
from html import escape
from pathlib import Path

METRICS = {
    "score": ("예시 종합점수", "높을수록 좋음"),
    "satisfaction": ("고객 만족도", "높을수록 좋음"),
    "store_satisfaction": ("가게 만족도(가정 지표)", "높을수록 좋음"),
    "profit": ("매장 이익", "높을수록 좋음"),
    "avg_wait": ("평균 조리 대기시간", "낮을수록 좋음"),
    "avg_decision": ("평균 메뉴 선택시간", "낮을수록 좋음"),
    "avg_kiosk_queue": ("평균 키오스크 대기시간", "낮을수록 좋음"),
    "avg_total_wait": ("평균 총 소요시간", "낮을수록 좋음"),
    "served": ("응대한 고객", "높을수록 좋음"),
    "accept_rate": ("추천 수락률", "높을수록 좋음"),
    "waste": ("남은 재고의 폐기 위험", "낮을수록 좋음"),
    "stockouts": ("주문 가능 메뉴 없는 고객", "낮을수록 좋음"),
}

DESCRIPTIONS = {
    "score": "고객 만족도×0.45 + 가게 만족도×0.35 + 응대율×0.10 − 평균 총 소요시간/(평균 총 소요시간+25)×0.10. 가정한 종합 기준이며 실측 효용이 아닙니다.",
    "satisfaction": "응대한 고객의 취향·방문 목적 적합도에 가격과 대기 불편을 반영한 가상 만족도의 평균(0~1). 식사 후 설문값이 아닙니다.",
    "store_satisfaction": "가상 이익 목표 달성도 75%와 남은 재고의 폐기 위험 감소 25%로 만든 가게 지표(0~1). 실제 점주 평가가 아닙니다.",
    "profit": "판매 메뉴의 가격에서 메뉴별 가상 원가를 뺀 하루 총액. 임대료·인건비 등의 고정비는 빠져 있습니다.",
    "avg_wait": "주문을 마친 순간부터 해당 메뉴의 조리가 끝날 때까지의 고객당 평균 시간(분).",
    "avg_decision": "키오스크에서 메뉴 선택을 시작한 뒤 주문할 때까지의 고객당 평균 시간(분). 추천 수락 여부에 따른 시간 차이는 가정입니다.",
    "avg_kiosk_queue": "매장에 도착한 뒤 키오스크 사용을 시작하기까지의 고객당 평균 대기(분).",
    "avg_total_wait": "키오스크 대기 + 메뉴 선택 + 조리 대기를 합친 응대 고객당 평균 시간(분).",
    "served": "해당 가상 영업일에 주문을 완료한 고객 수. 입력한 고객 수 대비 비율이 종합점수에 포함됩니다.",
    "accept_rate": "추천이 제시된 주문 중 제안된 메뉴가 선택된 비율. 실제 수락률이 아닌 행동 모형의 출력입니다.",
    "waste": "영업 종료 시 남은 재고에 메뉴별 폐기 위험을 적용한 가상 위험 지표. 실제 폐기량(kg)이 아닙니다.",
    "stockouts": "재고 부족 또는 예산·알레르기 제한으로 주문 가능한 메뉴가 없었던 고객 수. 순수 품절 건수가 아닙니다.",
}


def write_dashboard(path, rows, weights, hourly, *, ai_weights=None, ai_info=None, robust=None, random_refs=None,
                    validation=None, weight_benchmark=None, sensitivity=None,
                    contextual_comparison=None,
                    customers, train_days, test_days, start_hour=11, source_label="공공데이터"):
    averages = {policy: {key: sum(float(r[key]) for r in subset) / len(subset)
                         for key in METRICS}
                for policy in ("none", "popular", "customer", "fixed", "gemini")
                for subset in [[r for r in rows if r["policy"] == policy]]}
    payload = json.dumps({"averages": averages, "weights": weights, "aiWeights": ai_weights,
                          "aiInfo": ai_info, "robust": robust, "randomRefs": random_refs,
                          "validation": validation, "weightBenchmark": weight_benchmark,
                          "contextualComparison": contextual_comparison,
                          "sensitivity": sensitivity,
                          "descriptions": DESCRIPTIONS, "hourly": hourly,
                          "customers": customers, "trainDays": train_days, "testDays": test_days,
                          "metrics": METRICS, "startHour": start_hour, "sourceLabel": source_label}, ensure_ascii=False).replace("<", "\\u003c")
    html = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>키오스크 추천 시뮬레이션</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#172331;background:#f2f5f9}*{box-sizing:border-box}body{max-width:1100px;margin:auto;padding:28px 18px 70px}h1{font-size:clamp(26px,4vw,40px);margin:8px 0}h2{font-size:20px;margin:0 0 15px}p{line-height:1.6;color:#536374}.tag{color:#12616a;font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:white;padding:22px;border-radius:16px;box-shadow:0 3px 18px #1830470d;margin-top:16px}.wide{grid-column:1/-1}select{padding:10px;border:1px solid #b9c9d2;border-radius:9px;background:white;font-size:16px}.row{display:grid;grid-template-columns:105px 1fr 65px;gap:10px;align-items:center;margin:12px 0;font-size:14px}.bar{height:19px;background:#eaf0f3;border-radius:12px;overflow:hidden}.fill{height:100%;background:#40878d;border-radius:12px}.learned .fill{background:#ef9646}.hour{display:flex;align-items:end;gap:6px;height:145px}.hour>div{flex:1;text-align:center;font-size:12px;color:#536374}.hour i{display:block;background:#52a2aa;border-radius:5px 5px 0 0;min-height:2px}.hint{font-size:13px}.pill{display:inline-block;padding:5px 9px;border-radius:8px;background:#eef5f5;margin:4px;color:#24575c}strong{color:#172331}
</style></head><body><div class="tag">연구용 시뮬레이션 · 실행 결과</div><h1>키오스크 메뉴 추천 비교</h1><p>같은 가상 고객·가게 조건에서 추천 방식별 결과를 비교합니다. 만족도와 메뉴 선택시간은 가정한 함수로 계산하며 실제 매장 실적을 의미하지 않습니다.</p>
<div id="summary" class="card"></div><div class="grid"><section class="card wide"><h2>AI 가중치 학습 효과: 여러 가중치의 평균과 비교</h2><p class="hint">학습 전 동일 가중치 1개와 고정된 난수로 생성한 여러 가중치를 같은 평가용 영업일에 적용한 참고 분포입니다. 이는 <strong>현실 매장이나 모든 가능한 가중치의 평균이 아닙니다.</strong> 아래 최소 개선 기준은 연구자가 <strong>평가 전에</strong> 정하는 예시 기준이며 실제 영업 효용의 검증선이 아닙니다.</p><div id="weightBenchmark"></div></section><section class="card wide"><h2>보조 비교: 학습 전 동일 가중치</h2><p class="hint">두 방식 모두 메뉴를 추천합니다. 같은 가상 고객·가게·난수 조건으로 학습에 쓰지 않은 영업일만 비교했습니다. 양수/음수는 Gemini 값에서 동일 가중치 값을 뺀 결과입니다.</p><div id="validation"></div><p class="hint">95% 구간은 평가용 <strong>가상 영업일</strong>을 다시 뽑아 만든 점수 차이의 범위입니다. 현실에서의 효과나 통계적 유의성을 입증하는 구간이 아닙니다. 구간이 0을 지나면 시뮬레이션 내 개선 방향도 불확실합니다.</p></section><section class="card"><h2>추천 방식 비교</h2><label for="metric">비교할 지표　</label><select id="metric"></select><p id="direction" class="hint"></p><p id="metricDescription" class="hint"></p><div id="bars"></div></section><section class="card"><h2>학습 전 가중치 (각 25%)</h2><div id="weights"></div><p class="hint">사전 정보 없이 동일하게 둔 출발점입니다. 실측된 최적 가중치가 아닙니다.</p></section><section class="card"><h2>Gemini 학습 가중치</h2><div id="aiWeights"></div><p class="hint" id="aiNote"></p></section><section class="card wide"><h2>가정을 바꾸면 결론이 달라지는가?</h2><p class="hint">학습된 Gemini 가중치는 그대로 두고, 추천 수락 확률(기본의 0.5·1·1.5배), 메뉴 선택시간(0.5·1·2배), 고객/가게 만족도 점수 비율(55/25·45/35·35/45)을 바꾼 27개 가상 조건입니다. 나머지 점수 비율은 그대로 둡니다. 모든 조건은 동일한 평가 영업일과 참고 가중치 32개를 사용합니다.</p><div id="sensitivity"></div><p class="hint">이 27개는 연구자가 정한 조합이지 실제 환경의 발생 확률이 아닙니다. 각 조건의 95% 구간은 가상 영업일의 변동만 포함합니다. 여러 조건 중 좋은 것만 고르거나 27개 구간을 한꺼번에 엄격한 검정으로 해석하면 안 됩니다.</p></section><section class="card wide"><h2>추천 기능 자체의 효과: 추천 없음과 비교</h2><p class="hint">동일한 평가용 영업일의 평균 차이입니다. 종합점수의 %는 추천 없음 점수 대비 상대 변화이며, 실제 매장 효과의 추정치는 아닙니다.</p><div id="deltas"></div></section><section class="card wide"><h2>가정 변경에도 우세한가?</h2><p class="hint">고객 만족·고객당 이익·남은 재고 위험 감소·응대율·시간 효율의 가중치 격자와 선택시간 가정을 바꾼 결과입니다. 우세 비율은 계산한 가정 조합의 비율이며 현실 성공 확률이나 통계적 유의성이 아닙니다.</p><p id="randomRefs" class="hint"></p><div id="robust"></div><p class="hint">동률은 어느 쪽도 우세하지 않은 경우입니다. 최악 차이는 조합 중 가장 불리한 차이입니다.</p></section><section class="card wide"><h2>시간대별 고객 도착 비율</h2><p id="source"></p><div class="hour" id="hours"></div><p class="hint">생활인구의 시간대별 상대 비중을 사용하며, 전체 고객 수는 별도로 입력한 가정값입니다.</p></section></div>
<script>const data=PAYLOAD;const names={none:'추천 없음',popular:'인기 메뉴',customer:'취향 우선',fixed:'학습 전 동일 가중치',gemini:'Gemini 피드백 학습'};const weightNames=['취향','재고·폐기 위험','매장 이익','대기시간'];document.getElementById('summary').innerHTML=`<strong>가상 고객 ${data.customers}명 / 영업일</strong>　<span class="pill">가중치 학습 ${data.trainDays}일</span><span class="pill">별도 평가 ${data.testDays}일</span>`;const select=document.getElementById('metric');for(const [key,[name]] of Object.entries(data.metrics)){const o=document.createElement('option');o.value=key;o.textContent=name;select.append(o)};function update(){const key=select.value,max=Math.max(...Object.values(data.averages).map(x=>x[key]),.001);document.getElementById('direction').textContent=data.metrics[key][1]+' · 값은 평가 영업일의 평균';document.getElementById('metricDescription').textContent=data.descriptions[key];document.getElementById('bars').innerHTML=Object.entries(data.averages).map(([policy,v])=>`<div class="row ${policy}"><span>${names[policy]}</span><div class="bar"><div class="fill" style="width:${Math.max(0,v[key]/max*100)}%"></div></div><strong>${v[key].toFixed(4)}</strong></div>`).join('')}select.onchange=update;update();const bench=data.weightBenchmark;document.getElementById('weightBenchmark').innerHTML=`<p><strong>Gemini 종합점수 ${bench.ai_mean_score.toFixed(4)} / 참고 가중치 ${bench.reference_count}개의 평균 ${bench.reference_mean_score.toFixed(4)}</strong></p><p><strong>상대 증가율 ${bench.relative_gain_percent>=0?'+':''}${bench.relative_gain_percent.toFixed(2)}%</strong>　<span class="pill">가상 영업일 재표본 95% 구간 ${bench.lower_95_percent.toFixed(2)}% ~ ${bench.upper_95_percent.toFixed(2)}%</span><span class="pill">사전 최소 기준 ${bench.minimum_gain_percent.toFixed(2)}%</span></p><p><strong>판정: ${bench.meets_threshold?'설정한 가상 조건에서는 최소 증가 기준을 넘었습니다.':'설정한 가상 조건에서 최소 증가 기준을 넘었다고 판단할 수 없습니다.'}</strong></p><p class="hint">증가율 = (Gemini 평균 점수 ÷ 참고 가중치 평균 점수 − 1) × 100. 가상 영업일 재표본 구간의 하한이 사전 기준보다 클 때만 기준을 넘었다고 표시합니다. 참고 가중치의 개별 평균 점수 범위: ${bench.reference_mean_min.toFixed(4)} ~ ${bench.reference_mean_max.toFixed(4)}. Gemini는 ${bench.above_reference_count}/${bench.reference_count}개 가중치보다 높았고 ${bench.equal_reference_count}개와 같았습니다. 기준 1%는 실측 근거 없이 넣은 예시입니다. 현실에서의 효과는 이 결과만으로 입증할 수 없습니다.</p>`;const sens=data.sensitivity;document.getElementById('sensitivity').innerHTML=`<p><strong>평균 증가율 양수 ${sens.positive_average_count}/${sens.condition_count}개 조건</strong> · 가상 영업일 구간의 하한이 사전 최소 기준 ${sens.minimum_gain_percent.toFixed(1)}% 초과: <strong>${sens.meets_threshold_count}/${sens.condition_count}개 조건</strong></p><details><summary>27개 조건별 결과 보기</summary><div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;text-align:left"><thead><tr><th>수락 확률 배수</th><th>선택시간 배수</th><th>고객/가게 점수 비율</th><th>점수 증가율</th><th>영업일 재표본 95% 구간</th><th>기준 충족</th></tr></thead><tbody>${sens.conditions.map(c=>`<tr><td>${c.acceptance_multiplier}</td><td>${c.choice_time_multiplier}</td><td>${Math.round(c.customer_weight*100)}/${Math.round(c.store_weight*100)}</td><td>${c.relative_gain_percent.toFixed(2)}%</td><td>${c.lower_95_percent.toFixed(2)} ~ ${c.upper_95_percent.toFixed(2)}%</td><td>${c.meets_threshold?'예':'아니오'}</td></tr>`).join('')}</tbody></table></div></details>`;const check=data.validation.comparisons.gemini_vs_fixed;const main=check.score;const pass=main.lower_95>0;document.getElementById('validation').innerHTML=`<p><strong>가상 종합점수 평균 차이: ${main.mean_delta>=0?'+':''}${main.mean_delta.toFixed(4)}</strong>　<span class="pill">가상 영업일 재표본 구간 ${main.lower_95.toFixed(4)} ~ ${main.upper_95.toFixed(4)}</span><span class="pill">개선 ${main.better_days}/${data.validation.days}일 · 동일 ${main.equal_days}일</span></p><p class="hint">${pass?'동일 가중치보다 높은 방향이 나타났습니다. 최소 증가 기준 판정은 위 참고 가중치 비교를 보세요.':'이 평가에서는 AI의 일관된 개선을 확인하기 어렵습니다.'} 실제 매장 효과는 판단할 수 없습니다.</p><table style="border-collapse:collapse;width:100%;text-align:left"><thead><tr><th>지표</th><th>평균 차이</th><th>가상 영업일 재표본 95% 구간</th><th>개선일</th></tr></thead><tbody>${['satisfaction','store_satisfaction','profit','avg_decision','avg_total_wait','waste','served'].map(k=>{const v=check[k];return `<tr><td title="${data.descriptions[k]}" style="padding:8px;border-top:1px solid #ddd">${data.metrics[k][0]}</td><td>${v.mean_delta>=0?'+':''}${v.mean_delta.toFixed(3)}</td><td>${v.lower_95.toFixed(3)} ~ ${v.upper_95.toFixed(3)}</td><td>${v.better_days}/${data.validation.days}</td></tr>`}).join('')}</tbody></table><p class="hint">표의 값은 Gemini − 동일 가중치입니다. 이익은 가상 화폐 단위/일, 시간은 분/고객, 나머지는 각 지표 설명을 참고하세요. 각 지표의 이름에 마우스를 올리거나 아래 지표 선택창에서 계산 기준을 볼 수 있습니다.</p>`;document.getElementById('weights').innerHTML=data.weights.map((x,i)=>`<div class="row"><span>${weightNames[i]}</span><div class="bar"><div class="fill" style="width:${x*100}%"></div></div><strong>${(x*100).toFixed(1)}%</strong></div>`).join('');document.getElementById('aiWeights').innerHTML=data.aiWeights.map((x,i)=>`<div class="row"><span>${weightNames[i]}</span><div class="bar"><div class="fill" style="width:${x*100}%"></div></div><strong>${(x*100).toFixed(1)}%</strong></div>`).join('');document.getElementById('randomRefs').textContent=`Gemini는 평가용 고객에서 무작위로 만든 ${data.randomRefs.reference_count}개 가중치 중 ${data.randomRefs.better_than_references}개보다 우세, ${data.randomRefs.tied_with_references}개와 동률, ${data.randomRefs.worse_than_references}개보다 열세였습니다. 이 표본의 최적값이라는 뜻은 아닙니다.`;document.getElementById('deltas').innerHTML=Object.entries(data.averages).filter(([policy])=>policy!=='none').map(([policy,v])=>{const b=data.averages.none;const d=v.score-b.score;return `<p><strong>${names[policy]}</strong>: 종합점수 ${d>=0?'+':''}${d.toFixed(4)} (${(100*d/b.score).toFixed(1)}%), 고객 만족도 ${(v.satisfaction-b.satisfaction>=0?'+':'')+(v.satisfaction-b.satisfaction).toFixed(3)}, 가게 만족도 ${(v.store_satisfaction-b.store_satisfaction>=0?'+':'')+(v.store_satisfaction-b.store_satisfaction).toFixed(3)}, 메뉴 선택시간 ${(v.avg_decision-b.avg_decision).toFixed(2)}분, 매장 이익 ${(v.profit-b.profit).toFixed(1)}</p>`}).join('');document.getElementById('robust').innerHTML=Object.entries(data.robust).map(([policy,pairs])=>Object.entries(pairs).map(([base,r])=>`<div class="row" style="grid-template-columns:170px 1fr 165px"><span>${names[policy]} / ${names[base]}</span><div class="bar"><div class="fill" style="width:${(r.win_share*100).toFixed(1)}%"></div></div><strong>우세 ${(r.win_share*100).toFixed(1)}% · 동률 ${(r.tie_share*100).toFixed(1)}%</strong></div>`).join('')).join('');document.getElementById('aiNote').textContent=`Gemini API (${data.aiInfo.model}): 가상 고객별 피드백으로 ${data.aiInfo.api_calls}회 제안, 여러 가정에서 우세한 제안 ${data.aiInfo.accepted_updates}회 반영. 실제 고객 설문은 아닙니다. 0회 반영이면 출발 가중치와 결과가 같습니다.`;const hours=document.getElementById('hours');document.getElementById('source').textContent=data.hourly?data.sourceLabel:'가정한 도착시간 분포';if(data.hourly){const m=Math.max(...data.hourly);hours.innerHTML=data.hourly.map((v,i)=>`<div><i style="height:${v/m*110}px"></i>${data.startHour+i}시<br>${(v*100).toFixed(1)}%</div>`).join('')}else{hours.innerHTML='<p>공공데이터 입력 없이 실행했습니다. 도착시간은 시뮬레이션 내부에서 생성합니다.</p>'}</script></body></html>'''
    if ai_info and ai_info.get("optimizer") in ("bayesian", "genetic", "contextual"):
        method = {"bayesian": "베이지안 최적화", "genetic": "유전 알고리즘",
                  "contextual": "상황별 가중치 학습"}[ai_info["optimizer"]]
        html = html.replace('<div id="summary" class="card"></div>',
            '<div id="summary" class="card"></div><section class="card"><h2>학습일의 고객·가게 만족도 파레토 후보</h2><p class="hint">두 값 모두 학습용 가상 영업일 평균입니다. 어느 쪽도 동시에 개선할 수 없는 시험 후보만 표시하며, 표의 선택안은 관측된 두 목표 이상점까지의 정규화 거리가 가장 짧은 후보입니다. 평가용 영업일의 결과는 아래 비교를 보세요.</p><div id="paretoTable"></div></section>')
        start = html.index("document.getElementById('aiNote').textContent=")
        end = html.index(";const hours=", start)
        html = (html[:start] +
                "document.getElementById('aiNote').textContent=`" + method +
                ": 동일 가상 가게의 학습 영업일에 ${data.aiInfo.candidate_evaluations}개 가중치를 평가했습니다. "
                "이는 가상 데이터에 대한 탐색 결과이며 실제 매장 최적값이 아닙니다.`" +
                html[end:])
        html = html.replace("Gemini", method).replace("gemini:'Gemini 피드백 학습'",
                                                   "gemini:'" + method + " 학습'")
        if ai_info["optimizer"] == "contextual":
            html = html.replace("학습 가중치</h2>", "기준 상태의 가중치</h2>")
            html = html.replace("이는 가상 데이터에 대한 탐색 결과이며 실제 매장 최적값이 아닙니다.",
                "아래 가중치는 기준 상태에서만 적용됩니다. 실제 주문마다 고객·재고·조리 혼잡에 따라 달라집니다. 가상 데이터에 대한 탐색 결과입니다.")
            html = html.replace('<div id="summary" class="card"></div>',
                '<div id="summary" class="card"></div><section class="card"><h2>상황별 모델과 학습된 고정 가중치 비교</h2><p class="hint">두 방식은 같은 학습일·동일 후보 평가 횟수로 훈련하고 같은 별도 평가일에 적용했습니다. 아래 값은 상황별 모델에서 학습된 고정 가중치를 뺀 평균입니다. 가상 조건 안의 결과입니다.</p><div id="contextualComparison"></div></section>')
            contextual_script = """const cc=data.contextualComparison;document.getElementById('contextualComparison').innerHTML='<p>고객 만족도 '+cc.satisfaction.mean_delta.toFixed(4)+' · 가게 만족도 '+cc.store_satisfaction.mean_delta.toFixed(4)+' · 보조 종합점수 '+cc.score.mean_delta.toFixed(4)+'</p>';"""
            html = html.replace("const hours=document.getElementById('hours');",
                                contextual_script + "const hours=document.getElementById('hours');")
        table_script = """const front=data.aiInfo.pareto_frontier;document.getElementById('paretoTable').innerHTML='<div style="overflow-x:auto"><table><thead><tr><th>후보</th><th>고객 만족</th><th>가게 만족</th><th>취향</th><th>재고</th><th>이익</th><th>대기</th></tr></thead><tbody>'+front.map(p=>'<tr><td>'+p.trial+(p.trial===data.aiInfo.selected_trial?' (선택)':'')+'</td><td>'+p.customer.toFixed(4)+'</td><td>'+p.store.toFixed(4)+'</td>'+p.weights.map(w=>'<td>'+w.toFixed(3)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';"""
        html = html.replace("const hours=document.getElementById('hours');",
                            table_script + "const hours=document.getElementById('hours');")
    Path(path).write_text(html.replace("PAYLOAD", payload), encoding="utf-8")
