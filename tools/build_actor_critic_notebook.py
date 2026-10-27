"""New experiment entrypoint; never repurpose the old 1000-day checkpoint."""
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def build(revision='main'):
    for name in ('pilot','research','continual'):
        config=yaml.safe_load((ROOT/'config'/('screening_'+name+'.yaml')).read_text(encoding='utf-8'))
        config.update(algorithm='actor_critic',auto_cleanup=False,workers=1)
        if name=='pilot':
            config.update(visitors=5,scenarios={'balanced':{'customer':[.45,.25,.30],'store':[.75,.25]}})
        (ROOT/'config'/('actor_critic_'+name+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    notebook=json.loads((ROOT/'notebooks/run_automatic.ipynb').read_text(encoding='utf-8'))
    notebook['metadata']['colab']['name']='Kiosk_Actor_Critic.ipynb'
    notebook['cells'][0]['source']=['# 액터–크리틱 신경망 실험\n\n개인별: 고객 결과마다 TD 학습. 일별: 하루 결과로 1회 배치 학습. 기존 1000일 무작위 탐색 실험은 그대로 보존합니다. 이 노트북은 별도 실험을 생성합니다.\n']
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']).replace('REVISION = "main"',f'REVISION = "{revision}"')
            source=source.replace('"screening_"+PRESET','"actor_critic_"+PRESET')
            source=source.replace('TARGET_DAY = 0','TARGET_DAY = 1000')
            if 'run = run_session' in source:
                source=source.replace('run = run_session', 'assert config.algorithm == "actor_critic"\nrun = run_session')
                source=source.replace('run = run_session', 'if RESUME_ID.strip():\n    from kiosk_research.persistence import read_json\n    saved = read_json(DATA_ROOT / "experiments" / RESUME_ID.strip() / "metadata.json")\n    if saved["identity"]["config"].get("algorithm") != "actor_critic":\n        raise ValueError("기존 탐색 실험은 원래 노트북에서 재개하세요.")\nrun = run_session')
            cell['source']=source.splitlines(keepends=True)
            cell['outputs']=[];cell['execution_count']=None
        elif 'pilot`:' in ''.join(cell['source']):
            cell['source']=['## 3. 새 신경망 실험\n\n`pilot`: 8매장·균형 평가기준 1개·5고객·학습2/검증1/평가1일. 기본값으로 먼저 실행해 저장·평가를 확인합니다. `research`: 9개 평가기준을 사용하는 독립 반복 실험. `continual`: 주기 사이에 신경망을 이어 학습하며 기존 100학습일 수렴 조건을 사용합니다. TARGET_DAY=1000은 절대 상한입니다.\n\n기존 무작위 탐색 실험 ID를 RESUME_ID에 넣지 마세요. 재개 시 해당 신경망 실험의 원래 커밋과 ID를 사용하세요. 각 학습일의 개인별 결과는 업데이트 중인 정책의 결과이며, 모델 비교에는 동결된 검증·평가 결과를 사용합니다.\n']
    target=ROOT/'notebooks/run_actor_critic.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),str(target),'exec')
    return target


if __name__=='__main__':build()
