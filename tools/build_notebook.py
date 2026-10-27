"""Build the clean Colab entrypoint. All code cells are ordinary Python."""
import json
from pathlib import Path

BOOTSTRAP = r'''import os, re, sys, subprocess, tempfile
from pathlib import Path
REPOSITORY = "phj9187/koisk-simulation" #@param {type:"string"}
REVISION = "main" #@param {type:"string"}
if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", REPOSITORY):
    raise ValueError("저장소는 owner/name 형식이어야 합니다.")
if not re.fullmatch(r"[A-Za-z0-9_./-]+", REVISION) or REVISION.startswith("-"):
    raise ValueError("브랜치 또는 커밋 해시를 입력하세요.")
if any(n.startswith(("kiosk_research", "kiosk_sim", "kiosk_cloud")) for n in sys.modules):
    raise RuntimeError("이전 코드가 로드되어 있습니다. 런타임을 다시 시작한 뒤 실행하세요.")
token = None
try:
    from google.colab import userdata
    token = userdata.get("KIOSK_GITHUB_TOKEN")
except Exception:
    pass
CODE_ROOT = Path(tempfile.mkdtemp(prefix="kiosk-research-", dir="/content"))
with tempfile.TemporaryDirectory() as auth_dir:
    helper = Path(auth_dir) / "askpass.py"
    helper.write_text("#!/usr/bin/env python3\nimport os,sys\nprint('x-access-token' if 'username' in sys.argv[1].lower() else os.environ.get('KIOSK_EPHEMERAL_TOKEN',''))\n")
    helper.chmod(0o700)
    child_env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS=str(helper))
    if token:
        child_env["KIOSK_EPHEMERAL_TOKEN"] = token
    try:
        commands = [
            ["git", "init", str(CODE_ROOT)],
            ["git", "-C", str(CODE_ROOT), "remote", "add", "origin", "https://github.com/"+REPOSITORY+".git"],
            ["git", "-C", str(CODE_ROOT), "fetch", "--depth", "1", "origin", REVISION],
            ["git", "-C", str(CODE_ROOT), "checkout", "--detach", "FETCH_HEAD"],
        ]
        for command in commands:
            result = subprocess.run(command, env=child_env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("GitHub 읽기 실패. 저장소·버전·비공개 저장소 권한을 확인하세요. 토큰은 출력하지 않습니다.")
    finally:
        child_env.pop("KIOSK_EPHEMERAL_TOKEN", None)
        token = None
COMMIT = subprocess.check_output(["git","-C",str(CODE_ROOT),"rev-parse","HEAD"],text=True).strip()
subprocess.check_call([sys.executable,"-m","pip","install","-q","-r",str(CODE_ROOT/"requirements.txt")])
sys.path.insert(0,str(CODE_ROOT/"src"))
print("코드 경로:", CODE_ROOT)
print("실행 커밋:", COMMIT)
'''

CELLS = [
('markdown', '# λ 조건부 키오스크 파레토 연구 — 새 실행 노트북\n\n고정·상황별 **추천 가중치 함수**를 같은 학습 예산으로 비교합니다. 만족도 구성 가중치와 λ는 평가 조건이며 정책이 변경하지 못합니다. 기존 LinUCB/톰프슨 실험을 재개하는 파일이 아닙니다.\n\nCPU부터 사용하세요. Pro+도 실행시간·자원을 무제한 보장하지 않습니다.'),
('markdown', '## 1. 특정 GitHub 버전 불러오기\n\n비공개 저장소라면 왼쪽 열쇠 모양 Secrets에 `KIOSK_GITHUB_TOKEN`을 등록하고 노트북 접근을 허용하세요. 코드 읽기에는 해당 저장소 Contents 읽기, 결과 업로드에는 쓰기 권한이 필요합니다. 토큰을 코드나 채팅에 넣지 마세요.\n\n재개 시에는 metadata.json의 정확한 커밋을 REVISION에 입력하세요.'),
('code', BOOTSTRAP),
('markdown', '## 2. Google Drive 인증·연결\n\n아래 인증은 사용자 계정에서 승인합니다. Drive에는 원자료·모델·체크포인트·상세 결과를 저장합니다.'),
('code', 'from google.colab import drive\ndrive.mount("/content/drive")\nfrom kiosk_research.persistence import layout\nfrom kiosk_cloud.provenance import runtime_info\nDATA_ROOT = layout("/content/drive/MyDrive/AI-Kiosk-Data")\nprint("저장 루트:", DATA_ROOT)\nprint(runtime_info())\n'),
('markdown', '## 3. 실험 조건·실행·안전 중지·재개\n\n각 반복은 새 모델로 시작해 학습 → 동결 검증 → 동결 평가를 수행합니다. 반복 수 0은 중지 요청/런타임 종료까지 계속합니다. 기본값은 소규모 가상 검증용입니다. 실측 연구는 KREI 모드·64개 매장·원자료 경로를 선택하세요.\n\n버튼은 백그라운드 작업자를 사용하므로 실행 셀이 반환된 뒤에도 중지 요청이 가능합니다. 버튼이 응답하지 않으면 Drive의 `experiments/실험ID/STOP` 파일을 만드세요. `RUNNING.lock`은 이전 실행이 끝났음을 확인한 뒤에만 삭제하세요.\n\n원자료는 검증된 SHA256과 일치해야 합니다. 기존 `Kiosk_LHS/source_2025.xlsx` 경로를 직접 지정할 수 있습니다. 다른 자료나 가상 자료로 자동 대체하지 않습니다.'),
('code', 'from kiosk_research.ui import controls\npanel = controls(DATA_ROOT, REPOSITORY)\n'),
('markdown', '## 4. 중지 완료 후 결과 확인 (선택)\n\n결과 보기 버튼도 같은 기능입니다. 완전한 최종 평가 반복만 파레토·통계 비교에 포함됩니다. 5회 미만 반복은 기술통계만 제공합니다.'),
('code', 'run = panel["experiment"]\nif run is not None:\n    if panel["future"] is not None and not panel["future"].done():\n        print("먼저 안전 중지를 요청하고 완료를 기다려주세요.")\n    else:\n        from kiosk_research.analysis import export\n        summary = export(run)\n        print("실험 ID:", run.id)\n        print("완료 날짜:", run.state["day"])\n        print("요약:", summary)\n'),
('markdown', '## 5. GitHub 요약 기록\n\n업로드 버튼은 `results/실험ID-day-날짜` 새 브랜치만 만들며 기본 브랜치를 직접 수정하지 않습니다. 토큰이 없으면 건너뛰고 Drive에 요약을 보존합니다. 대용량 로그·모델·체크포인트는 GitHub에 올리지 않습니다.\n\n여러 결과 브랜치를 기본 브랜치에 병합할 때 registry CSV 충돌을 정리해야 할 수 있습니다. 자세한 가정과 한계는 `docs/research_methodology_ko.md`, 실행 방법은 `docs/colab_research_ko.md`를 참고하세요.'),
]

def build():
    cells=[]
    for kind,text in CELLS:
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    notebook={'cells':cells,'metadata':{'colab':{'name':'Kiosk_Lambda_Pareto_Research.ipynb'},'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
    path=Path(__file__).resolve().parents[1]/'notebooks'/'run_simulation.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')

if __name__=='__main__':build()


def build_automatic():
    """A Run-all entrypoint: mount -> pilot/research or resume -> durable outputs."""
    cells = [
        ('markdown', '# 키오스크 연구 자동 실행\n\nGitHub 코드 → Drive 연결 → 9개 평가 기준 실험 → 체크포인트 저장·중복 정리 → 결과 저장 순서로 실행합니다. 첫 실행은 pilot으로 검증하세요.\n\nColab 런타임 종료 후에는 계산도 중단됩니다. 이 노트북은 백그라운드 무제한 실행이나 연결 유지 우회를 하지 않습니다. 다시 연결한 뒤 같은 커밋과 실험 ID로 재개하세요.'),
        ('markdown', '## 1. GitHub 버전\n\n공개 저장소 읽기에는 토큰이 필요 없습니다. 비공개 저장소만 Colab Secrets에 `KIOSK_GITHUB_TOKEN`을 등록하세요. 재개 시 `REVISION`에 progress.json의 git_commit을 입력하세요.'),
        ('code', BOOTSTRAP),
        ('markdown', '## 2. Drive 연결\n\n처음 인증 요청은 직접 승인하세요. 공유 PC에서는 본인 계정인지 확인하세요.'),
        ('code', 'from google.colab import drive\ndrive.mount("/content/drive")\nfrom pathlib import Path\nDATA_ROOT = Path("/content/drive/MyDrive/AI-Kiosk-Data")\nDATA_ROOT.mkdir(parents=True, exist_ok=True)\n'),
        ('markdown', '## 3. 자동 실행 또는 재개\n\n`pilot`: 8매장·9기준·15방문자·학습2/검증1/평가1일·1반복. `research`: 8매장·9기준·30방문자·20/5/5일·5반복. 모두 가상 자료입니다. 실측 자료는 별도 검증된 설정이 필요합니다.\n\n`continual`: 같은 모델을 20일 학습·5일 검증·5일 평가 주기 사이에 이어갑니다. 중지 또는 런타임 종료까지 계속하며 주기마다 요약을 저장합니다. 자동 삭제는 꺼져 있습니다. 새 누적 실행은 최소 100학습 날짜 구간의 가중치·검증 점수 안정성으로 자동 중지합니다. TARGET_DAY는 재개해도 넘지 않는 절대 일수 상한입니다. 자세한 기준은 docs/stopping_and_linecharts_ko.md를 참고하세요.\n\n`RESUME_ID`를 지정하면 원래 설정을 사용합니다. 빈 ID로 재실행하면 새 실험이 생깁니다. 중지는 별도 셀에서 STOP 파일을 만들거나 실행 중단을 누를 수 있습니다. STOP은 오늘 완료 후 안전 중지, 강제 중단은 마지막 저장된 매장 부분부터 복구합니다.\n\nRUNNING.lock이 남은 경우 이전 런타임이 종료됐음을 확인한 뒤 해당 실험의 잠금 파일만 수동 삭제하세요. 활성 런타임의 잠금을 자동으로 지우지 않습니다.'),
        ('code', 'PRESET = "pilot" #@param ["pilot", "research", "continual"]\nRESUME_ID = "" #@param {type:"string"}\nDAY_BUDGET = 0 #@param {type:"integer"}\nTARGET_DAY = 0 #@param {type:"integer"}\n# TARGET_DAY: 절대 가상 날짜 상한, 0은 별도 상한 없음.\n# 0: 설정된 전체 일정. 양수: 지정 일수 후 안전 중지.\nfrom kiosk_research.config import Config\nfrom kiosk_research.session import run_session\nconfig = Config.load(CODE_ROOT / "config" / ("screening_"+PRESET+".yaml"))\nrun = run_session(DATA_ROOT, config=config, resume_id=RESUME_ID.strip() or None, day_budget=DAY_BUDGET or None, target_day=TARGET_DAY or None)\n'),
        ('markdown', '## 4. 진행 상황·저장 결과\n\n이 셀은 실행 완료 후 자동으로 표시됩니다. 실행 중에는 Drive의 latest_run.json 또는 experiments/실험ID/progress.json에서 마지막 갱신 시각·완료 날짜·체크포인트·완료 매장 수를 확인하세요. 파일의 running 표시가 오래됐다면 현재 런타임이 살아 있다는 보장은 아닙니다.'),
        ('code', 'import json\nfrom kiosk_research.session import status\nprint(json.dumps(status(DATA_ROOT), ensure_ascii=False, indent=2))\nfrom kiosk_research.persistence import read_json\nprint(read_json(run.directory / "session_result.json"))\n'),
        ('markdown', '## 5. 선택: GitHub 요약 업로드\n\n아래 설정을 true로 바꾼 경우에만 Colab Secrets의 쓰기 권한 토큰을 사용합니다. 원시 자료·체크포인트는 Drive에 남고 작은 요약만 새로운 results 브랜치에 올라갑니다. 이미 같은 요약을 올렸으면 저장된 업로드 기록을 보여줍니다.'),
        ('code', 'UPLOAD_SUMMARY = False #@param {type:"boolean"}\nif UPLOAD_SUMMARY:\n    from google.colab import userdata\n    from kiosk_research.github import upload_summary\n    summary = Path(read_json(run.directory / "session_result.json")["summary"])\n    if (summary / "github_upload.json").exists():\n        print(read_json(summary / "github_upload.json"))\n    else:\n        print(upload_summary(summary, REPOSITORY, userdata.get("KIOSK_GITHUB_TOKEN")))\n'),
    ]
    result=[]
    for kind,text in cells:
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        result.append(cell)
    notebook={'cells':result,'metadata':{'colab':{'name':'Kiosk_Automatic_Research.ipynb'},'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=Path(__file__).resolve().parents[1]/'notebooks'/'run_automatic.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')

if __name__=='__main__':build_automatic()
