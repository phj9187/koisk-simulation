"""Complete every pending job using the unchanged, fingerprinted research engine.

No wall-clock budget is applied. Colab/runtime termination remains external.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from kiosk_foodcourt.cli import atomic, run_job, signature, summarize
from kiosk_foodcourt.models import Settings


def run_until_complete(settings, root):
    settings.validate()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"signature": signature(settings), "settings": settings.to_dict()}
    manifest = root / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError("Research configuration/source differs from the saved experiment.")
    else:
        atomic(manifest, expected)
    tasks = [(r, m, a, v) for r in range(settings.repetitions)
             for m in settings.menu_counts for a in settings.attention_modes
             for v in settings.visitors]
    def result_path(task):
        r, m, a, v = task
        return root / "jobs" / f"r{r}_m{m}_{a}_v{v}" / "result.json"
    pending = [task for task in tasks if not result_path(task).exists()]
    completed = len(tasks) - len(pending)
    print(json.dumps({"mode": "until_complete", "time_limit": None,
                      "completed_jobs": completed, "pending_jobs": len(pending),
                      "total_jobs": len(tasks)}, ensure_ascii=False), flush=True)
    # The existing engine already accepts an infinite deadline. Its source and
    # scientific signature remain byte-for-byte unchanged, preserving resumes.
    payloads = [(settings.to_dict(), str(root), task, float("inf")) for task in pending]
    if settings.workers == 1:
        for payload in payloads:
            run_job(payload)
            completed += 1
            print(f"Completed jobs: {completed}/{len(tasks)}", flush=True)
    elif payloads:
        with ProcessPoolExecutor(max_workers=min(settings.workers, os.cpu_count() or 1)) as pool:
            futures = [pool.submit(run_job, payload) for payload in payloads]
            for future in as_completed(futures):
                future.result()
                completed += 1
                print(f"Completed jobs: {completed}/{len(tasks)}", flush=True)
    summarize(root, settings)
    print("All jobs and final/stress evaluations completed.", flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run_until_complete(Settings.load(args.config), args.root)


if __name__ == "__main__":
    main()
