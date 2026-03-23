"""Orchestrator: run the full spatial transferability experiment.

Phase 1: Data selection (already done via data_select.py)
Phase 2: Data download
Phase 3: Training — 8 jobs in parallel across 8 GPUs
Phase 4: Evaluation — 16 jobs in parallel across 8 GPUs
Phase 5: Analysis
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from config import CONFIG

SCRIPT_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_cmd(cmd, env=None, desc=""):
    """Run a command, print output, return success."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return False
    return True


def run_parallel(jobs, desc=""):
    """Run jobs in parallel. Each job is (cmd, env, desc)."""
    print(f"\n{'#'*60}")
    print(f"  PARALLEL: {desc} ({len(jobs)} jobs)")
    print(f"{'#'*60}")

    procs = []
    log_files = []
    for cmd, env, job_desc in jobs:
        log_path = SCRIPT_DIR / "outputs" / f"{job_desc.replace(' ', '_')}.log"
        log_f = open(log_path, "w")
        log_files.append((log_f, log_path, job_desc))
        print(f"  Launching: {job_desc} → {log_path}")
        p = subprocess.Popen(cmd, env=env, cwd=str(SCRIPT_DIR), stdout=log_f, stderr=subprocess.STDOUT)
        procs.append(p)

    # Wait for all
    for p, (log_f, log_path, job_desc) in zip(procs, log_files):
        p.wait()
        log_f.close()
        status = "OK" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
        print(f"  {job_desc}: {status}")
        if p.returncode != 0:
            # Print last 20 lines of log
            with open(log_path) as f:
                lines = f.readlines()
                print("  --- Last 20 lines ---")
                for line in lines[-20:]:
                    print(f"  {line.rstrip()}")

    failed = sum(1 for p in procs if p.returncode != 0)
    if failed:
        print(f"\n  WARNING: {failed}/{len(procs)} jobs failed")
    return failed == 0


def main():
    regions = list(CONFIG["regions"].keys())
    models = CONFIG["models"]
    base_env = os.environ.copy()

    # Phase 1: Data selection
    splits_exist = all(
        (CONFIG["splits_dir"] / f"{r}_split.json").exists()
        for r in regions
    )
    if not splits_exist:
        run_cmd([PYTHON, "data_select.py"], desc="Phase 1: Data selection")
    else:
        print("\nPhase 1: Splits already exist, skipping data_select.py")

    # Phase 2: Data download
    download_jobs = []
    for region in regions:
        env = base_env.copy()
        cmd = [PYTHON, "data_download.py", "--region", region]
        download_jobs.append((cmd, env, f"download_{region}"))

    run_parallel(download_jobs, desc="Phase 2: Data download")

    # Phase 3: Training
    train_jobs = []
    gpu_idx = 0
    for region in regions:
        for model_name in models:
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            cmd = [PYTHON, "train.py", "--model", model_name, "--region", region]
            train_jobs.append((cmd, env, f"train_{region}_{model_name}"))
            gpu_idx += 1

    run_parallel(train_jobs, desc="Phase 3: Training (8 GPUs)")

    # Phase 4: Evaluation
    eval_jobs = []
    gpu_idx = 0
    for train_region, test_region in CONFIG["experiment_matrix"]:
        for model_name in models:
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx % 8)
            cmd = [
                PYTHON, "evaluate.py",
                "--model", model_name,
                "--train-region", train_region,
                "--test-region", test_region,
            ]
            eval_jobs.append((cmd, env, f"eval_{train_region}_on_{test_region}_{model_name}"))
            gpu_idx += 1

    run_parallel(eval_jobs, desc="Phase 4: Evaluation (16 conditions)")

    # Phase 5: Analysis
    run_cmd([PYTHON, "analyze.py"], desc="Phase 5: Analysis")

    print("\n" + "="*60)
    print("  EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
