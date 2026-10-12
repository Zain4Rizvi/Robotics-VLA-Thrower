"""Run step 0, then one train, then the remaining rollouts.

Does not load xy into the state. Does not write outside connector.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
SRC = ROOT / "camera_text" / "checkpoints" / "checkpoints" / "008000" / "pretrained_model"
INIT = HERE / "init"
OUT = HERE / "checkpoints"
TRAIN = ROOT / "color" / "swap" / "datasets" / "train"
VAL = ROOT / "color" / "swap" / "datasets" / "val"
STEP0 = OUT / "checkpoints" / "000000" / "pretrained_model"


def gpu_python_pids() -> list[str]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    )
    rows = []
    for line in out.splitlines():
        if "python" in line.lower():
            rows.append(line.strip())
    return rows


def free_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def log_rows() -> int:
    path = OUT / "train_log.csv"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)


def copy_init() -> None:
    if (INIT / "model.safetensors").exists():
        print(f"init already at {INIT}", flush=True)
        return
    if not (SRC / "model.safetensors").exists():
        raise SystemExit(f"missing camera_text step-8000 weights {SRC}")
    if INIT.exists():
        raise SystemExit(f"incomplete init at {INIT}")
    print(f"copy {SRC} -> {INIT}", flush=True)
    shutil.copytree(SRC, INIT)


def copy_step0() -> None:
    if (STEP0 / "model.safetensors").exists():
        return
    if not (INIT / "model.safetensors").exists():
        raise SystemExit("init copy is missing")
    if STEP0.exists():
        shutil.rmtree(STEP0)
    STEP0.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(INIT, STEP0)
    print(f"copied init to {STEP0}", flush=True)


def run_py(args: list[str], log_name: str) -> int:
    stdout_path = HERE / f"{log_name}_stdout.txt"
    stderr_path = HERE / f"{log_name}_stderr.txt"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        proc = subprocess.Popen([str(PY), "-u", *args], cwd=str(REPO), stdout=stdout, stderr=stderr)
        code = proc.wait()
    print(f"{log_name} exit {code}", flush=True)
    return code


def start_train(batch: int) -> subprocess.Popen:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists")
    stdout_path = HERE / "stdout.txt"
    stderr_path = HERE / "stderr.txt"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(PY), "-u", str(HERE / "train.py"),
            "--dataset", str(TRAIN),
            "--val-dataset", str(VAL),
            "--output-dir", str(OUT),
            "--from-checkpoint", str(INIT),
            "--steps", "2000",
            "--save-freq", "500",
            "--val-every", "500",
            "--batch-size", str(batch),
            "--cut-minutes", "45",
            "--cut-to", "1000",
            "--measure-at", "500",
        ],
        cwd=str(REPO),
        stdout=stdout,
        stderr=stderr,
    )
    proc.stdout_file = stdout
    proc.stderr_file = stderr
    print(f"train pid {proc.pid} batch {batch}", flush=True)
    return proc


def close_train_logs(proc: subprocess.Popen) -> None:
    for handle in (getattr(proc, "stdout_file", None), getattr(proc, "stderr_file", None)):
        if handle is not None:
            handle.close()


def train() -> None:
    busy = gpu_python_pids()
    if busy:
        raise SystemExit(f"python already on the gpu: {busy}")
    batch = 8
    bumped = False
    ooms = 0
    while True:
        proc = start_train(batch)
        committed = False
        killed = False
        while proc.poll() is None:
            rows = log_rows()
            if not committed and not bumped and batch == 8 and rows >= 2:
                free = free_mb()
                print(f"free VRAM {free} MB after {rows} steps", flush=True)
                if free >= 1024:
                    bumped = True
                    killed = True
                    print("at least 1 GB free; stopping to try batch 16", flush=True)
                    kill_tree(proc.pid)
                    proc.wait(timeout=120)
                    break
                committed = True
                copy_step0()
            elif not committed and rows >= 2:
                committed = True
                copy_step0()
            time.sleep(15)
        close_train_logs(proc)
        if killed:
            shutil.rmtree(OUT, ignore_errors=True)
            batch = 16
            continue
        code = proc.returncode
        print(f"train exit {code} batch {batch}", flush=True)
        if code == 0:
            if not (STEP0 / "model.safetensors").exists():
                copy_step0()
            return
        if code == 2:
            ooms += 1
            shutil.rmtree(OUT, ignore_errors=True)
            if ooms >= 2:
                raise SystemExit("second CUDA OOM; stopping")
            batch = max(1, batch // 2)
            print(f"OOM; retry batch {batch}", flush=True)
            continue
        raise SystemExit(f"train failed with exit {code}")


def eval_args(require: str, skip_report: bool) -> list[str]:
    args = [
        str(HERE / "eval.py"),
        "--out", str(HERE),
        "--init", str(INIT),
        "--ckpt-root", str(OUT),
        "--seeds-file", str(TRAIN / "stack_seeds.json"),
        "--seeds-file", str(VAL / "stack_seeds.json"),
        "--require", require,
    ]
    if skip_report:
        args.append("--skip-report")
    return args


def main() -> None:
    if not (HERE / "PLAN.md").exists():
        raise SystemExit("PLAN.md is missing; refusing to start")
    if gpu_python_pids():
        raise SystemExit(f"python already on the gpu: {gpu_python_pids()}")
    code = run_py([str(HERE / "train.py"), "--self-check-only"], "self_check_train")
    if code != 0:
        raise SystemExit(code)
    code = run_py([str(HERE / "eval.py"), "--self-check-only"], "self_check_eval")
    if code != 0:
        raise SystemExit(code)
    copy_init()
    if gpu_python_pids():
        raise SystemExit(f"python already on the gpu: {gpu_python_pids()}")
    code = run_py(eval_args("0", True), "step0")
    if code != 0:
        raise SystemExit(code)
    train()
    budget = json.loads((HERE / "budget.json").read_text(encoding="utf-8"))
    steps = int(budget["steps"])
    require = ",".join(str(s) for s in range(0, steps + 1, 500))
    code = run_py(eval_args(require, False), "eval")
    if code != 0:
        raise SystemExit(code)
    print("connector finished", flush=True)


if __name__ == "__main__":
    main()
