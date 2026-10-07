"""Collect, probe, train 4000 steps, roll out, then the unseen-word rollout.

Stops if the expert gate or the probe gate fails. One retry at batch 4 on
an out-of-memory train. Does not use this morning's clock.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
INIT = ROOT / "sentence" / "cont" / "run1" / "checkpoints" / "checkpoints" / "008000" / "pretrained_model"
TRAIN = HERE / "datasets" / "train"
VAL = HERE / "datasets" / "val"


def env() -> dict:
    out = os.environ.copy()
    if Path("Z:/").exists():
        out["HF_HOME"] = r"Z:\hf_cache"
        out["UV_CACHE_DIR"] = r"Z:\uv_cache"
        out["TEMP"] = r"Z:\tmp"
        out["TMP"] = r"Z:\tmp"
        out["TORCH_HOME"] = r"Z:\hf_cache\torch"
        out["XDG_CACHE_HOME"] = r"Z:\hf_cache\xdg"
    else:
        for folder in (r"D:\tmp", r"D:\uv_cache", r"D:\torch", r"D:\xdg"):
            Path(folder).mkdir(parents=True, exist_ok=True)
        out["HF_HOME"] = r"C:\Users\Zain\.cache\huggingface"
        out["HUGGINGFACE_HUB_CACHE"] = r"C:\Users\Zain\.cache\huggingface\hub"
        out["HF_HUB_OFFLINE"] = "1"
        out["TRANSFORMERS_OFFLINE"] = "1"
        out["UV_CACHE_DIR"] = r"D:\uv_cache"
        out["TEMP"] = r"D:\tmp"
        out["TMP"] = r"D:\tmp"
        out["TORCH_HOME"] = r"D:\torch"
        out["XDG_CACHE_HOME"] = r"D:\xdg"
    out["MUJOCO_GL"] = "glfw"
    return out


def run(args: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(args), flush=True)
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.run([str(PY), "-u", *args], cwd=str(REPO), env=env(), stdout=out, stderr=subprocess.STDOUT)
    print(f"exit {proc.returncode} log {log}", flush=True)
    return proc.returncode


def main() -> None:
    if not (INIT / "model.safetensors").exists():
        raise SystemExit(f"missing init {INIT}")
    if not (TRAIN / "stack_seeds.json").exists():
        stale = HERE / "datasets" / "REPORT.md"
        if stale.exists():
            stale.unlink()
        code = run(
            [str(HERE / "collect.py"), "--train-out", str(TRAIN), "--val-out", str(VAL)],
            HERE / "datasets" / "collect_log.txt",
        )
        if code != 0 or not (TRAIN / "stack_seeds.json").exists():
            raise SystemExit(f"collect failed ({code})")
    if not (HERE / "probe" / "REPORT.md").exists():
        code = run(
            [
                str(HERE / "probe.py"),
                "--train", str(TRAIN),
                "--val", str(VAL),
                "--checkpoint", str(INIT),
                "--out", str(HERE / "probe"),
            ],
            HERE / "probe" / "stdout.txt",
        )
        if code != 0 or not (HERE / "probe" / "probe.json").exists():
            raise SystemExit(f"probe failed ({code})")
    gate = json.loads((HERE / "probe" / "probe.json").read_text(encoding="utf-8")).get("gate")
    if gate != "pass":
        print(f"probe gate {gate}; not training", flush=True)
        return
    ckpt = HERE / "train" / "checkpoints"
    if not (ckpt / "train_log.csv").exists():
        for batch in (8, 4):
            if ckpt.exists():
                shutil.rmtree(ckpt)
            code = run(
                [
                    str(HERE / "train.py"),
                    "--dataset", str(TRAIN),
                    "--val-dataset", str(VAL),
                    "--output-dir", str(ckpt),
                    "--from-checkpoint", str(INIT),
                    "--steps", "4000",
                    "--init-step", "8000",
                    "--batch-size", str(batch),
                ],
                HERE / "train" / "stdout.txt",
            )
            if code == 0:
                break
            if code != 2 or batch == 4:
                raise SystemExit(f"train failed ({code})")
            print("oom; retry batch 4", flush=True)
    if not (HERE / "train" / "eval_summary.json").exists():
        code = run(
            [
                str(HERE / "eval.py"),
                "--out", str(HERE / "train"),
                "--init", str(INIT),
                "--ckpt-root", str(ckpt),
                "--seeds-file", str(TRAIN / "stack_seeds.json"),
                "--seeds-file", str(VAL / "stack_seeds.json"),
                "--require", "0,1000,2000,3000,4000",
            ],
            HERE / "train" / "eval_stdout.txt",
        )
        if code != 0:
            raise SystemExit(f"eval failed ({code})")
    if not (HERE / "unseen_words" / "REPORT.md").exists():
        saved = sorted(p for p in (ckpt / "checkpoints").glob("[0-9]*") if (p / "pretrained_model" / "model.safetensors").exists())
        if not saved:
            raise SystemExit("no color checkpoint for the unseen rollout")
        code = run(
            [
                str(HERE / "unseen.py"),
                "--out", str(HERE / "unseen_words"),
                "--checkpoint", str(saved[-1] / "pretrained_model"),
                "--seeds-file", str(TRAIN / "stack_seeds.json"),
                "--seeds-file", str(VAL / "stack_seeds.json"),
            ],
            HERE / "unseen_words" / "stdout.txt",
        )
        if code != 0:
            raise SystemExit(f"unseen failed ({code})")
    print("colors finished", flush=True)


if __name__ == "__main__":
    main()
