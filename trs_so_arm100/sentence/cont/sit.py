"""Run the continuation sitting. One GPU job. Reports win over the next train."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(HERE))

from gate import (  # noqa: E402
    CAP_S,
    COLOR_START_S,
    RUN2_NEED_S,
    branch_text,
    color_request,
    decide,
    fit_color_steps,
    fits,
)

PY = REPO / ".venv" / "Scripts" / "python.exe"
INIT = ROOT / "sentence" / "use_note" / "checkpoints" / "checkpoints" / "002000" / "pretrained_model"
SWAP_TRAIN = ROOT / "color" / "swap" / "datasets" / "train"
SWAP_VAL = ROOT / "color" / "swap" / "datasets" / "val"
RUN1 = HERE / "run1"
RUN2 = HERE / "run2"
COLORS = HERE / "colors"
CLOCK = HERE / "clock.json"
LOG = HERE / "sit_log.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def apply_env() -> None:
    import os

    z = Path("Z:/")
    log(f"Test-Path Z:\\ -> {z.exists()}")
    if z.exists():
        os.environ["HF_HOME"] = r"Z:\hf_cache"
        os.environ["UV_CACHE_DIR"] = r"Z:\uv_cache"
        os.environ["TEMP"] = r"Z:\tmp"
        os.environ["TMP"] = r"Z:\tmp"
        os.environ["TORCH_HOME"] = r"Z:\hf_cache\torch"
        os.environ["XDG_CACHE_HOME"] = r"Z:\hf_cache\xdg"
    else:
        for folder in (r"D:\tmp", r"D:\uv_cache", r"D:\torch", r"D:\xdg"):
            Path(folder).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = r"C:\Users\Zain\.cache\huggingface"
        os.environ["HUGGINGFACE_HUB_CACHE"] = r"C:\Users\Zain\.cache\huggingface\hub"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["UV_CACHE_DIR"] = r"D:\uv_cache"
        os.environ["TEMP"] = r"D:\tmp"
        os.environ["TMP"] = r"D:\tmp"
        os.environ["TORCH_HOME"] = r"D:\torch"
        os.environ["XDG_CACHE_HOME"] = r"D:\xdg"
    os.environ["MUJOCO_GL"] = "glfw"
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


def prelude_text() -> str:
    if Path("Z:/").exists():
        return "\n".join(
            [
                '$env:HF_HOME="Z:\\hf_cache"',
                '$env:UV_CACHE_DIR="Z:\\uv_cache"',
                '$env:TEMP="Z:\\tmp"',
                '$env:TMP="Z:\\tmp"',
                '$env:TORCH_HOME="Z:\\hf_cache\\torch"',
                '$env:XDG_CACHE_HOME="Z:\\hf_cache\\xdg"',
                '$env:MUJOCO_GL="glfw"',
                "",
            ]
        )
    return "\n".join(
        [
            '$env:HF_HOME="C:\\Users\\Zain\\.cache\\huggingface"',
            '$env:HUGGINGFACE_HUB_CACHE="C:\\Users\\Zain\\.cache\\huggingface\\hub"',
            '$env:HF_HUB_OFFLINE="1"',
            '$env:TRANSFORMERS_OFFLINE="1"',
            '$env:UV_CACHE_DIR="D:\\uv_cache"',
            '$env:TEMP="D:\\tmp"',
            '$env:TMP="D:\\tmp"',
            '$env:TORCH_HOME="D:\\torch"',
            '$env:XDG_CACHE_HOME="D:\\xdg"',
            '$env:MUJOCO_GL="glfw"',
            "",
        ]
    )


def load_clock() -> dict:
    if CLOCK.exists():
        return json.loads(CLOCK.read_text(encoding="utf-8"))
    return {}


def save_clock(blob: dict) -> None:
    CLOCK.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def remaining(clock: dict) -> float:
    return CAP_S - (time.time() - float(clock["t0"]))


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and str(pid) in proc.stdout


def kill_tree(pid: int) -> None:
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$kids = Get-CimInstance Win32_Process | Where-Object {{ $_.ParentProcessId -eq {pid} }}; "
            f"foreach ($k in $kids) {{ Stop-Process -Id $k.ProcessId -Force -ErrorAction SilentlyContinue }}; "
            f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue",
        ],
        check=False,
    )


def last_step(run_dir: Path) -> int:
    path = run_dir / "checkpoints" / "train_log.csv"
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return 0
    return int(lines[-1].split(",")[0])


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def tail(path: Path, n: int = 60) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def start_hidden(args: list[str], stdout: Path, stderr: Path, pid_path: Path) -> subprocess.Popen:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    if pid_path.exists():
        pid_path.unlink()
    quoted = ", ".join("'" + str(a).replace("'", "''") + "'" for a in args)
    ps1 = stdout.parent / "_launch.ps1"
    ps1.write_text(
        prelude_text()
        + "\n"
        + f"$p = Start-Process -FilePath '{PY}' -ArgumentList @({quoted}) -WorkingDirectory '{REPO}' "
        + f"-RedirectStandardOutput '{stdout}' -RedirectStandardError '{stderr}' -WindowStyle Hidden -PassThru\n"
        + f"Set-Content -Path '{pid_path}' -Value $p.Id\n"
        + "$null = $p.WaitForExit()\n"
        + "if ($null -eq $p.ExitCode) { exit 1 }\n"
        + "exit $p.ExitCode\n",
        encoding="ascii",
    )
    return subprocess.Popen(["powershell", "-NoProfile", "-File", str(ps1)], cwd=str(REPO))


def wait_job(proc: subprocess.Popen, pid_path: Path, clock: dict, kill_at_cap: bool, cap_flag: Path, watch_csv: Path | None) -> str:
    pid = 0
    last_logged = -1
    while proc.poll() is None:
        if pid_path.exists() and pid == 0:
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
            except ValueError:
                pid = 0
        if kill_at_cap and "t0" in clock and time.time() - float(clock["t0"]) >= CAP_S:
            log("cap reached; stopping the running process")
            cap_flag.write_text("cap\n", encoding="utf-8")
            if pid:
                kill_tree(pid)
            proc.kill()
            return "cap"
        if watch_csv and watch_csv.exists():
            step = last_step(watch_csv.parent.parent) if watch_csv.name == "train_log.csv" else 0
            if watch_csv.name == "train_log.csv":
                lines = watch_csv.read_text(encoding="utf-8").splitlines()
                step = int(lines[-1].split(",")[0]) if len(lines) > 1 else 0
            if step != last_logged:
                rem_h = remaining(clock) / 3600 if "t0" in clock else -1
                log(f"step {step}  remaining {rem_h:.2f} h")
                last_logged = step
        time.sleep(30)
    code = proc.returncode
    if code == 0:
        return "ok"
    return f"exit {code}"


def launch_and_wait(
    args: list[str],
    folder: Path,
    clock: dict,
    kill_at_cap: bool,
    watch: bool,
    logs: tuple[str, str] = ("stdout.txt", "stderr.txt"),
) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    pid_path = folder / ("pid.txt" if logs[0] == "stdout.txt" else "eval_pid.txt")
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = 0
        if alive(pid):
            log(f"waiting for existing pid {pid}")
            while alive(pid):
                if kill_at_cap and time.time() - float(clock["t0"]) >= CAP_S:
                    (folder / "cap.txt").write_text("cap\n", encoding="utf-8")
                    kill_tree(pid)
                    return "cap"
                time.sleep(30)
            return "ok"
    proc = start_hidden(args, folder / logs[0], folder / logs[1], pid_path)
    log(f"started {' '.join(str(a) for a in args[:3])} -> {folder.name} {logs[0]}")
    return wait_job(
        proc,
        pid_path,
        clock,
        kill_at_cap,
        folder / "cap.txt",
        (folder / "checkpoints" / "train_log.csv") if watch else None,
    )


def expected_steps(steps: int, save_freq: int) -> str:
    vals = [0] + list(range(save_freq, steps + 1, save_freq))
    return ",".join(str(v) for v in vals)


def find_ckpt(run_dir: Path, step: int, step0: Path) -> Path:
    if int(step) == 0:
        return step0
    root = run_dir / "checkpoints" / "checkpoints"
    for path in root.glob("[0-9]*"):
        if int(path.name) == int(step) and (path / "pretrained_model" / "model.safetensors").exists():
            return path / "pretrained_model"
    raise SystemExit(f"missing pretrained_model for step {step} under {root}")


def append_gate(report: Path, decision: dict, started: bool, why: str) -> None:
    text = report.read_text(encoding="utf-8") if report.exists() else "# Phase 1\n"
    if "<!-- gate -->" not in text:
        text += "\n## Gate\n\n<!-- gate -->\n\n" + branch_text(decision, started, why) + "\n"
        report.write_text(text, encoding="utf-8")
    (report.parent / "gate.json").write_text(
        json.dumps({**decision, "started": started, "why": why}, indent=2),
        encoding="utf-8",
    )


def train_target(run_dir: Path, default_steps: int) -> int:
    budget = read_json(run_dir / "budget.json")
    if budget.get("steps"):
        return int(budget["steps"])
    meta = read_json(run_dir / "train_meta.json")
    return int(meta.get("requested_steps", default_steps))


def train_done(run_dir: Path, default_steps: int) -> bool:
    return last_step(run_dir) >= train_target(run_dir, default_steps) > 0


def crash_report(folder: Path, title: str) -> None:
    err = tail(folder / "stderr.txt") + "\n" + tail(folder / "oom.txt")
    (folder / "REPORT.md").write_text(
        f"# {title}\n\nThe run stopped before a finished rollout.\n\n```\n{err[-4000:]}\n```\n",
        encoding="utf-8",
    )


def run_train(clock: dict, folder: Path, args: list[str], default_steps: int) -> str:
    """Launch, retry OOM once at batch 4, retry a startup crash once. Return ok, cap, or stop."""
    attempt_key = folder.name + "_attempts"
    clock.setdefault(attempt_key, 0)
    while True:
        if (folder / "pid.txt").exists():
            try:
                pid = int((folder / "pid.txt").read_text(encoding="utf-8").strip())
            except ValueError:
                pid = 0
            if alive(pid):
                log(f"waiting for existing pid {pid}")
                while alive(pid):
                    if time.time() - float(clock["t0"]) >= CAP_S:
                        (folder / "cap.txt").write_text("cap\n", encoding="utf-8")
                        kill_tree(pid)
                        return "cap"
                    time.sleep(30)
        if train_done(folder, default_steps):
            return "ok"
        if (folder / "cap.txt").exists():
            return "cap"
        if remaining(clock) < 15 * 60 and clock[attempt_key] > 0:
            return "stop"
        out = folder / "checkpoints"
        if out.exists() and not train_done(folder, default_steps):
            step = last_step(folder)
            if step >= 100 or clock[attempt_key] >= 2:
                log(f"{folder.name} stopped at step {step}")
                return "partial" if step > 0 else "stop"
            log(f"removing partial {out} before retry")
            shutil.rmtree(out, ignore_errors=True)
        clock[attempt_key] += 1
        save_clock(clock)
        status = launch_and_wait(args, folder, clock, True, True)
        log(f"{folder.name} train status {status} step {last_step(folder)}")
        if status == "ok" or train_done(folder, default_steps):
            return "ok"
        if status == "cap" or (folder / "cap.txt").exists():
            return "cap"
        err = tail(folder / "stderr.txt").lower()
        oom = status.endswith("2") or "out of memory" in err or (folder / "oom.txt").exists()
        if oom and "--batch-size" in args:
            idx = args.index("--batch-size")
            if args[idx + 1] == "8" and clock[attempt_key] < 2:
                args[idx + 1] = "4"
                if out.exists():
                    shutil.rmtree(out, ignore_errors=True)
                log("oom; retry once at batch 4")
                continue
            crash_report(folder, "Out of memory")
            return "stop"
        if clock[attempt_key] < 2 and last_step(folder) < 100 and remaining(clock) > 3600:
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            log("startup crash; retry once")
            continue
        crash_report(folder, "Training stopped")
        return "partial" if last_step(folder) > 0 else "stop"


def run_eval(clock: dict, folder: Path, args: list[str]) -> bool:
    if (folder / "eval_summary.json").exists() and (folder / "success.png").exists() and (folder / "REPORT.md").exists():
        return True
    for attempt in (1, 2):
        status = launch_and_wait(args, folder, clock, False, False, logs=("eval_stdout.txt", "eval_stderr.txt"))
        log(f"eval {folder.name} {status}")
        if (folder / "eval_summary.json").exists() and (folder / "success.png").exists():
            n = 0
            summary = read_json(folder / "eval_summary.json")
            for block in summary.get("sentences", {}).values():
                for row in block.get("results", {}).values():
                    n += int(row.get("n", 10))
            if n and "sec_per_ep" not in clock:
                # measured on the first finished eval; later resumes are not timed here
                pass
            return True
        if attempt == 1:
            log("eval failed; retry once")
            continue
        crash_report(folder, "Eval stopped")
        return False
    return False


def measure_eval(clock: dict, fn) -> None:
    t0 = time.time()
    ok = fn()
    if ok and "sec_per_ep" not in clock:
        # Filled by the caller when it knows the episode count.
        clock["eval_timed_s"] = time.time() - t0
        save_clock(clock)
    return ok


def rate(clock: dict, run_dir: Path) -> float:
    budget = read_json(run_dir / "budget.json")
    if budget.get("minutes_per_1000"):
        clock["min_per_1000"] = float(budget["minutes_per_1000"])
        save_clock(clock)
        return float(budget["minutes_per_1000"])
    return float(clock.get("min_per_1000", 45))


def sec_per(clock: dict) -> float:
    return float(clock.get("sec_per_ep", 60))


def record_eval_time(clock: dict, started: float, episodes: int) -> None:
    if episodes <= 0:
        return
    clock["sec_per_ep"] = (time.time() - started) / episodes
    save_clock(clock)
    log(f"eval {clock['sec_per_ep']:.1f} s/episode")


def phase1_args(folder: Path, init: Path, steps: int, weight: float, init_step: int, batch: int, cut: float) -> list[str]:
    return [
        "-u",
        str(HERE / "train.py"),
        "--dataset",
        str(SWAP_TRAIN),
        "--val-dataset",
        str(SWAP_VAL),
        "--output-dir",
        str(folder / "checkpoints"),
        "--from-checkpoint",
        str(init),
        "--steps",
        str(steps),
        "--save-freq",
        "2000",
        "--val-every",
        "1000",
        "--batch-size",
        str(batch),
        "--green-weight",
        str(weight),
        "--init-step",
        str(init_step),
        "--cut-minutes",
        str(cut),
        "--cut-to",
        "6000",
    ]


def phase1_eval_args(folder: Path, init: Path, require: str) -> list[str]:
    args = [
        "-u",
        str(HERE / "eval.py"),
        "--out",
        str(folder),
        "--init",
        str(init),
        "--ckpt-root",
        str(folder / "checkpoints"),
        "--seeds-file",
        str(SWAP_TRAIN / "stack_seeds.json"),
        "--seeds-file",
        str(SWAP_VAL / "stack_seeds.json"),
    ]
    if require:
        args += ["--require", require]
    return args


def do_phase1_eval(clock: dict, folder: Path, init: Path, default_steps: int, strict: bool) -> bool:
    target = train_target(folder, default_steps) if strict else 0
    require = expected_steps(target, 2000) if strict and target else ""
    started = time.time()
    ok = run_eval(clock, folder, phase1_eval_args(folder, init, require))
    if ok:
        summary = read_json(folder / "eval_summary.json")
        n = 0
        for block in summary.get("sentences", {}).values():
            for row in block.get("results", {}).values():
                n += int(row.get("n", 0))
        record_eval_time(clock, started, n) if (time.time() - started) > 30 else None
    return ok


def maybe_run2(clock: dict, decision: dict) -> str:
    """Return 'ran', 'skipped', or 'stop'."""
    if decision["branch"] not in ("run2_same", "run2_weight"):
        return "skipped"
    if (RUN2 / "eval_summary.json").exists():
        return "ran"
    steps = int(decision["steps"])
    if remaining(clock) < RUN2_NEED_S:
        append_gate(
            RUN1 / "REPORT.md",
            decision,
            False,
            f"Run 2 was not started. {remaining(clock) / 3600:.2f} h remained, under 2.5 h.",
        )
        return "skipped"
    if not fits(steps, 2000, remaining(clock), rate(clock, RUN1), 20, sec_per(clock), 0):
        append_gate(
            RUN1 / "REPORT.md",
            decision,
            False,
            "Run 2 was not started. The prescribed steps and their rollout would pass the 8 h 30 min cap.",
        )
        return "skipped"
    init = find_ckpt(RUN1, int(decision["init_step"]), INIT)
    log(f"run 2 {decision['branch']} steps {steps} weight {decision['green_weight']} init {init}")
    append_gate(
        RUN1 / "REPORT.md",
        decision,
        True,
        f"Run 2 started from step {decision['init_step']} for {steps} steps, green weight {decision['green_weight']}.",
    )
    args = phase1_args(RUN2, init, steps, float(decision["green_weight"]), int(decision["init_step"]), 8, 0)
    status = run_train(clock, RUN2, args, steps)
    strict = status == "ok"
    if status == "stop" and not (RUN2 / "checkpoints" / "train_log.csv").exists():
        return "stop"
    if not do_phase1_eval(clock, RUN2, init, steps, strict):
        return "stop"
    return "ran"


def start_colors(clock: dict, decision: dict, run_dir: Path, step0: Path) -> str:
    if decision["branch"] not in ("colors_from_run1", "colors_from_run2"):
        return "no"
    if (COLORS / "train" / "eval_summary.json").exists():
        return "done"
    if (COLORS / "probe" / "REPORT.md").exists() and read_json(COLORS / "probe" / "probe.json").get("gate") == "fail":
        log("probe gate failed; not training colors")
        return "probe_fail"
    rem = remaining(clock)
    requested = color_request(rem)
    if requested is None:
        append_gate(
            run_dir / "REPORT.md",
            decision,
            False,
            f"Colors were not started. {rem / 3600:.2f} h remained, under 3 h.",
        )
        return "no"
    per = rate(clock, RUN1)
    steps = fit_color_steps(requested, rem, per, sec_per(clock), 50 * 60)
    if steps == 0:
        append_gate(
            run_dir / "REPORT.md",
            decision,
            False,
            "Colors were not started. Even 1000 steps plus the rollout would not finish before the cap.",
        )
        return "no"
    init_step = int(decision["init_step"])
    init = find_ckpt(run_dir, init_step, step0)
    why = (
        f"Colors start from phase-1 step {init_step} (`{init}`). "
        f"The clock asked for {requested} steps; the step rate left {steps}."
    )
    append_gate(run_dir / "REPORT.md", decision, True, why)
    log(why)
    if not (COLORS / "datasets" / "train" / "stack_seeds.json").exists():
        status = launch_and_wait(
            [
                "-u",
                str(COLORS / "collect.py"),
                "--train-out",
                str(COLORS / "datasets" / "train"),
                "--val-out",
                str(COLORS / "datasets" / "val"),
            ],
            COLORS / "datasets",
            clock,
            True,
            False,
        )
        log(f"collect {status}")
        if status.endswith("3") or (COLORS / "datasets" / "REPORT.md").exists() and not (COLORS / "datasets" / "train" / "stack_seeds.json").exists():
            log("expert gate failed")
            return "stop"
        if not (COLORS / "datasets" / "train" / "stack_seeds.json").exists():
            crash_report(COLORS / "datasets", "Collection stopped")
            return "stop"
    if not (COLORS / "probe" / "REPORT.md").exists():
        if remaining(clock) < 20 * 60:
            log("no time for the probe")
            return "stop"
        status = launch_and_wait(
            [
                "-u",
                str(COLORS / "probe.py"),
                "--train",
                str(COLORS / "datasets" / "train"),
                "--val",
                str(COLORS / "datasets" / "val"),
                "--checkpoint",
                str(init),
                "--out",
                str(COLORS / "probe"),
            ],
            COLORS / "probe",
            clock,
            True,
            False,
        )
        log(f"probe {status}")
        if not (COLORS / "probe" / "REPORT.md").exists():
            crash_report(COLORS / "probe", "Probe stopped")
            return "stop"
    gate = read_json(COLORS / "probe" / "probe.json").get("gate")
    if gate != "pass":
        log(f"probe gate {gate}; not training")
        return "probe_fail"
    rem = remaining(clock)
    steps = fit_color_steps(steps, rem, per, sec_per(clock), 0)
    if steps == 0:
        (COLORS / "train").mkdir(parents=True, exist_ok=True)
        (COLORS / "train" / "REPORT.md").write_text(
            "# Color train\n\nThe train was not started. 1000 steps plus the rollout would not finish before the cap.\n",
            encoding="utf-8",
        )
        return "no"
    if (COLORS / "train" / "eval_summary.json").exists():
        return "done"
    if not train_done(COLORS / "train", steps) and not (COLORS / "train" / "cap.txt").exists():
        note = (
            f"Requested by the clock at the color-phase start, then cut to {steps} steps "
            f"so the rollout still fits. Init phase-1 step {init_step}."
        )
        (COLORS / "train").mkdir(parents=True, exist_ok=True)
        (COLORS / "train" / "clock_cut.txt").write_text(note, encoding="utf-8")
        args = [
            "-u",
            str(COLORS / "train.py"),
            "--dataset",
            str(COLORS / "datasets" / "train"),
            "--val-dataset",
            str(COLORS / "datasets" / "val"),
            "--output-dir",
            str(COLORS / "train" / "checkpoints"),
            "--from-checkpoint",
            str(init),
            "--steps",
            str(steps),
            "--init-step",
            str(init_step),
            "--batch-size",
            "8",
        ]
        status = run_train(clock, COLORS / "train", args, steps)
        log(f"color train {status}")
        if status == "stop" and not (COLORS / "train" / "checkpoints" / "train_log.csv").exists():
            return "stop"
    strict = train_done(COLORS / "train", steps)
    require = expected_steps(train_target(COLORS / "train", steps), 1000) if strict else ""
    started = time.time()
    ok = run_eval(
        clock,
        COLORS / "train",
        [
            "-u",
            str(COLORS / "eval.py"),
            "--out",
            str(COLORS / "train"),
            "--init",
            str(init),
            "--ckpt-root",
            str(COLORS / "train" / "checkpoints"),
            "--seeds-file",
            str(COLORS / "datasets" / "train" / "stack_seeds.json"),
            "--seeds-file",
            str(COLORS / "datasets" / "val" / "stack_seeds.json"),
        ]
        + (["--require", require] if require else []),
    )
    if ok:
        if ok and (time.time() - started) > 30:
            summary = read_json(COLORS / "train" / "eval_summary.json")
            n = sum(int(row.get("n", 0)) for block in summary.get("sentences", {}).values() for row in block.get("results", {}).values())
            record_eval_time(clock, started, n)
        cut_note = (COLORS / "train" / "clock_cut.txt").read_text(encoding="utf-8") if (COLORS / "train" / "clock_cut.txt").exists() else ""
        if cut_note and "<!-- cut -->" not in (COLORS / "train" / "REPORT.md").read_text(encoding="utf-8"):
            report = COLORS / "train" / "REPORT.md"
            report.write_text(report.read_text(encoding="utf-8") + "\n<!-- cut -->\n\n" + cut_note + "\n", encoding="utf-8")
    return "done" if ok else "stop"


def start_unseen(clock: dict) -> str:
    if (COLORS / "unseen_words" / "REPORT.md").exists():
        return "done"
    ckpt_root = COLORS / "train" / "checkpoints" / "checkpoints"
    found = sorted(p for p in ckpt_root.glob("[0-9]*") if (p / "pretrained_model" / "model.safetensors").exists())
    if not found:
        return "no"
    last = found[-1] / "pretrained_model"
    need = 10 * sec_per(clock) + 300
    if remaining(clock) < need:
        report = COLORS / "train" / "REPORT.md"
        if report.exists() and "unseen words were not rolled out" not in report.read_text(encoding="utf-8"):
            report.write_text(
                report.read_text(encoding="utf-8")
                + f"\nUnseen words were not rolled out. {remaining(clock) / 3600:.2f} h remained.\n",
                encoding="utf-8",
            )
        return "no"
    log(f"unseen rollout {last}")
    status = launch_and_wait(
        [
            "-u",
            str(COLORS / "unseen.py"),
            "--out",
            str(COLORS / "unseen_words"),
            "--checkpoint",
            str(last),
            "--seeds-file",
            str(COLORS / "datasets" / "train" / "stack_seeds.json"),
            "--seeds-file",
            str(COLORS / "datasets" / "val" / "stack_seeds.json"),
        ],
        COLORS / "unseen_words",
        clock,
        False,
        False,
    )
    log(f"unseen {status}")
    return "done" if (COLORS / "unseen_words" / "REPORT.md").exists() else "stop"


def main() -> None:
    apply_env()
    if not (INIT / "model.safetensors").exists():
        raise SystemExit(f"missing init checkpoint {INIT}")
    if not (ROOT / "best" / "pretrained_model" / "model.safetensors").exists():
        raise SystemExit("best checkpoint is missing")
    clock = load_clock()
    if "t0" not in clock:
        clock["t0"] = time.time()
        save_clock(clock)
        log("t0 set at run 1 launch")
    else:
        log(f"resume, remaining {remaining(clock) / 3600:.2f} h")

    if not train_done(RUN1, 8000) and not (RUN1 / "cap.txt").exists() and not (RUN1 / "eval_summary.json").exists():
        args = phase1_args(RUN1, INIT, 8000, 1, 2000, int(clock.get("run1_batch", 8)), 45)
        if "run1_batch" not in clock:
            clock["run1_batch"] = 8
            save_clock(clock)
        status = run_train(clock, RUN1, args, 8000)
        log(f"run 1 {status}")
        if status == "stop" and not (RUN1 / "eval_summary.json").exists() and last_step(RUN1) == 0:
            log("run 1 did not train")
            return
    strict = train_done(RUN1, 8000)
    if not (RUN1 / "eval_summary.json").exists():
        if not do_phase1_eval(clock, RUN1, INIT, 8000, strict):
            log("run 1 eval failed")
            return
    rate(clock, RUN1)
    if not (RUN1 / "gate.json").exists():
        decision = decide(read_json(RUN1 / "eval_summary.json"), False)
        log(f"run 1 gate {decision['branch']} R={decision['R']} G={decision['G']}")
        outcome = maybe_run2(clock, decision)
        if outcome == "stop":
            log("stopped during run 2")
            return
        if outcome == "ran":
            decision = decide(read_json(RUN2 / "eval_summary.json"), True)
            log(f"run 2 gate {decision['branch']} R={decision['R']} G={decision['G']}")
            if decision["branch"] == "stop_missed":
                append_gate(
                    RUN2 / "REPORT.md",
                    decision,
                    False,
                    "The color phase was not started because green on red, or red on green, was still under 5.",
                )
                log("bar missed after run 2")
                return
            color_dir = RUN2
            step0 = find_ckpt(RUN1, int(read_json(RUN2 / "train_meta.json").get("init_step", decision.get("init_step", 0))), INIT)
        else:
            color_dir = RUN1
            step0 = INIT
            if (RUN1 / "gate.json").exists() and not read_json(RUN1 / "gate.json").get("started", True):
                log("run 2 was not started")
                if decision["branch"] in ("run2_same", "run2_weight"):
                    return
        if decision["branch"] in ("colors_from_run1", "colors_from_run2"):
            # step0 for run 2's step 0 is the run 1 checkpoint the run was inited from.
            if decision["branch"] == "colors_from_run2":
                step0 = Path(read_json(RUN2 / "train_meta.json").get("from_checkpoint", step0))
            result = start_colors(clock, decision, color_dir if decision["branch"] == "colors_from_run1" else RUN2, step0)
            log(f"colors {result}")
            if result == "done":
                start_unseen(clock)
    else:
        decision = read_json(RUN1 / "gate.json")
        branch = decision.get("branch", "")
        if branch in ("run2_same", "run2_weight") and decision.get("started") and not (RUN2 / "eval_summary.json").exists():
            steps = int(decision["steps"])
            init = find_ckpt(RUN1, int(decision["init_step"]), INIT)
            status = run_train(
                clock,
                RUN2,
                phase1_args(RUN2, init, steps, float(decision["green_weight"]), int(decision["init_step"]), 8, 0.0),
                steps,
            )
            log(f"resume run 2 {status}")
            if status != "stop" or (RUN2 / "checkpoints" / "train_log.csv").exists():
                do_phase1_eval(clock, RUN2, init, steps, status == "ok" or train_done(RUN2, steps))
        if (RUN2 / "eval_summary.json").exists():
            decision2 = decide(read_json(RUN2 / "eval_summary.json"), True)
            if not (RUN2 / "gate.json").exists() and decision2["branch"] == "stop_missed":
                append_gate(
                    RUN2 / "REPORT.md",
                    decision2,
                    False,
                    "The color phase was not started because green on red, or red on green, was still under 5.",
                )
            elif decision2["branch"] == "colors_from_run2" and not (COLORS / "unseen_words" / "REPORT.md").exists():
                step0 = Path(read_json(RUN2 / "train_meta.json").get("from_checkpoint", INIT))
                if start_colors(clock, decision2, RUN2, step0) == "done":
                    start_unseen(clock)
        elif branch == "colors_from_run1" and not (COLORS / "unseen_words" / "REPORT.md").exists():
            fresh = decide(read_json(RUN1 / "eval_summary.json"), False)
            if start_colors(clock, fresh, RUN1, INIT) == "done":
                start_unseen(clock)
    log(f"sit finished, remaining {remaining(clock) / 3600:.2f} h")


if __name__ == "__main__":
    main()
