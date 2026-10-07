"""Expert-only SmolVLA loop for this sitting.

Schedule checks live here. color/train.py assert_expert_only still refuses a
decay that is not 2000, and this file does not call it.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))

import view  # noqa: E402

VIDEO_BACKEND = "pyav"
GREEN = "stack the green cube on the red cube"
DECAY_LR = 2.5e-6
PEAK_LR = 1.0e-4


def load_color_helpers():
    path = ROOT / "color" / "train.py"
    spec = importlib.util.spec_from_file_location("color_train_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cosine_lambda(decay_steps: int, warmup: int, peak: float, floor: float):
    alpha = floor / peak

    def lr_lambda(current_step):
        def linear_warmup_schedule(step):
            if step <= 0:
                return 1 / (warmup + 1)
            frac = 1 - step / warmup
            return (1 / (warmup + 1) - 1) * frac + 1

        def cosine_decay_schedule(step):
            step = min(step, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
            return (1 - alpha) * cosine_decay + alpha

        if current_step < warmup:
            return linear_warmup_schedule(current_step)
        return cosine_decay_schedule(current_step)

    return lr_lambda


def assert_schedule(policy, steps: int, helpers) -> None:
    raw = helpers.unwrap(policy)
    cfg = raw.config
    if not cfg.freeze_vision_encoder or not cfg.train_expert_only:
        raise helpers.StopTrain(
            f"freeze_vision_encoder={cfg.freeze_vision_encoder} train_expert_only={cfg.train_expert_only}"
        )
    if cfg.scheduler_warmup_steps != 100 or int(cfg.scheduler_decay_steps) != int(steps):
        raise helpers.StopTrain(
            f"scheduler warmup {cfg.scheduler_warmup_steps} decay {cfg.scheduler_decay_steps} expected decay {steps}"
        )
    if abs(float(cfg.optimizer_lr) - PEAK_LR) > 1.0e-12:
        raise helpers.StopTrain(f"lr {cfg.optimizer_lr}")
    if abs(float(cfg.scheduler_decay_lr) - DECAY_LR) > 1.0e-15:
        raise helpers.StopTrain(f"decay lr {cfg.scheduler_decay_lr}")
    vlm = raw.model.vlm_with_expert
    model = vlm.get_vlm_model()
    bad = []
    for label, module in (
        ("vision", model.vision_model),
        ("connector", model.connector),
        ("text", model.text_model),
    ):
        for name, param in module.named_parameters():
            if param.requires_grad:
                bad.append(f"{label}.{name}")
    if bad:
        raise helpers.StopTrain("trainable prefix parameter: " + ", ".join(bad[:6]))
    if any(p.requires_grad for p in vlm.vlm.parameters()):
        raise helpers.StopTrain("a vlm parameter is trainable")
    n_expert = sum(p.numel() for p in vlm.lm_expert.parameters() if p.requires_grad)
    n_state = sum(p.numel() for p in raw.model.state_proj.parameters() if p.requires_grad)
    n = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    print(f"trainable {n} action_expert {n_expert} state_proj {n_state}", flush=True)
    if n_expert == 0 or n_state == 0:
        raise helpers.StopTrain("action expert or state_proj has no trainable parameters")
    src = (ROOT / "color" / "train.py").read_text(encoding="utf-8")
    if "scheduler_decay_steps != 2000" not in src:
        raise helpers.StopTrain("color/train.py assert_expert_only no longer refuses a decay other than 2000")


def frame_tasks(seeds_path: Path, n_xy: int) -> np.ndarray:
    rows = json.loads(seeds_path.read_text(encoding="utf-8"))
    tasks = []
    for row in rows:
        tasks.extend([row["sentence"]] * int(row["steps"]))
    if len(tasks) != n_xy:
        raise RuntimeError(f"frame tasks {len(tasks)} != xy rows {n_xy}")
    return np.asarray(tasks)


def retarget(held, policy, helpers, decay_steps: int) -> None:
    sched = held.get("sched")
    if sched is None or not getattr(sched, "lr_lambdas", None):
        raise helpers.StopTrain("cannot retarget the cosine; scheduler has no lr_lambdas")
    sched.lr_lambdas[0] = cosine_lambda(decay_steps, 100, PEAK_LR, DECAY_LR)
    raw = helpers.unwrap(policy)
    raw.config.scheduler_decay_steps = int(decay_steps)
    cfg = held.get("cfg")
    if cfg is not None:
        cfg.steps = int(decay_steps)
        if getattr(cfg, "policy", None) is not None:
            cfg.policy.scheduler_decay_steps = int(decay_steps)


def execute(
    *,
    out: Path,
    dataset: Path,
    val_dataset: Path,
    from_checkpoint: Path,
    steps: int,
    save_freq: int,
    val_every: int,
    batch_size: int,
    fill,
    logit_slice: tuple[int, int],
    zero_from: int,
    green_weight: float = 1.0,
    cut_minutes: float = 0.0,
    cut_to: int = 0,
    report_path: Path | None = None,
) -> None:
    helpers = load_color_helpers()
    if out.exists():
        raise SystemExit(f"{out} already exists")
    blocked = {"best", "use_note"}
    if blocked & set(out.parts) or "swap" in out.parts:
        raise SystemExit(f"refusing to train into {out}")
    ckpt_parts = set(from_checkpoint.parts)
    if "swap" in ckpt_parts and "checkpoints" in ckpt_parts:
        raise SystemExit(f"refusing to init from {from_checkpoint}")
    if not (from_checkpoint / "model.safetensors").exists():
        raise SystemExit(f"no model.safetensors in {from_checkpoint}")
    if green_weight != 1 and abs(green_weight - 2) > 1.0e-9:
        raise SystemExit(f"green weight {green_weight}")
    train_xy = helpers.load_xy(dataset / "xy.npz")
    val_xy = helpers.load_xy(val_dataset / "xy.npz")
    tasks = frame_tasks(dataset / "stack_seeds.json", len(train_xy))
    if green_weight != 1 and GREEN not in set(tasks.tolist()):
        raise SystemExit("green-on-red is not in the training frames")
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    if int(info["total_frames"]) != len(train_xy):
        raise SystemExit(f"dataset frames {info['total_frames']} != xy {len(train_xy)}")
    print(f"xy mean {view.XY_MEAN.tolist()} std {view.XY_STD.tolist()}", flush=True)
    used = helpers.gpu_used_mb()
    print(f"gpu memory used {used} MB -> batch {batch_size}", flush=True)

    argv = [
        "lerobot-train",
        f"--policy.path={from_checkpoint}",
        f"--dataset.repo_id=local/{dataset.name}",
        f"--dataset.root={dataset}",
        f"--dataset.video_backend={VIDEO_BACKEND}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.optimizer_lr=1.0e-4",
        "--policy.scheduler_decay_lr=2.5e-6",
        "--policy.scheduler_warmup_steps=100",
        f"--policy.scheduler_decay_steps={steps}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        "--log_freq=50",
        "--num_workers=0",
        f"--output_dir={out}",
        "--wandb.enable=false",
    ]
    print(" ".join(argv), flush=True)
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    import lerobot.scripts.lerobot_train as lt
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.collate import lerobot_collate_fn

    make = lt.make_policy

    def make_policy_fp32(*a, **k):
        policy = make(*a, **k).float()
        assert_schedule(policy, steps, helpers)
        return policy

    lt.make_policy = make_policy_fp32
    lt.update_last_checkpoint = lambda d: d
    orig_opt = lt.make_optimizer_and_scheduler

    held = {
        "xy": None,
        "logits": None,
        "checked": False,
        "hooked": False,
        "skip_hook": False,
        "green_weight": float(green_weight),
        "frame_tasks": tasks,
        "weights": None,
        "cfg": None,
        "sched": None,
        "limit": None,
        "t0": None,
        "val_s": 0.0,
        "logit_slice": logit_slice,
        "zero_from": zero_from,
    }

    def make_opt(cfg, policy):
        held["cfg"] = cfg
        return orig_opt(cfg, policy)

    lt.make_optimizer_and_scheduler = make_opt

    class BudgetCut(RuntimeError):
        def __init__(self, step: int):
            super().__init__(f"cut at {step}")
            self.step = step

    real_save = lt.save_checkpoint

    def save_checkpoint(*a, **k):
        real_save(*a, **k)
        step = k.get("step", a[1] if len(a) > 1 else None)
        if held.get("limit") and int(step) >= int(held["limit"]):
            raise BudgetCut(int(step))

    lt.save_checkpoint = save_checkpoint

    procs = {}
    make_pp = lt.make_pre_post_processors

    def make_pp_capture(*a, **k):
        helpers.keep_checkpoint_stats(k.get("preprocessor_overrides"))
        helpers.keep_checkpoint_stats(k.get("postprocessor_overrides"))
        procs["pre"], procs["post"] = make_pp(*a, **k)
        return procs["pre"], procs["post"]

    lt.make_pre_post_processors = make_pp_capture

    def pre_hook(_mod, inputs):
        if held["skip_hook"]:
            return
        got = inputs[0]
        if got.ndim == 3:
            got = got[:, -1, :]
        lo, hi = held["logit_slice"]
        z0 = held["zero_from"]
        exp_xy = held["xy"]
        exp_logits = held["logits"]
        if exp_xy is None or exp_logits is None or got.shape[-1] != 32:
            raise helpers.StopTrain(f"forward path state {tuple(got.shape)}")
        if not __import__("torch").allclose(got[:, 6:10], exp_xy, atol=1.0e-5):
            raise helpers.StopTrain("forward path dropped dims 6:10")
        if not __import__("torch").allclose(got[:, lo:hi], exp_logits, atol=1.0e-5):
            raise helpers.StopTrain(f"forward path dropped dims {lo}:{hi}")
        if float(got[:, z0:].abs().sum()) != 0:
            raise helpers.StopTrain(f"dims {z0}:32 are not zero at state_proj")
        held["hooked"] = True

    def validate(policy, step: int) -> None:
        import torch

        t0 = time.perf_counter()
        meta = LeRobotDatasetMetadata(f"local/{val_dataset.name}", root=val_dataset)
        dt = resolve_delta_timestamps(policy.config, meta)
        ds = LeRobotDataset(
            f"local/{val_dataset.name}",
            root=val_dataset,
            video_backend=VIDEO_BACKEND,
            delta_timestamps=dt,
        )
        policy.eval()
        losses = []
        buf = []
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(0)
            for i in range(len(ds)):
                buf.append(ds[int(i)])
                if len(buf) < batch_size and i != len(ds) - 1:
                    continue
                b = procs["pre"](helpers.float_images(lerobot_collate_fn(buf)))
                fill(b, val_xy, policy, held)
                losses.append(float(policy.forward(b)[0]))
                buf = []
        policy.train()
        val = float(np.mean(losses))
        helpers.append_row(out / "val_log.csv", ["step", "val_loss"], [step, val])
        print(f"[val] step {step}: val_loss={val:.4f} n={len(ds)}", flush=True)
        helpers.prune_training_state(out)
        held["val_s"] += time.perf_counter() - t0

    class GreenWeighter:
        def compute_batch_weights(self, batch):
            import torch

            w = held["weights"]
            if w is None:
                raise helpers.StopTrain("green weights were not set")
            if tuple(w.shape) != (batch["observation.state"].shape[0],):
                raise helpers.StopTrain(f"weight shape {tuple(w.shape)}")
            return w, {"green_weight": float(green_weight)}

        def get_stats(self):
            return {"green_weight": float(green_weight)}

    weighter = GreenWeighter() if green_weight != 1 else None
    st = {"step": 0}
    upd = lt.update_policy
    budget_path = out.parent / "budget.json"

    def write_budget(measured_steps: int, minutes: float | None, cut: bool) -> None:
        blob = {
            "requested_steps": int(steps),
            "steps": int(held["limit"] or steps),
            "cut": bool(cut),
            "minutes_per_1000": minutes,
            "measured_at_step": measured_steps,
            "green_weight": float(green_weight),
        }
        budget_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"budget {blob}", flush=True)

    def maybe_cut(policy) -> None:
        if cut_minutes <= 0 or st["step"] != 1000 or held.get("limit"):
            return
        elapsed = time.perf_counter() - held["t0"]
        minutes = elapsed / 60.0
        print(f"step rate {minutes:.2f} min per 1000 steps (wall clock, including val)", flush=True)
        if minutes > cut_minutes:
            held["limit"] = int(cut_to)
            retarget(held, policy, helpers, int(cut_to))
            write_budget(1000, minutes, True)
            print(f"cut to {cut_to} steps", flush=True)
        else:
            write_budget(1000, minutes, False)

    def update_policy(tracker, policy, batch, *a, **k):
        if held["t0"] is None:
            held["t0"] = time.perf_counter()
        if "lr_scheduler" in k:
            held["sched"] = k["lr_scheduler"]
        raw = helpers.unwrap(policy)
        fill(batch, train_xy, raw, held)
        if weighter is not None:
            import torch

            index = batch["index"].detach().reshape(-1).cpu().numpy().astype(np.int64)
            w = np.ones(len(index), np.float32)
            w[tasks[index] == GREEN] = np.float32(green_weight)
            held["weights"] = torch.as_tensor(w, device=batch["observation.state"].device)
        if not held["checked"]:
            helpers.assert_kept_stats(procs["pre"], from_checkpoint)
            helpers.assert_state_proj(policy)
            raw.model.state_proj.register_forward_pre_hook(pre_hook)
            held["checked"] = True
            print(
                f"state_proj {tuple(raw.model.state_proj.weight.shape)} "
                f"xy abs mean {float(held['xy'].abs().mean()):.3f} "
                f"logit abs mean {float(held['logits'].abs().mean()):.3f}",
                flush=True,
            )
        if weighter is not None:
            k["sample_weighter"] = weighter
        held["hooked"] = False
        tracker, out_dict = upd(tracker, policy, batch, *a, **k)
        if not held["hooked"]:
            raise helpers.StopTrain("state_proj was not called")
        st["step"] += 1
        m = tracker.metrics
        out.mkdir(parents=True, exist_ok=True)
        helpers.append_row(
            out / "train_log.csv",
            ["step", "loss", "lr", "grad_norm"],
            [st["step"], m["loss"].val, m["lr"].val, m["grad_norm"].val],
        )
        if st["step"] % val_every == 0:
            validate(raw, st["step"])
        maybe_cut(policy)
        return tracker, out_dict

    lt.update_policy = update_policy
    sys.argv = argv
    report = report_path or (out.parent / "REPORT.md")
    try:
        lt.main()
    except BudgetCut as e:
        print(f"stopped on the clock cut at step {e.step}", flush=True)
    except helpers.StopTrain as e:
        print(f"stopped: {e}", flush=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Training stopped\n\n"
            "Training stopped before a finished run.\n\n"
            f"{e}\n\n"
            "No rollout.\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from e
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(traceback.format_exc(), flush=True)
        shutil.rmtree(out, ignore_errors=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        (report.parent / "oom.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise SystemExit(2) from e
    finally:
        if out.exists():
            helpers.prune_training_state(out)
        print(json.dumps({"steps": st["step"], "batch": batch_size, "gpu_used_mb_at_start": used, "limit": held["limit"]}))


def self_check() -> None:
    lam = cosine_lambda(8000, 100, PEAK_LR, DECAY_LR)
    if abs(lam(8000) - (DECAY_LR / PEAK_LR)) > 1.0e-9:
        raise SystemExit(f"cosine floor {lam(8000)}")
    if not (0 < lam(50) < 1):
        raise SystemExit(f"warmup {lam(50)}")
    if lam(100) < 0.99:
        raise SystemExit(f"peak {lam(100)}")
    src = (ROOT / "color" / "train.py").read_text(encoding="utf-8")
    if "scheduler_decay_steps != 2000" not in src:
        raise SystemExit("assert_expert_only was weakened")
    print("harness self_check ok", flush=True)


if __name__ == "__main__":
    self_check()
