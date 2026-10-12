"""Train the connector, the action expert, and state_proj.

SigLIP and the text layers stay frozen. Dimensions 6:32 stay zero.
This file does not call harness.assert_schedule or harness.execute.
color/train.py assert_expert_only is not called. It still refuses a decay
other than 2000; this script checks its own schedule.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence" / "cont"))

from harness import retarget  # noqa: E402

_spec = importlib.util.spec_from_file_location("camera_text_zero_tail", ROOT / "camera_text" / "train.py")
_camera = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_camera)
abort_if_tail = _camera.abort_if_tail
zero_tail = _camera.zero_tail

VIDEO_BACKEND = "pyav"
ROLLOUT = set(range(30000, 30010))
RED = "stack the red cube on the green cube"
GREEN = "stack the green cube on the red cube"
BLOCKED = {"best", "use_note", "run1", "swap", "playground", "camera_text"}
PEAK_LR = 1.0e-4
DECAY_LR = 2.5e-6


def configure_fp32() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def connector_grad_alive(grads) -> bool:
    import torch

    for grad in grads:
        if grad is None:
            continue
        if float(torch.as_tensor(grad).detach().abs().sum()) != 0:
            return True
    return False


def self_check() -> None:
    import torch

    _camera.self_check()
    if connector_grad_alive([None, torch.zeros(2, 2)]):
        raise SystemExit("all-zero connector grads looked alive")
    if not connector_grad_alive([None, torch.zeros(2, 2), torch.tensor([0.0, 1.0e-8])]):
        raise SystemExit("nonzero connector grad looked dead")
    print("connector train self_check ok", flush=True)


def refuse_seeds(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    seeds = {int(row["seed"]) for row in rows}
    hit = sorted(seeds & ROLLOUT)
    if hit:
        raise SystemExit(f"rollout seeds are in the dataset: {hit}")
    counts = {RED: 0, GREEN: 0}
    for row in rows:
        sentence = row["sentence"]
        if sentence not in counts:
            raise SystemExit(f"unexpected sentence {sentence!r} in {path}")
        counts[sentence] += 1
    return counts


def refuse_path(path: Path, what: str) -> None:
    parts = set(path.resolve().parts)
    if BLOCKED & parts:
        raise SystemExit(f"refusing to {what} {path}")


def modules(policy, helpers):
    raw = helpers.unwrap(policy)
    vlm = raw.model.vlm_with_expert
    model = vlm.get_vlm_model()
    return raw, vlm, model


def assert_frozen_flags(model, helpers) -> None:
    if any(p.requires_grad for p in model.vision_model.parameters()):
        raise helpers.StopTrain("a vision-tower parameter is trainable")
    if any(p.requires_grad for p in model.text_model.parameters()):
        raise helpers.StopTrain("a text-layer parameter is trainable")
    if not any(p.requires_grad for p in model.connector.parameters()):
        raise helpers.StopTrain("no connector parameter is trainable")


def enable_connector(policy, helpers) -> None:
    raw, vlm, model = modules(policy, helpers)
    cfg = raw.config
    if not cfg.freeze_vision_encoder or not cfg.train_expert_only:
        raise helpers.StopTrain(
            f"freeze_vision_encoder={cfg.freeze_vision_encoder} train_expert_only={cfg.train_expert_only}"
        )
    for param in model.connector.parameters():
        param.requires_grad = True
    orig = vlm.train

    def train(mode: bool = True):
        orig(mode)
        for param in model.connector.parameters():
            param.requires_grad = True
        assert_frozen_flags(model, helpers)
        return vlm

    vlm.train = train
    assert_frozen_flags(model, helpers)


def assert_trainable(policy, helpers, steps: int) -> None:
    raw, vlm, model = modules(policy, helpers)
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
    assert_frozen_flags(model, helpers)
    n_conn = sum(p.numel() for p in model.connector.parameters() if p.requires_grad)
    n_expert = sum(p.numel() for p in vlm.lm_expert.parameters() if p.requires_grad)
    n_state = sum(p.numel() for p in raw.model.state_proj.parameters() if p.requires_grad)
    n = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    print(
        f"trainable {n} connector {n_conn} action_expert {n_expert} state_proj {n_state}",
        flush=True,
    )
    if n_conn == 0 or n_expert == 0 or n_state == 0:
        raise helpers.StopTrain("connector, action expert, or state_proj has no trainable parameters")
    weight = raw.model.state_proj.weight
    if tuple(weight.shape) != (960, 32):
        raise helpers.StopTrain(f"state_proj.weight.shape is {tuple(weight.shape)}")


def execute(args) -> None:
    from harness import load_color_helpers

    helpers = load_color_helpers()
    out = args.output_dir
    dataset = args.dataset
    val_dataset = args.val_dataset
    from_checkpoint = args.from_checkpoint
    steps = args.steps
    src = (ROOT / "color" / "train.py").read_text(encoding="utf-8")
    if "scheduler_decay_steps != 2000" not in src:
        raise SystemExit("color/train.py assert_expert_only no longer refuses a decay other than 2000")
    if out.exists():
        raise SystemExit(f"{out} already exists")
    refuse_path(out, "train into")
    refuse_path(from_checkpoint, "init from")
    if "connector" not in from_checkpoint.resolve().parts or from_checkpoint.name != "init":
        raise SystemExit(f"init must be the connector copy, not {from_checkpoint}")
    if not (from_checkpoint / "model.safetensors").exists():
        raise SystemExit(f"no model.safetensors in {from_checkpoint}")
    train_counts = refuse_seeds(dataset / "stack_seeds.json")
    val_counts = refuse_seeds(val_dataset / "stack_seeds.json")
    if train_counts != {RED: 40, GREEN: 40} or val_counts != {RED: 16, GREEN: 16}:
        raise SystemExit(f"sentence counts train {train_counts} val {val_counts}")
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    if list(info["features"]["observation.state"]["shape"]) != [6]:
        raise SystemExit(f"dataset state is not 6-d: {info['features']['observation.state']['shape']}")
    if info["features"]["observation.state"]["dtype"] != "float32":
        raise SystemExit("dataset state is not float32")
    used = helpers.gpu_used_mb()
    print(f"gpu memory used {used} MB -> batch {args.batch_size}", flush=True)
    print("dims 6:32 forced to zero; xy.npz is not read", flush=True)

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
        f"--batch_size={args.batch_size}",
        f"--steps={steps}",
        f"--save_freq={args.save_freq}",
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
        enable_connector(policy, helpers)
        assert_trainable(policy, helpers, steps)
        return policy

    lt.make_policy = make_policy_fp32
    lt.update_last_checkpoint = lambda d: d
    orig_opt = lt.make_optimizer_and_scheduler
    held = {
        "checked": False,
        "hooked": False,
        "grad_checked": False,
        "cfg": None,
        "sched": None,
        "limit": None,
        "t0": None,
    }

    def make_opt(cfg, policy):
        held["cfg"] = cfg
        optimizer, scheduler = orig_opt(cfg, policy)
        raw = helpers.unwrap(policy)
        connector_params = list(raw.model.vlm_with_expert.get_vlm_model().connector.parameters())
        ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if not any(id(p) in ids and p.requires_grad for p in connector_params):
            raise helpers.StopTrain("connector is not in the optimizer")
        orig_zero = optimizer.zero_grad

        def zero_grad(*za, **zk):
            if not held["grad_checked"]:
                alive = connector_grad_alive([p.grad for p in connector_params])
                held["grad_checked"] = True
                if not alive:
                    raise helpers.StopTrain("every connector gradient is None or zero")
                print("connector gradient is nonzero", flush=True)
            return orig_zero(*za, **zk)

        optimizer.zero_grad = zero_grad
        return optimizer, scheduler

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
        got = inputs[0]
        if got.ndim == 3:
            got = got[:, -1, :]
        try:
            abort_if_tail(got, "state_proj")
        except RuntimeError as e:
            raise helpers.StopTrain(str(e)) from e
        held["hooked"] = True

    def fill(batch) -> None:
        batch["observation.state"] = zero_tail(batch["observation.state"])

    def validate(policy, step: int) -> None:
        import torch

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
                if len(buf) < args.batch_size and i != len(ds) - 1:
                    continue
                b = procs["pre"](helpers.float_images(lerobot_collate_fn(buf)))
                fill(b)
                losses.append(float(policy.forward(b)[0]))
                buf = []
        policy.train()
        assert_frozen_flags(modules(policy, helpers)[2], helpers)
        val = float(np.mean(losses))
        helpers.append_row(out / "val_log.csv", ["step", "val_loss"], [step, val])
        print(f"[val] step {step}: val_loss={val:.4f} n={len(ds)}", flush=True)
        helpers.prune_training_state(out)

    st = {"step": 0}
    upd = lt.update_policy
    budget_path = out.parent / "budget.json"

    def write_budget(measured_steps: int, minutes: float, per_1000: float, cut: bool) -> None:
        blob = {
            "requested_steps": int(steps),
            "steps": int(held["limit"] or steps),
            "cut": bool(cut),
            "minutes_for_measure": minutes,
            "minutes_per_1000": per_1000,
            "measured_at_step": measured_steps,
            "batch_size": int(args.batch_size),
        }
        budget_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"budget {blob}", flush=True)

    def maybe_cut(policy) -> None:
        if st["step"] != args.measure_at or held.get("limit"):
            return
        minutes = (time.perf_counter() - held["t0"]) / 60.0
        per_1000 = minutes * (1000.0 / float(args.measure_at))
        print(
            f"step rate {per_1000:.2f} min per 1000 steps "
            f"(from {minutes:.2f} min for {args.measure_at} steps, including val)",
            flush=True,
        )
        if per_1000 > args.cut_minutes:
            held["limit"] = int(args.cut_to)
            retarget(held, policy, helpers, int(args.cut_to))
            write_budget(int(args.measure_at), minutes, per_1000, True)
            print(f"cut to {args.cut_to} steps", flush=True)
        else:
            write_budget(int(args.measure_at), minutes, per_1000, False)

    def update_policy(tracker, policy, batch, *a, **k):
        if held["t0"] is None:
            held["t0"] = time.perf_counter()
        if "lr_scheduler" in k:
            held["sched"] = k["lr_scheduler"]
        raw = helpers.unwrap(policy)
        fill(batch)
        if not held["checked"]:
            helpers.assert_kept_stats(procs["pre"], from_checkpoint)
            assert_trainable(policy, helpers, int(held["limit"] or steps))
            raw.model.state_proj.register_forward_pre_hook(pre_hook)
            held["checked"] = True
            tail = batch["observation.state"]
            tail = tail[:, -1, :] if tail.ndim == 3 else tail
            print(
                f"state_proj {tuple(raw.model.state_proj.weight.shape)} "
                f"dims 6:32 abs sum {float(tail[:, 6:].abs().sum()):.3f}",
                flush=True,
            )
        held["hooked"] = False
        tracker, out_dict = upd(tracker, policy, batch, *a, **k)
        if not held["hooked"]:
            raise helpers.StopTrain("state_proj was not called")
        if not held["grad_checked"]:
            raise helpers.StopTrain("connector gradient was not checked")
        assert_frozen_flags(modules(policy, helpers)[2], helpers)
        st["step"] += 1
        m = tracker.metrics
        out.mkdir(parents=True, exist_ok=True)
        helpers.append_row(
            out / "train_log.csv",
            ["step", "loss", "lr", "grad_norm"],
            [st["step"], m["loss"].val, m["lr"].val, m["grad_norm"].val],
        )
        if st["step"] % args.val_every == 0:
            validate(raw, st["step"])
        maybe_cut(policy)
        return tracker, out_dict

    lt.update_policy = update_policy
    sys.argv = argv
    report = out.parent / "REPORT.md"
    try:
        lt.main()
    except BudgetCut as e:
        print(f"stopped on the clock cut at step {e.step}", flush=True)
    except helpers.StopTrain as e:
        print(f"stopped: {e}", flush=True)
        report.write_text(
            "# Training stopped\n\n"
            "Training stopped before a finished run.\n\n"
            f"{e}\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from e
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(traceback.format_exc(), flush=True)
        shutil.rmtree(out, ignore_errors=True)
        (out.parent / "oom.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise SystemExit(2) from e
    finally:
        if out.exists():
            helpers.prune_training_state(out)
        print(json.dumps({"steps": st["step"], "batch": args.batch_size, "gpu_used_mb_at_start": used, "limit": held["limit"]}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path)
    p.add_argument("--val-dataset", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--from-checkpoint", type=Path)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--save-freq", type=int, default=500)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cut-minutes", type=float, default=45.0)
    p.add_argument("--cut-to", type=int, default=1000)
    p.add_argument("--measure-at", type=int, default=500)
    p.add_argument("--self-check-only", action="store_true")
    args = p.parse_args()
    configure_fp32()
    self_check()
    if args.self_check_only:
        return
    execute(args)


if __name__ == "__main__":
    main()
