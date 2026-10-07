"""Teach the action expert the probe's two logits.

State dims 6:10 stay red xy then green xy. Dims 10:12 are the frozen
classifier's logits. Dims 12:32 stay zero. Vision, language, the connector,
and the classifier stay frozen. Init is trs_so_arm100/best. Output dir must
not already exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))

import view  # noqa: E402
from note import batch_notes, configure_fp32, load_classifier, two_logits, verify_saved_notes, write_state  # noqa: E402
from train import (  # noqa: E402
    StopTrain,
    append_row,
    assert_expert_only,
    assert_kept_stats,
    assert_state_proj,
    float_images,
    gpu_used_mb,
    keep_checkpoint_stats,
    load_xy,
    prune_training_state,
    unwrap,
)

HERE = Path(__file__).resolve().parent
CLF = ROOT / "sentence" / "probe" / "classifier.npz"
NOTES = ROOT / "sentence" / "probe" / "notes.npz"
VIDEO_BACKEND = "pyav"


def self_check() -> None:
    state = torch.zeros(2, 1, 6)
    state[..., 0] = 1
    scored = torch.tensor([[1.0, 2, 3, 4], [5, 6, 7, 8]])
    logits = torch.tensor([[0.1, -0.1], [0.3, -0.3]])
    out = write_state(state, scored, logits)
    if out.shape != (2, 1, 32) or not torch.equal(out[:, 0, 6:10], scored):
        raise SystemExit("write_state xy failed")
    if not torch.equal(out[:, 0, 10:12], logits) or float(out[:, :, 12:].abs().sum()) != 0:
        raise SystemExit("write_state logits failed")
    if not torch.equal(out[:, 0, :6], state[:, 0, :6]):
        raise SystemExit("joints were overwritten")
    red = np.array([0.02, -0.2], np.float32)
    green = np.array([-0.03, -0.22], np.float32)
    got = view.zscore_xy(red, green)
    back = got * view.XY_STD + view.XY_MEAN
    if not np.allclose(back[:2], red, atol=1.0e-5) or not np.allclose(back[2:], green, atol=1.0e-5):
        raise SystemExit(f"body order {back}")
    verify_saved_notes(CLF, NOTES)
    print("self_check ok", flush=True)


def fill_batch(batch: dict, table: np.ndarray, policy, clf, held: dict) -> None:
    if "index" not in batch:
        raise StopTrain("batch has no index; cannot align cube xy")
    state = batch["observation.state"]
    index = batch["index"].detach().reshape(-1)
    if index.numel() != state.shape[0]:
        raise StopTrain(f"index {tuple(index.shape)} != state batch {state.shape[0]}")
    idx = index.cpu().numpy().astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= len(table):
        raise StopTrain(f"frame index {idx.min()}..{idx.max()} outside labels {len(table)}")
    held["skip_hook"] = True
    try:
        notes = batch_notes(policy, batch)
    finally:
        held["skip_hook"] = False
    logits_np = two_logits(notes, *clf)
    if not np.isfinite(logits_np).all():
        raise StopTrain("logits are not finite")
    scored = torch.from_numpy((table[idx] - view.XY_MEAN) / view.XY_STD)
    logits = torch.from_numpy(logits_np)
    out = write_state(state, scored, logits)
    if float(out[..., 12:].abs().sum()) != 0:
        raise StopTrain("dims 12:32 are not zero")
    batch["observation.state"] = out
    held["xy"] = (out[:, -1, 6:10] if out.ndim == 3 else out[:, 6:10]).detach()
    held["logits"] = (out[:, -1, 10:12] if out.ndim == 3 else out[:, 10:12]).detach()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--val-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--from-checkpoint", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    configure_fp32()
    self_check()
    clf = load_classifier(CLF)
    root = args.dataset
    vroot = args.val_dataset
    out = args.output_dir
    if out.exists():
        raise SystemExit(f"{out} already exists")
    if "best" in out.parts or "swap" in out.parts:
        raise SystemExit(f"refusing to train into {out}")
    if not (args.from_checkpoint / "model.safetensors").exists():
        raise SystemExit(f"no model.safetensors in {args.from_checkpoint}")
    seeds_path = root / "stack_seeds.json"
    blob = seeds_path.read_text(encoding="utf-8").lower()
    if "orange" in blob or "purple" in blob:
        raise SystemExit("held-out color words are in the training seeds")
    train_xy = load_xy(root / "xy.npz")
    val_xy = load_xy(vroot / "xy.npz")
    print(f"xy mean {view.XY_MEAN.tolist()} std {view.XY_STD.tolist()}", flush=True)
    used = gpu_used_mb()
    batch = args.batch_size
    print(f"gpu memory used {used} MB -> batch {batch}", flush=True)

    argv = [
        "lerobot-train",
        f"--policy.path={args.from_checkpoint}",
        f"--dataset.repo_id=local/{root.name}",
        f"--dataset.root={root}",
        f"--dataset.video_backend={VIDEO_BACKEND}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.optimizer_lr=1.0e-4",
        "--policy.scheduler_warmup_steps=100",
        f"--policy.scheduler_decay_steps={args.steps}",
        f"--batch_size={batch}",
        f"--steps={args.steps}",
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
        assert_expert_only(policy)
        connector = unwrap(policy).model.vlm_with_expert.get_vlm_model().connector
        if any(param.requires_grad for param in connector.parameters()):
            raise StopTrain("connector is trainable")
        return policy

    lt.make_policy = make_policy_fp32
    lt.update_last_checkpoint = lambda d: d

    procs = {}
    make_pp = lt.make_pre_post_processors

    def make_pp_capture(*a, **k):
        keep_checkpoint_stats(k.get("preprocessor_overrides"))
        keep_checkpoint_stats(k.get("postprocessor_overrides"))
        procs["pre"], procs["post"] = make_pp(*a, **k)
        return procs["pre"], procs["post"]

    lt.make_pre_post_processors = make_pp_capture

    held = {"xy": None, "logits": None, "checked": False, "hooked": False, "skip_hook": False}

    def pre_hook(_mod, inputs):
        if held["skip_hook"]:
            return
        got = inputs[0]
        if got.ndim == 3:
            got = got[:, -1, :]
        exp_xy = held["xy"]
        exp_logits = held["logits"]
        if exp_xy is None or exp_logits is None or got.shape[-1] != 32:
            raise StopTrain(f"forward path state {tuple(got.shape)}")
        if not torch.allclose(got[:, 6:10], exp_xy, atol=1.0e-5):
            raise StopTrain("forward path dropped dims 6:10")
        if not torch.allclose(got[:, 10:12], exp_logits, atol=1.0e-5):
            raise StopTrain("forward path dropped dims 10:12")
        if float(got[:, 12:].abs().sum()) != 0:
            raise StopTrain("dims 12:32 are not zero at state_proj")
        held["hooked"] = True

    def validate(policy, step: int) -> None:
        dt = resolve_delta_timestamps(policy.config, LeRobotDatasetMetadata(f"local/{vroot.name}", root=vroot))
        ds = LeRobotDataset(f"local/{vroot.name}", root=vroot, video_backend=VIDEO_BACKEND, delta_timestamps=dt)
        policy.eval()
        losses = []
        buf = []
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(0)
            for i in range(len(ds)):
                buf.append(ds[int(i)])
                if len(buf) < batch and i != len(ds) - 1:
                    continue
                b = procs["pre"](float_images(lerobot_collate_fn(buf)))
                fill_batch(b, val_xy, policy, clf, held)
                losses.append(float(policy.forward(b)[0]))
                buf = []
        policy.train()
        val = float(np.mean(losses))
        append_row(out / "val_log.csv", ["step", "val_loss"], [step, val])
        print(f"[val] step {step}: val_loss={val:.4f} n={len(ds)}", flush=True)
        prune_training_state(out)

    st = {"step": 0}
    upd = lt.update_policy

    def update_policy(tracker, policy, batch, *a, **k):
        raw = unwrap(policy)
        fill_batch(batch, train_xy, raw, clf, held)
        if not held["checked"]:
            assert_kept_stats(procs["pre"], args.from_checkpoint)
            assert_state_proj(policy)
            raw.model.state_proj.register_forward_pre_hook(pre_hook)
            held["checked"] = True
            print(
                f"state_proj {tuple(raw.model.state_proj.weight.shape)} "
                f"xy abs mean {float(held['xy'].abs().mean()):.3f} "
                f"logit abs mean {float(held['logits'].abs().mean()):.3f}",
                flush=True,
            )
        tracker, out_dict = upd(tracker, policy, batch, *a, **k)
        if not held["hooked"]:
            raise StopTrain("state_proj was not called")
        st["step"] += 1
        m = tracker.metrics
        out.mkdir(parents=True, exist_ok=True)
        append_row(
            out / "train_log.csv",
            ["step", "loss", "lr", "grad_norm"],
            [st["step"], m["loss"].val, m["lr"].val, m["grad_norm"].val],
        )
        if st["step"] % args.val_every == 0:
            validate(raw, st["step"])
        return tracker, out_dict

    lt.update_policy = update_policy
    sys.argv = argv
    report = HERE / "REPORT.md"
    try:
        lt.main()
    except StopTrain as e:
        print(f"stopped: {e}", flush=True)
        report.write_text(
            "# The note as two numbers\n\n"
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
        (HERE / "oom.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise SystemExit(2) from e
    finally:
        if out.exists():
            prune_training_state(out)
        print(json.dumps({"steps": st["step"], "batch": batch, "gpu_used_mb_at_start": used}))


if __name__ == "__main__":
    main()
