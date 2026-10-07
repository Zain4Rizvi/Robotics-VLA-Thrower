"""Fine-tune from a local checkpoint. True body xy in state dims 6:10.

Joints stay on the checkpoint's normalizer. Dims 10:32 stay zero.
Vision frozen. Language frozen. Output dir must not exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import view  # noqa: E402

VIDEO_BACKEND = "pyav"
HERE = Path(__file__).resolve().parent
STATE_DIM = 32


class StopTrain(RuntimeError):
    pass


def append_row(path: Path, header: list[str], row: list) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def gpu_used_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def float_images(batch: dict) -> dict:
    return {
        k: v.float() / 255.0 if k.startswith("observation.images.") and v.dtype == torch.uint8 else v
        for k, v in batch.items()
    }


def load_xy(path: Path) -> np.ndarray:
    z = np.load(path)
    table = np.concatenate([z["red_xy"], z["green_xy"]], axis=1).astype(np.float32)
    if table.ndim != 2 or table.shape[1] != 4:
        raise StopTrain(f"xy table {table.shape}")
    return table


def write_xy(state: torch.Tensor, scored: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] != 6:
        raise StopTrain(f"state last dim {state.shape[-1]}, expected 6 before the xy write")
    if scored.shape[0] != state.shape[0]:
        raise StopTrain(f"xy batch {scored.shape[0]} != state batch {state.shape[0]}")
    out = state.new_zeros(*state.shape[:-1], STATE_DIM)
    out[..., :6] = state
    xy = scored.to(dtype=state.dtype, device=state.device).view(scored.shape[0], *([1] * (out.ndim - 2)), 4)
    out[..., 6:10] = xy
    return out


def fill_batch(batch: dict, table: np.ndarray) -> torch.Tensor:
    if "index" not in batch:
        raise StopTrain("batch has no index; cannot align cube xy")
    state = batch["observation.state"]
    index = batch["index"].detach().reshape(-1)
    if index.numel() != state.shape[0]:
        raise StopTrain(f"index {tuple(index.shape)} != state batch {state.shape[0]}")
    idx = index.cpu().numpy().astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= len(table):
        raise StopTrain(f"frame index {idx.min()}..{idx.max()} outside labels {len(table)}")
    scored = torch.from_numpy((table[idx] - view.XY_MEAN) / view.XY_STD)
    out = write_xy(state, scored)
    if not bool(torch.any(out[..., 6:10] != 0)):
        raise StopTrain("dims 6:10 are all zero")
    if float(out[..., 10:].abs().sum()) != 0:
        raise StopTrain("dims 10:32 are not zero")
    batch["observation.state"] = out
    return out[:, -1, 6:10] if out.ndim == 3 else out[:, 6:10]


def unwrap(policy):
    return policy.module if hasattr(policy, "module") else policy


def assert_state_proj(policy) -> None:
    model = unwrap(policy).model
    weight = model.state_proj.weight
    if tuple(weight.shape) != (960, 32):
        raise StopTrain(f"state_proj.weight.shape is {tuple(weight.shape)}")
    if not all(p.requires_grad for p in model.state_proj.parameters()):
        raise StopTrain("state_proj does not require grad")
    if any(p.requires_grad for p in model.vlm_with_expert.vlm.parameters()):
        raise StopTrain("vision or language parameters are trainable")


def assert_expert_only(policy) -> None:
    cfg = unwrap(policy).config
    if not cfg.freeze_vision_encoder or not cfg.train_expert_only:
        raise StopTrain(
            f"freeze_vision_encoder={cfg.freeze_vision_encoder} train_expert_only={cfg.train_expert_only}"
        )
    if cfg.scheduler_warmup_steps != 100 or cfg.scheduler_decay_steps != 2000:
        raise StopTrain(
            f"scheduler warmup {cfg.scheduler_warmup_steps} decay {cfg.scheduler_decay_steps}"
        )
    if abs(float(cfg.optimizer_lr) - 1.0e-4) > 1.0e-12:
        raise StopTrain(f"lr {cfg.optimizer_lr}")
    vlm = unwrap(policy).model.vlm_with_expert
    if any(p.requires_grad for p in vlm.vlm.parameters()):
        raise StopTrain("vision or language parameters are trainable")
    n = sum(p.numel() for p in vlm.lm_expert.parameters() if p.requires_grad)
    if n == 0:
        raise StopTrain("action expert has no trainable parameters")
    print(f"action expert trainable: {n}", flush=True)


def prune_training_state(out: Path) -> None:
    ckpts = sorted((out / "checkpoints").glob("[0-9]*"))
    for c in ckpts[:-1]:
        shutil.rmtree(c / "training_state", ignore_errors=True)


def self_check() -> None:
    state = torch.zeros(2, 1, 6)
    state[..., 0] = 1
    scored = torch.tensor([[1.0, 2, 3, 4], [5, 6, 7, 8]])
    out = write_xy(state, scored)
    if out.shape != (2, 1, 32) or not torch.equal(out[:, 0, 6:10], scored) or float(out[:, :, 10:].abs().sum()) != 0:
        raise SystemExit("write_xy failed")
    if not torch.equal(out[:, 0, :6], state[:, 0, :6]):
        raise SystemExit("joints were overwritten")
    # Dims 6:8 stay the red body even when the sentence's source is the green body.
    red = np.array([0.02, -0.2], np.float32)
    green = np.array([-0.03, -0.22], np.float32)
    got = view.zscore_xy(red, green)
    back = got * view.XY_STD + view.XY_MEAN
    if not np.allclose(back[:2], red, atol=1.0e-5) or not np.allclose(back[2:], green, atol=1.0e-5):
        raise SystemExit(f"body order {back}")
    print("self_check ok", flush=True)


def assert_kept_stats(pre, ckpt: Path) -> None:
    from safetensors.torch import load_file

    files = sorted(ckpt.glob("*normalizer_processor.safetensors"))
    files = [f for f in files if "unnormalizer" not in f.name]
    if len(files) != 1:
        raise StopTrain(f"normalizer stats files {files}")
    ref = load_file(files[0])["observation.state.mean"].float().cpu().reshape(-1)
    got = None
    for step in pre.steps:
        stats = getattr(step, "stats", None)
        if isinstance(stats, dict) and "observation.state" in stats:
            got = torch.as_tensor(stats["observation.state"]["mean"]).float().cpu().reshape(-1)
            break
    if got is None:
        raise StopTrain("preprocessor has no observation.state mean")
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1.0e-5):
        raise StopTrain(f"joint normalizer was replaced (got {got.tolist()} ref {ref.tolist()})")
    print(f"joint normalizer kept {got.tolist()}", flush=True)


def keep_checkpoint_stats(overrides: dict | None) -> None:
    if not overrides:
        return
    for key in ("normalizer_processor", "unnormalizer_processor"):
        block = overrides.get(key)
        if isinstance(block, dict):
            block.pop("stats", None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--val-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--from-checkpoint", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--batch-size", type=int)
    args = p.parse_args()
    self_check()
    root = args.dataset
    vroot = args.val_dataset
    out = args.output_dir
    if out.exists():
        raise SystemExit(f"{out} already exists")
    if "best" in out.parts:
        raise SystemExit(f"refusing to train into {out}")
    seeds_path = root / "stack_seeds.json"
    if not seeds_path.exists():
        raise SystemExit(f"no stack_seeds.json in {root}")
    blob = seeds_path.read_text(encoding="utf-8").lower()
    if "orange" in blob or "purple" in blob:
        raise SystemExit("held-out color words are in the training seeds")
    if not (args.from_checkpoint / "model.safetensors").exists():
        raise SystemExit(f"no model.safetensors in {args.from_checkpoint}")
    train_xy = load_xy(root / "xy.npz")
    val_xy = load_xy(vroot / "xy.npz")
    print(f"xy mean {view.XY_MEAN.tolist()} std {view.XY_STD.tolist()}", flush=True)

    used = gpu_used_mb()
    batch = args.batch_size if args.batch_size is not None else (2 if used >= 800 else 8)
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

    held = {"xy": None, "checked": False, "hooked": False}

    def pre_hook(_mod, inputs):
        got = inputs[0]
        if got.ndim == 3:
            got = got[:, -1, :]
        exp = held["xy"]
        if exp is None or got.shape[-1] != STATE_DIM or got[:, 6:10].shape != exp.shape:
            raise StopTrain(
                f"forward path dropped dims 6:10 (got {tuple(got.shape)}, expected xy {None if exp is None else tuple(exp.shape)})"
            )
        if not torch.allclose(got[:, 6:10], exp, atol=1.0e-5):
            diff = float((got[:, 6:10] - exp).abs().max())
            raise StopTrain(f"forward path dropped dims 6:10 (max abs diff {diff})")
        if float(got[:, 10:].abs().sum()) != 0:
            raise StopTrain("dims 10:32 are not zero at state_proj")
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
                held["xy"] = fill_batch(b, val_xy).detach()
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
        held["xy"] = fill_batch(batch, train_xy).detach()
        if not held["checked"]:
            assert_kept_stats(procs["pre"], args.from_checkpoint)
            assert_state_proj(policy)
            unwrap(policy).model.state_proj.register_forward_pre_hook(pre_hook)
            held["checked"] = True
            print(
                f"state_proj {tuple(unwrap(policy).model.state_proj.weight.shape)} "
                f"xy abs mean {float(held['xy'].abs().mean()):.3f}",
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
            validate(unwrap(policy), st["step"])
        return tracker, out_dict

    lt.update_policy = update_policy
    sys.argv = argv
    report = out.parent / "REPORT.md"
    try:
        lt.main()
    except StopTrain as e:
        print(f"stopped: {e}", flush=True)
        report.write_text(
            "# Color words\n\n"
            "Training stopped before a finished run.\n\n"
            f"{e}\n\n"
            "No rollout. Dims 6:10 did not reach `state_proj`, or a required assert failed.\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from e
    finally:
        if out.exists():
            prune_training_state(out)
        print(json.dumps({"steps": st["step"], "batch": batch, "gpu_used_mb_at_start": used}))


if __name__ == "__main__":
    main()
