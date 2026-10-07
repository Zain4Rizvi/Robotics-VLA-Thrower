"""Closed-loop stack rollouts. Physics does not step while a chunk is computed."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))

import view  # noqa: E402
from expert import HOLD_STEPS, StackSim, sit_error, stacked  # noqa: E402

H, W = 480, 640
CAP = 400


def load_finetuned(path: Path, device: str):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(path)).to(device).float().eval()
    pre, post = make_pre_post_processors(
        policy.config,
        str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def inset(top: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    small = wrist[::3, ::3]
    h, w = small.shape[:2]
    frame = top.copy()
    frame[8 : 8 + h, 8 : 8 + w] = small
    return frame


def write_mp4(path: Path, frames: list[np.ndarray]) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.mp4")
    container = av.open(str(tmp), mode="w")
    stream = container.add_stream("libx264", rate=view.CTRL_HZ)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        video = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    tmp.replace(path)


def render_pair(sim: StackSim, renderer) -> tuple[np.ndarray, np.ndarray]:
    images = {}
    for name in ("camera1", "camera2"):
        renderer.update_scene(sim.data, camera=name)
        images[name] = renderer.render().copy()
    return images["camera1"], images["camera2"]


def _expect(filled):
    if filled.ndim == 3:
        return filled[:, -1, :].detach()
    return filled.detach()


def forward_chunk(policy, post, batch, filled, kind: str, checked: dict):
    import torch

    handle = None
    if not checked["ok"]:
        expect = _expect(filled)

        def hook(_mod, inputs, expect=expect, kind=kind):
            got = inputs[0]
            if got.ndim == 3:
                got = got[:, -1, :]
            if got.shape[-1] != 32 or not torch.allclose(got[:, 6:10], expect[:, 6:10], atol=1.0e-5):
                raise SystemExit("forward path dropped dims 6:10")
            if kind == "phase1":
                if not torch.allclose(got[:, 10:12], expect[:, 10:12], atol=1.0e-5):
                    raise SystemExit("forward path dropped dims 10:12")
                if float(got[:, 12:].abs().sum()) != 0:
                    raise SystemExit("dims 12:32 are not zero at state_proj")
            elif kind == "color":
                if not torch.allclose(got[:, 10:18], expect[:, 10:18], atol=1.0e-5):
                    raise SystemExit("forward path dropped dims 10:18")
                if float(got[:, 18:].abs().sum()) != 0:
                    raise SystemExit("dims 18:32 are not zero at state_proj")
            elif kind == "zero":
                if float(got[:, 10:].abs().sum()) != 0:
                    raise SystemExit("dims 10:32 are not zero at state_proj")
            else:
                raise SystemExit(f"kind {kind}")

        handle = policy.model.state_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
    finally:
        if handle is not None:
            handle.remove()
            checked["ok"] = True
    return post(actions)[0].detach().cpu().numpy()


def run_episode(sim, renderer, policy, seed, source, colors, task, predict_fn, record: bool, keep_still: bool) -> dict:
    sim.reset(seed, source=source, colors=colors or {"red": "red", "green": "green"})
    if sim.task() != task:
        raise SystemExit(f"task {sim.task()!r} != {task!r}")
    policy.reset()
    src0 = sim.pos(sim.source).copy()
    grasped = False
    dropped = False
    hold = 0
    closest = sit_error(sim)
    miss_still = None
    success_still = None
    if keep_still:
        top0, wrist0 = render_pair(sim, renderer)
        miss_still = inset(top0, wrist0)
    buf, ptr = None, 0
    video = []
    for _ in range(CAP):
        if buf is None or ptr >= view.REPLAN_EVERY or ptr >= len(buf):
            top, wrist = render_pair(sim, renderer)
            buf = predict_fn({"camera1": top, "camera2": wrist}, sim)
            ptr = 0
        else:
            top = None
        if record:
            if top is None:
                top, wrist = render_pair(sim, renderer)
            video.append(inset(top, wrist))
        ctrl = view.degrees_to_ctrl(buf[ptr], sim.limits)
        ptr += 1
        sim.step(ctrl)
        err = sit_error(sim)
        if err < closest:
            closest = err
            if keep_still:
                top, wrist = render_pair(sim, renderer)
                miss_still = inset(top, wrist)
        if sim.holding() and sim.pos(sim.source)[2] > src0[2] + 0.02:
            grasped = True
        if grasped and not sim.holding() and not stacked(sim):
            dropped = True
        hold = hold + 1 if stacked(sim) else 0
        if hold >= HOLD_STEPS:
            if keep_still:
                top, wrist = render_pair(sim, renderer)
                success_still = inset(top, wrist)
            break
    return {
        "seed": seed,
        "stack": hold >= HOLD_STEPS,
        "grasp": grasped,
        "drop": dropped and hold < HOLD_STEPS,
        "closest_m": closest,
        "task": task,
        "video": video,
        "still": success_still if hold >= HOLD_STEPS else miss_still,
    }


def tally(rows: list[dict]) -> dict:
    keep = [{k: v for k, v in r.items() if k not in ("video", "still")} for r in rows]
    return {
        "stacks": int(sum(r["stack"] for r in rows)),
        "grasps": int(sum(r["grasp"] for r in rows)),
        "drops": int(sum(r["drop"] for r in rows)),
        "closest_m_mean": float(np.mean([r["closest_m"] for r in rows])),
        "closest_m_best": float(min(r["closest_m"] for r in rows)),
        "n": len(rows),
        "episodes": keep,
    }


def cell(row: dict) -> str:
    return f"{row['closest_m_best'] * 100:.1f} cm best, {row['closest_m_mean'] * 100:.1f} cm mean"


def table(results: dict, denom: int) -> list[str]:
    lines = [
        f"| step | stacks/{denom} | grasps/{denom} | drops | closest |",
        "|---|---|---|---|---|",
    ]
    for step in sorted(results, key=int):
        r = results[step]
        lines.append(f"| {step} | {r['stacks']} | {r['grasps']} | {r['drops']} | {cell(r)} |")
    return lines


def saved_checkpoints(root: Path) -> list[tuple[int, Path]]:
    found = []
    ckpt_dir = root / "checkpoints"
    if ckpt_dir.exists():
        for path in sorted(ckpt_dir.glob("[0-9]*")):
            model = path / "pretrained_model"
            if (model / "model.safetensors").exists():
                found.append((int(path.name), model))
    return found


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_loss(path: Path, train_rows: list[dict], val_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if train_rows:
        ax.plot([int(r["step"]) for r in train_rows], [float(r["loss"]) for r in train_rows], label="train", color="C0")
    if val_rows:
        ax.plot(
            [int(r["step"]) for r in val_rows],
            [float(r["val_loss"]) for r in val_rows],
            "o-",
            label="val",
            color="C1",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_success(path: Path, summary: dict, series: list[tuple[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ymax = 0
    for tag, label in series:
        block = summary["sentences"][tag]["results"]
        xs = sorted(block, key=int)
        ys = [block[s]["stacks"] for s in xs]
        ymax = max(ymax, max(ys) if ys else 0)
        ax.plot([int(s) for s in xs], ys, "o-", label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("stacks")
    ax.set_ylim(-0.2, max(10.2, ymax + 0.2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pick_still(rows: list[dict]) -> np.ndarray:
    hits = [r for r in rows if r["stack"] and r["still"] is not None]
    if hits:
        return hits[0]["still"]
    misses = [r for r in rows if r["still"] is not None]
    if not misses:
        raise SystemExit("no still frame")
    return min(misses, key=lambda r: r["closest_m"])["still"]


def read_seeds(paths: list[Path]) -> set[int]:
    import json

    used = set()
    for path in paths:
        used |= {int(row["seed"]) for row in json.loads(path.read_text(encoding="utf-8"))}
    return used


def obs_batch(pre, images, degrees, task):
    import torch

    obs = {
        "observation.state": torch.from_numpy(degrees.astype(np.float32)),
        "task": task,
    }
    for name, img in images.items():
        obs[f"observation.images.{name}"] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return pre(obs)
