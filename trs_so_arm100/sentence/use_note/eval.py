"""Roll out the use_note checkpoints. Counts in eval_summary.json are the result.

Physics does not step while a chunk is computed. Dims 6:10 are live red xy then
green xy. Dims 10:12 are the frozen probe logits. Dims 12:32 stay zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))

import view  # noqa: E402
from expert import HOLD_STEPS, StackSim, sit_error, stacked  # noqa: E402
from note import batch_notes, configure_fp32, load_classifier, two_logits, write_state  # noqa: E402

H, W = 480, 640
CAP = 400
SEEDS = list(range(30000, 30010))
RED = "stack the red cube on the green cube"
GREEN = "stack the green cube on the red cube"
CLF = ROOT / "sentence" / "probe" / "classifier.npz"


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
    container = av.open(str(path), mode="w")
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


def render_pair(sim: StackSim, renderer) -> tuple[np.ndarray, np.ndarray]:
    images = {}
    for name in ("camera1", "camera2"):
        renderer.update_scene(sim.data, camera=name)
        images[name] = renderer.render().copy()
    return images["camera1"], images["camera2"]


def predict(policy, pre, post, images, degrees, task, xy, clf) -> np.ndarray:
    import torch

    obs = {
        "observation.state": torch.from_numpy(degrees.astype(np.float32)),
        "task": task,
    }
    for name, img in images.items():
        obs[f"observation.images.{name}"] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    batch = pre(obs)
    state = batch["observation.state"]
    notes = batch_notes(policy, batch)
    logits = torch.from_numpy(two_logits(notes, *clf))
    scored = torch.as_tensor(xy, dtype=state.dtype, device=state.device).reshape(1, 4)
    filled = write_state(state, scored, logits)
    batch["observation.state"] = filled
    if not predict.checked:
        expect_xy = (filled[:, -1, 6:10] if filled.ndim == 3 else filled[:, 6:10]).detach()
        expect_logits = (filled[:, -1, 10:12] if filled.ndim == 3 else filled[:, 10:12]).detach()

        def hook(_mod, inputs, expect_xy=expect_xy, expect_logits=expect_logits):
            got = inputs[0]
            if got.ndim == 3:
                got = got[:, -1, :]
            if got.shape[-1] != 32 or not torch.allclose(got[:, 6:10], expect_xy, atol=1.0e-5):
                raise SystemExit("forward path dropped dims 6:10")
            if not torch.allclose(got[:, 10:12], expect_logits, atol=1.0e-5):
                raise SystemExit("forward path dropped dims 10:12")
            if float(got[:, 12:].abs().sum()) != 0:
                raise SystemExit("dims 12:32 are not zero at state_proj")

        handle = policy.model.state_proj.register_forward_pre_hook(hook)
    else:
        handle = None
    try:
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
    finally:
        if handle is not None:
            handle.remove()
            predict.checked = True
    return post(actions)[0].detach().cpu().numpy()


predict.checked = False


def rollout(sim, policy, pre, post, seed, renderer, record, source, task, clf, keep_still: bool) -> dict:
    sim.reset(seed, source=source, colors={"red": "red", "green": "green"})
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
            xy = view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2])
            buf = predict(policy, pre, post, {"camera1": top, "camera2": wrist}, sim.degrees(), task, xy, clf)
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
    keep = []
    for r in rows:
        keep.append({k: v for k, v in r.items() if k not in ("video", "still")})
    return {
        "stacks": int(sum(r["stack"] for r in rows)),
        "grasps": int(sum(r["grasp"] for r in rows)),
        "drops": int(sum(r["drop"] for r in rows)),
        "closest_m_mean": float(np.mean([r["closest_m"] for r in rows])),
        "closest_m_best": float(min(r["closest_m"] for r in rows)),
        "episodes": keep,
    }


def saved_checkpoints(root: Path) -> list[tuple[int, Path]]:
    found = []
    ckpt_dir = root / "checkpoints"
    if ckpt_dir.exists():
        for path in sorted(ckpt_dir.glob("[0-9]*")):
            model = path / "pretrained_model"
            if (model / "model.safetensors").exists():
                found.append((int(path.name), model))
    return found


def read_seeds(paths: list[Path]) -> set[int]:
    used = set()
    for path in paths:
        used |= {int(row["seed"]) for row in json.loads(path.read_text(encoding="utf-8"))}
    return used


def cell(row: dict) -> str:
    return f"{row['closest_m_best'] * 100:.1f} cm best, {row['closest_m_mean'] * 100:.1f} cm mean"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_loss(root: Path, train_rows: list[dict], val_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
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
    fig.savefig(root / "loss.png", dpi=120)
    plt.close(fig)


def plot_success(root: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, label in (("red_on_green", "red on green"), ("green_on_red", "green on red")):
        block = summary["sentences"][tag]["results"]
        xs = sorted(block, key=int)
        ax.plot([int(s) for s in xs], [block[s]["stacks"] for s in xs], "o-", label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("stacks out of 10")
    ax.set_ylim(-0.2, 10.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "success.png", dpi=120)
    plt.close(fig)


def table(results: dict) -> list[str]:
    lines = [
        "| step | stacks/10 | grasps/10 | drops | closest |",
        "|---|---|---|---|---|",
    ]
    for step in sorted(results, key=int):
        r = results[step]
        lines.append(f"| {step} | {r['stacks']} | {r['grasps']} | {r['drops']} | {cell(r)} |")
    return lines


def write_report(root: Path, summary: dict) -> None:
    ckpt = root / "checkpoints"
    train_rows = read_csv(ckpt / "train_log.csv")
    val_rows = read_csv(ckpt / "val_log.csv") if (ckpt / "val_log.csv").exists() else []
    plot_loss(root, train_rows, val_rows)
    plot_success(root, summary)
    red = summary["sentences"]["red_on_green"]["results"]
    green = summary["sentences"]["green_on_red"]["results"]
    lines = [
        "# The note as two numbers",
        "",
        "The probe's val accuracy was 1.000 and its shuffled-val accuracy was 0.422, so this is the train that ran. "
        "The frozen classifier's two logits are written into state dims 10:12. Dims 6:10 stay the red body xy then the green body xy. Dims 12:32 stay zero.",
        "Vision, the text layers, the connector, and the classifier stayed frozen. The action expert and `state_proj` trained, in fp32, from `best/`.",
        f"Train loss was logged for {train_rows[-1]['step']} steps. Counts are from `eval_summary.json`, seeds {summary['seeds']}.",
        "",
        "## stack the red cube on the green cube",
        "",
        *table(red),
        "",
        "## stack the green cube on the red cube",
        "",
        *table(green),
        "",
        "Closest is the distance from the source cube center to the pose where it sits on the target cube.",
        "A stack is that pose, in contact, for 0.5 s, with the target cube still on the floor.",
        "",
        "![loss](loss.png)",
        "",
        "![success](success.png)",
        "",
        "![red on green](frames/red_on_green.png)",
        "",
        "![green on red](frames/green_on_red.png)",
        "",
        "The stills are step 2000. Each one is a success when that sentence stacked, otherwise the closest miss. "
        "Videos are the top camera with the wrist view inset: "
        "`videos/red_on_green_step0000.mp4`, `videos/red_on_green_step1000.mp4`, `videos/red_on_green_step2000.mp4`, "
        "`videos/green_on_red_step0000.mp4`, `videos/green_on_red_step1000.mp4`, `videos/green_on_red_step2000.mp4`.",
        "",
        f"Red on green stacked {red['2000']['stacks']} of 10 at step 2000. "
        f"Green on red stacked {green['2000']['stacks']} of 10 at step 2000.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def scenes() -> dict[str, list[dict]]:
    groups = {"red_on_green": [], "green_on_red": []}
    for seed in SEEDS:
        groups["red_on_green"].append({"seed": seed, "source": "red", "task": RED})
        groups["green_on_red"].append({"seed": seed, "source": "green", "task": GREEN})
    return groups


def pick_still(rows: list[dict]) -> np.ndarray:
    hits = [r for r in rows if r["stack"] and r["still"] is not None]
    if hits:
        return hits[0]["still"]
    misses = [r for r in rows if r["still"] is not None]
    if not misses:
        raise SystemExit("no still frame")
    return min(misses, key=lambda r: r["closest_m"])["still"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--ckpt-root", type=Path, required=True)
    p.add_argument("--seeds-file", type=Path, action="append", default=[])
    args = p.parse_args()
    overlap = set(SEEDS) & read_seeds(args.seeds_file)
    if overlap:
        raise SystemExit(f"eval seeds collide with train/val: {sorted(overlap)}")
    configure_fp32()
    clf = load_classifier(CLF)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("cuda is not available")
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=H, width=W)
    groups = scenes()
    ckpts = [(0, args.init), *saved_checkpoints(args.ckpt_root)]
    have = {step for step, _ in ckpts}
    if not {0, 1000, 2000} <= have:
        raise SystemExit(f"checkpoints present: {sorted(have)}")
    summary: dict = {"seeds": SEEDS, "sentences": {}}
    videos = args.out / "videos"
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for step, ckpt in ckpts:
        if step not in (0, 1000, 2000):
            continue
        predict.checked = False
        batch_notes.checked = False
        policy, pre, post = load_finetuned(ckpt, device)
        print(f"checkpoint step {step} {ckpt}", flush=True)
        for tag, group in groups.items():
            rows = []
            for i, scene in enumerate(group):
                record = i == 0
                row = rollout(
                    sim, policy, pre, post, scene["seed"], renderer, record,
                    scene["source"], scene["task"], clf, keep_still=(step == 2000),
                )
                rows.append(row)
                print(
                    f"seed {scene['seed']}: stack={row['stack']} grasp={row['grasp']} "
                    f"drop={row['drop']} closest_cm={row['closest_m'] * 100:.1f} task={scene['task']!r}",
                    flush=True,
                )
            video_frames = next(r["video"] for r in rows if r["video"])
            write_mp4(videos / f"{tag}_step{step:04d}.mp4", video_frames)
            for r in rows:
                r["video"] = []
            if step == 2000:
                Image.fromarray(pick_still(rows)).save(frames / f"{tag}.png")
            summary["sentences"].setdefault(tag, {"task": group[0]["task"], "results": {}})
            summary["sentences"][tag]["results"][str(step)] = tally(rows)
        del policy, pre, post
        torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name in (
        "videos/red_on_green_step0000.mp4",
        "videos/red_on_green_step1000.mp4",
        "videos/red_on_green_step2000.mp4",
        "videos/green_on_red_step0000.mp4",
        "videos/green_on_red_step1000.mp4",
        "videos/green_on_red_step2000.mp4",
        "frames/red_on_green.png",
        "frames/green_on_red.png",
    ):
        if not (args.out / name).exists():
            raise SystemExit(f"missing {name}")
    write_report(args.out, summary)
    renderer.close()
    red_n = summary["sentences"]["red_on_green"]["results"]["2000"]["stacks"]
    green_n = summary["sentences"]["green_on_red"]["results"]["2000"]["stacks"]
    print(f"step2000 red_on_green={red_n} green_on_red={green_n}", flush=True)
    print(f"wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
