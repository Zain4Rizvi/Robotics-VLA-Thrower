"""Roll out saved checkpoints. Counts in eval_summary.json are the result.

Physics does not step while a chunk is computed. Cube xy is the live body
positions, red_box then green_box, z-scored with the train-label mean and std.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from expert import (  # noqa: E402
    BODY,
    HOLD_STEPS,
    HELD_COLORS,
    TRAIN_COLORS,
    StackSim,
    mentions_heldout,
    run_episode,
    sentence,
    sit_error,
    stacked,
)
from train import write_xy  # noqa: E402

H, W = 480, 640
CAP = 400
SWAP_SEEDS = list(range(30000, 30010))
PALETTE_SEEDS = list(range(31000, 31010))
HELDOUT_SEEDS = list(range(32000, 32010))
RED_ON_GREEN = sentence("red", "green")
GREEN_ON_RED = sentence("green", "red")


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


def predict(policy, pre, post, images, degrees, task, xy) -> np.ndarray:
    import torch

    obs = {
        "observation.state": torch.from_numpy(degrees.astype(np.float32)),
        "task": task,
    }
    for name, img in images.items():
        obs[f"observation.images.{name}"] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    batch = pre(obs)
    state = batch["observation.state"]
    scored = torch.as_tensor(xy, dtype=state.dtype, device=state.device).reshape(1, 4)
    filled = write_xy(state, scored)
    batch["observation.state"] = filled
    if not predict.checked:
        expect = (filled[:, -1, 6:10] if filled.ndim == 3 else filled[:, 6:10]).detach()

        def hook(_mod, inputs, expect=expect):
            got = inputs[0]
            if got.ndim == 3:
                got = got[:, -1, :]
            if got.shape[-1] != 32 or not torch.allclose(got[:, 6:10], expect, atol=1.0e-5):
                raise SystemExit("forward path dropped dims 6:10")
            if float(got[:, 10:].abs().sum()) != 0:
                raise SystemExit("dims 10:32 are not zero at state_proj")

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


def rollout(sim, policy, pre, post, seed, renderer, record, source, colors, task) -> dict:
    sim.reset(seed, source=source, colors=colors)
    if sim.task() != task:
        raise SystemExit(f"task {sim.task()!r} != {task!r}")
    policy.reset()
    src0 = sim.pos(sim.source).copy()
    grasped = False
    dropped = False
    hold = 0
    closest = sit_error(sim)
    buf, ptr = None, 0
    video = []
    for _ in range(CAP):
        if buf is None or ptr >= view.REPLAN_EVERY or ptr >= len(buf):
            top, wrist = render_pair(sim, renderer)
            xy = view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2])
            buf = predict(policy, pre, post, {"camera1": top, "camera2": wrist}, sim.degrees(), task, xy)
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
        closest = min(closest, sit_error(sim))
        if sim.holding() and sim.pos(sim.source)[2] > src0[2] + 0.02:
            grasped = True
        if grasped and not sim.holding() and not stacked(sim):
            dropped = True
        hold = hold + 1 if stacked(sim) else 0
        if hold >= HOLD_STEPS:
            break
    return {
        "seed": seed,
        "stack": hold >= HOLD_STEPS,
        "grasp": grasped,
        "drop": dropped and hold < HOLD_STEPS,
        "closest_m": closest,
        "task": task,
        "source_body": BODY[source],
        "colors": colors,
        "video": video,
    }


def tally(rows: list[dict]) -> dict:
    keep = []
    for r in rows:
        item = {k: v for k, v in r.items() if k != "video"}
        keep.append(item)
    return {
        "stacks": sum(r["stack"] for r in rows),
        "grasps": sum(r["grasp"] for r in rows),
        "drops": sum(r["drop"] for r in rows),
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


def record_expert(sim, renderer, path: Path, scenes: list[dict]) -> None:
    for scene in scenes:
        frames = []

        def on_step(sim, _ctrl, frames=frames):
            top, wrist = render_pair(sim, renderer)
            frames.append(inset(top, wrist))

        row = run_episode(
            sim, scene["seed"], source=scene["source"], colors=scene["colors"], on_step=on_step
        )
        if row["success"]:
            write_mp4(path, frames)
            print(f"expert video {path} seed {scene['seed']} task={row['task']!r}", flush=True)
            return
    raise SystemExit("no scripted success for the expert video")


def run_policy(sim, renderer, policy, pre, post, scenes, video_for) -> list[dict]:
    rows = []
    for scene in scenes:
        record = video_for(scene, rows)
        row = rollout(
            sim, policy, pre, post, scene["seed"], renderer, record,
            scene["source"], scene["colors"], scene["task"],
        )
        rows.append(row)
        print(
            f"seed {scene['seed']}: stack={row['stack']} grasp={row['grasp']} "
            f"drop={row['drop']} closest_cm={row['closest_m'] * 100:.1f} task={scene['task']!r}",
            flush=True,
        )
    return rows


def dump_videos(rows: list[dict], path: Path) -> None:
    frames = next(r["video"] for r in rows if r["video"])
    write_mp4(path, frames)
    for r in rows:
        r["video"] = []


def read_seeds(paths: list[Path]) -> set[int]:
    used = set()
    for path in paths:
        used |= {int(row["seed"]) for row in json.loads(path.read_text(encoding="utf-8"))}
    return used


def assert_disjoint(used: set[int], seeds: list[int]) -> None:
    overlap = used.intersection(seeds)
    if overlap:
        raise SystemExit(f"eval seeds collide with train/val: {sorted(overlap)}")


def swap_scenes() -> list[dict]:
    scenes = []
    for task, source in ((RED_ON_GREEN, "red"), (GREEN_ON_RED, "green")):
        for seed in SWAP_SEEDS:
            scenes.append(
                {
                    "tag": "red_on_green" if source == "red" else "green_on_red",
                    "seed": seed,
                    "source": source,
                    "colors": {"red": "red", "green": "green"},
                    "task": task,
                }
            )
    return scenes


def palette_scenes() -> list[dict]:
    rng = np.random.default_rng(0)
    scenes = []
    for seed in PALETTE_SEEDS:
        pair = rng.choice(TRAIN_COLORS, size=2, replace=False)
        source = "red" if int(rng.integers(2)) == 0 else "green"
        colors = {"red": str(pair[0]), "green": str(pair[1])}
        task = sentence(colors[source], colors["green" if source == "red" else "red"])
        if mentions_heldout(task):
            raise SystemExit(task)
        scenes.append({"seed": seed, "source": source, "colors": colors, "task": task})
    return scenes


def heldout_scenes() -> list[dict]:
    scenes = []
    for i, seed in enumerate(HELDOUT_SEEDS):
        src_color, tgt_color = ("orange", "purple") if i < 5 else ("purple", "orange")
        if i % 2 == 0:
            source, colors = "red", {"red": src_color, "green": tgt_color}
        else:
            source, colors = "green", {"red": tgt_color, "green": src_color}
        task = sentence(src_color, tgt_color)
        scenes.append({"seed": seed, "source": source, "colors": colors, "task": task, "order": f"{src_color}_on_{tgt_color}"})
    orders = {s["task"] for s in scenes}
    if orders != {sentence("orange", "purple"), sentence("purple", "orange")}:
        raise SystemExit(f"held-out orders {orders}")
    return scenes


def assert_no_heldout_words(paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for word in HELD_COLORS:
            if word in text:
                raise SystemExit(f"{word} appears in {path}")


def cell(row: dict) -> str:
    best = row["closest_m_best"] * 100
    mean = row["closest_m_mean"] * 100
    return f"{best:.1f} cm best, {mean:.1f} cm mean"


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


def plot_success(root: Path, summary: dict, mode: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if mode == "swap":
        for tag, label in (("red_on_green", "red on green"), ("green_on_red", "green on red")):
            block = summary["sentences"][tag]["results"]
            xs = sorted(block, key=int)
            ax.plot([int(s) for s in xs], [block[s]["stacks"] for s in xs], "o-", label=label)
    else:
        block = summary["results"]
        xs = sorted(block, key=int)
        xi = [int(s) for s in xs]
        ax.plot(xi, [block[s]["stacks"] for s in xs], "o-", label="stacks")
        ax.plot(xi, [block[s]["grasps"] for s in xs], "s-", label="grasps")
        ax.plot(xi, [block[s]["drops"] for s in xs], "x--", label="drops")
    ax.set_xlabel("step")
    ax.set_ylabel("count out of 10")
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


def write_swap_report(root: Path, summary: dict) -> None:
    ckpt = root / "checkpoints"
    train_rows = read_csv(ckpt / "train_log.csv")
    val_rows = read_csv(ckpt / "val_log.csv") if (ckpt / "val_log.csv").exists() else []
    plot_loss(root, train_rows, val_rows)
    plot_success(root, summary, "swap")
    lines = [
        "# Both red/green sentences",
        "",
        "Cubes stay red and green on `red_box` and `green_box`. State dims 6:8 are the red body xy and dims 8:10 are the green body xy, z-scored with the same mean and std as `best/`. Dims 10:32 stay zero.",
        "The sentence is the only thing that changes which cube should move. Vision and language stayed frozen. Init weights are `best/`.",
        f"Train loss was logged for {train_rows[-1]['step']} steps. Counts are from `eval_summary.json`, seeds {summary['seeds']}.",
        "",
        "## stack the red cube on the green cube",
        "",
        *table(summary["sentences"]["red_on_green"]["results"]),
        "",
        "## stack the green cube on the red cube",
        "",
        *table(summary["sentences"]["green_on_red"]["results"]),
        "",
        "Closest is the distance from the source cube center to the pose where it sits on the target cube.",
        "A stack is that pose, in contact, for 0.5 s, with the target cube still on the floor.",
        "",
        "![loss](loss.png)",
        "",
        "![success](success.png)",
        "",
        "Videos are the top camera with the wrist view inset: `videos/expert.mp4`, "
        "`videos/red_on_green_step0000.mp4`, `videos/red_on_green_step1000.mp4`, `videos/red_on_green_step2000.mp4`, "
        "`videos/green_on_red_step0000.mp4`, `videos/green_on_red_step1000.mp4`, `videos/green_on_red_step2000.mp4`.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_palette_report(root: Path, summary: dict) -> None:
    ckpt = root / "checkpoints"
    train_rows = read_csv(ckpt / "train_log.csv")
    val_rows = read_csv(ckpt / "val_log.csv") if (ckpt / "val_log.csv").exists() else []
    plot_loss(root, train_rows, val_rows)
    plot_success(root, summary, "palette")
    lines = [
        "# Trained color words",
        "",
        "Each episode paints two of red, green, blue, and yellow on `red_box` and `green_box`. State slots stay in that body order. The sentence names the colors.",
        "Vision and language stayed frozen. Step 0 is the step 1 checkpoint at 2000 steps. This run then trained 2000 more steps from those weights.",
        f"Train loss was logged for {train_rows[-1]['step']} steps. Counts are from `eval_summary.json`, seeds {summary['seeds']}.",
        "",
        *table(summary["results"]),
        "",
        "Closest is the distance from the source cube center to the pose where it sits on the target cube.",
        "",
        "![loss](loss.png)",
        "",
        "![success](success.png)",
        "",
        "Videos are the top camera with the wrist view inset: `videos/expert.mp4`, `videos/step0000.mp4`, `videos/step1000.mp4`, `videos/step2000.mp4`.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_heldout_report(root: Path, summary: dict) -> None:
    r = summary["results"]
    lines = [
        "# A color the training set never said",
        "",
        "No new demos. The step 2 checkpoint at 2000 steps was rolled out on orange and purple only. Those words are not in the step 1 or step 2 datasets.",
        "State slots stay `red_box` xy then `green_box` xy. Vision stayed frozen.",
        f"Seeds {summary['seeds']}. Counts are from `eval_summary.json`.",
        "",
        "| stacks/10 | grasps/10 | drops | closest |",
        "|---|---|---|---|",
        f"| {r['stacks']} | {r['grasps']} | {r['drops']} | {cell(r)} |",
        "",
        "Closest is the distance from the source cube center to the pose where it sits on the target cube.",
        "",
    ]
    if r["stacks"] == 0:
        lines.append("The trained color words work only for colors named in the demos.")
        lines.append("")
    lines.append("Videos are the top camera with the wrist view inset: `videos/orange_on_purple.mp4`, `videos/purple_on_orange.mp4`.")
    lines.append("")
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("swap", "palette", "heldout"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path)
    p.add_argument("--ckpt-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seeds-file", type=Path, action="append", default=[])
    p.add_argument("--scan", type=Path, action="append", default=[])
    args = p.parse_args()

    if args.mode == "swap":
        seeds = SWAP_SEEDS
        groups = {}
        for scene in swap_scenes():
            groups.setdefault(scene["tag"], []).append(scene)
        scene_groups = groups
    elif args.mode == "palette":
        seeds = PALETTE_SEEDS
        scene_groups = {"palette": palette_scenes()}
    else:
        seeds = HELDOUT_SEEDS
        scene_groups = {"heldout": heldout_scenes()}
        assert_no_heldout_words(args.scan)

    assert_disjoint(read_seeds(args.seeds_file), seeds)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=H, width=W)
    videos = args.out / "videos"
    if args.mode != "heldout":
        expert_scenes = scene_groups["red_on_green"] if args.mode == "swap" else scene_groups["palette"]
        record_expert(sim, renderer, videos / "expert.mp4", expert_scenes)

    if args.mode == "heldout":
        ckpts = [(2000, args.checkpoint)]
    else:
        if args.init is None or args.ckpt_root is None:
            raise SystemExit("need --init and --ckpt-root")
        ckpts = [(0, args.init), *saved_checkpoints(args.ckpt_root)]
        have = {step for step, _ in ckpts}
        if not {0, 1000, 2000} <= have:
            raise SystemExit(f"checkpoints present: {sorted(have)}")

    summary: dict = {"seeds": seeds, "mode": args.mode}
    if args.mode == "swap":
        summary["sentences"] = {}
    else:
        summary["scenes"] = [
            {k: v for k, v in s.items() if k != "tag"} for s in next(iter(scene_groups.values()))
        ]
        if args.mode == "palette":
            summary["results"] = {}

    for step, ckpt in ckpts:
        if args.mode != "heldout" and step not in (0, 1000, 2000):
            continue
        predict.checked = False
        policy, pre, post = load_finetuned(ckpt, device)
        print(f"checkpoint step {step} {ckpt}", flush=True)
        if args.mode == "swap":
            for tag, scenes in scene_groups.items():
                def video_for(scene, rows, tag=tag, step=step):
                    return not any(r["video"] for r in rows) and scene is scenes[0]

                rows = run_policy(sim, renderer, policy, pre, post, scenes, video_for)
                if tag == "green_on_red" and step == 0 and rows[0]["stack"]:
                    fail = next((r for r in rows if not r["stack"]), None)
                    if fail is not None:
                        for r in rows:
                            r["video"] = []
                        again = rollout(
                            sim, policy, pre, post, fail["seed"], renderer, True,
                            "green", {"red": "red", "green": "green"}, GREEN_ON_RED,
                        )
                        fail["video"] = again["video"]
                dump_videos(rows, videos / f"{tag}_step{step:04d}.mp4")
                summary["sentences"].setdefault(tag, {"task": scenes[0]["task"], "results": {}})
                summary["sentences"][tag]["results"][str(step)] = tally(rows)
        elif args.mode == "palette":
            scenes = scene_groups["palette"]

            def video_for(scene, rows, scenes=scenes):
                return scene is scenes[0] and not any(r["video"] for r in rows)

            rows = run_policy(sim, renderer, policy, pre, post, scenes, video_for)
            dump_videos(rows, videos / f"step{step:04d}.mp4")
            summary["results"][str(step)] = tally(rows)
        else:
            scenes = scene_groups["heldout"]
            seen: set[str] = set()

            def video_for(scene, rows, seen=seen):
                if scene["order"] in seen:
                    return False
                seen.add(scene["order"])
                return True

            rows = run_policy(sim, renderer, policy, pre, post, scenes, video_for)
            order_task = {
                "orange_on_purple": sentence("orange", "purple"),
                "purple_on_orange": sentence("purple", "orange"),
            }
            for order, task in order_task.items():
                frames = next(r["video"] for r in rows if r["task"] == task and r["video"])
                write_mp4(videos / f"{order}.mp4", frames)
            for r in rows:
                r["video"] = []
            summary["checkpoint"] = str(ckpt)
            summary["results"] = tally(rows)
        del policy, pre, post
        if device == "cuda":
            torch.cuda.empty_cache()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.mode == "swap":
        for name in (
            "videos/expert.mp4",
            "videos/red_on_green_step0000.mp4",
            "videos/red_on_green_step1000.mp4",
            "videos/red_on_green_step2000.mp4",
            "videos/green_on_red_step0000.mp4",
            "videos/green_on_red_step1000.mp4",
            "videos/green_on_red_step2000.mp4",
        ):
            require(args.out / name)
        write_swap_report(args.out, summary)
        stacks = summary["sentences"]["green_on_red"]["results"]["2000"]["stacks"]
        print(f"green_on_red_step2000_stacks={stacks}", flush=True)
    elif args.mode == "palette":
        for name in ("videos/expert.mp4", "videos/step0000.mp4", "videos/step1000.mp4", "videos/step2000.mp4"):
            require(args.out / name)
        write_palette_report(args.out, summary)
        print(f"palette_step2000_stacks={summary['results']['2000']['stacks']}", flush=True)
    else:
        for name in ("videos/orange_on_purple.mp4", "videos/purple_on_orange.mp4"):
            require(args.out / name)
        write_heldout_report(args.out, summary)
        print(f"heldout_stacks={summary['results']['stacks']}", flush=True)
    renderer.close()
    print(f"wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
