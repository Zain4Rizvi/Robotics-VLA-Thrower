"""Eleven color pairs. The expert gate runs before any dataset is written.

Held out of train and val: stack the yellow cube on the blue cube.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(HERE))

from expert import BODY, StackSim, mentions_heldout, run_episode, self_check  # noqa: E402
from layout import HELD_SENTENCE, PAIRS, forbidden, paint, self_check as layout_check  # noqa: E402

H, W = 480, 640
VIDEO_BACKEND = "pyav"
GATE_SEEDS = set(range(60000, 60020)) | set(range(61000, 61020))
ROLLOUT = set(range(30000, 30010)) | set(range(33000, 33010)) | set(range(34000, 34010))


def create_dataset(root: Path):
    from lerobot.configs.video import VideoEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    img = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    return LeRobotDataset.create(
        repo_id=f"local/{root.name}",
        fps=30,
        features={
            "observation.images.camera1": img,
            "observation.images.camera2": img,
            "observation.state": {"dtype": "float32", "shape": (6,), "names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
            "action": {"dtype": "float32", "shape": (6,), "names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
        },
        root=root,
        robot_type="so_arm100",
        use_videos=True,
        image_writer_threads=2,
        video_backend=VIDEO_BACKEND,
        rgb_encoder=VideoEncoderConfig(vcodec="h264", crf=28),
    )


def gate(sim: StackSim, pairs: list[tuple[str, str]], seed0: int, label: str) -> int:
    n_ok = 0
    for i, seed in enumerate(range(seed0, seed0 + 20)):
        src, tgt = pairs[i % len(pairs)]
        body = "red" if i % 2 == 0 else "green"
        other = "green" if body == "red" else "red"
        colors = {body: src, other: tgt}
        row = run_episode(sim, seed, source=body, colors=colors)
        n_ok += int(row["success"])
        if row["task"] != f"stack the {src} cube on the {tgt} cube":
            raise SystemExit(f"gate task {row['task']!r}")
        if forbidden(row["task"]) and label != "yellow_on_blue":
            raise SystemExit(f"gate wrote a held-out sentence {row['task']}")
        print(
            f"{label} seed {seed}: success={row['success']} body={body} task={row['task']!r} "
            f"phase={row['phase']} closest_cm={row['closest_m'] * 100:.1f}",
            flush=True,
        )
    print(f"{label} gate {n_ok}/20", flush=True)
    return n_ok


def collect_split(out: Path, n_each: int, seed0: int) -> None:
    if out.exists():
        raise SystemExit(f"{out} already exists")
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=H, width=W)
    ds = None
    rows = []
    reds: list[np.ndarray] = []
    greens: list[np.ndarray] = []
    counts = {pair: 0 for pair in PAIRS}
    seed = seed0
    done = False
    try:
        ds = create_dataset(out)
        while any(v < n_each for v in counts.values()):
            pair = next(p for p in PAIRS if counts[p] < n_each)
            src, tgt = pair
            body, colors = paint(seed, src, tgt)
            buf: list[tuple] = []

            def on_step(sim, ctrl, buf=buf):
                images = {}
                for name in ("camera1", "camera2"):
                    renderer.update_scene(sim.data, camera=name)
                    images[name] = renderer.render().copy()
                buf.append(
                    (
                        images,
                        sim.degrees().copy(),
                        sim.action_degrees(ctrl).copy(),
                        sim.red_pos()[:2].copy(),
                        sim.green_pos()[:2].copy(),
                    )
                )

            row = run_episode(sim, seed, source=body, colors=colors, on_step=on_step)
            task = row["task"]
            if forbidden(task) or mentions_heldout(task) or task == HELD_SENTENCE:
                raise SystemExit(f"refusing to write {task}")
            if task != f"stack the {src} cube on the {tgt} cube":
                raise SystemExit(f"task {task} != {src} on {tgt}")
            print(f"seed {seed}: success={row['success']} task={task!r} ({sum(counts.values())} saved)", flush=True)
            if row["success"]:
                if len(buf) != row["steps"]:
                    raise SystemExit(f"frames {len(buf)} != steps {row['steps']}")
                for images, state, action, red_xy, green_xy in buf:
                    ds.add_frame(
                        {
                            "observation.images.camera1": images["camera1"],
                            "observation.images.camera2": images["camera2"],
                            "observation.state": state,
                            "action": action,
                            "task": task,
                        }
                    )
                    reds.append(red_xy)
                    greens.append(green_xy)
                ds.save_episode(parallel_encoding=False)
                rows.append(
                    {
                        "episode": len(rows),
                        "seed": seed,
                        "sentence": task,
                        "source_color": src,
                        "target_color": tgt,
                        "source_body": BODY[body],
                        "colors": colors,
                        "steps": row["steps"],
                    }
                )
                counts[pair] += 1
            seed += 1
            if seed > seed0 + n_each * len(PAIRS) * 8:
                raise SystemExit(f"too many failures before {n_each} of each pair")
        done = True
    finally:
        if ds is not None:
            ds.finalize()
        renderer.close()
        if not done and out.exists():
            shutil.rmtree(out, ignore_errors=True)
    if any(v != n_each for v in counts.values()):
        raise SystemExit(f"counts {counts}")
    text = json.dumps(rows)
    if forbidden(text):
        raise SystemExit("held-out sentence landed in the seeds file")
    seeds = {int(r["seed"]) for r in rows}
    if seeds & (GATE_SEEDS | ROLLOUT):
        raise SystemExit("dataset seeds collide with a gate or a rollout")
    np.savez(
        out / "xy.npz",
        red_xy=np.stack(reds).astype(np.float32),
        green_xy=np.stack(greens).astype(np.float32),
    )
    (out / "stack_seeds.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} episodes to {out}", flush=True)


def write_gate_report(parent: Path, eleven: int, held: int) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "REPORT.md").write_text(
        "# Expert gate\n\n"
        f"Eleven-pair gate on seeds 60000–60019: {eleven}/20.\n\n"
        f"Yellow on blue gate on seeds 61000–61019: {held}/20.\n\n"
        "Below 18/20. No dataset was written.\n",
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train-out", type=Path, required=True)
    p.add_argument("--val-out", type=Path, required=True)
    args = p.parse_args()
    self_check()
    layout_check()
    if args.train_out.exists() or args.val_out.exists():
        raise SystemExit("dataset out already exists")
    sim = StackSim()
    sim.assert_paint_is_visual()
    eleven = gate(sim, list(PAIRS), 60000, "eleven")
    held = gate(sim, [("yellow", "blue")], 61000, "yellow_on_blue")
    if eleven < 18 or held < 18:
        write_gate_report(args.train_out.parent, eleven, held)
        raise SystemExit(3)
    collect_split(args.train_out, 10, 40000)
    collect_split(args.val_out, 4, 50000)
    train_seeds = {int(r["seed"]) for r in json.loads((args.train_out / "stack_seeds.json").read_text(encoding="utf-8"))}
    val_seeds = {int(r["seed"]) for r in json.loads((args.val_out / "stack_seeds.json").read_text(encoding="utf-8"))}
    if train_seeds & val_seeds:
        raise SystemExit("val seeds overlap train")
    print("collect ok", flush=True)


if __name__ == "__main__":
    main()
