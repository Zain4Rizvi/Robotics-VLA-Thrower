"""Successful stack demonstrations. --out must not exist. CPU only.

One dataset holds every sentence in the step. Val seeds are the caller's job.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert import BODY, TRAIN_COLORS, StackSim, mentions_heldout, run_episode, self_check  # noqa: E402

H, W = 480, 640
VIDEO_BACKEND = "pyav"


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


def palette_spec(seed: int) -> tuple[str, dict[str, str]]:
    rng = np.random.default_rng(seed)
    pair = rng.choice(TRAIN_COLORS, size=2, replace=False)
    source = "red" if int(rng.integers(2)) == 0 else "green"
    colors = {"red": str(pair[0]), "green": str(pair[1])}
    return source, colors


def collect(out: Path, mode: str, n: int, seed0: int) -> None:
    self_check()
    if out.exists():
        raise SystemExit(f"{out} already exists")
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=H, width=W)
    ds = None
    rows = []
    reds: list[np.ndarray] = []
    greens: list[np.ndarray] = []
    seed = seed0
    counts = {"red": 0, "green": 0}
    done = False
    try:
        ds = create_dataset(out)
        while True:
            if mode == "swap":
                if counts["red"] < n:
                    source, colors = "red", {"red": "red", "green": "green"}
                elif counts["green"] < n:
                    source, colors = "green", {"red": "red", "green": "green"}
                else:
                    break
            else:
                if len(rows) >= n:
                    break
                source, colors = palette_spec(seed)
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

            row = run_episode(sim, seed, source=source, colors=colors, on_step=on_step)
            task = row["task"]
            if mentions_heldout(task):
                raise SystemExit(f"held-out color in task {task}")
            ok = row["success"]
            print(
                f"seed {seed}: success={ok} task={task!r} ({len(rows)} saved)",
                flush=True,
            )
            if ok:
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
                        "source_body": BODY[source],
                        "colors": colors,
                        "steps": row["steps"],
                    }
                )
                if mode == "swap":
                    counts[source] += 1
            seed += 1
            if seed > seed0 + max(n, 1) * 8:
                raise SystemExit(f"too many failures before {n} successes")
        done = True
    finally:
        if ds is not None:
            ds.finalize()
        renderer.close()
        if not done and out.exists():
            shutil.rmtree(out, ignore_errors=True)
    if mode == "swap":
        tasks = {r["sentence"] for r in rows}
        if tasks != {"stack the red cube on the green cube", "stack the green cube on the red cube"}:
            raise SystemExit(f"swap tasks {tasks}")
        if counts["red"] != n or counts["green"] != n:
            raise SystemExit(f"counts {counts}")
    text = json.dumps(rows)
    if mentions_heldout(text):
        raise SystemExit("held-out color landed in the seeds file")
    np.savez(
        out / "xy.npz",
        red_xy=np.stack(reds).astype(np.float32),
        green_xy=np.stack(greens).astype(np.float32),
    )
    (out / "stack_seeds.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} episodes to {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("swap", "palette"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed0", type=int, required=True)
    args = p.parse_args()
    collect(args.out, args.mode, args.n, args.seed0)


if __name__ == "__main__":
    main()
