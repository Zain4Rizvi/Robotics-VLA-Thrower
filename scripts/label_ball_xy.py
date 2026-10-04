"""Replay a dataset's stored actions and record the target ball's xy.

The live `headcam` is the overhead camera. `train_head` / `val_head` front videos are the older
shoulder view, so a pixel compare against `image_front` fails even when the replay is the same
scene. The wrist video and the joint state do match, and those are the abort checks: if either
drifts, the ball positions would not belong to these frames. The front images written beside the
labels are the live camera, which is what eval will show the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np

from openarm_vla.config import EnvConfig
from openarm_vla.constants import COLORS, REPO_ROOT
from openarm_vla.data import open_dataset
from openarm_vla.env.throw_env import ThrowEnv

# Wrist video is AV1. A wrong scene is far above this; compression sits around 1-4.
WRIST_MAE_MAX = 8.0
STATE_ABS_MAX = 1.0e-2


def _as_int(x) -> int:
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


def label(dataset: Path, out: Path) -> None:
    ds = open_dataset(dataset)
    seeds = json.loads((dataset / "openarm_seeds.json").read_text(encoding="utf-8"))
    n_ep = int(ds.meta.total_episodes)
    if len(seeds) != n_ep:
        raise SystemExit(f"{dataset.name}: {len(seeds)} seeds, {n_ep} episodes")

    actions = np.asarray(ds.hf_dataset["action"], np.float32)
    states = np.asarray(ds.hf_dataset["observation.state"], np.float32)
    n = len(actions)
    out.parent.mkdir(parents=True, exist_ok=True)
    xy = np.empty((n, 2), np.float32)
    color = np.empty(n, np.int64)
    reset = np.zeros(n, bool)
    color_of = {c: i for i, c in enumerate(COLORS)}
    front = np.memmap(out.with_name(out.stem + "_front.u8"), dtype=np.uint8, mode="w+", shape=(n, 256, 256, 3))

    video = dataset / ds.meta.get_video_file_path(0, "observation.images.image_wrist")
    container = av.open(str(video))
    frames = container.decode(video=0)

    env = ThrowEnv(EnvConfig.from_yaml(REPO_ROOT / "configs" / "env.yaml"), render_mode="rgb_array")
    worst_mae = 0.0
    worst_state = 0.0
    cursor = 0
    try:
        for ep_i, row in enumerate(seeds):
            ep = ds.meta.episodes[ep_i]
            start, stop = _as_int(ep["dataset_from_index"]), _as_int(ep["dataset_to_index"])
            if start != cursor:
                raise SystemExit(f"episode {ep_i} starts at {start}, replay is at {cursor}")
            obs, _ = env.reset(seed=int(row["seed"]), options={"instruction": row["instruction"]})
            if env.task["ball_color"] != row["ball_color"] or env.task["instruction"] != row["instruction"]:
                raise SystemExit(f"episode {ep_i}: reset task {env.task['ball_color']!r} != seed {row['ball_color']!r}")
            col = color_of[row["ball_color"]]
            for t, idx in enumerate(range(start, stop)):
                try:
                    vid = next(frames).to_ndarray(format="rgb24")
                except StopIteration as e:
                    raise SystemExit(f"wrist video ended at dataset frame {idx}") from e
                if vid.shape != obs["image_wrist"].shape:
                    raise SystemExit(f"frame {idx}: wrist video {vid.shape} != render {obs['image_wrist'].shape}")
                mae = float(np.abs(vid.astype(np.int16) - obs["image_wrist"].astype(np.int16)).mean())
                state_err = float(np.max(np.abs(states[idx] - obs["state"])))
                worst_mae = max(worst_mae, mae)
                worst_state = max(worst_state, state_err)
                if mae > WRIST_MAE_MAX or state_err > STATE_ABS_MAX:
                    raise SystemExit(
                        f"frame {idx} ep {ep_i} t {t}: wrist mae {mae:.2f} (max {WRIST_MAE_MAX}), "
                        f"state abs {state_err:.3e} (max {STATE_ABS_MAX})"
                    )
                front[idx] = obs["image_front"]
                xy[idx] = env.ball_pos(env.task["ball_body"])[:2]
                color[idx] = col
                reset[idx] = t == 0
                obs, _, _, _, _ = env.step(actions[idx])
            cursor = stop
            print(f"ep {ep_i}/{n_ep} frames {stop - start} worst_wrist_mae {worst_mae:.2f} worst_state {worst_state:.2e}", flush=True)
        try:
            next(frames)
        except StopIteration:
            pass
        else:
            raise SystemExit("wrist video has frames after the last dataset frame")
    finally:
        container.close()
        env.close()
        front.flush()

    if cursor != n:
        raise SystemExit(f"replayed {cursor} frames, dataset has {n}")
    np.savez(out, xy=xy, color=color, reset=reset)
    print(
        f"wrote {out} n={n} resets={int(reset.sum())} xy_mean={xy.mean(0).round(3).tolist()} "
        f"worst_wrist_mae={worst_mae:.2f}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    label(args.dataset, args.out)


if __name__ == "__main__":
    main()
