"""Roll a fine-tuned checkpoint on stored peg-socket seeds.

Uses zeroshot_eval.rollout. The step cap is 900 because a successful expert
episode is about 700-860 control steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zeroshot_eval as z
from openarm_vla.constants import LEFT_ACTUATORS, RIGHT_ACTUATORS, RIGHT_JOINTS
from smoke_load import XML


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=900)
    args = p.parse_args()
    z.MAX_STEPS = args.max_steps

    rows = json.loads(Path(args.seeds_file).read_text(encoding="utf-8"))
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    key_id = int(model.key("home").id)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    right_act = np.array([model.actuator(n).id for n in RIGHT_ACTUATORS], dtype=int)
    ids = {
        "key": key_id,
        "weld": int(model.eq("grasp_right_peg").id),
        "left": np.array([model.actuator(n).id for n in LEFT_ACTUATORS], dtype=int),
        "right": right_act,
        "qpos": np.array([model.joint(n).qposadr[0] for n in RIGHT_JOINTS], dtype=int),
        "qvel": np.array([model.joint(n).dofadr[0] for n in RIGHT_JOINTS], dtype=int),
        "finger": int(model.joint("openarm_right_finger_joint1").qposadr[0]),
        "ee": int(model.body("openarm_right_ee_base_link").id),
        "peg": int(model.body("peg").id),
        "socket": int(model.body("socket").id),
        "peg_geom": int(model.geom("peg_geom").id),
        "fingers": {
            int(model.body("openarm_right_ee_inner_finger").id),
            int(model.body("openarm_right_ee_outer_finger").id),
        },
    }
    low = np.array([model.actuator_ctrlrange[i, 0] for i in right_act], dtype=np.float64)
    high = np.array([model.actuator_ctrlrange[i, 1] for i in right_act], dtype=np.float64)

    from openarm_vla.policies.smolvla import SmolVLAAdapter

    policy = SmolVLAAdapter(args.checkpoint, device="cuda")
    import torch

    weight = next(policy.policy.parameters())
    print(f"checkpoint={args.checkpoint} dtype={weight.dtype} max_steps={z.MAX_STEPS}", flush=True)
    if weight.dtype != torch.float32:
        raise SystemExit(f"expected float32 weights, got {weight.dtype}")

    renderer = mujoco.Renderer(model, height=z.IMG, width=z.IMG)
    episodes = []
    for row in rows:
        peg_xy = np.asarray(row["peg_xy"], dtype=np.float64)
        sock_xy = np.asarray(row["socket_xy"], dtype=np.float64)
        print(f"seed={row['seed']} peg={np.round(peg_xy, 4)} socket={np.round(sock_xy, 4)}", flush=True)
        frames: list[np.ndarray] = []
        result = z.rollout(
            model, data, policy, renderer, ids, low, high, peg_xy, sock_xy, frames
        )
        result["seed"] = int(row["seed"])
        episodes.append(result)
        path = out / "videos" / f"seed{row['seed']}_{result['failure']}.mp4"
        imageio.mimsave(path, frames, fps=25)
        print(
            f"seed={row['seed']} failure={result['failure']} grasped={int(result['grasped'])} "
            f"min_d={result['min_hand_to_peg_m']:.4f} steps={result['steps']} frames={len(frames)}",
            flush=True,
        )
    renderer.close()

    dists = [e["min_hand_to_peg_m"] for e in episodes]
    summary = {
        "checkpoint": args.checkpoint,
        "seeds_file": args.seeds_file,
        "max_steps": args.max_steps,
        "n": len(episodes),
        "insertions": sum(e["failure"] == "success" for e in episodes),
        "grasp_count": sum(bool(e["grasped"]) for e in episodes),
        "min_hand_to_peg_m": float(min(dists)),
        "median_episode_min_hand_to_peg_m": float(np.median(dists)),
        "episodes": episodes,
    }
    text = json.dumps(summary, indent=2)
    (out / "eval_summary.json").write_text(text + "\n", encoding="utf-8")
    print(
        f"done n={summary['n']} insertions={summary['insertions']} "
        f"grasp_count={summary['grasp_count']} "
        f"min_hand_to_peg_m={summary['min_hand_to_peg_m']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
