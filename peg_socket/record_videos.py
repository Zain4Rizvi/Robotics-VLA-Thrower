"""A few tablecam clips. Expert seeds 0 and 3 insert; 62 does not.

The VLA clips are lerobot/smolvla_base. That model does not place the peg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_peg_expert import draw, prepare
from peg_expert import HOLD_STEPS, PegExpert, PegSim
from smoke_load import CHECKPOINT, _rgb
from zeroshot_eval import (
    SEED,
    draw_layouts,
    pedestal_aabb,
    rollout,
)

EXPERT_SEEDS = (0, 3, 62)
N_VLA = 2
EXPERT_DIR = Path(__file__).resolve().parent / "findings" / "expert" / "videos"
VLA_DIR = Path(__file__).resolve().parent / "findings" / "zeroshot" / "videos"


def save(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=25)
    print(f"wrote {path} frames={len(frames)}", flush=True)


def record_expert() -> None:
    sim = PegSim()
    expert = PegExpert()
    renderer = mujoco.Renderer(sim.model, height=256, width=256)
    for seed in EXPERT_SEEDS:
        rng = np.random.default_rng(seed)
        peg_xy, sock_xy = draw(rng)
        issue = prepare(sim, peg_xy, sock_xy)
        if issue or not expert.reset(sim):
            raise SystemExit(f"seed {seed} not runnable: {issue}")
        frames: list[np.ndarray] = []
        ever = False
        hold = 0
        inserted = False
        for t, action in enumerate(expert._actions):
            sim.step(action)
            if t % 2 == 0:
                frames.append(_rgb(renderer, sim.data, "tablecam"))
            if sim.holding():
                ever = True
            if sim.finger_qpos() < -0.45 and sim.seated():
                hold += 1
                inserted = inserted or hold >= HOLD_STEPS
            else:
                hold = 0
        tag = "inserted" if inserted else ("never_grasped" if not ever else "missed")
        save(EXPERT_DIR / f"expert_seed{seed}_{tag}.mp4", frames)
    renderer.close()


def record_vla() -> None:
    from openarm_vla.constants import LEFT_ACTUATORS, RIGHT_ACTUATORS, RIGHT_JOINTS
    from openarm_vla.policies.smolvla import SmolVLAAdapter
    from zeroshot_eval import XML

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ped_box = pedestal_aabb(model, data)
    peg_geoms = {int(model.geom("peg_geom").id)}
    sock_geoms = {int(model.geom(f"socket_{i}").id) for i in range(8)}
    table_geom = int(model.geom("table_top").id)
    cam_id = int(model.camera("tablecam").id)
    cam = (
        data.cam_xpos[cam_id].copy(),
        data.cam_xmat[cam_id].reshape(3, 3).copy(),
        float(model.cam_fovy[cam_id]),
    )
    layouts, _ = draw_layouts(model, data, np.random.default_rng(SEED), ped_box, cam, peg_geoms, sock_geoms, table_geom)
    right_act = np.array([model.actuator(n).id for n in RIGHT_ACTUATORS], dtype=int)
    ids = {
        "key": 0,
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
    policy = SmolVLAAdapter(CHECKPOINT, device="cuda")
    renderer = mujoco.Renderer(model, height=256, width=256)
    for i, (peg_xy, sock_xy) in enumerate(layouts[:N_VLA]):
        frames: list[np.ndarray] = []
        row = rollout(model, data, policy, renderer, ids, low, high, peg_xy, sock_xy, frames)
        tag = "inserted" if row["failure"] == "success" else row["failure"]
        save(VLA_DIR / f"vla_base_ep{i}_{tag}.mp4", frames)
        print(f"vla ep={i} {row['failure']} grasped={int(row['grasped'])}", flush=True)
    renderer.close()


if __name__ == "__main__":
    record_expert()
    record_vla()
