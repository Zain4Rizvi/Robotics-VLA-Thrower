"""Load lerobot/smolvla_base on the peg-socket scene and step a few chunks.

Base actions are meaningless. This only checks that one episode accepts them.
"""

from pathlib import Path

import mujoco
import numpy as np

from openarm_vla.constants import (
    HOME_CTRL_LEFT,
    LEFT_ACTUATORS,
    N_SUBSTEPS,
    RIGHT_ACTUATORS,
    RIGHT_JOINTS,
)
from openarm_vla.policies.smolvla import SmolVLAAdapter

XML = Path(__file__).resolve().parent / "peg_socket_scene.xml"
CHECKPOINT = "lerobot/smolvla_base"
INSTRUCTION = "place the peg in the hole"
N_CHUNKS = 4


def _state(data, jnt_qpos, jnt_qvel, finger_qpos) -> np.ndarray:
    return np.concatenate(
        [
            data.qpos[jnt_qpos],
            data.qvel[jnt_qvel],
            [data.qpos[finger_qpos]],
        ]
    ).astype(np.float32)


def _rgb(renderer, data, camera: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    key_id = int(model.key("home").id)
    print(f"scene={XML}")
    print(f"key_name=home key_id={key_id} nq={model.nq}")
    if key_id != 0:
        raise SystemExit(f"home key id is {key_id}, expected 0")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    weld = int(model.eq("grasp_right_peg").id)
    data.eq_active[weld] = 0
    mujoco.mj_forward(model, data)
    print(f"grasp_right_peg_active={int(data.eq_active[weld])}")
    print(f"qpos_max_abs_err={float(np.max(np.abs(data.qpos - model.key_qpos[key_id]))):.3e}")

    right_act = np.array([model.actuator(n).id for n in RIGHT_ACTUATORS], dtype=int)
    left_act = np.array([model.actuator(n).id for n in LEFT_ACTUATORS], dtype=int)
    jnt_qpos = np.array([model.joint(n).qposadr[0] for n in RIGHT_JOINTS], dtype=int)
    jnt_qvel = np.array([model.joint(n).dofadr[0] for n in RIGHT_JOINTS], dtype=int)
    finger_qpos = int(model.joint("openarm_right_finger_joint1").qposadr[0])
    low = np.array([model.actuator_ctrlrange[i, 0] for i in right_act], dtype=np.float64)
    high = np.array([model.actuator_ctrlrange[i, 1] for i in right_act], dtype=np.float64)
    data.ctrl[left_act] = HOME_CTRL_LEFT

    for name in ("tablecam", "camera_wrist_right", "frontcam"):
        print(f"camera {name} id={int(model.camera(name).id)}")

    renderer = mujoco.Renderer(model, height=256, width=256)
    policy = SmolVLAAdapter(CHECKPOINT, device="cuda")
    print(f"checkpoint={CHECKPOINT}")
    param = next(policy.policy.parameters())
    print(f"policy_dtype={param.dtype} device={param.device}")

    for i in range(N_CHUNKS):
        front = _rgb(renderer, data, "tablecam")
        wrist = _rgb(renderer, data, "camera_wrist_right")
        state = _state(data, jnt_qpos, jnt_qvel, finger_qpos)
        obs = {
            "image_front": front,
            "image_wrist": wrist,
            "state": state,
            "instruction": INSTRUCTION,
        }
        if i == 0:
            print(f"instruction={obs['instruction']!r}")
            print("obs_keys=" + ",".join(obs.keys()))
            for key in ("image_front", "image_wrist", "state"):
                arr = obs[key]
                print(f"obs.{key} shape={tuple(arr.shape)} dtype={arr.dtype}")
            print("map observation.images.image_front=tablecam")
            print("map observation.images.image_wrist=camera_wrist_right")
            print("frontcam_used=0")

        actions = policy.predict_chunk(obs)
        finite = bool(np.isfinite(actions).all())
        print(
            f"chunk={i} action_shape={tuple(actions.shape)} dtype={actions.dtype} "
            f"finite={int(finite)} min={float(np.min(actions)):.4f} max={float(np.max(actions)):.4f}"
        )
        clipped = np.clip(np.asarray(actions, dtype=np.float64), low, high)
        for action in clipped:
            data.ctrl[left_act] = HOME_CTRL_LEFT
            data.ctrl[right_act] = action
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)
        q_ok = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        print(f"chunk={i} stepped={actions.shape[0]} qpos_finite={int(q_ok)}")
        if not q_ok:
            raise RuntimeError(f"non-finite state after chunk {i}")

    renderer.close()
    print(f"chunks={N_CHUNKS} crashed=0")


if __name__ == "__main__":
    main()
