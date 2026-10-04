"""Reset the peg-socket scene copy to keyframe home and save the two cameras."""

from pathlib import Path

import mujoco
import numpy as np
from imageio.v2 import imwrite

XML = Path(__file__).resolve().parent / "peg_socket_scene.xml"
OUT = Path(__file__).resolve().parent / "findings" / "scene"
HOME_ARM = np.array(
    [0.0, 0.0, 0.0, 1.570796, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    key_id = int(model.key("home").id)
    print(f"key_name=home key_id={key_id} nq={model.nq} nv={model.nv}")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    qerr = float(np.max(np.abs(data.qpos - model.key_qpos[key_id])))
    left_err = float(np.max(np.abs(data.qpos[0:9] - HOME_ARM)))
    right_err = float(np.max(np.abs(data.qpos[9:18] - HOME_ARM)))
    print(f"qpos_max_abs_err={qerr:.3e} left_arm_err={left_err:.3e} right_arm_err={right_err:.3e}")
    print(f"left_qpos={np.array2string(data.qpos[0:9], precision=6)}")
    print(f"right_qpos={np.array2string(data.qpos[9:18], precision=6)}")

    jid = int(model.joint("socket_free").id)
    adr = int(model.jnt_qposadr[jid])
    body = int(model.body("socket").id)
    home_xpos = data.xpos[body].copy()
    print(f"socket_free_qposadr={adr} home_xpos={np.array2string(home_xpos, precision=6)}")

    data.qpos[adr] += 0.08
    mujoco.mj_forward(model, data)
    moved = data.xpos[body].copy()
    delta = moved - home_xpos
    print(f"moved_xpos={np.array2string(moved, precision=6)} delta={np.array2string(delta, precision=6)}")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    restored = data.xpos[body].copy()
    restore_err = float(np.max(np.abs(restored - home_xpos)))
    print(f"restored_xpos={np.array2string(restored, precision=6)} restore_err={restore_err:.3e}")

    peg = data.xpos[int(model.body("peg").id)]
    print(f"peg_xpos={np.array2string(peg, precision=6)}")
    weld = int(model.eq("grasp_right_peg").id)
    print(f"grasp_right_peg_active={int(data.eq_active[weld])}")

    renderer = mujoco.Renderer(model, height=256, width=256)
    for cam, fname in (("tablecam", "tablecam.png"), ("camera_wrist_right", "wrist.png")):
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        imwrite(OUT / fname, rgb)
        print(f"{cam} shape={tuple(rgb.shape)} dtype={rgb.dtype} mean={float(rgb.mean()):.2f} -> {OUT / fname}")
    renderer.close()


if __name__ == "__main__":
    main()
