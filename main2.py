"""Interactive viewer for the SO-ARM100. Move each joint from the Control sliders."""

from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent / "trs_so_arm100" / "scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for name in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"):
        j = model.joint(name)
        lo, hi = model.jnt_range[j.id]
        print(f"{name}: {lo:.3f} .. {hi:.3f} rad")
    print("Control sliders are on the right. Space pauses.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
