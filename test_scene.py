"""Open the MuJoCo viewer on one scene from v2/pedestal.

    uv run python test_scene.py throw_scene
    uv run python test_scene.py v2/pedestal/catch_scene.xml
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

SCENES = Path(__file__).resolve().parent / "v2" / "pedestal"


def scene_path(name: str) -> Path:
    path = Path(name)
    if not path.is_file():
        path = SCENES / name
    if path.suffix != ".xml":
        path = path.with_suffix(".xml")
    return path


def main():
    if len(sys.argv) != 2:
        names = "\n".join(sorted(p.name for p in SCENES.glob("*.xml")))
        raise SystemExit(f"usage: python test_scene.py <scene>\n{names}")
    xml = scene_path(sys.argv[1])
    if not xml.is_file():
        raise SystemExit(f"no such scene: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    mujoco.viewer.launch(model, mujoco.MjData(model))


if __name__ == "__main__":
    main()
