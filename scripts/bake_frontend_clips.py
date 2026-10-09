"""Record kinematic tapes for the static frontend. CPU only.

Writes frontend/public/clips and a copy of the scene XML and meshes under
frontend/public/mujoco. Playback steps the recorded qpos. It does not run physics.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "frontend" / "public"
MUJOCO = PUBLIC / "mujoco"
CLIPS = PUBLIC / "clips"

sys.path.insert(0, str(REPO / "trs_so_arm100" / "color"))
sys.path.insert(0, str(REPO / "trs_so_arm100"))
sys.path.insert(0, str(REPO / "peg_socket"))
sys.path.insert(0, str(REPO))

from expert import StackSim, run_episode  # noqa: E402
import view  # noqa: E402


RED = "stack the red cube on the green cube"
GREEN = "stack the green cube on the red cube"


def _rel(path: Path) -> str:
    return path.relative_to(MUJOCO).as_posix()


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".xml":
        text = src.read_text(encoding="utf-8")
        text = text.replace('shadowsize="8192"', 'shadowsize="2048"')
        dest.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(src, dest)


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _inject_stack_cameras() -> None:
    mat = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    quat_text = " ".join(f"{value:.8g}" for value in quat)
    arm = MUJOCO / "trs_so_arm100" / "so_arm100.xml"
    arm_text = arm.read_text(encoding="utf-8")
    needle = '<body name="Fixed_Jaw" pos="0 -0.0601 0" euler="0 1.57079 0">'
    camera = (
        f'{needle}\n'
        f'                <camera name="camera2" pos="0 -0.025 0.02" quat="{quat_text}" fovy="65"/>'
    )
    if needle not in arm_text:
        raise SystemExit("Fixed_Jaw body not found in so_arm100.xml")
    arm.write_text(arm_text.replace(needle, camera, 1), encoding="utf-8")

    scene = MUJOCO / "trs_so_arm100" / "scene.xml"
    scene_text = scene.read_text(encoding="utf-8")
    scene_text = scene_text.replace(
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n'
        '    <camera name="camera1" pos="0 -0.18 0.48" fovy="55"/>',
        1,
    )
    scene.write_text(scene_text, encoding="utf-8")


def stage_assets() -> dict[str, list[str]]:
    if MUJOCO.exists():
        shutil.rmtree(MUJOCO)
    mapping = {
        REPO / "trs_so_arm100" / "scene.xml": MUJOCO / "trs_so_arm100" / "scene.xml",
        REPO / "trs_so_arm100" / "so_arm100.xml": MUJOCO / "trs_so_arm100" / "so_arm100.xml",
        REPO / "v2" / "pedestal" / "throw_multi_scene.xml": MUJOCO / "v2" / "pedestal" / "throw_multi_scene.xml",
        REPO / "v2" / "pedestal" / "openarm_pedestal.xml": MUJOCO / "v2" / "pedestal" / "openarm_pedestal.xml",
        REPO / "v2" / "openarm_bimanual.xml": MUJOCO / "v2" / "openarm_bimanual.xml",
        REPO / "peg_socket" / "peg_socket_scene.xml": MUJOCO / "peg_socket" / "peg_socket_scene.xml",
    }
    for src, dest in mapping.items():
        _copy_file(src, dest)
    _copy_tree(REPO / "trs_so_arm100" / "assets", MUJOCO / "trs_so_arm100" / "assets")
    _copy_tree(REPO / "v2" / "assets", MUJOCO / "v2" / "assets")
    _inject_stack_cameras()
    files = sorted(_rel(path) for path in MUJOCO.rglob("*") if path.is_file())
    groups = {
        "stack": [name for name in files if name.startswith("trs_so_arm100/")],
        "openarm": [name for name in files if name.startswith("v2/")],
        "peg": [name for name in files if name.startswith("v2/") or name.startswith("peg_socket/")],
    }
    (MUJOCO / "files.json").write_text(json.dumps(groups), encoding="utf-8")
    return groups


def _snapshot(model) -> tuple[list[dict], list[dict]]:
    geoms = []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if not name:
            continue
        geoms.append({"name": name, "rgba": [round(float(value), 5) for value in model.geom_rgba[index]]})
    bodies = []
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or ""
        if not name:
            continue
        bodies.append(
            {
                "name": name,
                "pos": [round(float(value), 6) for value in model.body_pos[index]],
                "quat": [round(float(value), 6) for value in model.body_quat[index]],
            }
        )
    return geoms, bodies


def _write_tape(clip_id: str, frames: list[np.ndarray]) -> dict:
    tape = np.stack(frames).astype(np.float32)
    path = CLIPS / f"{clip_id}.bin"
    path.write_bytes(tape.tobytes())
    return {"file": f"clips/{clip_id}.bin", "frames": int(tape.shape[0]), "nq": int(tape.shape[1])}


def bake_stack() -> list[dict]:
    sim = StackSim()
    frames: list[np.ndarray] = []
    original = sim.step

    def wrapped(ctrl: np.ndarray) -> None:
        original(ctrl)
        frames.append(np.asarray(sim.data.qpos, dtype=np.float32).copy())

    sim.step = wrapped
    clips = []
    jobs = (("red", RED, 50000), ("green", GREEN, 51000))
    for source, instruction, seed0 in jobs:
        found = 0
        seed = seed0
        while found < 4:
            if seed > seed0 + 40:
                raise SystemExit(f"only {found} successes for {instruction}")
            frames.clear()
            row = run_episode(sim, seed, source=source, colors={"red": "red", "green": "green"})
            print(
                f"stack seed {seed} source={source} success={row['success']} steps={row['steps']}",
                flush=True,
            )
            if row["success"] and row["task"] == instruction and frames:
                clip_id = f"stack_{source}_{found}"
                geoms, bodies = _snapshot(sim.model)
                clips.append(
                    {
                        "id": clip_id,
                        "instruction": instruction,
                        "seed": seed,
                        "rgba": geoms,
                        "bodies": bodies,
                        **_write_tape(clip_id, frames),
                    }
                )
                found += 1
            seed += 1
    return clips


def bake_throw() -> list[dict]:
    from openarm_vla.config import EnvConfig
    from openarm_vla.env.throw_env import ThrowEnv
    from openarm_vla.expert.throw_expert import ThrowExpert

    env = ThrowEnv(EnvConfig(render=False), render_mode=None)
    expert = ThrowExpert()
    clips = []
    seed = 1
    while len(clips) < 8:
        if seed > 40:
            raise SystemExit(f"only {len(clips)} throw successes")
        env.reset(seed=seed)
        expert.reset(env)
        frames = [np.asarray(env.data.qpos, dtype=np.float32).copy()]
        success = False
        for _ in range(env.cfg.max_episode_steps):
            _obs, reward, terminated, truncated, _info = env.step(expert.act(env))
            frames.append(np.asarray(env.data.qpos, dtype=np.float32).copy())
            if terminated and reward > 0:
                success = True
                break
            if truncated:
                break
        print(f"throw seed {seed} success={success} frames={len(frames)}", flush=True)
        if success:
            clip_id = f"throw_{len(clips)}"
            geoms, bodies = _snapshot(env.model)
            clips.append(
                {
                    "id": clip_id,
                    "instruction": str(env.task["instruction"]),
                    "seed": seed,
                    "rgba": geoms,
                    "bodies": bodies,
                    **_write_tape(clip_id, frames),
                }
            )
        seed += 1
    env.close()
    return clips


def bake_peg() -> list[dict]:
    from eval_peg_expert import draw, prepare
    from peg_expert import HOLD_STEPS, PegExpert, PegSim

    sim = PegSim()
    expert = PegExpert()
    clips = []
    seed = 1
    while len(clips) < 4:
        if seed > 80:
            raise SystemExit(f"only {len(clips)} peg successes")
        rng = np.random.default_rng(seed)
        peg_xy, sock_xy = draw(rng)
        issue = prepare(sim, peg_xy, sock_xy)
        if issue or not expert.reset(sim):
            print(f"peg seed {seed} skipped", flush=True)
            seed += 1
            continue
        frames = [np.asarray(sim.data.qpos, dtype=np.float32).copy()]
        hold = 0
        inserted = False
        for action in expert._actions:
            sim.step(action)
            frames.append(np.asarray(sim.data.qpos, dtype=np.float32).copy())
            if sim.finger_qpos() < -0.45 and sim.seated():
                hold += 1
                inserted = hold >= HOLD_STEPS
            else:
                hold = 0
        print(f"peg seed {seed} inserted={inserted} frames={len(frames)}", flush=True)
        if inserted:
            clip_id = f"peg_{len(clips)}"
            geoms, bodies = _snapshot(sim.model)
            clips.append(
                {
                    "id": clip_id,
                    "instruction": "place the peg in the hole",
                    "seed": seed,
                    "rgba": geoms,
                    "bodies": bodies,
                    **_write_tape(clip_id, frames),
                }
            )
        seed += 1
    return clips


def main() -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    print("staging meshes", flush=True)
    stage_assets()
    catalog = {
        "stack": {
            "xml": "trs_so_arm100/scene.xml",
            "fps": view.CTRL_HZ,
            "cameras": ["camera1", "camera2"],
            "lookat": [0.0, -0.18, 0.14],
            "distance": 0.72,
            "azimuth": 132.0,
            "elevation": -22.0,
            "clips": bake_stack(),
        },
        "openarm": {
            "xml": "v2/pedestal/throw_multi_scene.xml",
            "fps": 50,
            "cameras": ["headcam", "camera_wrist_right"],
            "lookat": [0.4, -0.2, 0.35],
            "distance": 1.7,
            "azimuth": 130.0,
            "elevation": -24.0,
            "clips": bake_throw(),
        },
        "peg": {
            "xml": "peg_socket/peg_socket_scene.xml",
            "fps": 50,
            "cameras": ["tablecam", "camera_wrist_right"],
            "lookat": [0.28, -0.28, 0.4],
            "distance": 1.35,
            "azimuth": 140.0,
            "elevation": -28.0,
            "clips": bake_peg(),
        },
    }
    (CLIPS / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    print("wrote frontend/public/clips/catalog.json", flush=True)


if __name__ == "__main__":
    main()
