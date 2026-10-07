"""MuJoCo viewer for trs_so_arm100/best, the best stacking checkpoint.

Type an instruction and Enter. r draws a new cube layout and homes the arm. q quits.
Cube xy is filled into state dims 6:10 the same way the checkpoint was trained.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene.xml"
JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float64)
CTRL_HZ = 30
REPLAN_EVERY = 10
INSTRUCTION = "stack the red cube on the green cube"
HALF = 0.015
X_RANGE = (-0.12, 0.12)
Y_RANGE = (-0.28, -0.15)
MIN_SEP = 0.08
# Train-label mean and std of (red_x, red_y, green_x, green_y), from see/labels/train.npz.
XY_MEAN = np.array([0.0071532782, -0.21075277, -0.0050198748, -0.21059522], np.float32)
XY_STD = np.array([0.073098943, 0.035696767, 0.069526479, 0.038455848], np.float32)


def add_cameras(spec: mujoco.MjSpec) -> None:
    top = spec.worldbody.add_camera()
    top.name = "camera1"
    top.pos = [0, -0.18, 0.48]
    top.fovy = 55
    mat = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    wrist = spec.body("Fixed_Jaw").add_camera()
    wrist.name = "camera2"
    wrist.pos = [0.0, -0.025, 0.02]
    wrist.quat = quat
    wrist.fovy = 65


def qpos_to_degrees(qpos: np.ndarray) -> np.ndarray:
    return SIGNS * np.degrees(qpos)


def degrees_to_ctrl(degrees: np.ndarray, limits: np.ndarray) -> np.ndarray:
    q = np.radians(SIGNS * degrees)
    return np.clip(q, limits[:, 0], limits[:, 1]).astype(np.float64)


def draw(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([X_RANGE[0], Y_RANGE[0]])
    hi = np.array([X_RANGE[1], Y_RANGE[1]])
    for _ in range(1000):
        red = rng.uniform(lo, hi)
        green = rng.uniform(lo, hi)
        if np.linalg.norm(red - green) >= MIN_SEP:
            return red, green
    raise RuntimeError("no separated cube pair")


class StackSim:
    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(SCENE))
        add_cameras(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.home = m.key("home").id
        self.red_body = int(m.body("red_box").id)
        self.green_body = int(m.body("green_box").id)
        self.red_q, self.green_q = 6, 13
        self.red_v, self.green_v = 6, 12
        self.nsub = max(1, round((1 / CTRL_HZ) / m.opt.timestep))
        self.limits = np.stack([m.jnt_range[m.joint(n).id] for n in JOINTS])
        self.reset(0)

    def reset(self, seed: int) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home)
        red, green = draw(np.random.default_rng(seed))
        self._place(self.red_q, self.red_v, red)
        self._place(self.green_q, self.green_v, green)
        for _ in range(20):
            self.step(self.data.qpos[:6].copy())

    def _place(self, qadr: int, vadr: int, xy: np.ndarray) -> None:
        self.data.qpos[qadr : qadr + 3] = (xy[0], xy[1], HALF + 0.001)
        self.data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[vadr : vadr + 6] = 0.0

    def step(self, ctrl: np.ndarray) -> None:
        q = np.clip(np.asarray(ctrl, dtype=np.float64), self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[:] = q
        for _ in range(self.nsub):
            mujoco.mj_step(self.model, self.data)

    def red_pos(self) -> np.ndarray:
        return self.data.xpos[self.red_body].copy()

    def green_pos(self) -> np.ndarray:
        return self.data.xpos[self.green_body].copy()

    def degrees(self) -> np.ndarray:
        return qpos_to_degrees(self.data.qpos[:6]).astype(np.float32)


def zscore_xy(red, green) -> np.ndarray:
    raw = np.concatenate([np.asarray(red, np.float32), np.asarray(green, np.float32)], axis=-1)
    return ((raw - XY_MEAN) / XY_STD).astype(np.float32)


def write_xy(state, scored):
    out = state.new_zeros(*state.shape[:-1], 32)
    out[..., :6] = state[..., :6]
    xy = scored.to(dtype=state.dtype, device=state.device).reshape(scored.shape[0], *([1] * (out.ndim - 2)), 4)
    out[..., 6:10] = xy
    return out


def load_policy(path: Path, device: str):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(path)).to(device).float().eval()
    pre, post = make_pre_post_processors(
        policy.config,
        str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def predict(policy, pre, post, images: dict[str, np.ndarray], degrees: np.ndarray, task: str, xy: np.ndarray) -> np.ndarray:
    import torch

    obs = {
        "observation.state": torch.from_numpy(degrees.astype(np.float32)),
        "task": task,
    }
    for name, img in images.items():
        obs[f"observation.images.{name}"] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    batch = pre(obs)
    state = batch["observation.state"]
    scored = torch.as_tensor(xy, dtype=state.dtype, device=state.device).reshape(1, 4)
    batch["observation.state"] = write_xy(state, scored)
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    return post(actions)[0].detach().cpu().numpy()


def read_stdin(cmds: queue.Queue) -> None:
    try:
        while True:
            cmds.put(input().strip())
    except EOFError:
        cmds.put("q")


def main() -> None:
    import torch

    ckpt = ROOT / "best" / "pretrained_model"
    if not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"missing {ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {ckpt} on {device} ...", flush=True)
    policy, pre, post = load_policy(ckpt, device)
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    rng = np.random.default_rng()

    def new_layout() -> None:
        seed = int(rng.integers(0, 2**31))
        sim.reset(seed)
        policy.reset()
        print(f"layout seed {seed}", flush=True)

    new_layout()
    print("type an instruction + Enter, 'r' = new layout, 'q' = quit", flush=True)
    print(f"instruction -> {INSTRUCTION!r}", flush=True)

    cmds: queue.Queue = queue.Queue()
    threading.Thread(target=read_stdin, args=(cmds,), daemon=True).start()
    task = INSTRUCTION
    buf, ptr = None, 0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            cmd = None if cmds.empty() else cmds.get()
            if cmd == "q":
                break
            if cmd == "r":
                new_layout()
                buf = None
            elif cmd:
                task = cmd
                policy.reset()
                buf = None
                print(f"instruction -> {task!r}", flush=True)

            if task and (buf is None or ptr >= REPLAN_EVERY or ptr >= len(buf)):
                images = {}
                for name in ("camera1", "camera2"):
                    renderer.update_scene(sim.data, camera=name)
                    images[name] = renderer.render().copy()
                xy = zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2])
                buf = predict(policy, pre, post, images, sim.degrees(), task, xy)
                ptr = 0
            if buf is not None:
                ptr += 1
                sim.step(degrees_to_ctrl(buf[ptr - 1], sim.limits))
            viewer.sync()
            time.sleep(1 / CTRL_HZ)
    renderer.close()


if __name__ == "__main__":
    main()
