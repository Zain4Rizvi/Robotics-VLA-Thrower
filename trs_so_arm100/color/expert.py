"""Scripted stack. It may read cube poses. The policy will not.

Cubes stay on bodies red_box and green_box. Recolor is geom rgba only.
State slots stay that body order: red xy, then green xy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import view  # noqa: E402
from openarm_vla.expert.ik import dls_ik  # noqa: E402

CUBE = 0.030
HOLD_STEPS = 15
OPEN = 0.42
CLOSED = -0.12
RELEASE = 0.35
OFFSET = np.array([-0.0162, -0.1004, 0.0])
TRAIN_COLORS = ("red", "green", "blue", "yellow")
HELD_COLORS = ("orange", "purple")
RGBA = {
    "red": np.array([0.8, 0.15, 0.1, 1.0]),
    "green": np.array([0.1, 0.7, 0.15, 1.0]),
    "blue": np.array([0.15, 0.25, 0.85, 1.0]),
    "yellow": np.array([0.9, 0.8, 0.1, 1.0]),
    "orange": np.array([0.9, 0.4, 0.05, 1.0]),
    "purple": np.array([0.55, 0.15, 0.7, 1.0]),
}
BODY = {"red": "red_box", "green": "green_box"}


def sentence(source: str, target: str) -> str:
    return f"stack the {source} cube on the {target} cube"


def other(body: str) -> str:
    return "green" if body == "red" else "red"


def mentions_heldout(text: str) -> bool:
    low = text.lower()
    return "orange" in low or "purple" in low


def self_check() -> None:
    raw = np.array([0.1, -0.2, -0.1, -0.22], np.float32)
    back = view.zscore_xy(raw[:2], raw[2:]) * view.XY_STD + view.XY_MEAN
    if not np.allclose(back, raw, atol=1.0e-5):
        raise SystemExit(f"xy roundtrip {back} vs {raw}")
    if sentence("red", "green") != "stack the red cube on the green cube":
        raise SystemExit("sentence")
    if sentence("green", "red") != "stack the green cube on the red cube":
        raise SystemExit("sentence")
    if set(TRAIN_COLORS) & set(HELD_COLORS):
        raise SystemExit("held-out colors overlap the training palette")
    print("self_check ok", flush=True)


class StackSim:
    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(view.SCENE))
        view.add_cameras(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.ik = mujoco.MjData(self.model)
        m = self.model
        self.home = m.key("home").id
        self.dofs = np.array([m.joint(n).dofadr[0] for n in view.JOINTS[:5]], dtype=int)
        self.jaw_body = int(m.body("Fixed_Jaw").id)
        self.red_body = int(m.body("red_box").id)
        self.green_body = int(m.body("green_box").id)
        self.red_geom = int(m.geom("red_box").id)
        self.green_geom = int(m.geom("green_box").id)
        self.floor_geom = int(m.geom("floor").id)
        self.red_q, self.green_q = 6, 13
        self.red_v, self.green_v = 6, 12
        self.fingers = set()
        for i in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or ""
            if body in ("Fixed_Jaw", "Moving_Jaw"):
                self.fingers.add(i)
        self.nsub = max(1, round((1 / view.CTRL_HZ) / m.opt.timestep))
        self.limits = np.stack([m.jnt_range[m.joint(n).id] for n in view.JOINTS])
        self.source = "red"
        self.colors = {"red": "red", "green": "green"}
        self.reset(0)
        self.ik.qpos[:] = self.data.qpos
        mujoco.mj_forward(m, self.ik)
        self.rot_down = self.ik.xmat[self.jaw_body].reshape(3, 3).copy()

    def paint(self, colors: dict[str, str]) -> None:
        for body, name in colors.items():
            gid = self.red_geom if body == "red" else self.green_geom
            self.model.geom_rgba[gid] = RGBA[name]

    def assert_paint_is_visual(self) -> None:
        floor_mat = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, int(self.model.geom_matid[self.floor_geom])
        )
        if floor_mat != "groundplane":
            raise SystemExit(f"floor material {floor_mat}")
        friction = {
            name: self.model.geom_friction[self.model.geom(name).id].copy() for name in ("red_box", "green_box")
        }
        mass = {name: float(self.model.body_mass[self.model.body(name).id]) for name in ("red_box", "green_box")}
        rgba = {name: self.model.geom_rgba[self.model.geom(name).id].copy() for name in ("red_box", "green_box")}
        self.paint({"red": "orange", "green": "purple"})
        for name in ("red_box", "green_box"):
            if not np.allclose(self.model.geom_friction[self.model.geom(name).id], friction[name]):
                raise SystemExit(f"{name} friction changed")
            if float(self.model.body_mass[self.model.body(name).id]) != mass[name]:
                raise SystemExit(f"{name} mass changed")
        if mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, int(self.model.geom_matid[self.floor_geom])
        ) != "groundplane":
            raise SystemExit("floor material changed")
        self.model.geom_rgba[self.red_geom] = rgba["red_box"]
        self.model.geom_rgba[self.green_geom] = rgba["green_box"]

    def reset(self, seed: int, source: str = "red", colors: dict[str, str] | None = None) -> None:
        if source not in BODY:
            raise SystemExit(f"source {source}")
        self.source = source
        self.colors = dict(colors or {"red": "red", "green": "green"})
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home)
        red, green = view.draw(np.random.default_rng(seed))
        self._place(self.red_q, self.red_v, red)
        self._place(self.green_q, self.green_v, green)
        self.paint(self.colors)
        for _ in range(20):
            self.step(self.data.qpos[:6].copy())

    def _place(self, qadr: int, vadr: int, xy: np.ndarray) -> None:
        self.data.qpos[qadr : qadr + 3] = (xy[0], xy[1], view.HALF + 0.001)
        self.data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[vadr : vadr + 6] = 0.0

    def step(self, ctrl: np.ndarray) -> None:
        q = np.clip(np.asarray(ctrl, dtype=np.float64), self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[:] = q
        for _ in range(self.nsub):
            mujoco.mj_step(self.model, self.data)

    def pinch(self) -> np.ndarray:
        d = self.data
        return d.xpos[self.jaw_body] + d.xmat[self.jaw_body].reshape(3, 3) @ OFFSET

    def _body(self, which: str) -> int:
        return self.red_body if which == "red" else self.green_body

    def _geom(self, which: str) -> int:
        return self.red_geom if which == "red" else self.green_geom

    def pos(self, which: str) -> np.ndarray:
        return self.data.xpos[self._body(which)].copy()

    def red_pos(self) -> np.ndarray:
        return self.pos("red")

    def green_pos(self) -> np.ndarray:
        return self.pos("green")

    def _touching(self, a: int, bset: set[int]) -> bool:
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == a and g2 in bset or g2 == a and g1 in bset:
                return True
        return False

    def holding(self) -> bool:
        return self._touching(self._geom(self.source), self.fingers)

    def on_target(self) -> bool:
        return self._touching(self._geom(self.source), {self._geom(other(self.source))})

    def degrees(self) -> np.ndarray:
        return view.qpos_to_degrees(self.data.qpos[:6]).astype(np.float32)

    def action_degrees(self, ctrl: np.ndarray) -> np.ndarray:
        return view.qpos_to_degrees(np.asarray(ctrl, dtype=np.float64)).astype(np.float32)

    def task(self) -> str:
        src = self.colors[self.source]
        tgt = self.colors[other(self.source)]
        return sentence(src, tgt)


def _ik(sim: StackSim, target: np.ndarray, max_dq: float) -> np.ndarray:
    sim.ik.qpos[:] = sim.data.qpos
    sim.ik.qvel[:] = 0.0
    q, _ = dls_ik(
        sim.model, sim.ik, sim.jaw_body, OFFSET, sim.dofs, target, sim.rot_down,
        iters=3, damping=2.0e-3, max_dq=max_dq, rot_weight=0.4,
    )
    return q


def _cmd(sim: StackSim, arm_q: np.ndarray, jaw: float) -> np.ndarray:
    ctrl = sim.data.ctrl.copy()
    ctrl[:5] = arm_q
    ctrl[5] = np.clip(jaw, ctrl[5] - 0.06, ctrl[5] + 0.06)
    return ctrl


def stacked(sim: StackSim) -> bool:
    src, tgt = sim.pos(sim.source), sim.pos(other(sim.source))
    xy = float(np.linalg.norm(src[:2] - tgt[:2]))
    gap = float((src[2] - view.HALF) - (tgt[2] + view.HALF))
    return xy < view.HALF and -0.004 < gap < 0.003 and tgt[2] < 0.025 and not sim.holding()


def sit_error(sim: StackSim) -> float:
    tgt = sim.pos(other(sim.source))
    goal = np.array([tgt[0], tgt[1], tgt[2] + CUBE])
    return float(np.linalg.norm(sim.pos(sim.source) - goal))


def run_episode(sim: StackSim, seed: int, source: str = "red", colors: dict[str, str] | None = None, on_step=None) -> dict:
    sim.reset(seed, source=source, colors=colors)
    src0 = sim.pos(sim.source).copy()
    phases = ("hover", "down", "close", "lift", "carry", "lower", "open", "clear", "hold")
    limit = {"hover": 90, "down": 90, "close": 30, "lift": 80, "carry": 160, "lower": 100, "open": 30, "clear": 40, "hold": 60}
    phase_i = 0
    phase_n = 0
    grasped = False
    dropped = False
    hold = 0
    retries = 0
    closest = sit_error(sim)
    lock = src0.copy()
    frozen = sim.data.ctrl.copy()
    clear_goal = src0.copy()
    clear_free = 0
    steps = 0

    while phase_i < len(phases):
        name = phases[phase_i]
        src, tgt = sim.pos(sim.source), sim.pos(other(sim.source))
        if name == "hover":
            target = np.array([src[0], src[1], src[2] + 0.06])
            jaw, slow = OPEN, 0.05
        elif name == "down":
            target = np.array([lock[0], lock[1], src[2]])
            jaw, slow = OPEN, 0.02
        elif name == "clear":
            if phase_n == 0:
                clear_goal = sim.pinch().copy()
                clear_goal[2] = float(src[2] + 0.045)
            target = clear_goal.copy()
            jaw, slow = 0.75, 0.012
        elif name in ("lift", "carry", "lower"):
            goal_z = {"lift": tgt[2] + 0.08, "carry": tgt[2] + 0.08, "lower": tgt[2] + CUBE}[name]
            goal_xy = lock[:2] if name == "lift" else tgt[:2]
            goal = np.array([goal_xy[0], goal_xy[1], goal_z])
            target = goal + (sim.pinch() - src)
            xy_err = float(np.linalg.norm(src[:2] - goal_xy))
            jaw = CLOSED
            slow = 0.015 if xy_err < 0.03 or name == "lower" else 0.035
        else:
            target = None
            jaw = CLOSED if name == "close" else RELEASE
            slow = 0.02
        if name == "open" and phase_n == 0:
            frozen = sim.data.ctrl.copy()
        if name in ("open", "hold"):
            ctrl = frozen.copy()
            if name != "hold":
                ctrl[5] = np.clip(jaw, sim.data.ctrl[5] - 0.05, sim.data.ctrl[5] + 0.05)
                frozen[5] = ctrl[5]
        elif target is None:
            ctrl = sim.data.qpos[:6].copy()
            ctrl[5] = np.clip(jaw, sim.data.ctrl[5] - 0.04, sim.data.ctrl[5] + 0.04)
        else:
            ctrl = _cmd(sim, _ik(sim, target, slow), jaw)
        if on_step is not None:
            on_step(sim, ctrl)
        steps += 1
        sim.step(ctrl)
        closest = min(closest, sit_error(sim))
        if sim.holding() and sim.pos(sim.source)[2] > src0[2] + 0.02:
            grasped = True
        if grasped and not sim.holding() and name not in ("open", "clear", "hold") and not stacked(sim):
            dropped = True
        done = False
        src = sim.pos(sim.source)
        tgt = sim.pos(other(sim.source))
        if name == "hover":
            done = np.linalg.norm(sim.pinch()[:2] - src[:2]) < 0.008 and sim.pinch()[2] > src[2] + 0.04
        elif name == "down":
            dist = float(np.linalg.norm(sim.pinch() - src))
            done = dist < 0.007
            # IK can stall a few millimetres outside the 7 mm gate while still on the cube.
            if not done and phase_n + 1 >= limit[name] and dist < 0.011:
                done = True
            if np.linalg.norm(src[:2] - lock[:2]) > 0.01 and retries < 2:
                retries += 1
                lock = src.copy()
                phase_i, phase_n = 0, 0
                continue
        elif name == "close":
            done = sim.holding() and phase_n > 10
            if phase_n >= limit[name] - 1 and retries < 2:
                retries += 1
                phase_i, phase_n = 0, 0
                continue
        elif name == "lift":
            if phase_n == 0:
                lock = src.copy()
            done = sim.holding() and src[2] > tgt[2] + 0.055
        elif name == "carry":
            xy = float(np.linalg.norm(src[:2] - tgt[:2]))
            high = src[2] > tgt[2] + 0.05
            done = xy < 0.01 and high
            # IK can stall a couple of millimetres outside the 1 cm gate while still over the cube.
            if not done and phase_n + 1 >= limit[name] and xy < view.HALF and high:
                done = True
        elif name == "lower":
            done = sim.on_target() and np.linalg.norm(src[:2] - tgt[:2]) < 0.012
        elif name == "open":
            done = phase_n > 12
        elif name == "clear":
            clear_free = clear_free + 1 if not sim.holding() else 0
            done = clear_free >= 5 and sim.pinch()[2] > clear_goal[2] - 0.008
        else:
            hold = hold + 1 if stacked(sim) else 0
            done = hold >= HOLD_STEPS
        phase_n += 1
        if done:
            if name == "hover":
                lock = sim.pos(sim.source).copy()
            if name == "clear":
                frozen = sim.data.ctrl.copy()
            phase_i += 1
            phase_n = 0
            continue
        if phase_n >= limit[name]:
            break

    ok = hold >= HOLD_STEPS
    src, tgt = sim.pos(sim.source), sim.pos(other(sim.source))
    return {
        "seed": seed,
        "success": ok,
        "grasped": grasped,
        "dropped": dropped and not ok,
        "phase": phases[min(phase_i, len(phases) - 1)],
        "closest_m": closest,
        "hold": hold,
        "steps": steps,
        "source": sim.source,
        "source_body": BODY[sim.source],
        "colors": dict(sim.colors),
        "task": sim.task(),
        "xy_cm": float(np.linalg.norm(src[:2] - tgt[:2]) * 100),
        "dz_cm": float((src[2] - tgt[2]) * 100),
    }


def gate(n: int = 20, seed0: int = 40000) -> int:
    self_check()
    sim = StackSim()
    sim.assert_paint_is_visual()
    n_ok = 0
    per = {"red": 0, "green": 0}
    for i, seed in enumerate(range(seed0, seed0 + n)):
        source = "red" if i < n // 2 else "green"
        row = run_episode(sim, seed, source=source)
        n_ok += int(row["success"])
        per[source] += int(row["success"])
        print(
            f"seed {seed}: success={row['success']} source={source} phase={row['phase']} "
            f"hold={row['hold']} closest_cm={row['closest_m'] * 100:.1f} steps={row['steps']}",
            flush=True,
        )
    half = n // 2
    print(f"gate {n_ok}/{n} red {per['red']}/{half} green {per['green']}/{half}", flush=True)
    return n_ok


if __name__ == "__main__":
    ok = gate()
    if ok < 18:
        raise SystemExit(f"expert gate {ok}/20")
