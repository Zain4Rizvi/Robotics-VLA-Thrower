"""Open-loop peg-in-hole expert.

Plans a joint-target sequence from the true peg and socket poses, then replays it.
Phases: hover, open, descend, close, lift, carry, lower through the hole, open, retreat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from openarm_vla.constants import (
    GRASP_OFFSET,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_CTRL_LEFT,
    RIGHT_ACTUATORS,
    RIGHT_JOINTS,
    THROW_READY_RIGHT,
)
from openarm_vla.expert.ballistic import quintic_interp
from openarm_vla.expert.ik import dls_ik

XML = Path(__file__).resolve().parent / "peg_socket_scene.xml"

HZ = 50
N_SUBSTEPS = 20
TABLE_Z = 0.40
PEG_HALF = 0.045
PEG_R = 0.014
SOCKET_OUTER = 0.034
# Inner apothem 20 mm minus peg radius 14 mm. A seated peg can sit against a wall.
INSERT_R = 0.007
HOLD_STEPS = 25  # 0.5 s at 50 Hz
# Right-arm workspace on the table. Home gripper occupies y around -0.15.
X_RANGE = (0.18, 0.36)
Y_RANGE = (-0.45, -0.18)
MIN_SEP = 0.12  # peg center to socket center; open fingers clear the octagon
CLEARANCE = 0.01  # gap to pedestal and the home grippers
Q_READY = np.asarray(THROW_READY_RIGHT, dtype=np.float64)


def _column(aim: np.ndarray, z_top: float) -> tuple[list[np.ndarray], list[float]]:
    """Vertical waypoints at aim xy, from the carry height down to aim z."""
    z_bot = float(aim[2])
    zs = [float(z_top)]
    z = float(z_top)
    while z - 0.01 > z_bot:
        z -= 0.01
        zs.append(z)
    if abs(zs[-1] - z_bot) > 1.0e-4:
        zs.append(z_bot)
    return [np.array([aim[0], aim[1], z]) for z in zs], zs

LEFT_ACTUATORS = (
    "left_joint1_ctrl",
    "left_joint2_ctrl",
    "left_joint3_ctrl",
    "left_joint4_ctrl",
    "left_joint5_ctrl",
    "left_joint6_ctrl",
    "left_joint7_ctrl",
    "left_finger1_ctrl",
)


@dataclass
class ExpertConfig:
    approach_height: float = 0.08  # hover above the grasp point
    grasp_above_table: float = 0.070  # fingertips stay above the 5 cm walls
    carry_clearance: float = 0.12  # peg bottom this far above the table while carrying
    lower_fast: float = 0.03  # m/s until the peg nears the rim
    lower_slow: float = 0.008  # m/s through the hole
    slow_band: float = 0.05  # last metres use lower_slow
    ik_iters: int = 120
    ik_damping: float = 1.0e-4
    max_dq: float = 0.12
    rot_weight: float = 0.3
    ik_tol: float = 0.003


class PegSim:
    """Scene loader for the peg-socket copy. Not a shared env."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.right_act = np.array([m.actuator(n).id for n in RIGHT_ACTUATORS], dtype=int)
        self.left_act = np.array([m.actuator(n).id for n in LEFT_ACTUATORS], dtype=int)
        self.jnt_qpos = np.array([m.joint(n).qposadr[0] for n in RIGHT_JOINTS], dtype=int)
        self.jnt_dof = np.array([m.joint(n).dofadr[0] for n in RIGHT_JOINTS], dtype=int)
        self.ee = int(m.body("openarm_right_ee_base_link").id)
        self.offset = np.array(GRASP_OFFSET, dtype=np.float64)
        self.peg = int(m.body("peg").id)
        self.socket = int(m.body("socket").id)
        self.peg_geom = int(m.geom("peg_geom").id)
        self.socket_geoms = [int(m.geom(f"socket_{i}").id) for i in range(8)]
        self.peg_qadr = int(m.joint("peg_free").qposadr[0])
        self.peg_dadr = int(m.joint("peg_free").dofadr[0])
        self.sock_qadr = int(m.joint("socket_free").qposadr[0])
        self.sock_dadr = int(m.joint("socket_free").dofadr[0])
        self.finger_qadr = int(m.joint("openarm_right_finger_joint1").qposadr[0])
        self.finger_bodies = {
            int(m.body(n).id)
            for n in ("openarm_right_ee_inner_finger", "openarm_right_ee_outer_finger")
        }
        self.weld = int(m.eq("grasp_right_peg").id)
        self.home_key = int(m.key("home").id)
        self.act_low = np.array([m.actuator_ctrlrange[i, 0] for i in self.right_act])
        self.act_high = np.array([m.actuator_ctrlrange[i, 1] for i in self.right_act])
        self.home_left = np.array(HOME_CTRL_LEFT, dtype=np.float64)
        arm = []
        for i in range(m.ngeom):
            if m.geom_contype[i] == 0:
                continue
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or ""
            if body.startswith("openarm_"):
                arm.append(i)
        ped = int(m.geom("openarm_body_link0_collision").id)
        self.obstacles = np.array([ped, *arm], dtype=int)
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_n = mujoco.mj_stateSize(m, self._state_spec)
        self.q_ready = np.asarray(THROW_READY_RIGHT, dtype=np.float64)
        self.reset_home()
        saved = self.data.qpos.copy()
        self.data.qpos[self.jnt_qpos] = self.q_ready
        mujoco.mj_forward(m, self.data)
        self.rot_down = self.data.xmat[self.ee].reshape(3, 3).copy()
        self.x0 = float(np.arctan2(self.rot_down[1, 0], self.rot_down[0, 0]))
        self.data.qpos[:] = saved
        mujoco.mj_forward(m, self.data)

    def reset_home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.data.eq_active[self.weld] = 0
        self.data.ctrl[self.left_act] = self.home_left
        mujoco.mj_forward(self.model, self.data)

    def place(self, peg_xy: np.ndarray, sock_xy: np.ndarray) -> None:
        self._set_free(self.peg_qadr, self.peg_dadr, peg_xy, TABLE_Z + PEG_HALF)
        self._set_free(self.sock_qadr, self.sock_dadr, sock_xy, TABLE_Z)
        mujoco.mj_forward(self.model, self.data)

    def _set_free(self, qadr: int, dadr: int, xy: np.ndarray, z: float) -> None:
        self.data.qpos[qadr : qadr + 3] = (xy[0], xy[1], z)
        self.data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dadr : dadr + 6] = 0.0

    def settle(self, n: int = 15) -> None:
        hold = np.append(self.right_qpos(), GRIPPER_CLOSED)
        for _ in range(n):
            self.step(hold)

    def step(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float64), self.act_low, self.act_high)
        self.data.ctrl[self.left_act] = self.home_left
        self.data.ctrl[self.right_act] = action
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    def save(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.zeros(self._state_n)
        mujoco.mj_getState(self.model, self.data, s, self._state_spec)
        return s, self.data.ctrl.copy()

    def restore(self, saved: tuple[np.ndarray, np.ndarray]) -> None:
        s, ctrl = saved
        mujoco.mj_setState(self.model, self.data, s, self._state_spec)
        self.data.ctrl[:] = ctrl
        self.data.eq_active[self.weld] = 0
        mujoco.mj_forward(self.model, self.data)

    def right_qpos(self) -> np.ndarray:
        return self.data.qpos[self.jnt_qpos].copy()

    def finger_qpos(self) -> float:
        return float(self.data.qpos[self.finger_qadr])

    def peg_pos(self) -> np.ndarray:
        return self.data.xpos[self.peg].copy()

    def socket_pos(self) -> np.ndarray:
        return self.data.xpos[self.socket].copy()

    def grasp_point(self) -> np.ndarray:
        d = self.data
        return d.xpos[self.ee] + d.xmat[self.ee].reshape(3, 3) @ self.offset

    def peg_up(self) -> float:
        return float(self.data.xmat[self.peg].reshape(3, 3)[2, 2])

    def holding(self) -> bool:
        touching = set()
        g = self.peg_geom
        for c in self.data.contact[: self.data.ncon]:
            if g in (c.geom1, c.geom2):
                other = int(c.geom2 if c.geom1 == g else c.geom1)
                touching.add(int(self.model.geom_bodyid[other]))
        return self.finger_bodies <= touching

    def finger_lowest_z(self) -> float:
        """World z of the lowest vertex on the right finger collision meshes."""
        m, d = self.model, self.data
        low = np.inf
        for i in range(m.ngeom):
            if int(m.geom_bodyid[i]) not in self.finger_bodies or m.geom_contype[i] == 0:
                continue
            mesh_id = int(m.geom_dataid[i])
            if mesh_id < 0:
                low = min(low, float(d.geom_xpos[i, 2] - m.geom_rbound[i]))
                continue
            va = int(m.mesh_vertadr[mesh_id])
            vn = int(m.mesh_vertnum[mesh_id])
            verts = m.mesh_vert[va : va + vn]
            rotated = verts @ d.geom_xmat[i].reshape(3, 3).T
            low = min(low, float(rotated[:, 2].min() + d.geom_xpos[i, 2]))
        return float(low)

    def fingers_touch_socket(self) -> bool:
        socks = set(self.socket_geoms)
        for c in self.data.contact[: self.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if not (pair & socks):
                continue
            other = (pair - socks).pop() if len(pair & socks) == 1 else None
            if other is None:
                continue
            if int(self.model.geom_bodyid[other]) in self.finger_bodies:
                return True
        return False

    def seated(self) -> bool:
        """Peg center inside the socket opening, upright, about 6 mm of the hole axis."""
        peg = self.peg_pos()
        sock = self.socket_pos()
        radial = float(np.linalg.norm(peg[:2] - sock[:2]))
        z_rel = float(peg[2] - sock[2])
        return radial <= INSERT_R and 0.040 <= z_rel <= 0.051 and self.peg_up() > 0.97

    def hole_dist(self) -> float:
        return float(np.linalg.norm(self.peg_pos()[:2] - self.socket_pos()[:2]))

    def _gap(self, geom: int, other: int) -> float:
        fromto = np.zeros(6)
        return float(mujoco.mj_geomDistance(self.model, self.data, geom, other, 0.25, fromto))

    def layout_issue(self, peg_xy: np.ndarray, sock_xy: np.ndarray) -> str | None:
        """None when the draw is legal. Checked after place(), before the expert runs."""
        sep = float(np.linalg.norm(np.asarray(peg_xy) - np.asarray(sock_xy)))
        if sep < MIN_SEP:
            return "too_close"
        if sep < PEG_R + 0.020:
            return "already_seated"
        objects = [self.peg_geom, *self.socket_geoms]
        for g in objects:
            for o in self.obstacles:
                if self._gap(g, int(o)) < CLEARANCE:
                    return "pedestal_or_gripper"
        # On the tabletop: centers over the top, upright, bottoms at the surface.
        peg, sock = self.peg_pos(), self.socket_pos()
        if abs(peg[2] - (TABLE_Z + PEG_HALF)) > 0.01 or abs(sock[2] - TABLE_Z) > 0.01:
            return "off_table"
        if self.peg_up() < 0.98:
            return "peg_tipped"
        x_lo, x_hi = 0.47 - 0.41, 0.47 + 0.41
        y_lo, y_hi = -0.55, 0.55
        for p, margin in ((peg, PEG_R + 0.005), (sock, SOCKET_OUTER + 0.005)):
            if not (x_lo + margin <= p[0] <= x_hi - margin and y_lo + margin <= p[1] <= y_hi - margin):
                return "off_table"
        return None


class PegExpert:
    def __init__(self, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        self._actions: list[np.ndarray] = []
        self._i = 0
        self.reachable = False
        self.ik_err = 1.0
        self.release_step = 0
        self.prefix_len = 0
        self.offset = np.zeros(3)
        self.finger_drop = 0.02
        self.yaw = 0.0
        self.planned_peg = np.zeros(3)
        self.planned_sock = np.zeros(3)
        self.finger_drop = 0.0

    def reset(self, sim: PegSim) -> bool:
        cfg = self.cfg
        saved = sim.save()
        q0 = sim.right_qpos()
        peg = sim.peg_pos()
        sock = sim.socket_pos()
        dz = cfg.grasp_above_table - PEG_HALF
        grasp = peg + np.array([0.0, 0.0, dz])
        hover = grasp + np.array([0.0, 0.0, cfg.approach_height])
        lift = grasp + np.array([0.0, 0.0, cfg.carry_clearance])
        above = np.array([sock[0], sock[1], lift[2]])
        carry = []
        for t in (1 / 3, 2 / 3, 1.0):
            xy = lift[:2] * (1.0 - t) + above[:2] * t
            carry.append(np.array([xy[0], xy[1], lift[2]]))
        points = [hover, grasp, lift, *carry]
        qs, err, yaw = self._search(sim, points)
        self.ik_err = err
        self.yaw = yaw
        sim.restore(saved)
        if qs is None or err > cfg.ik_tol:
            self.reachable = False
            self._actions = []
            return False

        q_hover, q_grasp, q_lift = qs[0], qs[1], qs[2]
        q_above = qs[-1]
        prefix = self._approach(q0, q_hover, q_grasp, q_lift, q_above)
        for a in prefix:
            sim.step(a)
        held = sim.holding()
        self.offset = sim.peg_pos() - sim.grasp_point()
        self.finger_drop = float(sim.grasp_point()[2] - sim.finger_lowest_z())
        if not held:
            q_now = sim.right_qpos()
            suffix = self._hold(q_now, 0.4, GRIPPER_OPEN)
            self.release_step = len(prefix)
            self.prefix_len = len(prefix)
            self.planned_peg = sim.peg_pos().copy()
            self.planned_sock = sim.socket_pos().copy()
            self._actions = prefix + suffix
            self.reachable = True
            sim.restore(saved)
            self._i = 0
            return True

        mid = sim.save()
        sock = sim.socket_pos()
        sock_xy = sock[:2].copy()
        desired = np.array([sock[0], sock[1], sock[2] + PEG_HALF])
        seat = desired - self.offset
        # Stop while the finger meshes are still above the 50 mm walls.
        z_clear = float(sock[2] + 0.050 + 0.010 + self.finger_drop)
        z_top = float(sim.grasp_point()[2])
        z_bot = max(z_clear, float(seat[2]))
        aim = np.array([seat[0], seat[1], z_bot])
        rot = self._rot(sim, yaw)
        q_seed = sim.right_qpos()
        best = (np.inf, None, aim.copy(), err, 0)
        for _ in range(4):
            sim.restore(mid)
            pts, zs = _column(aim, z_top)
            q_lower, low_err = self._chain(sim, pts, rot, q_seed)
            if q_lower is None or low_err > cfg.ik_tol:
                break
            retreat_pt = np.array([aim[0], aim[1], float(aim[2]) + cfg.carry_clearance])
            q_retreat, ret_err = self._chain(sim, [retreat_pt], rot, q_lower[-1])
            if q_retreat is None or ret_err > cfg.ik_tol:
                break
            suffix = self._lower(q_above, q_lower, zs, float(aim[2]), q_retreat[0])
            closed = self._lower_until_open
            sim.restore(mid)
            for a in suffix[:closed]:
                sim.step(a)
            peg_err = sim.peg_pos()[:2] - sock_xy
            sag_z = float(sim.grasp_point()[2] - aim[2])
            miss = float(np.linalg.norm(peg_err))
            if miss < best[0]:
                best = (miss, suffix, aim.copy(), max(err, low_err, ret_err), closed)
            if miss < 0.0025 and abs(sag_z) < 0.003 and sim.holding():
                break
            if not sim.holding() or miss > 0.03:
                break
            aim = aim - np.array([peg_err[0], peg_err[1], sag_z])
        self.ik_err = best[3]
        self.aim = best[2]
        self._lower_until_open = best[4]
        if best[1] is None:
            sim.restore(saved)
            self.reachable = False
            self._actions = []
            return False
        # planned peg is the in-hand pose at the carry, for the replay check
        sim.restore(mid)
        self.planned_peg = sim.peg_pos().copy()
        self.planned_sock = sim.socket_pos().copy()
        self.release_step = len(prefix) + self._lower_until_open
        self.prefix_len = len(prefix)
        self._actions = prefix + best[1]
        self.reachable = True
        self._i = 0
        sim.restore(saved)
        return True

    def _approach(self, q0, q_hover, q_grasp, q_lift, q_above) -> list[np.ndarray]:
        q_up = q0.copy()
        q_up[3] = 2.4  # fold the elbow up before crossing the table
        out: list[np.ndarray] = []
        out += self._move(q0, q_up, 0.45, GRIPPER_CLOSED)
        out += self._move(q_up, Q_READY, 0.65, GRIPPER_CLOSED)
        out += self._move(Q_READY, q_hover, 0.70, GRIPPER_CLOSED)
        out += self._hold(q_hover, 0.45, GRIPPER_OPEN)
        out += self._move(q_hover, q_grasp, 1.0, GRIPPER_OPEN)
        out += self._hold(q_grasp, 0.70, GRIPPER_CLOSED)
        out += self._move(q_grasp, q_lift, 0.70, GRIPPER_CLOSED)
        out += self._move(q_lift, q_above, 0.90, GRIPPER_CLOSED)
        out += self._hold(q_above, 0.30, GRIPPER_CLOSED)
        return out

    def _lower(self, q_start, q_lower, zs, z_bot, q_retreat) -> list[np.ndarray]:
        cfg = self.cfg
        out: list[np.ndarray] = []
        out += self._move(q_start, q_lower[0], 0.35, GRIPPER_CLOSED)
        for i, (a, b) in enumerate(zip(q_lower, q_lower[1:])):
            dist = max(abs(float(zs[i] - zs[i + 1])), 0.002)
            speed = cfg.lower_slow if abs(float(zs[i + 1]) - z_bot) < cfg.slow_band else cfg.lower_fast
            out += self._move(a, b, max(0.12, dist / speed), GRIPPER_CLOSED)
        self._lower_until_open = len(out) + round(0.20 * HZ)
        out += self._hold(q_lower[-1], 0.20, GRIPPER_CLOSED)
        out += self._hold(q_lower[-1], 1.00, GRIPPER_OPEN)
        out += self._move(q_lower[-1], q_retreat, 0.90, GRIPPER_OPEN)
        out += self._hold(q_retreat, 0.80, GRIPPER_OPEN)
        return out

    def _search(self, sim: PegSim, points: list[np.ndarray]):
        prefer = -2.09
        offsets = np.linspace(0.0, np.pi, 12, endpoint=False)
        yaws = []
        for o in offsets:
            yaws.append(prefer + float(o))
            if o > 1e-9:
                yaws.append(prefer - float(o))
        best = (np.inf, None, prefer)
        for yaw in yaws:
            qs, err = self._chain(sim, points, self._rot(sim, yaw), sim.q_ready)
            if err < best[0]:
                best = (err, qs, yaw)
            if err < 0.002:
                break
        return best[1], best[0], best[2]

    def _chain(self, sim: PegSim, points, rot, q_seed) -> tuple[list[np.ndarray] | None, float]:
        cfg = self.cfg
        sim.data.qpos[sim.jnt_qpos] = q_seed
        qs: list[np.ndarray] = []
        err = 0.0
        for p in points:
            q, e = dls_ik(
                sim.model,
                sim.data,
                sim.ee,
                sim.offset,
                sim.jnt_dof,
                p,
                rot,
                cfg.ik_iters,
                cfg.ik_damping,
                cfg.max_dq,
                cfg.rot_weight,
            )
            qs.append(q)
            err = max(err, e)
            if err > 0.02:
                return None, err
        return qs, err

    def _rot(self, sim: PegSim, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw - sim.x0), np.sin(yaw - sim.x0)
        yaw_m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return yaw_m @ sim.rot_down

    def _move(self, a, b, dur, grip) -> list[np.ndarray]:
        n = max(1, round(dur * HZ))
        z = np.zeros_like(a)
        qs = quintic_interp(a, b, z, z, n / HZ, n + 1)
        return [np.append(q, grip) for q in qs[1:]]

    def _hold(self, q, dur, grip) -> list[np.ndarray]:
        n = max(1, round(dur * HZ))
        a = np.append(np.asarray(q, dtype=np.float64), grip)
        return [a.copy() for _ in range(n)]

    def act(self) -> np.ndarray:
        a = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        return a.astype(np.float32)
