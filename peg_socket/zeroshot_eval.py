"""Zero-shot rollout of lerobot/smolvla_base on 20 legal peg-socket layouts.

Base actions are meaningless. This records a baseline, not a working policy.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from openarm_vla.constants import (
    CONTROL_HZ,
    GRASP_OFFSET,
    GRIP_CLOSE_CMD,
    HOME_CTRL_LEFT,
    LEFT_ACTUATORS,
    N_SUBSTEPS,
    RIGHT_ACTUATORS,
    RIGHT_JOINTS,
)
from openarm_vla.policies.smolvla import SmolVLAAdapter
from smoke_load import CHECKPOINT, INSTRUCTION, XML, _rgb, _state

OUT = Path(__file__).resolve().parent / "findings" / "zeroshot"
N_EPISODES = 20
MAX_STEPS = 400
REPLAN_EVERY = 8
HOLD_STEPS = int(0.5 * CONTROL_HZ)
SEED = 0
IMG = 256
PIXEL_MARGIN = 16
# Fingertip half-span when fully open is ~7.5 cm. Socket outer radius is 3.0 cm.
# 12 cm leaves a gap so the fingers can open around the peg.
MIN_PEG_SOCKET_M = 0.12
PEDESTAL_MARGIN_M = 0.005
PEG_R = 0.014
SOCKET_R = 0.030
PEG_Z = 0.445
SOCKET_Z = 0.40
INSERT_R = 0.006
INSERT_Z = (0.0, 0.05)
# tablecam is the face camera (between the shoulders, a little toward +y).
# The pixel test is the real camera gate; this box is only the draw range.
SAMPLE_X = (0.20, 0.50)
SAMPLE_Y = (-0.46, 0.02)
TABLE_C = np.array([0.47, 0.0])
TABLE_H = np.array([0.41, 0.55])


def _aabb_corners(model, data, gid: int) -> np.ndarray:
    center = model.geom_aabb[gid, :3]
    half = model.geom_aabb[gid, 3:]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local = center + signs * half
    rot = data.geom_xmat[gid].reshape(3, 3)
    return data.geom_xpos[gid] + local @ rot.T


def _xy_aabb(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def pedestal_aabb(model, data) -> tuple[float, float, float, float]:
    gid = int(model.geom("openarm_body_link0_collision").id)
    return _xy_aabb(_aabb_corners(model, data, gid))


def dist_to_rect(xy: np.ndarray, rect: tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = rect
    cx = min(max(float(xy[0]), xmin), xmax)
    cy = min(max(float(xy[1]), ymin), ymax)
    return float(np.hypot(xy[0] - cx, xy[1] - cy))


def project(cam_pos, cam_mat, fovy_deg, point) -> np.ndarray:
    pc = cam_mat.T @ (np.asarray(point, dtype=np.float64) - cam_pos)
    depth = -pc[2]
    fy = 0.5 * IMG / np.tan(0.5 * np.deg2rad(fovy_deg))
    return np.array([IMG / 2 + fy * (pc[0] / depth), IMG / 2 - fy * (pc[1] / depth)])


def in_frame(cam_pos, cam_mat, fovy, xy, z, radius) -> bool:
    lo, hi = PIXEL_MARGIN, IMG - 1 - PIXEL_MARGIN
    for k in range(8):
        ang = k * np.pi / 4
        rim = (xy[0] + radius * np.cos(ang), xy[1] + radius * np.sin(ang), z)
        u, v = project(cam_pos, cam_mat, fovy, rim)
        if not (lo <= u <= hi and lo <= v <= hi):
            return False
    return True


def on_table(xy: np.ndarray, radius: float) -> bool:
    lo = TABLE_C - TABLE_H + radius
    hi = TABLE_C + TABLE_H - radius
    return bool(lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1])


def reject_reason(peg_xy, sock_xy, ped_box, cam) -> str | None:
    peg_xy = np.asarray(peg_xy, dtype=np.float64)
    sock_xy = np.asarray(sock_xy, dtype=np.float64)
    if not on_table(peg_xy, PEG_R) or not on_table(sock_xy, SOCKET_R):
        return "off_table"
    if dist_to_rect(peg_xy, ped_box) < PEG_R + PEDESTAL_MARGIN_M:
        return "peg_pedestal"
    if dist_to_rect(sock_xy, ped_box) < SOCKET_R + PEDESTAL_MARGIN_M:
        return "socket_pedestal"
    if float(np.hypot(*(peg_xy - sock_xy))) < MIN_PEG_SOCKET_M:
        return "too_close"
    cam_pos, cam_mat, fovy = cam
    if not in_frame(cam_pos, cam_mat, fovy, peg_xy, PEG_Z, PEG_R):
        return "peg_camera"
    if not in_frame(cam_pos, cam_mat, fovy, sock_xy, SOCKET_Z, SOCKET_R):
        return "socket_camera"
    return None


def blocked_contact(model, data, peg_geoms: set[int], sock_geoms: set[int], table_geom: int) -> str | None:
    """Peg and socket may rest on the table. Any other contact is a bad footprint."""
    objects = peg_geoms | sock_geoms
    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        hit = g1 in objects or g2 in objects
        if not hit:
            continue
        if g1 in peg_geoms and g2 in sock_geoms or g2 in peg_geoms and g1 in sock_geoms:
            return "peg_socket_contact"
        other = g2 if g1 in objects else g1
        if other == table_geom or other in objects:
            continue
        return "gripper" if "finger" in model.geom(other).name or "ee_" in model.geom(other).name else "contact"
    return None


def draw_layouts(model, data, rng, ped_box, cam, peg_geoms, sock_geoms, table_geom):
    layouts = []
    reasons: Counter[str] = Counter()
    while len(layouts) < N_EPISODES:
        peg = rng.uniform((SAMPLE_X[0], SAMPLE_Y[0]), (SAMPLE_X[1], SAMPLE_Y[1]))
        sock = rng.uniform((SAMPLE_X[0], SAMPLE_Y[0]), (SAMPLE_X[1], SAMPLE_Y[1]))
        reason = reject_reason(peg, sock, ped_box, cam)
        if reason is None:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            set_free(model, data, "peg_free", peg, PEG_Z)
            set_free(model, data, "socket_free", sock, SOCKET_Z)
            mujoco.mj_forward(model, data)
            reason = blocked_contact(model, data, peg_geoms, sock_geoms, table_geom)
        if reason is not None:
            reasons[reason] += 1
            if sum(reasons.values()) > 20000:
                raise RuntimeError(f"could not draw {N_EPISODES} layouts: {reasons}")
            continue
        layouts.append((peg, sock))
    return layouts, reasons


def set_free(model, data, name: str, xy, z: float) -> None:
    jid = int(model.joint(name).id)
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr : adr + 3] = (xy[0], xy[1], z)
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    vadr = int(model.jnt_dofadr[jid])
    data.qvel[vadr : vadr + 6] = 0.0


def grasp_point(data, ee_body: int) -> np.ndarray:
    rot = data.xmat[ee_body].reshape(3, 3)
    return data.xpos[ee_body] + rot @ np.asarray(GRASP_OFFSET)


def fingers_on_peg(model, data, peg_geom: int, fingers: set[int]) -> bool:
    touching = set()
    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 == peg_geom:
            touching.add(int(model.geom_bodyid[g2]))
        elif g2 == peg_geom:
            touching.add(int(model.geom_bodyid[g1]))
    return fingers <= touching


def peg_in_socket(peg: np.ndarray, socket: np.ndarray) -> bool:
    if float(np.hypot(peg[0] - socket[0], peg[1] - socket[1])) > INSERT_R:
        return False
    dz = float(peg[2] - socket[2])
    return INSERT_Z[0] <= dz <= INSERT_Z[1]


def classify(ever: bool, dropped: bool, released: bool, timed_out: bool, success: bool) -> str:
    if success:
        return "success"
    if not ever:
        return "never_grasped"
    if dropped and not released:
        return "dropped"
    if released:
        return "missed"
    if timed_out:
        return "timeout"
    return "missed"


def rollout(
    model,
    data,
    policy,
    renderer,
    ids,
    low,
    high,
    peg_xy,
    sock_xy,
    frames=None,
    replan_every: int | None = None,
    instruction: str | None = None,
    traces: list | None = None,
) -> dict:
    mujoco.mj_resetDataKeyframe(model, data, ids["key"])
    data.eq_active[ids["weld"]] = 0
    set_free(model, data, "peg_free", peg_xy, PEG_Z)
    set_free(model, data, "socket_free", sock_xy, SOCKET_Z)
    data.ctrl[ids["left"]] = HOME_CTRL_LEFT
    mujoco.mj_forward(model, data)
    if frames is not None:
        frames.append(_rgb(renderer, data, "tablecam"))

    buf = np.zeros((0, 8), np.float32)
    ptr = 0
    steps = 0
    hold = 0
    ever = False
    was = False
    dropped = False
    released = False
    success = False
    min_d = float("inf")
    timed_pred = True
    every = REPLAN_EVERY if replan_every is None else replan_every
    text = INSTRUCTION if instruction is None else instruction
    prev_action = None

    while steps < MAX_STEPS:
        new_chunk = False
        if ptr >= every or ptr >= len(buf):
            obs = {
                "image_front": _rgb(renderer, data, "tablecam"),
                "image_wrist": _rgb(renderer, data, "camera_wrist_right"),
                "state": _state(data, ids["qpos"], ids["qvel"], ids["finger"]),
                "instruction": text,
            }
            t0 = time.perf_counter()
            buf = policy.predict_chunk(obs)
            if timed_pred:
                print(
                    f"predict_s={time.perf_counter() - t0:.2f} "
                    f"action_shape={tuple(buf.shape)} dtype={buf.dtype}",
                    flush=True,
                )
                timed_pred = False
                if traces is not None and buf.shape[0] != policy.chunk_size:
                    raise RuntimeError(
                        f"predict_chunk returned {buf.shape[0]} steps, keep is {policy.chunk_size}"
                    )
            ptr = 0
            new_chunk = True
        ptr += 1
        action = np.clip(np.asarray(buf[min(ptr - 1, len(buf) - 1)], dtype=np.float64), low, high)
        if traces is not None and prev_action is not None:
            traces.append(
                {
                    "step": steps + 1,
                    "l2": float(np.linalg.norm(action - prev_action)),
                    "boundary": new_chunk,
                }
            )
        prev_action = action.copy()
        data.ctrl[ids["left"]] = HOME_CTRL_LEFT
        data.ctrl[ids["right"]] = action
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)
        steps += 1
        if frames is not None and steps % 2 == 0:
            frames.append(_rgb(renderer, data, "tablecam"))

        peg = data.xpos[ids["peg"]]
        sock = data.xpos[ids["socket"]]
        dist = float(np.linalg.norm(grasp_point(data, ids["ee"]) - peg))
        if dist < min_d:
            min_d = dist
        grasping = fingers_on_peg(model, data, ids["peg_geom"], ids["fingers"])
        gripper_open = float(data.ctrl[ids["right"][7]]) < GRIP_CLOSE_CMD
        if gripper_open and ever:
            released = True
        if grasping:
            ever = True
            was = True
        elif was and not released:
            dropped = True
            was = False
        if peg_in_socket(peg, sock):
            hold += 1
        else:
            hold = 0
        if hold >= HOLD_STEPS:
            success = True
            break
        if steps % 80 == 0:
            print(f"  step={steps} min_d={min_d:.4f} grasped={int(ever)}", flush=True)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f"non-finite qpos at step {steps}")

    failure = classify(ever, dropped, released, steps >= MAX_STEPS and not success, success)
    return {
        "failure": failure,
        "grasped": bool(ever),
        "min_hand_to_peg_m": min_d,
        "steps": steps,
        "peg_xy": [float(peg_xy[0]), float(peg_xy[1])],
        "socket_xy": [float(sock_xy[0]), float(sock_xy[1])],
    }


def write_summary(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    (OUT / "eval_summary.json").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    key_id = int(model.key("home").id)
    if key_id != 0:
        raise SystemExit(f"home key id is {key_id}, expected 0")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    ped_box = pedestal_aabb(model, data)
    peg_geoms = {int(model.geom("peg_geom").id)}
    sock_geoms = {int(model.geom(f"socket_{i}").id) for i in range(8)}
    table_geom = int(model.geom("table_top").id)
    cam_id = int(model.camera("tablecam").id)
    cam = (data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].reshape(3, 3).copy(), float(model.cam_fovy[cam_id]))
    home_peg = data.xpos[int(model.body("peg").id)][:2].copy()
    home_sock = data.xpos[int(model.body("socket").id)][:2].copy()
    home_reason = reject_reason(home_peg, home_sock, ped_box, cam)
    if home_reason is None:
        home_reason = blocked_contact(model, data, peg_geoms, sock_geoms, table_geom)
    print(f"home_legal={int(home_reason is None)} reason={home_reason}", flush=True)
    if home_reason is not None:
        raise SystemExit(f"home layout is illegal: {home_reason}")

    rng = np.random.default_rng(SEED)
    layouts, reasons = draw_layouts(model, data, rng, ped_box, cam, peg_geoms, sock_geoms, table_geom)
    print(f"rejected={sum(reasons.values())} {dict(reasons)}", flush=True)
    print(
        f"bounds sample_x={SAMPLE_X} sample_y={SAMPLE_Y} min_sep={MIN_PEG_SOCKET_M} "
        f"pixel_margin={PIXEL_MARGIN} pedestal_margin={PEDESTAL_MARGIN_M}",
        flush=True,
    )
    print(f"pedestal_aabb={tuple(round(v, 4) for v in ped_box)}", flush=True)

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=IMG, width=IMG)
    front = _rgb(renderer, data, "tablecam")
    uv = project(cam[0], cam[1], cam[2], data.xpos[int(model.body("peg").id)])
    pix = front[int(round(uv[1])), int(round(uv[0]))]
    print(f"home_peg_uv={tuple(np.round(uv, 1))} pixel={tuple(int(c) for c in pix)}", flush=True)
    if not (int(pix[2]) > int(pix[0]) and int(pix[2]) > 100):
        raise SystemExit(f"tablecam projection missed the peg: uv={uv} pixel={pix}")

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

    policy = SmolVLAAdapter(CHECKPOINT, device="cuda")
    import torch

    weight = next(policy.policy.parameters())
    print(f"checkpoint={CHECKPOINT} dtype={weight.dtype} device={weight.device}", flush=True)
    if weight.dtype != torch.float32:
        raise SystemExit(f"expected float32 weights, got {weight.dtype}")

    bounds = {
        "sample_x": list(SAMPLE_X),
        "sample_y": list(SAMPLE_Y),
        "table_center_xy": TABLE_C.tolist(),
        "table_half_extents_xy": TABLE_H.tolist(),
        "peg_z": PEG_Z,
        "socket_z": SOCKET_Z,
        "peg_radius_m": PEG_R,
        "socket_outer_radius_m": SOCKET_R,
        "min_peg_socket_m": MIN_PEG_SOCKET_M,
        "home_gripper": "reject a draw if the peg or socket contacts any geom except table_top",
        "pedestal_margin_m": PEDESTAL_MARGIN_M,
        "pedestal_xy_aabb": list(ped_box),
        "camera": "tablecam",
        "pixel_margin": PIXEL_MARGIN,
        "image_size": IMG,
        "insert_radius_m": INSERT_R,
        "insert_dz_m": list(INSERT_Z),
        "hold_s": 0.5,
        "max_steps": MAX_STEPS,
        "chunk": 16,
        "replan_every": REPLAN_EVERY,
    }
    episodes = []
    payload = {
        "checkpoint": CHECKPOINT,
        "instruction": INSTRUCTION,
        "n": 0,
        "insertions": 0,
        "grasp_count": 0,
        "min_hand_to_peg_m": None,
        "median_episode_min_hand_to_peg_m": None,
        "seed": SEED,
        "rejected_draws": int(sum(reasons.values())),
        "rejected_by": dict(reasons),
        "bounds": bounds,
        "episodes": episodes,
    }

    for i, (peg_xy, sock_xy) in enumerate(layouts):
        t0 = time.perf_counter()
        print(f"ep={i} peg={np.round(peg_xy, 4)} socket={np.round(sock_xy, 4)}", flush=True)
        row = rollout(model, data, policy, renderer, ids, low, high, peg_xy, sock_xy)
        row["i"] = i
        episodes.append(row)
        dists = [e["min_hand_to_peg_m"] for e in episodes]
        payload["n"] = len(episodes)
        payload["insertions"] = sum(e["failure"] == "success" for e in episodes)
        payload["grasp_count"] = sum(e["grasped"] for e in episodes)
        payload["min_hand_to_peg_m"] = float(min(dists))
        payload["median_episode_min_hand_to_peg_m"] = float(np.median(dists))
        write_summary(payload)
        print(
            f"ep={i} failure={row['failure']} grasped={int(row['grasped'])} "
            f"min_d={row['min_hand_to_peg_m']:.4f} steps={row['steps']} "
            f"elapsed_s={time.perf_counter() - t0:.1f} "
            f"insertions={payload['insertions']} grasp_count={payload['grasp_count']}",
            flush=True,
        )

    renderer.close()
    print(
        f"done n={payload['n']} insertions={payload['insertions']} "
        f"grasp_count={payload['grasp_count']} "
        f"min_hand_to_peg_m={payload['min_hand_to_peg_m']:.4f} "
        f"median_episode_min_hand_to_peg_m={payload['median_episode_min_hand_to_peg_m']:.4f} crashed=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
