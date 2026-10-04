"""Collect successful peg-in-hole expert rollouts. CPU only. --out must not exist.

The layout draw is eval_peg_expert.draw / prepare (the step 1 legal draw). This
script adds one filter: both objects' geom bounds must fall inside tablecam
with a 16 px margin on the 256 image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_peg_expert import _cause, draw, prepare
from peg_expert import HOLD_STEPS, MIN_SEP, PEG_R, SOCKET_OUTER, X_RANGE, Y_RANGE, PegExpert, PegSim

INSTRUCTION = "place the peg in the hole"
IMG = 256
MARGIN = 16
FINDINGS = Path(__file__).resolve().parent / "findings" / "demos"


def state15(sim: PegSim) -> np.ndarray:
    return np.concatenate(
        [
            sim.data.qpos[sim.jnt_qpos],
            sim.data.qvel[sim.jnt_dof],
            [sim.data.qpos[sim.finger_qadr]],
        ]
    ).astype(np.float32)


def project(sim: PegSim, point: np.ndarray) -> tuple[float, float, float]:
    cam = int(sim.model.camera("tablecam").id)
    pos = sim.data.cam_xpos[cam]
    mat = sim.data.cam_xmat[cam].reshape(3, 3)
    fovy = float(sim.model.cam_fovy[cam])
    pc = mat.T @ (np.asarray(point, dtype=np.float64) - pos)
    depth = float(-pc[2])
    fy = 0.5 * IMG / np.tan(0.5 * np.deg2rad(fovy))
    u = IMG / 2 + fy * (pc[0] / depth)
    v = IMG / 2 - fy * (pc[1] / depth)
    return float(u), float(v), depth


def _corners(model, data, gid: int) -> np.ndarray:
    center = model.geom_aabb[gid, :3]
    half = model.geom_aabb[gid, 3:]
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    local = center + signs * half
    rot = data.geom_xmat[gid].reshape(3, 3)
    return data.geom_xpos[gid] + local @ rot.T


def tablecam_inside(sim: PegSim) -> bool:
    """Peg cylinder and socket wall boxes, including their tops, inside the margin."""
    lo, hi = MARGIN, IMG - 1 - MARGIN
    gids = [sim.peg_geom, *sim.socket_geoms]
    for gid in gids:
        for point in _corners(sim.model, sim.data, gid):
            u, v, depth = project(sim, point)
            if depth <= 1.0e-6 or not (lo <= u <= hi and lo <= v <= hi):
                return False
    return True


def peg_center_px(sim: PegSim) -> tuple[float, float]:
    u, v, _ = project(sim, sim.peg_pos())
    return u, v


def check_camera(sim: PegSim) -> None:
    """Project one legal layout and compare the peg center to the blue pixels."""
    FINDINGS.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(sim.model, height=IMG, width=IMG)
    seed = 1000
    while seed < 1200:
        rng = np.random.default_rng(seed)
        peg_xy, sock_xy = draw(rng)
        issue = prepare(sim, peg_xy, sock_xy)
        if issue is None and tablecam_inside(sim):
            break
        seed += 1
    else:
        raise SystemExit("no in-frame layout in seeds 1000..1199")
    u, v = peg_center_px(sim)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(sim.data, camera="tablecam")
    seg = renderer.render()
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    mask = (seg[:, :, 0] == sim.peg_geom) & (seg[:, :, 1] == geom)
    if not mask.any():
        raise SystemExit(f"seed {seed}: peg geom is not in the tablecam frame")
    ys, xs = np.nonzero(mask)
    cu, cv = float(xs.mean()), float(ys.mean())
    err = float(np.hypot(cu - u, cv - v))
    renderer.disable_segmentation_rendering()
    renderer.update_scene(sim.data, camera="tablecam")
    frame = renderer.render().copy()
    mark = frame.copy()
    ui, vi = int(round(u)), int(round(v))
    mark[max(0, vi - 2) : vi + 3, max(0, ui - 8) : ui + 9] = (255, 0, 0)
    mark[max(0, vi - 8) : vi + 9, max(0, ui - 2) : ui + 3] = (255, 0, 0)
    out = FINDINGS / "camera_check.png"
    _write_png(out, mark)
    print(
        f"camera_check seed={seed} peg_px=({u:.1f},{v:.1f}) seg=({cu:.1f},{cv:.1f}) "
        f"err_px={err:.2f} n={int(mask.sum())} "
        f"peg_xyz={np.round(sim.peg_pos(), 3)} cam={np.round(sim.data.cam_xpos[int(sim.model.camera('tablecam').id)], 3)}",
        flush=True,
    )
    if err > 8.0:
        raise SystemExit(f"tablecam projection is {err:.1f} px off the peg")
    renderer.close()


def _write_png(path: Path, frame: np.ndarray) -> None:
    import struct
    import zlib

    raw = b"".join(b"\x00" + frame[y].tobytes() for y in range(frame.shape[0]))
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", frame.shape[1], frame.shape[0], 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def replay(sim: PegSim, expert: PegExpert, ds, renderer) -> tuple[bool, str]:
    sock0 = sim.socket_pos()[:2].copy()
    ever = False
    lost = False
    min_dist = 1.0e9
    drift = 0.0
    finger_hits = 0
    hold = 0
    inserted = False
    for t, action in enumerate(expert._actions):
        action = np.asarray(action, dtype=np.float32)
        renderer.update_scene(sim.data, camera="tablecam")
        front = renderer.render().copy()
        renderer.update_scene(sim.data, camera="camera_wrist_right")
        wrist = renderer.render().copy()
        ds.add_frame(
            {
                "observation.images.image_front": front,
                "observation.images.image_wrist": wrist,
                "observation.state": state15(sim),
                "action": action,
                "task": INSTRUCTION,
            }
        )
        sim.step(action)
        holding = sim.holding()
        if holding:
            ever = True
        elif ever and t < expert.release_step:
            lost = True
        dist = sim.hole_dist()
        min_dist = min(min_dist, dist)
        drift = max(drift, float(np.linalg.norm(sim.socket_pos()[:2] - sock0)))
        if sim.fingers_touch_socket():
            finger_hits += 1
        if sim.finger_qpos() < -0.45 and sim.seated():
            hold += 1
            if hold >= HOLD_STEPS:
                inserted = True
        else:
            hold = 0
    peg = sim.peg_pos()
    cause, _reason = _cause(
        inserted, ever, lost, float(np.linalg.norm(peg[:2] - sim.socket_pos()[:2])), peg, sim.peg_up(), drift, finger_hits, min_dist
    )
    return inserted, cause or "success"


def dump_counts(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-success", type=int, default=50)
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--seed-end", type=int, required=True)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--check-camera", action="store_true")
    args = p.parse_args()
    sim = PegSim()
    if args.check_camera:
        check_camera(sim)
        return

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    from openarm_vla.data import create_dataset

    FINDINGS.mkdir(parents=True, exist_ok=True)
    expert = PegExpert()
    ds = create_dataset(out, repo_id=f"local/{out.name}")
    renderer = mujoco.Renderer(sim.model, height=IMG, width=IMG)
    rows = []
    rejected: dict[str, int] = {}
    failed: dict[str, int] = {}
    counts_path = FINDINGS / f"{args.tag}_rejects.json"
    seed = args.seed_start
    while len(rows) < args.n_success and seed < args.seed_end:
        rng = np.random.default_rng(seed)
        peg_xy, sock_xy = draw(rng)
        issue = prepare(sim, peg_xy, sock_xy)
        if issue:
            rejected[issue] = rejected.get(issue, 0) + 1
            print(f"seed={seed} reject {issue}", flush=True)
            seed += 1
            continue
        if not tablecam_inside(sim):
            rejected["tablecam_edge"] = rejected.get("tablecam_edge", 0) + 1
            print(f"seed={seed} reject tablecam_edge", flush=True)
            seed += 1
            continue
        if not expert.reset(sim):
            rejected["unreachable"] = rejected.get("unreachable", 0) + 1
            print(f"seed={seed} reject unreachable ik_mm={expert.ik_err * 1000:.1f}", flush=True)
            seed += 1
            continue
        inserted, cause = replay(sim, expert, ds, renderer)
        if inserted:
            ds.save_episode()
            rows.append(
                {
                    "episode_index": len(rows),
                    "seed": seed,
                    "instruction": INSTRUCTION,
                    "ball_color": "peg",
                    "bin_color": "socket",
                    "peg_xy": [round(float(v), 4) for v in peg_xy],
                    "socket_xy": [round(float(v), 4) for v in sock_xy],
                }
            )
            print(f"saved {len(rows)}/{args.n_success} seed={seed}", flush=True)
        else:
            ds.clear_episode_buffer()
            failed[cause] = failed.get(cause, 0) + 1
            print(f"skip {cause} seed={seed}", flush=True)
        seed += 1
        if len(rows) % 5 == 0:
            dump_counts(
                counts_path,
                {"saved": len(rows), "next_seed": seed, "rejected": rejected, "failed": failed},
            )
    ds.finalize()
    renderer.close()
    (out / "openarm_seeds.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    payload = {
        "tag": args.tag,
        "saved": len(rows),
        "n_success": args.n_success,
        "seed_start": args.seed_start,
        "seed_end_exclusive": args.seed_end,
        "next_seed": seed,
        "seeds": [r["seed"] for r in rows],
        "rejected": rejected,
        "failed": failed,
        "legal": {
            "x": list(X_RANGE),
            "y": list(Y_RANGE),
            "min_peg_socket_sep_m": MIN_SEP,
            "clearance_m": 0.01,
            "tablecam_margin_px": MARGIN,
            "image": IMG,
            "peg_radius_m": PEG_R,
            "socket_outer_m": SOCKET_OUTER,
        },
    }
    dump_counts(counts_path, payload)
    print(f"wrote {len(rows)} episodes to {out}", flush=True)
    if len(rows) < args.n_success:
        raise SystemExit(f"short {len(rows)}/{args.n_success}")


if __name__ == "__main__":
    main()
