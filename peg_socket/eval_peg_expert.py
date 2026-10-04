"""Roll the peg expert on legal random layouts and write the summary json.

Does not collect demonstrations. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from peg_expert import HOLD_STEPS, X_RANGE, Y_RANGE, MIN_SEP, PegExpert, PegSim

OUT = Path(__file__).resolve().parent / "findings" / "expert"
LOG = OUT / "eval.log"


def draw(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    peg = rng.uniform(np.array([X_RANGE[0], Y_RANGE[0]]), np.array([X_RANGE[1], Y_RANGE[1]]))
    sock = rng.uniform(np.array([X_RANGE[0], Y_RANGE[0]]), np.array([X_RANGE[1], Y_RANGE[1]]))
    return peg, sock


def rollout(sim: PegSim, expert: PegExpert) -> dict:
    sock0 = sim.socket_pos()[:2].copy()
    ever = False
    lost = False
    min_dist = 1e9
    min_after = 1e9
    drift = 0.0
    finger_hits = 0
    hold = 0
    inserted = False
    diverge = None
    for t, action in enumerate(expert._actions):
        sim.step(action)
        if t + 1 == expert.prefix_len:
            diverge = float(np.linalg.norm(sim.peg_pos() - expert.planned_peg))
        holding = sim.holding()
        if holding:
            ever = True
        elif ever and t < expert.release_step:
            lost = True
        dist = sim.hole_dist()
        min_dist = min(min_dist, dist)
        if t >= expert.release_step:
            min_after = min(min_after, dist)
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
    sock = sim.socket_pos()
    final = float(np.linalg.norm(peg[:2] - sock[:2]))
    cause, reason = _cause(inserted, ever, lost, final, peg, sim.peg_up(), drift, finger_hits, min_dist)
    return {
        "inserted": inserted,
        "cause": cause,
        "reason": reason,
        "final_mm": round(final * 1000.0, 1),
        "min_mm": round(min_dist * 1000.0, 1),
        "min_after_release_mm": None if min_after > 1e8 else round(min_after * 1000.0, 1),
        "peg_xyz": [round(float(v), 4) for v in peg],
        "socket_xyz": [round(float(v), 4) for v in sock],
        "peg_up": round(sim.peg_up(), 3),
        "socket_drift_mm": round(drift * 1000.0, 1),
        "finger_socket_contacts": finger_hits,
        "ever_grasped": ever,
        "lost_before_release": lost,
        "plan_replay_mm": None if diverge is None else round(diverge * 1000.0, 2),
        "ik_mm": round(expert.ik_err * 1000.0, 2),
        "offset_mm": [round(float(v) * 1000.0, 2) for v in expert.offset],
        "finger_drop_mm": round(expert.finger_drop * 1000.0, 1),
        "steps": len(expert._actions),
        "yaw": round(expert.yaw, 3),
    }


def _cause(inserted, ever, lost, final, peg, up, drift, hits, min_dist):
    if inserted:
        return None, ""
    drift_mm = drift * 1000.0
    if not ever:
        return (
            "never_grasped",
            f"fingers never both touched the peg (closest {min_dist * 1000:.0f} mm, socket drift {drift_mm:.0f} mm)",
        )
    if lost:
        return (
            "dropped",
            f"lost finger contact before release (closest {min_dist * 1000:.0f} mm, socket drift {drift_mm:.0f} mm)",
        )
    return (
        "missed",
        f"peg ended {final * 1000:.0f} mm from the hole center at z={peg[2]:.3f}, upright={up:.2f}, "
        f"finger-socket contacts {hits}, socket drift {drift_mm:.0f} mm",
    )


def log(msg: str, fh) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def prepare(sim: PegSim, peg_xy, sock_xy) -> str | None:
    sim.reset_home()
    sim.place(peg_xy, sock_xy)
    issue = sim.layout_issue(peg_xy, sock_xy)
    if issue:
        return issue
    before = sim.peg_pos().copy()
    sim.settle()
    if sim.layout_issue(sim.peg_pos()[:2], sim.socket_pos()[:2]):
        return "settled_" + (sim.layout_issue(sim.peg_pos()[:2], sim.socket_pos()[:2]) or "bad")
    if float(np.linalg.norm(sim.peg_pos()[:2] - before[:2])) > 0.005 or sim.peg_up() < 0.98:
        return "knocked_at_home"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-seeds", type=int, default=400)
    p.add_argument("--home", action="store_true", help="run the keyframe layout only")
    p.add_argument("--seeds", type=str, default="", help="comma-separated seeds; do not write summary.json")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sim = PegSim()
    expert = PegExpert()
    fh = open(LOG, "w", encoding="utf-8")
    if args.home:
        peg = np.array([0.36, -0.10])
        sock = np.array([0.36, -0.26])
        issue = prepare(sim, peg, sock)
        log(f"home issue={issue} peg={sim.peg_pos()} sock={sim.socket_pos()}", fh)
        ok = expert.reset(sim)
        aim = getattr(expert, "aim", None)
        log(
            f"reachable={ok} ik_mm={expert.ik_err * 1000:.2f} actions={len(expert._actions)} "
            f"aim={None if aim is None else np.round(aim, 4)} "
            f"drop={expert.finger_drop*1000:.1f}",
            fh,
        )
        if not ok:
            fh.close()
            return
        result = rollout(sim, expert)
        log(json.dumps(result), fh)
        fh.close()
        return

    trials = []
    rejected: dict[str, int] = {}
    if args.seeds:
        wanted = [int(s) for s in args.seeds.split(",") if s]
    else:
        wanted = None
    seed = 0
    scanned = 0
    while True:
        if wanted is not None:
            if scanned >= len(wanted):
                break
            seed = wanted[scanned]
        elif len(trials) >= args.n or seed >= args.max_seeds:
            break
        scanned += 1
        rng = np.random.default_rng(seed)
        peg_xy, sock_xy = draw(rng)
        issue = prepare(sim, peg_xy, sock_xy)
        if issue:
            rejected[issue] = rejected.get(issue, 0) + 1
            log(f"seed={seed} reject {issue}", fh)
            if wanted is None:
                seed += 1
            continue
        if not expert.reset(sim):
            rejected["unreachable"] = rejected.get("unreachable", 0) + 1
            log(f"seed={seed} reject unreachable ik_mm={expert.ik_err * 1000:.1f}", fh)
            if wanted is None:
                seed += 1
            continue
        result = rollout(sim, expert)
        result["seed"] = seed
        result["peg_xy"] = [round(float(v), 4) for v in peg_xy]
        result["socket_xy"] = [round(float(v), 4) for v in sock_xy]
        trials.append(result)
        flag = "OK" if result["inserted"] else result["cause"]
        log(
            f"seed={seed} {flag} final_mm={result['final_mm']} min_mm={result['min_mm']} "
            f"ik_mm={result['ik_mm']} drift_mm={result['socket_drift_mm']} "
            f"replay_mm={result['plan_replay_mm']} offset={result['offset_mm']} {result['reason']}",
            fh,
        )
        if wanted is None:
            seed += 1

    inserted = sum(1 for t in trials if t["inserted"])
    summary = {
        "inserted": inserted,
        "attempts": len(trials),
        "insertions": f"{inserted}/{len(trials)}",
        "instruction": "place the peg in the hole",
        "seeds": [t["seed"] for t in trials],
        "rejected": rejected,
        "seeds_scanned": seed if wanted is None else scanned,
        "legal": {
            "x": list(X_RANGE),
            "y": list(Y_RANGE),
            "min_peg_socket_sep_m": MIN_SEP,
            "clearance_m": 0.01,
            "ik_tol_m": expert.cfg.ik_tol,
            "means": (
                "Both objects upright on the tabletop, at least 1 cm clear of the pedestal and of "
                "either gripper at the home pose, peg center at least 0.12 m from the socket center "
                "so the open fingers clear the octagon, not already seated, and reachable by IK within 3 mm."
            ),
        },
        "failures": [
            {"seed": t["seed"], "cause": t["cause"], "reason": t["reason"]}
            for t in trials
            if not t["inserted"]
        ],
        "trials": trials,
    }
    path = OUT / "summary.json"
    if wanted is None:
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"inserted {inserted}/{len(trials)} -> {path}", fh)
    else:
        log(f"inserted {inserted}/{len(trials)} (no summary write)", fh)
    fh.close()


if __name__ == "__main__":
    main()
