"""Play a face-probe checkpoint with three chunk lengths. Does not train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zeroshot_eval as z
from openarm_vla.constants import LEFT_ACTUATORS, RIGHT_ACTUATORS, RIGHT_JOINTS
from smoke_load import XML

SEEDS = [30003, 30006, 30013, 30014, 30032]
CKPT = "checkpoints/peg_face_probe/best/pretrained_model"
SEEDS_FILE = "data/datasets/peg_face_val/openarm_seeds.json"
FOLDERS = ("exec8", "exec16", "exec50")


def _stats(traces: list[dict]) -> dict:
    inside = [t["l2"] for t in traces if not t["boundary"]]
    boundary = [t["l2"] for t in traces if t["boundary"]]

    def med(xs: list[float]):
        return None if not xs else round(float(np.median(xs)), 6)

    def end(xs: list[float], q: float):
        return None if not xs else round(float(np.quantile(xs, q)), 6)

    return {
        "n_inside": len(inside),
        "n_boundary": len(boundary),
        "median_l2_inside": med(inside),
        "median_l2_boundary": med(boundary),
        "p95_l2_inside": end(inside, 0.95),
        "max_l2_inside": None if not inside else round(float(max(inside)), 6),
        "max_l2_boundary": None if not boundary else round(float(max(boundary)), 6),
    }


def _self_check() -> None:
    s = _stats(
        [
            {"l2": 1.0, "boundary": False},
            {"l2": 3.0, "boundary": False},
            {"l2": 10.0, "boundary": True},
        ]
    )
    assert s["median_l2_inside"] == 2.0 and s["median_l2_boundary"] == 10.0 and s["n_boundary"] == 1


def run(args) -> None:
    if args.execute > args.keep:
        raise SystemExit(f"execute {args.execute} exceeds keep {args.keep}")
    rows = json.loads(Path(args.seeds_file).read_text(encoding="utf-8"))
    got = [int(r["seed"]) for r in rows]
    if got != SEEDS:
        raise SystemExit(f"seeds {got} != {SEEDS}")
    z.MAX_STEPS = args.max_steps
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    key_id = int(model.key("home").id)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
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

    from openarm_vla.policies.smolvla import SmolVLAAdapter

    policy = SmolVLAAdapter(args.checkpoint, device="cuda")
    policy.chunk_size = args.keep
    import torch

    weight = next(policy.policy.parameters())
    print(
        f"checkpoint={args.checkpoint} dtype={weight.dtype} keep={policy.chunk_size} "
        f"execute={args.execute} max_steps={z.MAX_STEPS}",
        flush=True,
    )
    if weight.dtype != torch.float32:
        raise SystemExit(f"expected float32 weights, got {weight.dtype}")
    if SmolVLAAdapter.chunk_size != 16:
        raise SystemExit(f"class chunk_size changed: {SmolVLAAdapter.chunk_size}")

    renderer = mujoco.Renderer(model, height=z.IMG, width=z.IMG)
    episodes = []
    pooled: list[dict] = []

    def write_summary() -> None:
        dists = [e["min_hand_to_peg_m"] for e in episodes]
        summary = {
            "checkpoint": args.checkpoint,
            "seeds_file": args.seeds_file,
            "keep": args.keep,
            "execute": args.execute,
            "max_steps": args.max_steps,
            "n": len(episodes),
            "insertions": sum(e["failure"] == "success" for e in episodes),
            "grasp_count": sum(bool(e["grasped"]) for e in episodes),
            "min_hand_to_peg_m": None if not dists else float(min(dists)),
            "median_episode_min_hand_to_peg_m": None if not dists else float(np.median(dists)),
            **_stats(pooled),
            "episodes": episodes,
        }
        (out / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        peg_xy = np.asarray(row["peg_xy"], dtype=np.float64)
        sock_xy = np.asarray(row["socket_xy"], dtype=np.float64)
        print(f"seed={row['seed']} peg={np.round(peg_xy, 4)} socket={np.round(sock_xy, 4)}", flush=True)
        frames: list[np.ndarray] = []
        traces: list[dict] = []
        result = z.rollout(
            model,
            data,
            policy,
            renderer,
            ids,
            low,
            high,
            peg_xy,
            sock_xy,
            frames,
            replan_every=args.execute,
            instruction=row["instruction"],
            traces=traces,
        )
        if int(data.eq_active[ids["weld"]]) != 0:
            raise SystemExit("grasp_right_peg turned on")
        bsteps = [t["step"] for t in traces if t["boundary"]]
        if result["steps"] > args.execute:
            if not bsteps or bsteps[0] != args.execute + 1:
                raise SystemExit(f"seed={row['seed']} first boundary {bsteps[:1]} != {args.execute + 1}")
            gaps = np.diff(bsteps)
            if len(gaps) and not np.all(gaps == args.execute):
                raise SystemExit(f"seed={row['seed']} boundary gaps {gaps[:8].tolist()} != {args.execute}")
        result["seed"] = int(row["seed"])
        result["step"] = [t["step"] for t in traces]
        result["step_l2"] = [round(t["l2"], 6) for t in traces]
        result["boundary"] = [bool(t["boundary"]) for t in traces]
        result.update(_stats(traces))
        episodes.append(result)
        pooled.extend(traces)
        path = out / "videos" / f"seed{row['seed']}.mp4"
        imageio.mimsave(path, frames, fps=25)
        write_summary()
        print(
            f"seed={row['seed']} failure={result['failure']} grasped={int(result['grasped'])} "
            f"min_d={result['min_hand_to_peg_m']:.4f} steps={result['steps']} "
            f"median_inside={result['median_l2_inside']} median_boundary={result['median_l2_boundary']}",
            flush=True,
        )
    renderer.close()
    summary = json.loads((out / "eval_summary.json").read_text(encoding="utf-8"))
    print(
        f"done n={summary['n']} insertions={summary['insertions']} grasp_count={summary['grasp_count']} "
        f"min_hand_to_peg_m={summary['min_hand_to_peg_m']:.4f} "
        f"median_l2_inside={summary['median_l2_inside']} median_l2_boundary={summary['median_l2_boundary']}",
        flush=True,
    )


def plot(root: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", r"Z:\tmp\mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = []
    for name in FOLDERS:
        path = root / name / "eval_summary.json"
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    x = np.arange(len(FOLDERS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.bar(x - width / 2, [s["median_l2_inside"] for s in summaries], width, label="median inside a prediction")
    ax.bar(x + width / 2, [s["median_l2_boundary"] for s in summaries], width, label="median at a boundary")
    ax.set_xticks(x, list(FOLDERS))
    ax.set_ylabel("joint-target step L2 (rad)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "jumps.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, summary in zip(FOLDERS, summaries):
        ep = next(e for e in summary["episodes"] if e["seed"] == 30006)
        ax.plot(ep["step"], ep["step_l2"], label=name, linewidth=0.8)
    ax.set_xlabel("control step")
    ax.set_ylabel("joint-target step L2 (rad)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "trace_seed30006.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.bar(x - width / 2, [s["min_hand_to_peg_m"] * 100 for s in summaries], width, label="minimum")
    ax.bar(
        x + width / 2,
        [s["median_episode_min_hand_to_peg_m"] * 100 for s in summaries],
        width,
        label="median of the 5 episodes",
    )
    ax.set_xticks(x, list(FOLDERS))
    ax.set_ylabel("closest hand-to-peg (cm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "closest.png", dpi=120)
    plt.close(fig)
    print(f"wrote {root / 'jumps.png'} {root / 'trace_seed30006.png'} {root / 'closest.png'}", flush=True)


def main() -> None:
    _self_check()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CKPT)
    p.add_argument("--seeds-file", default=SEEDS_FILE)
    p.add_argument("--keep", type=int)
    p.add_argument("--execute", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--plot", type=Path)
    args = p.parse_args()
    if args.plot is not None:
        plot(args.plot)
        return
    if args.keep is None or args.execute is None or args.out is None:
        raise SystemExit("need --keep, --execute, and --out")
    run(args)


if __name__ == "__main__":
    main()
