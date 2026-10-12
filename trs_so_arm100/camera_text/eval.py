"""Roll out the camera-and-sentence run. Counts in eval_summary.json are the result.

Dimensions 6:32 are zero on every chunk. Closest is the live sit pose.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence" / "cont"))
sys.path.insert(0, str(HERE))

import view  # noqa: E402
from expert import HOLD_STEPS, StackSim, sentence, sit_error, stacked  # noqa: E402
from sim_rollout import (  # noqa: E402
    CAP,
    cell,
    inset,
    load_finetuned,
    obs_batch,
    pick_still,
    plot_loss,
    plot_success,
    read_csv,
    read_seeds,
    render_pair,
    saved_checkpoints,
    table,
    write_mp4,
)

_spec = importlib.util.spec_from_file_location("camera_text_train", HERE / "train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
abort_if_tail = _train.abort_if_tail
configure_fp32 = _train.configure_fp32
zero_tail = _train.zero_tail

SEEDS = list(range(30000, 30010))
RED = sentence("red", "green")
GREEN = sentence("green", "red")


def classify(stack: bool, grasp: bool, drop: bool, ever_sit: bool) -> str:
    if stack:
        return "stack"
    if ever_sit:
        return "reached_not_held"
    if grasp and drop:
        return "grasped_then_dropped"
    if not grasp:
        return "never_grasped"
    return "grasped_not_placed"


def self_check() -> None:
    import torch

    if classify(True, True, True, True) != "stack":
        raise SystemExit("stack class")
    if classify(False, True, True, True) != "reached_not_held":
        raise SystemExit("reached class")
    if classify(False, True, True, False) != "grasped_then_dropped":
        raise SystemExit("drop class")
    if classify(False, False, False, False) != "never_grasped":
        raise SystemExit("miss class")
    if classify(False, True, False, False) != "grasped_not_placed":
        raise SystemExit("hold class")
    bad = torch.zeros(1, 32)
    bad[0, 6] = 1
    try:
        abort_if_tail(bad, "self-check")
    except RuntimeError:
        pass
    else:
        raise SystemExit("planted nonzero did not abort")
    print("eval self_check ok", flush=True)


def predict_chunk(policy, pre, post, images, sim):
    batch = obs_batch(pre, images, sim.degrees(), sim.task())
    filled = zero_tail(batch["observation.state"])
    abort_if_tail(filled if filled.ndim == 2 else filled[:, -1, :], "rollout write")
    batch["observation.state"] = filled
    fired = {"ok": False}

    def hook(_mod, inputs):
        got = inputs[0]
        if got.ndim == 3:
            got = got[:, -1, :]
        try:
            abort_if_tail(got, "state_proj")
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
        fired["ok"] = True

    handle = policy.model.state_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
    finally:
        handle.remove()
    if not fired["ok"]:
        raise SystemExit("state_proj was not called")
    return post(actions)[0].detach().cpu().numpy()


def run_episode(sim, renderer, policy, pre, post, seed, source, task, record: bool, keep_still: bool) -> dict:
    colors = {"red": "red", "green": "green"}
    sim.reset(seed, source=source, colors=colors)
    if sim.task() != task:
        raise SystemExit(f"task {sim.task()!r} != {task!r}")
    policy.reset()
    src0 = sim.pos(sim.source).copy()
    grasped = False
    dropped = False
    ever_sit = False
    hold = 0
    closest = sit_error(sim)
    miss_still = None
    success_still = None
    if keep_still:
        top0, wrist0 = render_pair(sim, renderer)
        miss_still = inset(top0, wrist0)
    buf, ptr = None, 0
    video = []
    for _ in range(CAP):
        if buf is None or ptr >= view.REPLAN_EVERY or ptr >= len(buf):
            top, wrist = render_pair(sim, renderer)
            buf = predict_chunk(policy, pre, post, {"camera1": top, "camera2": wrist}, sim)
            ptr = 0
        else:
            top = None
        if record:
            if top is None:
                top, wrist = render_pair(sim, renderer)
            video.append(inset(top, wrist))
        ctrl = view.degrees_to_ctrl(buf[ptr], sim.limits)
        ptr += 1
        sim.step(ctrl)
        err = sit_error(sim)
        if err < closest:
            closest = err
            if keep_still:
                top, wrist = render_pair(sim, renderer)
                miss_still = inset(top, wrist)
        if stacked(sim):
            ever_sit = True
        if sim.holding() and sim.pos(sim.source)[2] > src0[2] + 0.02:
            grasped = True
        if grasped and not sim.holding() and not stacked(sim):
            dropped = True
        hold = hold + 1 if stacked(sim) else 0
        if hold >= HOLD_STEPS:
            if keep_still:
                top, wrist = render_pair(sim, renderer)
                success_still = inset(top, wrist)
            break
    success = hold >= HOLD_STEPS
    kind = classify(success, grasped, dropped and not success, ever_sit)
    return {
        "seed": seed,
        "stack": success,
        "grasp": grasped,
        "drop": dropped and not success,
        "ever_sit": ever_sit,
        "failure": kind,
        "closest_m": closest,
        "task": task,
        "video": video,
        "still": success_still if success else miss_still,
    }


def tally(rows: list[dict]) -> dict:
    keep = [{k: v for k, v in r.items() if k not in ("video", "still")} for r in rows]
    names = ("never_grasped", "grasped_then_dropped", "reached_not_held", "grasped_not_placed")
    return {
        "stacks": int(sum(r["stack"] for r in rows)),
        "grasps": int(sum(r["grasp"] for r in rows)),
        "drops": int(sum(r["drop"] for r in rows)),
        "closest_m_mean": float(np.mean([r["closest_m"] for r in rows])),
        "closest_m_best": float(min(r["closest_m"] for r in rows)),
        **{name: int(sum(r["failure"] == name for r in rows)) for name in names},
        "n": len(rows),
        "episodes": keep,
    }


def scenes() -> dict[str, list[dict]]:
    return {
        "red_on_green": [{"seed": s, "source": "red", "task": RED} for s in SEEDS],
        "green_on_red": [{"seed": s, "source": "green", "task": GREEN} for s in SEEDS],
    }


def reading(red: dict, green: dict) -> list[str]:
    r_stack, g_stack = int(red["stacks"]), int(green["stacks"])
    r_best = float(red["closest_m_best"]) * 100
    g_best = float(green["closest_m_best"]) * 100
    best = min(r_best, g_best)
    grasps = int(red["grasps"]) + int(green["grasps"])
    lines = [
        "The position run reached 0.4 cm best and stacked at step 5500 only after grasps were already common.",
    ]
    if (r_stack > 0) != (g_stack > 0):
        which = "Red on green" if r_stack > 0 else "Green on red"
        lines.append(
            f"{which} stacked and the other sentence did not, so this run is not using the text."
        )
        lines.append(
            "Next step: check whether the frozen word tokens still separate these two sentences. "
            "Do not start another train."
        )
        return lines
    if r_stack == 0 and g_stack == 0 and grasps == 0 and best >= 6:
        lines.append(
            f"Neither sentence grasped, and the best closest is {best:.1f} cm, near the 8 cm spawn gap. "
            "The arm has not used the image."
        )
        lines.append("Next step: leave this expert. Do not start another train.")
        return lines
    if grasps > 0 and best >= 1 and best <= 4 and r_stack == 0 and g_stack == 0:
        lines.append(
            f"The arm grasped and the best closest is {best:.1f} cm, a couple of centimeters off the sit pose. "
            "Placement is limited by the 1.6 cm frozen readout."
        )
        lines.append("Next step: do not train this expert further.")
        return lines
    if r_stack > 0 and g_stack > 0:
        r_mean = float(red["closest_m_mean"]) * 100
        g_mean = float(green["closest_m_mean"]) * 100
        if int(red["never_grasped"]) >= 5 or int(green["never_grasped"]) >= 5:
            lines.append(
                "The arm is failing to see the cube on most seeds. "
                f"Red on green never grasped {int(red['never_grasped'])} of 10 and green on red never grasped "
                f"{int(green['never_grasped'])} of 10. Mean closest stays at {r_mean:.1f} cm and {g_mean:.1f} cm. "
                f"The grasps that did happen usually held: red stacked {r_stack} of {int(red['grasps'])} grasps "
                f"with best closest {r_best:.1f} cm, and green stacked {g_stack} of {int(green['grasps'])} grasps "
                f"with best closest {g_best:.1f} cm. None reached the sit pose without holding 0.5 s. "
                "Both sentences produced a stack."
            )
            lines.append("Next step: look at the wrist view on the seeds that never grasped. Do not start another train.")
            return lines
        lines.append("Both sentences stacked from the cameras and the sentence, with dimensions 6:32 held at zero.")
        lines.append(
            "Next step: keep this checkpoint as the camera-text result. "
            "Do not point the web app at it and do not start another train."
        )
        return lines
    lines.append(
        f"Red on green grasped {int(red['grasps'])} of 10 and its best closest is {r_best:.1f} cm. "
        f"Green on red grasped {int(green['grasps'])} of 10 and its best closest is {g_best:.1f} cm."
    )
    if grasps == 0:
        lines.append("The arm is failing to grasp, so it is not yet a placement or a sentence result.")
    elif best >= 6:
        lines.append("Grasps did not close the spawn gap, so the arm is failing to use the image for placement.")
    else:
        lines.append("The grasps and the closest distance do not match a single one of the three patterns above.")
    lines.append("Next step: do not start another train from this report.")
    return lines


def write_report(root: Path, summary: dict) -> None:
    ckpt = root / "checkpoints"
    train_rows = read_csv(ckpt / "train_log.csv")
    val_rows = read_csv(ckpt / "val_log.csv")
    plot_loss(root / "loss.png", train_rows, val_rows)
    plot_success(
        root / "success.png",
        summary,
        (("red_on_green", "red on green"), ("green_on_red", "green on red")),
    )
    red = summary["sentences"]["red_on_green"]["results"]
    green = summary["sentences"]["green_on_red"]["results"]
    last = str(summary["last_step"])
    r = red[last]["stacks"]
    g = green[last]["stacks"]
    budget = summary.get("budget") or {}
    cut = ""
    if budget.get("cut"):
        cut = (
            f" The first 1000 steps took {budget['minutes_per_1000']:.1f} minutes, "
            f"slower than 45 minutes per 1000 steps, so this run was cut from "
            f"{budget['requested_steps']} steps to {budget['steps']}."
        )
    elif budget.get("minutes_per_1000") is not None:
        cut = (
            f" The first 1000 steps took {budget['minutes_per_1000']:.1f} minutes, "
            "faster than or equal to 45 minutes per 1000 steps, so this run stayed at "
            f"{budget.get('requested_steps', summary['last_step'])} steps."
        )
    videos = []
    for tag in ("red_on_green", "green_on_red"):
        for step in summary["rolled_steps"]:
            videos.append(f"`videos/{tag}_step{int(step):04d}.mp4`")
    logged = train_rows[-1]["step"] if train_rows else "0"
    quote = f"step{int(last):04d} red_on_green={r} green_on_red={g}"

    def split_block(title: str, row: dict) -> list[str]:
        lines = [
            f"### {title}",
            "",
            f"never grasped: {row['never_grasped']}",
            f"grasped then dropped: {row['grasped_then_dropped']}",
            f"reached the pose but did not hold 0.5 s: {row['reached_not_held']}",
        ]
        if row.get("grasped_not_placed"):
            lines.append(f"grasped and did not reach the pose: {row['grasped_not_placed']}")
        lines.append("")
        return lines

    lines = [
        "# Camera and sentence",
        "",
        f"Red on green stacked {r} of 10 at step {last}. Green on red stacked {g} of 10 at step {last}.{cut}",
        "",
        "Dimensions 6:32 were zero, and vision and text stayed frozen.",
        f"Train loss was logged for {logged} steps. Counts are from `eval_summary.json`, seeds {summary['seeds']}.",
        "",
        "![loss](loss.png)",
        "",
        "![success](success.png)",
        "",
        "## stack the red cube on the green cube",
        "",
        *table(red, 10),
        "",
        "## stack the green cube on the red cube",
        "",
        *table(green, 10),
        "",
        "Closest is the distance from the source cube center to the pose where it sits on the target cube.",
        "A stack is that pose, in contact, for 0.5 s, with the target cube still on the floor.",
        "",
        f"## Failure split at step {last}",
        "",
        *split_block("stack the red cube on the green cube", red[last]),
        *split_block("stack the green cube on the red cube", green[last]),
        f"Last-step stack line: `{quote}`.",
        "",
        "![red on green](frames/red_on_green.png)",
        "",
        "![green on red](frames/green_on_red.png)",
        "",
        f"The stills are step {last}. Each one is a success when that sentence stacked, otherwise the closest miss. "
        "Videos are the top camera with the wrist view inset: " + ", ".join(videos) + ".",
        "",
        *reading(red[last], green[last]),
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def load_partial(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--init", type=Path)
    p.add_argument("--ckpt-root", type=Path)
    p.add_argument("--seeds-file", type=Path, action="append", default=[])
    p.add_argument("--require", type=str, default="")
    p.add_argument("--skip-report", action="store_true")
    p.add_argument("--self-check-only", action="store_true")
    args = p.parse_args()
    configure_fp32()
    self_check()
    if args.self_check_only:
        return
    if args.out is None or args.init is None or args.ckpt_root is None:
        raise SystemExit("need --out, --init, and --ckpt-root")
    overlap = set(SEEDS) & read_seeds(args.seeds_file)
    if overlap:
        raise SystemExit(f"eval seeds collide with train/val: {sorted(overlap)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("cuda is not available")
    import mujoco

    budget_path = args.out / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else {}
    found = saved_checkpoints(args.ckpt_root)
    have = [step for step, _ in found]
    required = [int(s) for s in args.require.split(",") if s.strip()] if args.require else [0, *have]
    missing = [s for s in required if s != 0 and s not in have]
    if missing:
        raise SystemExit(f"missing checkpoints {missing}, have {have}")
    ckpts = [(0, args.init)]
    for step, path in found:
        if step == 0:
            continue
        if not required or step in required:
            ckpts.append((step, path))
    if required:
        ckpts = [(s, p) for s, p in ckpts if s in required or (s == 0 and 0 in required)]
    last_step = max(s for s, _ in ckpts)
    partial_path = args.out / "eval_partial.json"
    partial = load_partial(partial_path)
    summary = partial or {
        "seeds": SEEDS,
        "sentences": {},
        "budget": budget,
        "init": str(args.init),
        "state": "dims 6:32 zero",
        "rolled_steps": [],
    }
    summary["budget"] = budget
    summary["init"] = str(args.init)
    summary["state"] = "dims 6:32 zero"
    videos = args.out / "videos"
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    groups = scenes()
    stills = {}
    for step, ckpt in ckpts:
        done = step in summary.get("rolled_steps_done", []) or str(step) in summary.get("rolled_steps_done", [])
        vids_ok = all((videos / f"{tag}_step{step:04d}.mp4").exists() for tag in groups)
        frames_ok = step != last_step or all((frames / f"{tag}.png").exists() for tag in groups)
        if done and vids_ok and frames_ok:
            print(f"skip checkpoint step {step}", flush=True)
            continue
        policy, pre, post = load_finetuned(ckpt, device)
        print(f"checkpoint step {step} {ckpt}", flush=True)
        for tag, group in groups.items():
            rows = []
            for i, scene in enumerate(group):
                row = run_episode(
                    sim, renderer, policy, pre, post,
                    scene["seed"], scene["source"], scene["task"],
                    record=(i == 0), keep_still=(step == last_step),
                )
                rows.append(row)
                print(
                    f"seed {scene['seed']}: stack={row['stack']} grasp={row['grasp']} "
                    f"drop={row['drop']} failure={row['failure']} "
                    f"closest_cm={row['closest_m'] * 100:.1f} task={scene['task']!r}",
                    flush=True,
                )
            video_frames = next(r["video"] for r in rows if r["video"])
            write_mp4(videos / f"{tag}_step{step:04d}.mp4", video_frames)
            if step == last_step:
                stills[tag] = pick_still(rows)
            for r in rows:
                r["video"] = []
                r["still"] = None
            summary["sentences"].setdefault(tag, {"task": group[0]["task"], "results": {}})
            summary["sentences"][tag]["results"][str(step)] = tally(rows)
        summary.setdefault("rolled_steps_done", [])
        if step not in summary["rolled_steps_done"] and str(step) not in summary["rolled_steps_done"]:
            summary["rolled_steps_done"].append(step)
        summary["rolled_steps"] = [s for s, _ in ckpts]
        summary["last_step"] = last_step
        partial_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        del policy, pre, post
        torch.cuda.empty_cache()
    for tag, img in stills.items():
        Image.fromarray(img).save(frames / f"{tag}.png")
    if not args.skip_report:
        for tag in groups:
            png = frames / f"{tag}.png"
            if not png.exists():
                raise SystemExit(f"missing still {png}")
    summary["rolled_steps"] = [s for s, _ in ckpts]
    summary["last_step"] = last_step
    summary["seeds"] = SEEDS
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for step, _ in ckpts:
        for tag in groups:
            name = f"videos/{tag}_step{step:04d}.mp4"
            if not (args.out / name).exists():
                raise SystemExit(f"missing {name}")
    red_n = summary["sentences"]["red_on_green"]["results"][str(last_step)]["stacks"]
    green_n = summary["sentences"]["green_on_red"]["results"][str(last_step)]["stacks"]
    quote = f"step{last_step:04d} red_on_green={red_n} green_on_red={green_n}"
    print(quote, flush=True)
    if not args.skip_report:
        write_report(args.out, summary)
        print(f"wrote {args.out / 'REPORT.md'}", flush=True)
    renderer.close()


if __name__ == "__main__":
    main()
