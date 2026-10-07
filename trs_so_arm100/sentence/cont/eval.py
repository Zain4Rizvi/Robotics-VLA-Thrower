"""Roll out a phase-1 run. Counts in eval_summary.json are the result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))
sys.path.insert(0, str(ROOT / "sentence" / "cont"))

import view  # noqa: E402
from expert import sentence  # noqa: E402
from note import batch_notes, configure_fp32, load_classifier, two_logits, write_state  # noqa: E402
from sim_rollout import (  # noqa: E402
    cell,
    forward_chunk,
    load_finetuned,
    obs_batch,
    pick_still,
    plot_loss,
    plot_success,
    read_csv,
    read_seeds,
    run_episode,
    saved_checkpoints,
    table,
    tally,
    write_mp4,
)

SEEDS = list(range(30000, 30010))
RED = sentence("red", "green")
GREEN = sentence("green", "red")
CLF_PATH = ROOT / "sentence" / "probe" / "classifier.npz"
USE_NOTE = ROOT / "sentence" / "use_note" / "eval_summary.json"


def make_predict(policy, pre, post, clf, checked):
    def predict(images, sim):
        batch = obs_batch(pre, images, sim.degrees(), sim.task())
        notes = batch_notes(policy, batch)
        logits = torch.from_numpy(two_logits(notes, *clf))
        xy = torch.as_tensor(view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2]), dtype=batch["observation.state"].dtype)
        filled = write_state(batch["observation.state"], xy.reshape(1, 4), logits)
        if float(filled[..., 12:].abs().sum()) != 0:
            raise SystemExit("dims 12:32 are not zero")
        batch["observation.state"] = filled
        return forward_chunk(policy, post, batch, filled, "phase1", checked)

    return predict


def scenes() -> dict[str, list[dict]]:
    return {
        "red_on_green": [{"seed": s, "source": "red", "task": RED, "colors": {"red": "red", "green": "green"}} for s in SEEDS],
        "green_on_red": [{"seed": s, "source": "green", "task": GREEN, "colors": {"red": "red", "green": "green"}} for s in SEEDS],
    }


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
    prev = summary["use_note_step2000"]
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
    weight = summary.get("green_weight", 1)
    weight_line = ""
    if weight != 1:
        weight_line = (
            f" Green-on-red frames are weighted by {weight:g} in the training loss. "
            "The weighted mean divides by the sum of the weights. Val is unweighted. "
            f"Init step {summary.get('init_step')}."
        )
    videos = []
    for tag in ("red_on_green", "green_on_red"):
        for step in summary["rolled_steps"]:
            videos.append(f"`videos/{tag}_step{int(step):04d}.mp4`")
    logged = train_rows[-1]["step"] if train_rows else "0"
    lines = [
        "# Restart the cosine",
        "",
        f"Red on green stacked {r} of 10 at step {last}. Green on red stacked {g} of 10 at step {last}. "
        f"Red on green is at least 5: {str(r >= 5).lower()}. Green on red is at least 5: {str(g >= 5).lower()}.",
        "",
        f"The use_note step-2000 rollout, from `use_note/eval_summary.json`, was red on green {prev['red']} of 10 "
        f"and green on red {prev['green']} of 10. Step 0 here is a new rollout of the init weights "
        f"(`{summary.get('init')}`) on seeds 30000–30009. The table below is this rollout.{cut}{weight_line}",
        "",
        "Dims 6:10 stay the red body xy then the green body xy. Dims 10:12 are the frozen probe logits. "
        "Dims 12:32 stay zero. Vision, the text layers, the connector, and the classifier stayed frozen.",
        f"Train loss was logged for {logged} steps. Counts are from `eval_summary.json`, seeds {summary['seeds']}.",
        f"Last-step stack line: `step{int(last):04d} red_on_green={r} green_on_red={g}`.",
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
        "![loss](loss.png)",
        "",
        "![success](success.png)",
        "",
        "![red on green](frames/red_on_green.png)",
        "",
        "![green on red](frames/green_on_red.png)",
        "",
        f"The stills are step {last}. Each one is a success when that sentence stacked, otherwise the closest miss. "
        "Videos are the top camera with the wrist view inset: " + ", ".join(videos) + ".",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def load_partial(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--ckpt-root", type=Path, required=True)
    p.add_argument("--seeds-file", type=Path, action="append", default=[])
    p.add_argument("--require", type=str, default="")
    args = p.parse_args()
    overlap = set(SEEDS) & read_seeds(args.seeds_file)
    if overlap:
        raise SystemExit(f"eval seeds collide with train/val: {sorted(overlap)}")
    configure_fp32()
    clf = load_classifier(CLF_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("cuda is not available")
    import mujoco

    from expert import StackSim

    prev = json.loads(USE_NOTE.read_text(encoding="utf-8"))
    use_note = {
        "red": int(prev["sentences"]["red_on_green"]["results"]["2000"]["stacks"]),
        "green": int(prev["sentences"]["green_on_red"]["results"]["2000"]["stacks"]),
    }
    budget_path = args.out / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else {}
    meta_path = args.out / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    found = saved_checkpoints(args.ckpt_root)
    have = [step for step, _ in found]
    required = [int(s) for s in args.require.split(",") if s.strip()] if args.require else [0, *have]
    missing = [s for s in required if s != 0 and s not in have]
    if missing:
        raise SystemExit(f"missing checkpoints {missing}, have {have}")
    ckpts = [(0, args.init)]
    for step, path in found:
        if not required or step in required:
            ckpts.append((step, path))
    if required:
        ckpts = [(s, p) for s, p in ckpts if s in required]
    last_step = max(s for s, _ in ckpts)
    partial_path = args.out / "eval_partial.json"
    partial = load_partial(partial_path)
    summary = partial or {
        "seeds": SEEDS,
        "sentences": {},
        "use_note_step2000": use_note,
        "budget": budget,
        "green_weight": meta.get("green_weight", 1),
        "init_step": meta.get("init_step", 0),
        "init": str(args.init),
        "rolled_steps": [],
    }
    summary["budget"] = budget
    summary["use_note_step2000"] = use_note
    videos = args.out / "videos"
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    groups = scenes()
    stills = {}
    for step, ckpt in ckpts:
        done = str(step) in summary.get("rolled_steps_done", [])
        vids_ok = all((videos / f"{tag}_step{step:04d}.mp4").exists() for tag in groups)
        if done and vids_ok and not (step == last_step and any(not (frames / f"{tag}.png").exists() for tag in groups)):
            print(f"skip checkpoint step {step}", flush=True)
            continue
        batch_notes.checked = False
        checked = {"ok": False}
        policy, pre, post = load_finetuned(ckpt, device)
        predict = make_predict(policy, pre, post, clf, checked)
        print(f"checkpoint step {step} {ckpt}", flush=True)
        for tag, group in groups.items():
            rows = []
            for i, scene in enumerate(group):
                row = run_episode(
                    sim, renderer, policy, scene["seed"], scene["source"], scene["colors"], scene["task"],
                    predict, record=(i == 0), keep_still=(step == last_step),
                )
                rows.append(row)
                print(
                    f"seed {scene['seed']}: stack={row['stack']} grasp={row['grasp']} "
                    f"drop={row['drop']} closest_cm={row['closest_m'] * 100:.1f} task={scene['task']!r}",
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
    if last_step == max(s for s, _ in ckpts):
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
    write_report(args.out, summary)
    renderer.close()
    red_n = summary["sentences"]["red_on_green"]["results"][str(last_step)]["stacks"]
    green_n = summary["sentences"]["green_on_red"]["results"][str(last_step)]["stacks"]
    print(f"step{last_step:04d} red_on_green={red_n} green_on_red={green_n}", flush=True)
    print(f"wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
