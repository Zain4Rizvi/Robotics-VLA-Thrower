"""Roll out yellow on blue, blue on yellow, and red on green. Counts come from eval_summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))
sys.path.insert(0, str(ROOT / "sentence" / "cont"))
sys.path.insert(0, str(HERE))

import view  # noqa: E402
from expert import StackSim, other  # noqa: E402
from layout import eight_logits, load_color_classifier, paint, sentence, write_color_state  # noqa: E402
from note import batch_notes, configure_fp32  # noqa: E402
from sim_rollout import (  # noqa: E402
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

CLF = HERE / "probe" / "classifier.npz"
YELLOW_SEEDS = list(range(33000, 33010))
RED_SEEDS = list(range(30000, 30010))
SERIES = (
    ("yellow_on_blue", "yellow on blue"),
    ("blue_on_yellow", "blue on yellow"),
    ("red_on_green", "red on green"),
)


def scenes() -> dict[str, list[dict]]:
    yellow, blue, red = [], [], []
    for seed in YELLOW_SEEDS:
        body, colors = paint(seed, "yellow", "blue")
        yellow.append({"seed": seed, "source": body, "colors": colors, "task": sentence("yellow", "blue")})
        blue.append(
            {
                "seed": seed,
                "source": other(body),
                "colors": colors,
                "task": sentence("blue", "yellow"),
            }
        )
    for seed in RED_SEEDS:
        red.append(
            {
                "seed": seed,
                "source": "red",
                "colors": {"red": "red", "green": "green"},
                "task": sentence("red", "green"),
            }
        )
    return {"yellow_on_blue": yellow, "blue_on_yellow": blue, "red_on_green": red}


def make_predict(policy, pre, post, pack, checked):
    def predict(images, sim):
        batch = obs_batch(pre, images, sim.degrees(), sim.task())
        notes = batch_notes(policy, batch)
        logits = torch.from_numpy(eight_logits(notes, pack))
        xy = torch.as_tensor(
            view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2]),
            dtype=batch["observation.state"].dtype,
        )
        filled = write_color_state(
            batch["observation.state"], xy.reshape(1, 4), logits[:, :4], logits[:, 4:]
        )
        if float(filled[..., 18:].abs().sum()) != 0:
            raise SystemExit("dims 18:32 are not zero")
        batch["observation.state"] = filled
        return forward_chunk(policy, post, batch, filled, "color", checked)

    return predict


def write_report(root: Path, summary: dict) -> None:
    ckpt = root / "checkpoints"
    train_rows = read_csv(ckpt / "train_log.csv")
    val_rows = read_csv(ckpt / "val_log.csv")
    plot_loss(root / "loss.png", train_rows, val_rows)
    plot_success(root / "success.png", summary, list(SERIES))
    last = str(summary["last_step"])
    counts = []
    for tag, label in SERIES:
        n = summary["sentences"][tag]["results"][last]["stacks"]
        counts.append(f"{label} {n} of 10")
    init = summary.get("init") or {}
    videos = []
    for tag, _label in SERIES:
        for step in summary["rolled_steps"]:
            videos.append(f"`videos/{tag}_step{int(step):04d}.mp4`")
    lines = [
        "# Color pairs",
        "",
        f"At step {last}: {counts[0]}, {counts[1]}, and {counts[2]}.",
        "",
        "Yellow on blue was not in the training set. Blue on yellow was. "
        "Red on green uses the bodies' own names on seeds 30000–30009. "
        f"Step 0 is the phase-1 checkpoint `{init.get('from_checkpoint')}` "
        f"(phase-1 step {init.get('init_step')}) with the color logits in dims 10:18. "
        "It is not the same measurement as run 1's red-on-green count.",
        "",
        "Dims 6:10 stay red_box xy then green_box xy. Dims 10:14 are the source-color logits and "
        "dims 14:18 are the target-color logits, class order red, green, blue, yellow. Dims 18:32 stay zero. "
        "The state does not say which body is the source.",
        f"Counts are from `eval_summary.json`. Last-step stack line: `step{int(last):04d} "
        + " ".join(f"{tag}={summary['sentences'][tag]['results'][last]['stacks']}" for tag, _ in SERIES)
        + "`.",
        "",
    ]
    for tag, label in SERIES:
        lines += [f"## {label}", "", *table(summary["sentences"][tag]["results"], 10), ""]
    lines += [
        "Closest is the distance from the source cube center to the pose where it sits on the target cube.",
        "A stack is that pose, in contact, for 0.5 s, with the target cube still on the floor.",
        "Yellow on blue above 0 is the result this phase is looking for. 5 of 10 is not required.",
        "",
        "![loss](loss.png)",
        "",
        "![success](success.png)",
        "",
        "![yellow on blue](frames/yellow_on_blue.png)",
        "",
        "![blue on yellow](frames/blue_on_yellow.png)",
        "",
        "![red on green](frames/red_on_green.png)",
        "",
        f"The stills are step {last}. Each one is a success when that sentence stacked, otherwise the closest miss. "
        "Videos are the top camera with the wrist view inset: " + ", ".join(videos) + ".",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--ckpt-root", type=Path, required=True)
    p.add_argument("--seeds-file", type=Path, action="append", default=[])
    p.add_argument("--require", type=str, default="")
    args = p.parse_args()
    used = read_seeds(args.seeds_file) if args.seeds_file else set()
    overlap = (set(YELLOW_SEEDS) | set(RED_SEEDS)) & used
    if overlap:
        raise SystemExit(f"eval seeds collide with the dataset: {sorted(overlap)}")
    configure_fp32()
    pack = load_color_classifier(CLF)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("cuda is not available")
    init_path = args.out / "init.json"
    init = json.loads(init_path.read_text(encoding="utf-8")) if init_path.exists() else {"from_checkpoint": str(args.init)}
    found = saved_checkpoints(args.ckpt_root)
    have = [step for step, _ in found]
    required = [int(s) for s in args.require.split(",") if s.strip()] if args.require else [0, *have]
    missing = [s for s in required if s != 0 and s not in have]
    if missing:
        raise SystemExit(f"missing checkpoints {missing}, have {have}")
    ckpts = [(0, args.init)] + [(s, path) for s, path in found if not required or s in required]
    if required:
        ckpts = [(s, path) for s, path in ckpts if s in required]
    last_step = max(s for s, _ in ckpts)
    partial_path = args.out / "eval_partial.json"
    summary = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {"sentences": {}}
    summary["init"] = init
    summary["seeds"] = {"yellow_on_blue": YELLOW_SEEDS, "blue_on_yellow": YELLOW_SEEDS, "red_on_green": RED_SEEDS}
    videos = args.out / "videos"
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    sim = StackSim()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    groups = scenes()
    stills = {}
    for step, ckpt in ckpts:
        done = step in summary.get("rolled_steps_done", [])
        vids_ok = all((videos / f"{tag}_step{step:04d}.mp4").exists() for tag, _ in SERIES)
        png_ok = step != last_step or all((frames / f"{tag}.png").exists() for tag, _ in SERIES)
        if done and vids_ok and png_ok:
            print(f"skip checkpoint step {step}", flush=True)
            continue
        batch_notes.checked = False
        checked = {"ok": False}
        policy, pre, post = load_finetuned(ckpt, device)
        predict = make_predict(policy, pre, post, pack, checked)
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
                    f"{tag} seed {scene['seed']}: stack={row['stack']} grasp={row['grasp']} "
                    f"drop={row['drop']} closest_cm={row['closest_m'] * 100:.1f}",
                    flush=True,
                )
            write_mp4(videos / f"{tag}_step{step:04d}.mp4", next(r["video"] for r in rows if r["video"]))
            if step == last_step:
                stills[tag] = pick_still(rows)
            summary.setdefault("sentences", {}).setdefault(tag, {"task": group[0]["task"], "results": {}})
            summary["sentences"][tag]["results"][str(step)] = tally(rows)
        summary.setdefault("rolled_steps_done", [])
        if step not in summary["rolled_steps_done"]:
            summary["rolled_steps_done"].append(step)
        summary["rolled_steps"] = [s for s, _ in ckpts]
        summary["last_step"] = last_step
        partial_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        del policy, pre, post
        torch.cuda.empty_cache()
    for tag, img in stills.items():
        Image.fromarray(img).save(frames / f"{tag}.png")
    summary["rolled_steps"] = [s for s, _ in ckpts]
    summary["last_step"] = last_step
    summary["init"] = init
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for step, _ in ckpts:
        for tag, _label in SERIES:
            if not (args.out / f"videos/{tag}_step{step:04d}.mp4").exists():
                raise SystemExit(f"missing video {tag} {step}")
    for tag, _label in SERIES:
        if not (frames / f"{tag}.png").exists():
            raise SystemExit(f"missing still {tag}")
    write_report(args.out, summary)
    renderer.close()
    last = str(last_step)
    bits = [f"{tag}={summary['sentences'][tag]['results'][last]['stacks']}" for tag, _ in SERIES]
    print(f"step{last_step:04d} " + " ".join(bits), flush=True)
    print(f"wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
