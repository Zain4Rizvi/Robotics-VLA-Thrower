"""Orange and purple, one rollout, no training. Dims 10:18 stay zero."""

from __future__ import annotations

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
from expert import StackSim  # noqa: E402
from layout import forbidden, paint, sentence, write_color_state  # noqa: E402
from note import configure_fp32  # noqa: E402
from sim_rollout import (  # noqa: E402
    cell,
    forward_chunk,
    load_finetuned,
    obs_batch,
    run_episode,
    tally,
    write_mp4,
)

SEEDS = list(range(34000, 34010))


def scene(seed: int) -> dict:
    if seed < 34005:
        src, tgt = "orange", "purple"
    else:
        src, tgt = "purple", "orange"
    body, colors = paint(seed, src, tgt)
    return {"seed": seed, "source": body, "colors": colors, "task": sentence(src, tgt), "tag": f"{src}_on_{tgt}"}


def make_predict(policy, pre, post, checked):
    def predict(images, sim):
        import torch as th

        batch = obs_batch(pre, images, sim.degrees(), sim.task())
        xy = th.as_tensor(
            view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2]),
            dtype=batch["observation.state"].dtype,
        )
        zeros = th.zeros(1, 4, dtype=batch["observation.state"].dtype)
        filled = write_color_state(batch["observation.state"], xy.reshape(1, 4), zeros, zeros)
        if float(filled[..., 10:].abs().sum()) != 0:
            raise SystemExit("dims 10:32 are not zero")
        batch["observation.state"] = filled
        return forward_chunk(policy, post, batch, filled, "zero", checked)

    return predict


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seeds-file", type=Path, action="append", default=[])
    args = p.parse_args()
    for path in args.seeds_file:
        blob = path.read_text(encoding="utf-8").lower()
        if "orange" in blob or "purple" in blob:
            raise SystemExit(f"orange or purple is in {path}")
    configure_fp32()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("cuda is not available")
    if (args.out / "eval_summary.json").exists() and (args.out / "REPORT.md").exists():
        print("unseen report already written", flush=True)
        return
    groups: dict[str, list] = {}
    for seed in SEEDS:
        row = scene(seed)
        groups.setdefault(row["tag"], []).append(row)
    sim = StackSim()
    sim.assert_paint_is_visual()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    policy, pre, post = load_finetuned(args.checkpoint, device)
    checked = {"ok": False}
    predict = make_predict(policy, pre, post, checked)
    summary = {"seeds": SEEDS, "checkpoint": str(args.checkpoint), "color_dims": "10:18 zero", "sentences": {}}
    videos = args.out / "videos"
    rows_by = {}
    for tag, group in groups.items():
        rows = []
        for scene_row in group:
            record = scene_row["seed"] in (34000, 34005)
            row = run_episode(
                sim, renderer, policy, scene_row["seed"], scene_row["source"], scene_row["colors"],
                scene_row["task"], predict, record=record, keep_still=False,
            )
            if forbidden(row["task"]) is False:
                raise SystemExit("unseen task lost its color word")
            rows.append(row)
            print(
                f"seed {scene_row['seed']}: stack={row['stack']} grasp={row['grasp']} "
                f"drop={row['drop']} closest_cm={row['closest_m'] * 100:.1f} task={scene_row['task']!r}",
                flush=True,
            )
        video = next((r["video"] for r in rows if r["video"]), None)
        if video is None:
            raise SystemExit(f"no video for {tag}")
        name = "orange_on_purple.mp4" if tag == "orange_on_purple" else "purple_on_orange.mp4"
        write_mp4(videos / name, video)
        rows_by[tag] = tally(rows)
        summary["sentences"][tag] = {"task": group[0]["task"], "results": rows_by[tag]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    total_stacks = sum(block["stacks"] for block in rows_by.values())
    lines = [
        "# Unseen words",
        "",
        "Dimensions 10:18 were zero. The four-way probe has no orange class and no purple class. "
        "The sentence in the frozen prefix is the only color cue. This rollout was not trained.",
        "",
        "Seeds 34000–34004 are orange on purple. Seeds 34005–34009 are purple on orange. "
        "Each sentence has 5 seeds.",
        "",
        "| sentence | stacks/5 | grasps/5 | drops | closest |",
        "|---|---|---|---|---|",
    ]
    for tag in ("orange_on_purple", "purple_on_orange"):
        r = rows_by[tag]
        lines.append(
            f"| {summary['sentences'][tag]['task']} | {r['stacks']} | {r['grasps']} | {r['drops']} | {cell(r)} |"
        )
    lines += [
        "",
        "Videos: `videos/orange_on_purple.mp4` (seed 34000), `videos/purple_on_orange.mp4` (seed 34005).",
        "",
    ]
    if total_stacks == 0:
        lines.append(
            "All 10 rollouts missed. The color codes work for words the probe was fit on, "
            "and orange and purple were not a training target."
        )
        lines.append("")
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    renderer.close()
    print(f"unseen stacks {total_stacks}", flush=True)
    print(f"wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
