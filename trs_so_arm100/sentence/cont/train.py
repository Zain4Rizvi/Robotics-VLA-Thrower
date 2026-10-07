"""Restart the cosine from a phase-1 checkpoint.

Dims 6:10 stay red xy then green xy. Dims 10:12 are the frozen probe logits.
Dims 12:32 stay zero. Vision, the text layers, the connector, and the classifier
stay frozen. The schedule is checked here. color/train.py assert_expert_only is
not called.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))
sys.path.insert(0, str(ROOT / "sentence" / "cont"))

import view  # noqa: E402
from harness import execute, self_check as harness_check  # noqa: E402
from note import batch_notes, configure_fp32, load_classifier, two_logits, verify_saved_notes, write_state  # noqa: E402

CLF = ROOT / "sentence" / "probe" / "classifier.npz"
NOTES = ROOT / "sentence" / "probe" / "notes.npz"
ROLLOUT = set(range(30000, 30010))


def self_check() -> None:
    harness_check()
    state = __import__("torch").zeros(2, 1, 6)
    scored = __import__("torch").tensor([[1.0, 2, 3, 4], [5, 6, 7, 8]])
    logits = __import__("torch").tensor([[0.1, -0.1], [0.3, -0.3]])
    out = write_state(state, scored, logits)
    if not __import__("torch").equal(out[:, 0, 10:12], logits) or float(out[:, :, 12:].abs().sum()) != 0:
        raise SystemExit("write_state logits failed")
    red = np.array([0.02, -0.2], np.float32)
    green = np.array([-0.03, -0.22], np.float32)
    got = view.zscore_xy(red, green)
    back = got * view.XY_STD + view.XY_MEAN
    if not np.allclose(back[:2], red, atol=1.0e-5) or not np.allclose(back[2:], green, atol=1.0e-5):
        raise SystemExit(f"body order {back}")
    verify_saved_notes(CLF, NOTES)
    print("train self_check ok", flush=True)


def make_fill(clf):
    def fill(batch, table, policy, held) -> None:
        import torch

        if "index" not in batch:
            from harness import load_color_helpers

            raise load_color_helpers().StopTrain("batch has no index; cannot align cube xy")
        state = batch["observation.state"]
        index = batch["index"].detach().reshape(-1)
        if index.numel() != state.shape[0]:
            raise RuntimeError(f"index {tuple(index.shape)} != state batch {state.shape[0]}")
        idx = index.cpu().numpy().astype(np.int64)
        if int(idx.min()) < 0 or int(idx.max()) >= len(table):
            raise RuntimeError(f"frame index {idx.min()}..{idx.max()} outside labels {len(table)}")
        held["skip_hook"] = True
        try:
            notes = batch_notes(policy, batch)
        finally:
            held["skip_hook"] = False
        logits_np = two_logits(notes, *clf)
        if not np.isfinite(logits_np).all():
            raise RuntimeError("logits are not finite")
        scored = torch.from_numpy((table[idx] - view.XY_MEAN) / view.XY_STD)
        logits = torch.from_numpy(logits_np)
        out = write_state(state, scored, logits)
        if float(out[..., 12:].abs().sum()) != 0:
            raise RuntimeError("dims 12:32 are not zero")
        batch["observation.state"] = out
        held["xy"] = (out[:, -1, 6:10] if out.ndim == 3 else out[:, 6:10]).detach()
        held["logits"] = (out[:, -1, 10:12] if out.ndim == 3 else out[:, 10:12]).detach()

    return fill


def refuse_seeds(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    blob = path.read_text(encoding="utf-8").lower()
    if "orange" in blob or "purple" in blob:
        raise SystemExit("held-out color words are in the training seeds")
    seeds = {int(row["seed"]) for row in rows}
    hit = sorted(seeds & ROLLOUT)
    if hit:
        raise SystemExit(f"rollout seeds are in the dataset: {hit}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--val-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--from-checkpoint", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--save-freq", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--green-weight", type=float, default=1.0)
    p.add_argument("--init-step", type=int, default=0)
    p.add_argument("--cut-minutes", type=float, default=0.0)
    p.add_argument("--cut-to", type=int, default=6000)
    p.add_argument("--self-check-only", action="store_true")
    args = p.parse_args()
    configure_fp32()
    try:
        self_check()
    except Exception as e:
        report = args.output_dir.parent / "REPORT.md"
        if not args.self_check_only:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                "# Phase 1 stopped before training\n\n"
                "The phase-1 classifier did not score val accuracy 1.0, or a self-check failed.\n\n"
                f"{e}\n\n"
                "No rollout.\n",
                encoding="utf-8",
            )
        raise SystemExit(1) from e
    if args.self_check_only:
        return
    refuse_seeds(args.dataset / "stack_seeds.json")
    refuse_seeds(args.val_dataset / "stack_seeds.json")
    meta = {
        "from_checkpoint": str(args.from_checkpoint),
        "init_step": args.init_step,
        "green_weight": args.green_weight,
        "requested_steps": args.steps,
        "save_freq": args.save_freq,
        "val_every": args.val_every,
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir.parent / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    clf = load_classifier(CLF)
    execute(
        out=args.output_dir,
        dataset=args.dataset,
        val_dataset=args.val_dataset,
        from_checkpoint=args.from_checkpoint,
        steps=args.steps,
        save_freq=args.save_freq,
        val_every=args.val_every,
        batch_size=args.batch_size,
        fill=make_fill(clf),
        logit_slice=(10, 12),
        zero_from=12,
        green_weight=args.green_weight,
        cut_minutes=args.cut_minutes,
        cut_to=args.cut_to,
        report_path=args.output_dir.parent / "REPORT.md",
    )


if __name__ == "__main__":
    main()
