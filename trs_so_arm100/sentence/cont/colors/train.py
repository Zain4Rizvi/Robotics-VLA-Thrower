"""Color-word train. Source logits in dims 10:14, target logits in dims 14:18.

Does not call color/train.py assert_expert_only. Yellow on blue, orange, and
purple must not appear in the train or val seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))
sys.path.insert(0, str(ROOT / "sentence" / "cont"))
sys.path.insert(0, str(HERE))

import view  # noqa: E402
from harness import execute  # noqa: E402
from layout import (  # noqa: E402
    PAIRS,
    eight_logits,
    forbidden,
    load_color_classifier,
    self_check as layout_check,
    write_color_state,
)
from note import batch_notes, configure_fp32  # noqa: E402

PROBE = HERE / "probe" / "probe.json"
CLF = HERE / "probe" / "classifier.npz"


def check_seeds(path: Path, n_each: int) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(rows).lower()
    if forbidden(blob):
        raise SystemExit(f"held-out sentence or word in {path}")
    counts = {}
    for row in rows:
        key = (row["source_color"], row["target_color"])
        counts[key] = counts.get(key, 0) + 1
        if row["sentence"] != f"stack the {row['source_color']} cube on the {row['target_color']} cube":
            raise SystemExit(f"sentence mismatch in {path}")
    if set(counts) != set(PAIRS) or any(v != n_each for v in counts.values()):
        raise SystemExit(f"pair counts {counts}")


def make_fill(pack):
    def fill(batch, table, policy, held) -> None:
        import torch

        if "index" not in batch:
            raise RuntimeError("batch has no index; cannot align cube xy")
        state = batch["observation.state"]
        index = batch["index"].detach().reshape(-1)
        idx = index.cpu().numpy().astype(np.int64)
        if int(idx.min()) < 0 or int(idx.max()) >= len(table):
            raise RuntimeError(f"frame index outside xy table {len(table)}")
        held["skip_hook"] = True
        try:
            notes = batch_notes(policy, batch)
        finally:
            held["skip_hook"] = False
        logits_np = eight_logits(notes, pack)
        if logits_np.shape[1] != 8 or not np.isfinite(logits_np).all():
            raise RuntimeError(f"logits {logits_np.shape}")
        scored = torch.from_numpy((table[idx] - view.XY_MEAN) / view.XY_STD)
        logits = torch.from_numpy(logits_np)
        src = logits[:, :4]
        tgt = logits[:, 4:]
        out = write_color_state(state, scored, src, tgt)
        if float(out[..., 18:].abs().sum()) != 0:
            raise RuntimeError("dims 18:32 are not zero")
        batch["observation.state"] = out
        held["xy"] = (out[:, -1, 6:10] if out.ndim == 3 else out[:, 6:10]).detach()
        held["logits"] = (out[:, -1, 10:18] if out.ndim == 3 else out[:, 10:18]).detach()

    return fill


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--val-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--from-checkpoint", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--init-step", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    configure_fp32()
    layout_check()
    if not PROBE.exists():
        raise SystemExit("color probe report json is missing")
    blob = json.loads(PROBE.read_text(encoding="utf-8"))
    if blob.get("gate") != "pass":
        raise SystemExit(f"probe gate is {blob.get('gate')}; not training")
    check_seeds(args.dataset / "stack_seeds.json", 10)
    check_seeds(args.val_dataset / "stack_seeds.json", 4)
    meta = {
        "from_checkpoint": str(args.from_checkpoint),
        "init_step": args.init_step,
        "requested_steps": args.steps,
        "save_freq": 1000,
        "val_every": 500,
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir.parent / "init.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pack = load_color_classifier(CLF)
    execute(
        out=args.output_dir,
        dataset=args.dataset,
        val_dataset=args.val_dataset,
        from_checkpoint=args.from_checkpoint,
        steps=args.steps,
        save_freq=1000,
        val_every=500,
        batch_size=args.batch_size,
        fill=make_fill(pack),
        logit_slice=(10, 18),
        zero_from=18,
        green_weight=1.0,
        cut_minutes=0.0,
        report_path=args.output_dir.parent / "REPORT.md",
    )


if __name__ == "__main__":
    main()
