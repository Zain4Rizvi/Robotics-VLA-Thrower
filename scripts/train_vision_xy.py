"""Train SigLIP so the front-camera tokens can locate the named ball.

The linear readout is discarded. The saved checkpoint is smolvla_base with only the vision weights
replaced, so a later expert-only run freezes this tower and trains the action expert.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from openarm_vla.constants import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_smolvla import unfreeze_vision  # noqa: E402

GATE_CM = 5.0
VISION_LR = 1.0e-5
HEAD_LR = 1.0e-3
BATCH = 4


def load_front(labels: Path, n: int) -> np.ndarray:
    path = labels.with_name(labels.stem + "_front.u8")
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, 256, 256, 3))
    return mm


def batch_forward(policy, head, images, color, index, device):
    import torch
    from lerobot.policies.common.vla_utils import resize_with_pad

    img = torch.from_numpy(np.ascontiguousarray(images[index])).permute(0, 3, 1, 2).float().div_(255)
    img = resize_with_pad(img.to(device), 512, 512, pad_value=0) * 2 - 1
    tokens = policy.model.vlm_with_expert.embed_image(img)
    if tokens.shape[1] != 64:
        raise RuntimeError(f"expected 64 front tokens, got {tuple(tokens.shape)}")
    pooled = tokens.mean(dim=1)
    oh = torch.zeros(len(index), 5, device=device)
    oh[torch.arange(len(index), device=device), torch.as_tensor(color[index], device=device)] = 1
    return head(torch.cat([pooled, oh], dim=-1))


def reset_cm(policy, head, images, xy, color, reset, device) -> float:
    import torch

    idx = np.flatnonzero(reset)
    preds = []
    with torch.no_grad():
        for j in range(0, len(idx), BATCH):
            preds.append(batch_forward(policy, head, images, color, idx[j : j + BATCH], device))
    pred = torch.cat(preds)
    target = torch.from_numpy(xy[idx]).to(device)
    return float((pred - target).norm(dim=-1).mean()) * 100


def main():
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import save_model

    p = argparse.ArgumentParser()
    p.add_argument("--train-labels", type=Path, default=REPO_ROOT / "artifacts" / "vision_xy" / "train_head.npz")
    p.add_argument("--val-labels", type=Path, default=REPO_ROOT / "artifacts" / "vision_xy" / "val_head.npz")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "vision_xy")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--val-every", type=int, default=100)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists")

    train = np.load(args.train_labels)
    val = np.load(args.val_labels)
    train_img = load_front(args.train_labels, len(train["xy"]))
    val_img = load_front(args.val_labels, len(val["xy"]))

    base = snapshot_download("lerobot/smolvla_base")
    policy = SmolVLAPolicy.from_pretrained(base).float().cuda()
    unfreeze_vision(policy)
    connector = policy.model.vlm_with_expert.get_vlm_model().connector
    if any(p.requires_grad for p in connector.parameters()):
        raise RuntimeError("connector is trainable")
    policy.train()
    device = torch.device("cuda")
    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    head = torch.nn.Linear(960 + 5, 2).to(device)
    opt = torch.optim.AdamW(
        [{"params": vision.parameters(), "lr": VISION_LR}, {"params": head.parameters(), "lr": HEAD_LR}],
        foreach=False,
    )

    log_path = REPO_ROOT / "artifacts" / "vision_xy" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(train["xy"])
    order = np.arange(n)
    rng = np.random.default_rng(0)
    cursor = n  # force a shuffle on step 0
    best_cm = float("inf")
    best_state = None
    bad = 0
    for step in range(1, args.steps + 1):
        if cursor + BATCH > n:
            rng.shuffle(order)
            cursor = 0
        index = order[cursor : cursor + BATCH]
        cursor += BATCH
        pred = batch_forward(policy, head, train_img, train["color"], index, device)
        target = torch.from_numpy(train["xy"][index]).to(device)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vision.parameters(), 1.0)
        opt.step()
        row = [step, float(loss.detach()), ""]
        if step % args.val_every == 0:
            policy.eval()
            cm = reset_cm(policy, head, val_img, val["xy"], val["color"], val["reset"], device)
            policy.train()
            row[2] = f"{cm:.4f}"
            improved = cm < best_cm - 0.3
            if cm < best_cm:
                best_cm = cm
                best_state = {k: v.detach().cpu().clone() for k, v in vision.state_dict().items()}
            bad = 0 if improved else bad + 1
            print(f"step {step} loss {float(loss):.5f} val_reset_cm {cm:.2f} best {best_cm:.2f}", flush=True)
        elif step % 10 == 0:
            print(f"step {step} loss {float(loss):.5f}", flush=True)
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        if row[2] and float(row[2]) < GATE_CM:
            break
        if row[2] and step >= 400 and bad >= 8:
            print(f"plateau at {best_cm:.2f} cm", flush=True)
            break

    probe = {"val_reset_cm": best_cm, "passed": best_cm < GATE_CM, "gate_cm": GATE_CM}
    probe_path = REPO_ROOT / "artifacts" / "vision_xy" / "probe.json"
    if best_state is None:
        probe["passed"] = False
        probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
        raise SystemExit("no validation")
    vision.load_state_dict(best_state)
    probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(json.dumps(probe), flush=True)
    if not probe["passed"]:
        raise SystemExit(f"val reset error {best_cm:.2f} cm is not under {GATE_CM:.0f} cm; not saving, not training the expert")

    dest = args.out / "pretrained_model"
    shutil.copytree(base, dest)
    save_model(policy, str(dest / "model.safetensors"))
    print(f"saved {dest}", flush=True)


if __name__ == "__main__":
    main()
