"""Train SigLIP so the front-camera tokens can locate the named ball.

The readout is a linear map on the 8x8 connector grid (64 tokens, not averaged)
plus a color one-hot. It is thrown away. A checkpoint is written only when the
held-out reset error is under 5 cm.

--frozen fits that readout on a frozen tower and exits. The tokens are cached
under artifacts/vision_xy/ so a crash during the fit does not re-run SigLIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from openarm_vla.constants import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_smolvla import unfreeze_vision  # noqa: E402

GATE_CM = 5.0
VISION_LR = 1.0e-5
HEAD_LR = 1.0e-3
BATCH = 4
GRID = 8
DIM = 960
N_TOK = GRID * GRID
N_COLOR = 5
FEAT = N_TOK * DIM
HOURS = 2.0
ART = REPO_ROOT / "artifacts" / "vision_xy"


def load_front(labels: Path, n: int) -> np.ndarray:
    path = labels.with_name(labels.stem + "_front.u8")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(n, 256, 256, 3))


def one_hot(color, device):
    import torch

    color = torch.as_tensor(color, device=device)
    oh = torch.zeros(len(color), N_COLOR, device=device)
    oh[torch.arange(len(color), device=device), color] = 1
    return oh


def make_head():
    import torch

    head = torch.nn.Linear(FEAT + N_COLOR, 2)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def apply_head(head, tokens, color):
    flat = tokens.reshape(tokens.shape[0], GRID, GRID, DIM).reshape(tokens.shape[0], FEAT)
    return head(torch_cat(flat / FEAT**0.5, one_hot(color, tokens.device)))


def torch_cat(flat, oh):
    import torch

    return torch.cat([flat, oh], dim=-1)


def image_batch(images, index, device):
    import torch
    from lerobot.policies.common.vla_utils import resize_with_pad

    frame = np.ascontiguousarray(images[index]).copy()
    img = torch.from_numpy(frame).permute(0, 3, 1, 2).float().div_(255)
    return resize_with_pad(img.to(device), 512, 512, pad_value=0) * 2 - 1


def embed(policy, images, index, device):
    tokens = policy.model.vlm_with_expert.embed_image(image_batch(images, index, device))
    if tokens.shape[1:] != (N_TOK, DIM):
        raise RuntimeError(f"expected {(N_TOK, DIM)} front tokens, got {tuple(tokens.shape)}")
    return tokens


def mean_cm(pred, target) -> float:
    return float((pred.detach() - target.detach()).norm(dim=-1).mean()) * 100


def color_only_cm(src_xy, src_color, src_mask, dst_xy, dst_color, dst_mask) -> float:
    means = []
    for c in range(N_COLOR):
        m = src_mask & (src_color == c)
        if not np.any(m):
            raise RuntimeError(f"color {c} has no reset frames")
        means.append(src_xy[m].mean(0))
    idx = np.flatnonzero(dst_mask)
    pred = np.stack(means)[dst_color[idx]]
    return float(np.linalg.norm(pred - dst_xy[idx], axis=1).mean() * 100)


def reset_cm(policy, head, images, xy, color, reset, device) -> float:
    import torch

    idx = np.flatnonzero(reset)
    preds = []
    with torch.no_grad():
        for j in range(0, len(idx), BATCH):
            preds.append(apply_head(head, embed(policy, images, idx[j : j + BATCH], device), color[idx[j : j + BATCH]]))
    pred = torch.cat(preds)
    target = torch.from_numpy(np.ascontiguousarray(xy[idx])).to(device)
    return mean_cm(pred, target)


def load_policy(trainable: bool):
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    base = snapshot_download("lerobot/smolvla_base")
    policy = SmolVLAPolicy.from_pretrained(base).float().cuda()
    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    if trainable:
        unfreeze_vision(policy)
        connector = policy.model.vlm_with_expert.get_vlm_model().connector
        if any(p.requires_grad for p in connector.parameters()):
            raise RuntimeError("connector is trainable")
        policy.train()
    else:
        policy.eval()
        if any(p.requires_grad for p in vision.parameters()):
            raise RuntimeError("vision is trainable in the frozen probe")
    return policy, base


def extract(policy, images, path: Path, device) -> None:
    import torch

    n = len(images)
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(n, N_TOK, DIM))
    t0 = time.perf_counter()
    with torch.inference_mode():
        for j in range(0, n, BATCH):
            stop = min(j + BATCH, n)
            mm[j:stop] = embed(policy, images, np.arange(j, stop), device).half().cpu().numpy()
            if j % 400 == 0:
                rate = (j + BATCH) / max(time.perf_counter() - t0, 1e-3)
                print(f"extract {j}/{n} {rate:.1f} img/s", flush=True)
    mm.flush()


def feature_stats(path: Path, n: int):
    mm = np.memmap(path, dtype=np.float16, mode="r", shape=(n, N_TOK, DIM))
    total = np.zeros(FEAT, np.float64)
    total_sq = np.zeros(FEAT, np.float64)
    peak = 0.0
    for j in range(0, n, 256):
        x = np.asarray(mm[j : j + 256], dtype=np.float64).reshape(-1, FEAT)
        total += x.sum(0)
        total_sq += np.square(x).sum(0)
        peak = max(peak, float(np.abs(x).max()))
    mean = total / n
    var = total_sq / n - np.square(mean)
    std = np.sqrt(np.maximum(var, 0))
    std[std < 1e-3] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), peak


def flat_grid(tokens, mean, std):
    flat = (tokens.float().reshape(tokens.shape[0], FEAT) - mean) / std
    return (flat / FEAT**0.5).clamp(-1, 1)


def predict_xy(image_w, color_w, tokens, color, mean, std):
    flat = flat_grid(tokens, mean, std)
    return flat @ image_w.T + one_hot(color, tokens.device) @ color_w.T


def cm_xy(image_w, color_w, tokens, color, mean, std, xy, rows) -> float:
    import torch

    rows = torch.as_tensor(rows, device=tokens.device)
    pred = predict_xy(image_w, color_w, tokens[rows], color[rows], mean, std)
    target = torch.from_numpy(np.ascontiguousarray(xy[rows.detach().cpu().numpy()])).to(pred.device)
    return mean_cm(pred, target)


def color_matrix(xy, color, rows, device):
    import torch

    means = []
    for c in range(N_COLOR):
        picked = rows[color[rows] == c]
        if len(picked) == 0:
            raise RuntimeError(f"color {c} missing from the fit resets")
        means.append(xy[picked].mean(0))
    return torch.from_numpy(np.stack(means).astype(np.float32).T).to(device)


def train_readout(tokens, color, mean, std, xy, fit_idx, hold_idx, watch_idx, color_w, wd):
    import torch

    device = tokens.device
    image_w = torch.nn.Parameter(torch.zeros(2, FEAT, device=device))
    init_hold = cm_xy(image_w, color_w, tokens, color, mean, std, xy, hold_idx)
    init_train = cm_xy(image_w, color_w, tokens, color, mean, std, xy, watch_idx)
    best = (init_hold, -1, image_w.detach().cpu().clone(), init_train)
    print(f"wd {wd:g} epoch -1 holdout_reset_cm {init_hold:.2f} train_reset_cm {init_train:.2f}", flush=True)
    opt = torch.optim.AdamW([image_w], lr=1.0e-3, weight_decay=wd, foreach=False)
    y = torch.from_numpy(np.ascontiguousarray(xy[fit_idx])).to(device)
    fit = torch.as_tensor(fit_idx, device=device)
    bad = 0
    for epoch in range(30):
        perm = torch.randperm(len(fit), device=device)
        for i in range(0, len(fit), 256):
            b = perm[i : i + 256]
            pred = predict_xy(image_w, color_w, tokens[fit[b]], color[fit[b]], mean, std)
            loss = torch.nn.functional.mse_loss(pred, y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([image_w], 1.0)
            opt.step()
        hold_cm = cm_xy(image_w, color_w, tokens, color, mean, std, xy, hold_idx)
        train_cm = cm_xy(image_w, color_w, tokens, color, mean, std, xy, watch_idx)
        if hold_cm < best[0] - 0.05:
            best = (hold_cm, epoch, image_w.detach().cpu().clone(), train_cm)
            bad = 0
        else:
            bad += 1
        if epoch % 5 == 0 or bad >= 8:
            print(f"wd {wd:g} epoch {epoch} holdout_reset_cm {hold_cm:.2f} train_reset_cm {train_cm:.2f}", flush=True)
        if bad >= 8 and epoch >= 10:
            break
    image_w = torch.nn.Parameter(best[2].to(device))
    return image_w, best[0], best[1], best[3]


def run_frozen(train, val, train_labels, val_labels) -> None:
    import torch

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    n = len(train["xy"])
    tok_path = ART / "train_tokens.f16"
    val_path = ART / "val_reset_tokens.npy"
    ok = tok_path.with_suffix(".ok")
    if not (ok.exists() and ok.read_text(encoding="utf-8").strip() == str(n) and tok_path.exists()):
        policy, _ = load_policy(False)
        print("extract train tokens", flush=True)
        extract(policy, load_front(train_labels, n), tok_path, device)
        vidx = np.flatnonzero(val["reset"])
        got = []
        with torch.inference_mode():
            for j in range(0, len(vidx), BATCH):
                got.append(embed(policy, load_front(val_labels, len(val["xy"])), vidx[j : j + BATCH], device).half().cpu().numpy())
        np.save(val_path, np.concatenate(got))
        del policy
        torch.cuda.empty_cache()
        ok.write_text(str(n), encoding="utf-8")
    mean, std, peak = feature_stats(tok_path, n)
    print(f"token peak {peak:.1f}", flush=True)
    if peak > 60000:
        raise RuntimeError("tokens do not fit in float16")
    gpu = torch.empty(n, N_TOK, DIM, dtype=torch.float16, device=device)
    mm = np.memmap(tok_path, dtype=np.float16, mode="r", shape=(n, N_TOK, DIM))
    for j in range(0, n, 256):
        gpu[j : j + 256] = torch.from_numpy(np.array(mm[j : j + 256], copy=True))
    del mm
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    color = torch.from_numpy(np.ascontiguousarray(train["color"])).to(device)
    ep = np.cumsum(train["reset"].astype(np.int64)) - 1
    hold = (ep % 5) == 0
    hold_reset = np.flatnonzero(hold & train["reset"])
    print(f"holdout resets {len(hold_reset)}", flush=True)
    fit_idx = np.flatnonzero(~hold)
    fit_reset = np.flatnonzero(~hold & train["reset"])
    color_w = color_matrix(train["xy"], train["color"], fit_reset, device)
    chosen = None
    tried = []
    for wd in (0.0, 1e-3, 1e-2, 0.1):
        image_w, hold_cm, epoch, train_cm = train_readout(
            gpu, color, mean_t, std_t, train["xy"], fit_idx, hold_reset, fit_reset, color_w, wd
        )
        tried.append({"weight_decay": wd, "epoch": epoch, "holdout_reset_cm": hold_cm, "train_reset_cm": train_cm})
        print(f"wd {wd:g} best_epoch {epoch} holdout_reset_cm {hold_cm:.2f} train_reset_cm {train_cm:.2f}", flush=True)
        if chosen is None or hold_cm < chosen[0]:
            chosen = (hold_cm, train_cm, wd, epoch, image_w.detach().cpu().clone())
    image_w = torch.nn.Parameter(chosen[4].to(device))
    val_tok = torch.from_numpy(np.load(val_path)).to(device)
    val_color = torch.from_numpy(np.ascontiguousarray(val["color"][val["reset"]])).to(device)
    val_cm = cm_xy(image_w, color_w, val_tok, val_color, mean_t, std_t, val["xy"][val["reset"]], np.arange(len(val_tok)))
    color_cm = color_only_cm(train["xy"], train["color"], train["reset"], val["xy"], val["color"], val["reset"])
    probe = {
        "val_reset_cm": val_cm,
        "train_reset_cm": chosen[1],
        "holdout_reset_cm": chosen[0],
        "color_only_val_reset_cm": color_cm,
        "weight_decay": chosen[2],
        "epoch": chosen[3],
        "tried": tried,
        "passed": val_cm < GATE_CM,
        "gate_cm": GATE_CM,
        "readout": "linear 8x8x960 + color",
    }
    path = ART / "grid_frozen.json"
    path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(json.dumps(probe), flush=True)


def run_train(args, train, val, train_img, val_img) -> None:
    import torch

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    policy, base = load_policy(True)
    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    head = make_head().to(device)
    opt = torch.optim.AdamW(
        [{"params": vision.parameters(), "lr": VISION_LR}, {"params": head.parameters(), "lr": HEAD_LR}],
        foreach=False,
    )
    log_path = ART / "grid_train_log.csv"
    if log_path.exists():
        raise SystemExit(f"{log_path} already exists")
    n = len(train["xy"])
    order = np.arange(n)
    rng = np.random.default_rng(0)
    cursor = n
    best_cm = float("inf")
    best_state = None
    mem = 0
    t0 = time.monotonic()
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "loss", "val_reset_cm", "train_reset_cm"])
    for step in range(1, args.steps + 1):
        if cursor + BATCH > n:
            rng.shuffle(order)
            cursor = 0
        index = order[cursor : cursor + BATCH]
        cursor += BATCH
        pred = apply_head(head, embed(policy, train_img, index, device), train["color"][index])
        target = torch.from_numpy(np.ascontiguousarray(train["xy"][index])).to(device)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(vision.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        row = [step, float(loss.detach()), "", ""]
        if step % args.val_every == 0:
            policy.eval()
            cm = reset_cm(policy, head, val_img, val["xy"], val["color"], val["reset"], device)
            train_cm = reset_cm(policy, head, train_img, train["xy"], train["color"], train["reset"], device)
            policy.train()
            row[2] = f"{cm:.4f}"
            row[3] = f"{train_cm:.4f}"
            if cm < best_cm:
                best_cm = cm
                best_state = {k: v.detach().cpu().clone() for k, v in vision.state_dict().items()}
            mem = mem + 1 if train_cm < 1.0 and cm > 8.0 else 0
            print(
                f"step {step} loss {float(loss.detach()):.5f} val_reset_cm {cm:.2f} "
                f"train_reset_cm {train_cm:.2f} best {best_cm:.2f}",
                flush=True,
            )
        elif step % 10 == 0:
            print(f"step {step} loss {float(loss.detach()):.5f}", flush=True)
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        if row[2] and float(row[2]) < GATE_CM:
            break
        if row[2] and step >= 400 and mem >= 3:
            print(f"memorized at {best_cm:.2f} cm", flush=True)
            break
        if time.monotonic() - t0 > HOURS * 3600:
            print("time cap", flush=True)
            break

    probe = {"val_reset_cm": best_cm, "passed": best_cm < GATE_CM, "gate_cm": GATE_CM, "readout": "linear 8x8x960 + color"}
    probe_path = ART / "grid_probe.json"
    if best_state is None:
        probe["passed"] = False
        probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
        raise SystemExit("no validation")
    vision.load_state_dict(best_state)
    probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(json.dumps(probe), flush=True)
    if not probe["passed"]:
        raise SystemExit(f"val reset error {best_cm:.2f} cm is not under {GATE_CM:.0f} cm; not saving")
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists")
    dest = args.out / "pretrained_model"
    shutil.copytree(base, dest)
    from safetensors.torch import save_model

    save_model(policy, str(dest / "model.safetensors"))
    print(f"saved {dest}", flush=True)


def self_check() -> None:
    import torch

    color = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    xy = np.zeros((10, 2), np.float32)
    xy[:, 0] = color
    mask = np.ones(10, bool)
    got = color_only_cm(xy, color, mask, xy, color, mask)
    if got > 1e-5:
        raise SystemExit(f"color baseline {got}")

    head = make_head()
    head.weight.data.normal_(0, 1e-3)
    tokens = torch.randn(2, N_TOK, DIM, requires_grad=True)
    apply_head(head, tokens, np.array([0, 1])).sum().backward()
    if torch.allclose(tokens.grad[:, 0], tokens.grad[:, 63]):
        raise SystemExit("readout treats every token the same")
    print("self_check ok", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-labels", type=Path, default=ART / "train_head.npz")
    p.add_argument("--val-labels", type=Path, default=ART / "val_head.npz")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "vision_xy")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    train = np.load(args.train_labels)
    val = np.load(args.val_labels)
    if args.frozen:
        run_frozen(train, val, args.train_labels, args.val_labels)
        return
    run_train(args, train, val, load_front(args.train_labels, len(train["xy"])), load_front(args.val_labels, len(val["xy"])))


if __name__ == "__main__":
    main()
