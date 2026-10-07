"""Does the frozen 16th-layer word note tell red from green?

No arm rollout. No training. Loads trs_so_arm100/best in fp32, eval, frozen.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import view  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "probe"
CKPT = HERE.parent / "best" / "pretrained_model"
TRAIN_SEEDS = HERE.parent / "color" / "swap" / "datasets" / "train" / "stack_seeds.json"
VAL_SEEDS = HERE.parent / "color" / "swap" / "datasets" / "val" / "stack_seeds.json"
RED = "stack the red cube on the green cube"
GREEN = "stack the green cube on the red cube"
SENTENCES = ((RED, 0, "red_box"), (GREEN, 1, "green_box"))
ROLLOUT = set(range(30000, 30010))
WIDTH = 960
FRONT = 64
WRIST = 64
C = 1.0


class ShapeStop(RuntimeError):
    def __init__(self, shapes: dict):
        super().__init__(json.dumps(shapes))
        self.shapes = shapes


def unique_seeds(path: Path) -> list[int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    seeds = sorted({int(row["seed"]) for row in rows})
    dropped = sorted(set(seeds) & ROLLOUT)
    if dropped:
        print(f"dropped rollout seeds from {path.name}: {dropped}", flush=True)
    return [s for s in seeds if s not in ROLLOUT]


def self_check() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(30, 3)) + np.array([4.0, 0.0, 0.0])
    x1 = rng.normal(size=(30, 3)) + np.array([-4.0, 0.0, 0.0])
    x = np.vstack([x0, x1])
    y = np.array([0] * 30 + [1] * 30)
    w, b, info = fit_logistic(x, y)
    pred, _ = predict_scores(x, w, b)
    acc = float((pred == y).mean())
    if acc < 0.95 or not info["success"]:
        raise SystemExit(f"logistic self_check acc={acc} info={info}")

    theta = np.zeros(x.shape[1] + 1)
    theta[0] = 0.3
    _, grad = _loss_grad(theta, x, np.where(y == 1, 1.0, -1.0), 1.0 / C)
    eps = 1.0e-6
    for i in (0, len(theta) - 1):
        up = theta.copy()
        up[i] += eps
        down = theta.copy()
        down[i] -= eps
        num = (_loss_grad(up, x, np.where(y == 1, 1.0, -1.0), 1.0 / C)[0] - _loss_grad(down, x, np.where(y == 1, 1.0, -1.0), 1.0 / C)[0]) / (2 * eps)
        if abs(num - grad[i]) > 1.0e-5:
            raise SystemExit(f"logistic gradient {i}: numeric {num} analytic {grad[i]}")
    print("self_check ok", flush=True)


def _loss_grad(theta: np.ndarray, x: np.ndarray, y_pm: np.ndarray, alpha: float):
    w, b = theta[:-1], float(theta[-1])
    score = x @ w + b
    yz = y_pm * score
    neg = -yz
    softplus = np.where(neg > 0, neg + np.log1p(np.exp(-neg)), np.log1p(np.exp(neg)))
    loss = float(softplus.sum() + 0.5 * alpha * np.dot(w, w))
    # d softplus(-y score) / d score = sigmoid(-y score) * (-y)
    sig = np.where(neg >= 0, 1.0 / (1.0 + np.exp(-neg)), np.exp(neg) / (1.0 + np.exp(neg)))
    ds = sig * (-y_pm)
    grad = np.concatenate([x.T @ ds + alpha * w, [ds.sum()]])
    return loss, grad


def fit_logistic(x: np.ndarray, y: np.ndarray, c: float = C):
    from scipy.optimize import minimize

    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.int64)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(f"labels {np.unique(y)}")
    y_pm = np.where(y == 1, 1.0, -1.0)
    alpha = 1.0 / c

    def fun(theta):
        return _loss_grad(theta, x, y_pm, alpha)

    res = minimize(fun, np.zeros(x.shape[1] + 1), method="L-BFGS-B", jac=True, options={"maxiter": 1000, "ftol": 1e-15})
    info = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "loss": float(res.fun),
        "message": str(res.message),
    }
    return res.x[:-1].copy(), float(res.x[-1]), info


def predict_scores(x: np.ndarray, w: np.ndarray, b: float):
    score = np.asarray(x, np.float64) @ w + b
    return (score >= 0.0).astype(np.int64), score


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise RuntimeError("a note is the zero vector")
    return float(1.0 - np.dot(a, b) / (na * nb))


def standardize(train: np.ndarray, other: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std_safe = std.copy()
    std_safe[std_safe == 0.0] = 1.0
    return (train - mean) / std_safe, (other - mean) / std_safe, mean, std


def choose_gate(val_acc: float, shuf_acc: float) -> tuple[str, str]:
    if val_acc >= 0.90 and shuf_acc <= 0.60:
        return (
            "works",
            "The note works: val accuracy is at least 0.90 and shuffled-val accuracy is at most 0.60. Step 2 runs. Step 3 does not.",
        )
    reasons = []
    if val_acc <= 0.65:
        reasons.append("val accuracy is at most 0.65")
    if abs(val_acc - shuf_acc) <= 0.10:
        reasons.append("val accuracy is within 0.10 of the shuffled-val accuracy")
    if reasons:
        why = " and ".join(reasons)
        return (
            "fails",
            f"The note fails: {why}. Step 3 runs. Step 2 does not.",
        )
    return (
        "partial",
        "The note is only partly separated, so neither train is justified yet.",
    )


def capture(sim, renderer):
    images = {}
    for name in ("camera1", "camera2"):
        renderer.update_scene(sim.data, camera=name)
        images[name] = renderer.render().copy()
    xy = view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2])
    return images, sim.degrees(), xy


def build_batch(pre, images, degrees, task, xy):
    import torch

    obs = {
        "observation.state": torch.from_numpy(degrees.astype(np.float32)),
        "task": task,
    }
    for name, img in images.items():
        obs[f"observation.images.{name}"] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    batch = pre(obs)
    state = batch["observation.state"]
    scored = torch.as_tensor(xy, dtype=state.dtype, device=state.device).reshape(1, 4)
    batch["observation.state"] = view.write_xy(state, scored)
    return batch


def word_note(policy, batch) -> tuple[np.ndarray, dict]:
    import torch
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model = policy.model
    vlm = model.vlm_with_expert
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    text_model = vlm.get_vlm_model().text_model
    saved_norm = text_model.norm
    text_model.norm = torch.nn.Identity()
    try:
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            outputs, _ = vlm.forward(
                attention_mask=att_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
    finally:
        text_model.norm = saved_norm

    hidden = outputs[0]
    n_lang = int(lang_tokens.shape[-1])
    seq = int(hidden.shape[1])
    n_img = seq - n_lang - 1
    shapes = {
        "hidden": list(hidden.shape),
        "n_images_in_batch": len(images),
        "n_img_tokens": n_img,
        "n_lang_positions": n_lang,
        "lang_tokens": list(lang_tokens.shape),
        "lang_mask_sum": int(lang_masks.sum().item()),
        "state": list(state.shape),
        "num_vlm_layers": int(vlm.num_vlm_layers),
        "n_text_layers": len(text_model.layers),
        "add_image_special_tokens": bool(policy.config.add_image_special_tokens),
        "prefix_length": int(policy.config.prefix_length),
    }
    if hidden.shape[-1] != WIDTH or hidden.ndim != 3:
        raise ShapeStop(shapes)
    if n_img != FRONT + WRIST or n_lang < 1 or shapes["n_text_layers"] != 16:
        raise ShapeStop(shapes)
    if shapes["add_image_special_tokens"] or shapes["prefix_length"] != 0:
        raise ShapeStop(shapes)
    mask = lang_masks.bool()
    if mask.ndim != 2 or mask.shape[0] != 1:
        shapes["lang_mask"] = list(lang_masks.shape)
        raise ShapeStop(shapes)
    words = hidden[:, n_img : n_img + n_lang, :]
    picked = words[0][mask[0]]
    shapes["n_word_tokens"] = int(picked.shape[0])
    shapes["word_width"] = int(picked.shape[-1]) if picked.ndim == 2 else None
    if picked.shape[0] == 0 or picked.shape[-1] != WIDTH:
        raise ShapeStop(shapes)
    note = picked.mean(dim=0).detach().float().cpu().numpy()
    shapes["decoded"] = _decode(policy, lang_tokens[0], mask[0])
    return note, shapes


def _decode(policy, token_ids, mask) -> str:
    try:
        tok = policy.model.vlm_with_expert.processor.tokenizer
        ids = token_ids[mask].detach().cpu().tolist()
        return tok.decode(ids)
    except Exception as exc:
        return f"<decode failed: {exc}>"


def camera_token_counts(policy, batch) -> list[int]:
    images, _ = policy.prepare_images(batch)
    vlm = policy.model.vlm_with_expert
    counts = []
    with __import__("torch").no_grad():
        for img in images:
            counts.append(int(vlm.embed_image(img).shape[1]))
    return counts


def annotate(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
    pad = 6
    heights, widths = [], []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    block_w = max(widths) + pad * 2
    block_h = sum(heights) + pad * (len(lines) + 1)
    draw.rectangle((8, 8, 8 + block_w, 8 + block_h), fill=(0, 0, 0, 190))
    y = 8 + pad
    for line, h in zip(lines, heights):
        draw.text((8 + pad, y), line, font=font, fill=(255, 255, 255, 255))
        y += h + pad
    return np.asarray(Image.alpha_composite(im, overlay).convert("RGB"))


def plot_accuracy(path: Path, train_acc: float, val_acc: float, shuf_acc: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["train", "val", "shuffled val"]
    vals = [train_acc, val_acc, shuf_acc]
    ax.bar(labels, vals, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_distance(path: Path, train_d: np.ndarray, val_d: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = float(max(train_d.max(), val_d.max()))
    bins = np.linspace(0, hi * 1.05 + 1.0e-6, 16)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    axes[0].hist(train_d, bins=bins, color="#4c78a8")
    axes[0].set_title("train")
    axes[1].hist(val_d, bins=bins, color="#f58518")
    axes[1].set_title("val")
    for ax in axes:
        ax.set_xlabel("cosine distance")
    axes[0].set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_notes(path: Path, train_z: np.ndarray, y_train: np.ndarray, val_z: np.ndarray, y_val: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mu = train_z.mean(axis=0)
    _, _, vt = np.linalg.svd(train_z - mu, full_matrices=False)
    comp = vt[:2]
    tr = (train_z - mu) @ comp.T
    va = (val_z - mu) @ comp.T
    fig, ax = plt.subplots(figsize=(6, 5))
    styles = (
        (tr, y_train, "o", "train"),
        (va, y_val, "^", "val"),
    )
    colors = {0: "#c0392b", 1: "#1e8449"}
    names = {0: "red source", 1: "green source"}
    for pts, labels, marker, split in styles:
        for lab in (0, 1):
            m = labels == lab
            ax.scatter(pts[m, 0], pts[m, 1], c=colors[lab], marker=marker, label=f"{split}, {names[lab]}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_shape_report(shapes: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"stopped": "word span", "shapes": shapes}
    (OUT / "probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Can the frozen sentence tell red from green?",
        "",
        "The word-token span could not be used, or its width is not 960. No classifier was fit. No training.",
        "",
        "```",
        json.dumps(shapes, indent=2),
        "```",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(blob: dict, frame_name: str, decoded: dict) -> None:
    dist = blob["cosine_distances"]
    train_mean = float(np.mean(dist["train"]))
    val_mean = float(np.mean(dist["val"]))
    quote = (
        f"accuracies train={blob['train_accuracy']:.6f} val={blob['val_accuracy']:.6f} "
        f"shuffled_val={blob['shuffled_val_accuracy']:.6f} gate={blob['gate']}"
    )
    lines = [
        "# Can the frozen sentence tell red from green?",
        "",
        f"Train accuracy {blob['train_accuracy']:.3f} ({blob['train_correct']}/{blob['n_train_notes']}). "
        f"Val accuracy {blob['val_accuracy']:.3f} ({blob['val_correct']}/{blob['n_val_notes']}). "
        f"Shuffled-val accuracy {blob['shuffled_val_accuracy']:.3f} ({blob['shuffled_val_correct']}/{blob['n_val_notes']}). "
        f"{blob['gate_text']}",
        "",
        "The note is the mean of the word-token hidden states after the 16th text layer, before the final text norm. "
        f"Width {blob['word_width']}. The {FRONT} front-camera tokens, the {WRIST} wrist tokens, and the state token are left out. "
        f"Each note averages {blob['n_word_tokens']} tokens whose language attention mask is 1. "
        f"The language block is {blob['n_lang_positions']} positions, padding included in the block and left out of the mean.",
        "",
        f"Train seeds {blob['n_train_seeds']}. Val seeds {blob['n_val_seeds']}. Seeds 30000–30009 were not fit. "
        "The classifier is L2 logistic regression, C=1.0, on notes standardized with the train mean and std. "
        "The label is the source body the sentence names. `red_box` is class 0 and `green_box` is class 1. "
        "The shuffled classifier refits on the train labels permuted with seed 0 and is scored on the true val labels.",
        "",
        f"Mean cosine distance between the two notes on the same frame: train {train_mean:.4f}, val {val_mean:.4f}. "
        "Distance is 1 minus the cosine of the raw notes, before standardization.",
        "",
        f"First-seed tokens: red sentence `{decoded['red']}`, green sentence `{decoded['green']}`.",
        "",
        "![accuracy](accuracy.png)",
        "",
        "![distance](distance.png)",
        "",
        "![notes](notes.png)",
        "",
        f"![frame](frames/{frame_name})",
        "",
        "`frames/` has four val seeds. Both sentences and the classifier's call are written on the same top-camera still. "
        "The cubes are that seed's reset layout.",
        "",
        "Accuracies and distances are from `probe.json`.",
        "",
        "```",
        quote,
        "```",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def collect(policy, pre, sim, renderer, seeds: list[int], frame_seeds: set[int]):
    notes_red = []
    notes_green = []
    frames = {}
    decoded = {}
    shapes = None
    counts = None
    for i, seed in enumerate(seeds):
        sim.reset(seed)
        images, degrees, xy = capture(sim, renderer)
        pair = {}
        for task, _label, _name in SENTENCES:
            batch = build_batch(pre, images, degrees, task, xy)
            if counts is None:
                counts = camera_token_counts(policy, batch)
                print(f"per-camera tokens {counts}", flush=True)
                if counts != [FRONT, WRIST]:
                    raise ShapeStop({"per_camera_tokens": counts})
            note, shapes = word_note(policy, batch)
            pair[task] = note
            if seed == seeds[0] and task not in decoded:
                decoded[task] = shapes["decoded"]
                print(f"tokens {task!r} -> {shapes['decoded']!r} n_words={shapes['n_word_tokens']}", flush=True)
        notes_red.append(pair[RED])
        notes_green.append(pair[GREEN])
        if seed in frame_seeds:
            frames[seed] = images["camera1"]
        if i % 10 == 0 or i + 1 == len(seeds):
            print(f"seed {seed} ({i + 1}/{len(seeds)})", flush=True)
    return np.stack(notes_red), np.stack(notes_green), frames, decoded, shapes


def main() -> None:
    self_check()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    OUT.mkdir(parents=True, exist_ok=True)
    train_seeds = unique_seeds(TRAIN_SEEDS)
    val_seeds = unique_seeds(VAL_SEEDS)
    overlap = sorted(set(train_seeds) & set(val_seeds))
    if overlap:
        raise SystemExit(f"train/val seeds overlap {overlap[:8]}")
    frame_seeds = val_seeds[:4]
    print(f"train seeds {len(train_seeds)} val seeds {len(val_seeds)} frames {frame_seeds}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("cuda is not available")
    print(f"loading {CKPT}", flush=True)
    policy, pre, _post = view.load_policy(CKPT, device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    sim = view.StackSim()
    renderer = __import__("mujoco").Renderer(sim.model, height=480, width=640)
    try:
        tr_red, tr_green, _frames, decoded_tr, shapes = collect(policy, pre, sim, renderer, train_seeds, set())
        va_red, va_green, frames, decoded_va, _shapes = collect(policy, pre, sim, renderer, val_seeds, set(frame_seeds))
    except ShapeStop as exc:
        write_shape_report(exc.shapes)
        print(f"stopped: word span {exc.shapes}", flush=True)
        return

    decoded = {"red": decoded_tr[RED], "green": decoded_tr[GREEN]}
    x_train = np.empty((len(train_seeds) * 2, WIDTH), np.float64)
    y_train = np.empty(len(train_seeds) * 2, np.int64)
    for i in range(len(train_seeds)):
        x_train[2 * i] = tr_red[i]
        x_train[2 * i + 1] = tr_green[i]
        y_train[2 * i] = 0
        y_train[2 * i + 1] = 1
    x_val = np.empty((len(val_seeds) * 2, WIDTH), np.float64)
    y_val = np.empty(len(val_seeds) * 2, np.int64)
    for i in range(len(val_seeds)):
        x_val[2 * i] = va_red[i]
        x_val[2 * i + 1] = va_green[i]
        y_val[2 * i] = 0
        y_val[2 * i + 1] = 1

    train_d = np.array([cosine_distance(tr_red[i], tr_green[i]) for i in range(len(train_seeds))])
    val_d = np.array([cosine_distance(va_red[i], va_green[i]) for i in range(len(val_seeds))])
    z_train, z_val, mean, std = standardize(x_train, x_val)
    w, b, info = fit_logistic(z_train, y_train)
    if not info["success"]:
        raise RuntimeError(f"logistic fit did not converge: {info}")
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y_train)
    w_s, b_s, info_s = fit_logistic(z_train, y_shuf)
    if not info_s["success"]:
        raise RuntimeError(f"shuffled logistic fit did not converge: {info_s}")

    pred_train, score_train = predict_scores(z_train, w, b)
    pred_val, score_val = predict_scores(z_val, w, b)
    pred_shuf, _ = predict_scores(z_val, w_s, b_s)
    train_correct = int((pred_train == y_train).sum())
    val_correct = int((pred_val == y_val).sum())
    shuf_correct = int((pred_shuf == y_val).sum())
    train_acc = train_correct / len(y_train)
    val_acc = val_correct / len(y_val)
    shuf_acc = shuf_correct / len(y_val)
    gate, gate_text = choose_gate(val_acc, shuf_acc)
    print(
        f"accuracies train={train_acc:.6f} val={val_acc:.6f} shuffled_val={shuf_acc:.6f} gate={gate}",
        flush=True,
    )
    print(f"fit {info}", flush=True)
    print(f"shuffled fit {info_s}", flush=True)

    calls = {}
    for i, seed in enumerate(val_seeds):
        calls[seed] = (int(pred_val[2 * i]), int(pred_val[2 * i + 1]))

    frame_dir = OUT / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    names = {0: "red_box", 1: "green_box"}
    for seed in frame_seeds:
        red_call, green_call = calls[seed]
        lines = [
            RED,
            f"call: {names[red_call]}",
            GREEN,
            f"call: {names[green_call]}",
        ]
        Image.fromarray(annotate(frames[seed], lines)).save(frame_dir / f"{seed}.png")

    plot_accuracy(OUT / "accuracy.png", train_acc, val_acc, shuf_acc)
    plot_distance(OUT / "distance.png", train_d, val_d)
    plot_notes(OUT / "notes.png", z_train, y_train, z_val, y_val)
    np.savez(
        OUT / "classifier.npz",
        weight=w,
        bias=np.array([b]),
        mean=mean,
        std=std,
        classes=np.array(["red_box", "green_box"]),
    )
    np.savez(
        OUT / "notes.npz",
        train_red=tr_red,
        train_green=tr_green,
        val_red=va_red,
        val_green=va_green,
        train_seeds=np.array(train_seeds),
        val_seeds=np.array(val_seeds),
    )

    blob = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "shuffled_val_accuracy": shuf_acc,
        "train_correct": train_correct,
        "val_correct": val_correct,
        "shuffled_val_correct": shuf_correct,
        "n_train_seeds": len(train_seeds),
        "n_val_seeds": len(val_seeds),
        "n_train_notes": int(len(y_train)),
        "n_val_notes": int(len(y_val)),
        "cosine_distances": {
            "train": train_d.tolist(),
            "val": val_d.tolist(),
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
        },
        "gate": gate,
        "gate_text": gate_text,
        "word_width": WIDTH,
        "n_word_tokens": shapes["n_word_tokens"],
        "n_lang_positions": shapes["n_lang_positions"],
        "per_camera_tokens": [FRONT, WRIST],
        "fit": info,
        "shuffled_fit": info_s,
        "n_zero_std": int((std == 0).sum()),
        "decoded": decoded,
        "val_decoded": {"red": decoded_va[RED], "green": decoded_va[GREEN]},
    }
    (OUT / "probe.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    write_report(blob, f"{frame_seeds[0]}.png", decoded)
    print(f"wrote {OUT / 'REPORT.md'}", flush=True)
    renderer.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "crash.txt").write_text(err, encoding="utf-8")
        raise
