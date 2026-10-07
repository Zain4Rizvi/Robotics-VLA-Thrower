"""Four-way color probe. One reset frame per episode. Does not touch sentence/probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASE1 = ROOT / "sentence" / "probe"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))
sys.path.insert(0, str(ROOT / "sentence"))
sys.path.insert(0, str(ROOT / "sentence" / "cont"))
sys.path.insert(0, str(HERE))

from layout import CLASSES, class_index, load_color_classifier, paint, self_check as layout_check  # noqa: E402
from note import batch_notes, configure_fp32  # noqa: E402

C = 1.0
HELD_SEEDS = list(range(33000, 33010))


def _loss_grad(theta: np.ndarray, x: np.ndarray, y: np.ndarray, n_class: int, alpha: float):
    n, d = x.shape
    w = theta[: n_class * d].reshape(n_class, d)
    b = theta[n_class * d :]
    scores = x @ w.T + b
    m = scores.max(axis=1, keepdims=True)
    exp = np.exp(scores - m)
    lse = m.ravel() + np.log(exp.sum(axis=1))
    correct = scores[np.arange(n), y]
    reg = 0.5 * alpha * float(np.sum(w * w))
    loss = float((lse - correct).sum() + reg)
    prob = exp / exp.sum(axis=1, keepdims=True)
    ds = prob
    ds[np.arange(n), y] -= 1.0
    gw = ds.T @ x + alpha * w
    gb = ds.sum(axis=0)
    return loss, np.concatenate([gw.ravel(), gb])


def fit_multinomial(x: np.ndarray, y: np.ndarray, n_class: int = 4, c: float = C):
    from scipy.optimize import minimize

    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.int64)
    if set(np.unique(y).tolist()) - set(range(n_class)):
        raise RuntimeError(f"labels {np.unique(y)}")
    alpha = 1.0 / c

    def fun(theta):
        return _loss_grad(theta, x, y, n_class, alpha)

    res = minimize(
        fun,
        np.zeros(n_class * x.shape[1] + n_class),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-12},
    )
    w = res.x[: n_class * x.shape[1]].reshape(n_class, x.shape[1])
    b = res.x[n_class * x.shape[1] :]
    return w, b, {"success": bool(res.success), "nit": int(res.nit), "loss": float(res.fun), "message": str(res.message)}


def accuracy(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    pred = np.argmax(x @ w.T + b, axis=1)
    return float((pred == y).mean())


def standardize(train: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std_safe = std.copy()
    std_safe[std_safe == 0.0] = 1.0
    return mean, std_safe


def apply_std(x, mean, std):
    return (x - mean) / std


def self_check() -> None:
    layout_check()
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for k in range(4):
        xs.append(rng.normal(size=(25, 6)) + np.eye(6)[k] * 4)
        ys.append(np.full(25, k))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    w, b, info = fit_multinomial(x, y)
    acc = accuracy(x, y, w, b)
    if acc < 0.95 or not info["success"]:
        raise SystemExit(f"multinomial self_check acc={acc} info={info}")
    theta = np.zeros(4 * 6 + 4)
    _, grad = _loss_grad(theta, x, y, 4, 1.0)
    eps = 1.0e-6
    up = theta.copy()
    up[0] += eps
    down = theta.copy()
    down[0] -= eps
    num = (_loss_grad(up, x, y, 4, 1.0)[0] - _loss_grad(down, x, y, 4, 1.0)[0]) / (2 * eps)
    if abs(num - grad[0]) > 1.0e-4:
        raise SystemExit(f"multinomial gradient numeric {num} analytic {grad[0]}")
    print("probe self_check ok", flush=True)


def float_images(batch: dict) -> dict:
    import torch

    return {
        k: v.float() / 255.0 if k.startswith("observation.images.") and v.dtype == torch.uint8 else v
        for k, v in batch.items()
    }


def frame_note(policy, pre, item) -> np.ndarray:
    from lerobot.utils.collate import lerobot_collate_fn

    batch = pre(float_images(lerobot_collate_fn([item])))
    return batch_notes(policy, batch)[0]


def held_notes(policy, pre, sim, renderer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sim_rollout import obs_batch, render_pair

    notes, src, tgt = [], [], []
    for seed in HELD_SEEDS:
        body, colors = paint(seed, "yellow", "blue")
        sim.reset(seed, source=body, colors=colors)
        if sim.task() != "stack the yellow cube on the blue cube":
            raise SystemExit(f"held task {sim.task()}")
        top, wrist = render_pair(sim, renderer)
        batch = obs_batch(pre, {"camera1": top, "camera2": wrist}, sim.degrees(), sim.task())
        notes.append(batch_notes(policy, batch)[0])
        src.append(class_index("yellow"))
        tgt.append(class_index("blue"))
    return np.stack(notes), np.asarray(src), np.asarray(tgt)


def capture_split(policy, pre, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rows = json.loads((root / "stack_seeds.json").read_text(encoding="utf-8"))
    ds = LeRobotDataset(f"local/{root.name}", root=root, video_backend="pyav")
    if len(ds) != sum(int(r["steps"]) for r in rows):
        raise SystemExit(f"{root} frames {len(ds)} != seeds steps")
    notes, src, tgt = [], [], []
    start = 0
    for row in rows:
        if row["sentence"] != f"stack the {row['source_color']} cube on the {row['target_color']} cube":
            raise SystemExit(f"seed row sentence {row['sentence']}")
        note = frame_note(policy, pre, ds[start])
        notes.append(note)
        src.append(class_index(row["source_color"]))
        tgt.append(class_index(row["target_color"]))
        start += int(row["steps"])
    return np.stack(notes), np.asarray(src), np.asarray(tgt)


def plot_accuracy(path: Path, scores: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "source train",
        "source val",
        "source held-out",
        "target train",
        "target val",
        "target held-out",
    ]
    vals = [
        scores["train_source"],
        scores["val_source"],
        scores["heldout_source"],
        scores["train_target"],
        scores["val_target"],
        scores["heldout_target"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color=["#4c78a8", "#f58518", "#54a24b"] * 2)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="0.9")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.tick_params(axis="x", labelrotation=20)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(out: Path, scores: dict) -> None:
    gate = scores["gate"]
    lead = (
        f"Train source {scores['train_source']:.3f}, val source {scores['val_source']:.3f}, "
        f"held-out source {scores['heldout_source']:.3f}. "
        f"Train target {scores['train_target']:.3f}, val target {scores['val_target']:.3f}, "
        f"held-out target {scores['heldout_target']:.3f}."
    )
    if gate == "pass":
        decision = "The gate passed: val and the held-out sentence are at least 0.90 on source and on target."
    else:
        decision = (
            "The gate missed. The frozen notes did not separate the color words. "
            "Color training was not started. The text layers stay frozen."
        )
    text = "\n".join(
        [
            "# Color probe",
            "",
            lead,
            "",
            decision,
            "",
            "Class order is red, green, blue, yellow. One reset frame per train episode, one per val episode, "
            f"and one reset frame for each of seeds {HELD_SEEDS[0]}–{HELD_SEEDS[-1]} with the yellow-on-blue sentence. "
            "Those held-out frames are not in the dataset. The note is the word-token mean, width 960.",
            "",
            "![accuracy](accuracy.png)",
            "",
            f"Source fit: {scores['source_fit']}. Target fit: {scores['target_fit']}.",
            "",
        ]
    )
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    import argparse

    import mujoco
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--self-check-only", action="store_true")
    args = p.parse_args()
    if args.self_check_only:
        self_check()
        return
    if args.out.resolve() == PHASE1.resolve():
        raise SystemExit("refusing to overwrite sentence/probe")
    if (args.out / "probe.json").exists() and (args.out / "REPORT.md").exists() and (args.out / "classifier.npz").exists():
        print("probe already written", flush=True)
        return
    self_check()
    configure_fp32()
    from expert import StackSim
    from sim_rollout import load_finetuned

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("cuda is not available")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        policy, pre, _post = load_finetuned(args.checkpoint, device)
        policy.eval()
        train_x, train_s, train_t = capture_split(policy, pre, args.train)
        val_x, val_s, val_t = capture_split(policy, pre, args.val)
        sim = StackSim()
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        held_x, held_s, held_t = held_notes(policy, pre, sim, renderer)
        renderer.close()
        del policy
        torch.cuda.empty_cache()
    except RuntimeError as e:
        (args.out / "REPORT.md").write_text(
            "# Color probe\n\nThe word-note capture failed.\n\n"
            f"{e}\n\nNo color training.\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from e
    if train_x.shape[1] != 960:
        raise SystemExit(f"note width {train_x.shape}")
    mean, std = standardize(train_x)
    z_train = apply_std(train_x, mean, std)
    z_val = apply_std(val_x, mean, std)
    z_held = apply_std(held_x, mean, std)
    sw, sb, sinfo = fit_multinomial(z_train, train_s)
    tw, tb, tinfo = fit_multinomial(z_train, train_t)
    scores = {
        "classes": list(CLASSES),
        "train_source": accuracy(z_train, train_s, sw, sb),
        "val_source": accuracy(z_val, val_s, sw, sb),
        "heldout_source": accuracy(z_held, held_s, sw, sb),
        "train_target": accuracy(z_train, train_t, tw, tb),
        "val_target": accuracy(z_val, val_t, tw, tb),
        "heldout_target": accuracy(z_held, held_t, tw, tb),
        "n_train": int(len(train_x)),
        "n_val": int(len(val_x)),
        "heldout_frames": int(len(held_x)),
        "source_fit": sinfo,
        "target_fit": tinfo,
        "checkpoint": str(args.checkpoint),
    }
    passed = (
        scores["val_source"] >= 0.90
        and scores["val_target"] >= 0.90
        and scores["heldout_source"] >= 0.90
        and scores["heldout_target"] >= 0.90
    )
    scores["gate"] = "pass" if passed else "fail"
    np.savez(
        args.out / "classifier.npz",
        mean=mean,
        std=std,
        source_weight=sw,
        source_bias=sb,
        target_weight=tw,
        target_bias=tb,
    )
    np.savez(
        args.out / "notes.npz",
        train_notes=train_x.astype(np.float32),
        train_source=train_s,
        train_target=train_t,
        val_notes=val_x.astype(np.float32),
        val_source=val_s,
        val_target=val_t,
        held_notes=held_x.astype(np.float32),
        held_source=held_s,
        held_target=held_t,
    )
    pack = load_color_classifier(args.out / "classifier.npz")
    from layout import eight_logits

    got = eight_logits(val_x, pack)
    src_acc = float((np.argmax(got[:, :4], axis=1) == val_s).mean())
    tgt_acc = float((np.argmax(got[:, 4:], axis=1) == val_t).mean())
    if abs(src_acc - scores["val_source"]) > 1.0e-6 or abs(tgt_acc - scores["val_target"]) > 1.0e-6:
        raise SystemExit(f"reloaded classifier {src_acc} {tgt_acc} != {scores['val_source']} {scores['val_target']}")
    (args.out / "probe.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    plot_accuracy(args.out / "accuracy.png", scores)
    write_report(args.out, scores)
    print(f"probe gate {scores['gate']} val {scores['val_source']:.3f} {scores['val_target']:.3f}", flush=True)


if __name__ == "__main__":
    main()
