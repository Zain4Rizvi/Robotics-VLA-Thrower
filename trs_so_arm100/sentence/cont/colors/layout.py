"""Color pairs, paint, and the eight logits. Bodies stay red_box then green_box."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "color"))

import view  # noqa: E402
from expert import RGBA, other, sentence  # noqa: E402

CLASSES = ("red", "green", "blue", "yellow")
PAIRS = (
    ("red", "green"),
    ("red", "blue"),
    ("red", "yellow"),
    ("green", "red"),
    ("green", "blue"),
    ("green", "yellow"),
    ("blue", "red"),
    ("blue", "green"),
    ("blue", "yellow"),
    ("yellow", "red"),
    ("yellow", "green"),
)
HELD = ("yellow", "blue")
HELD_SENTENCE = sentence(*HELD)


def class_index(name: str) -> int:
    if name not in CLASSES:
        raise RuntimeError(f"color {name} is not in {CLASSES}")
    return CLASSES.index(name)


def forbidden(text: str) -> bool:
    low = text.lower()
    if "orange" in low or "purple" in low:
        return True
    return HELD_SENTENCE in low


def paint(seed: int, source_color: str, target_color: str) -> tuple[str, dict[str, str]]:
    """Which body wears the source color. A fresh generator, so cube xy stays view.draw(seed)."""
    source_body = "red" if int(np.random.default_rng(seed).integers(2)) == 0 else "green"
    colors = {source_body: source_color, other(source_body): target_color}
    return source_body, colors


def write_color_state(state, scored, source_logits, target_logits):
    import torch

    if state.shape[-1] != 6:
        raise RuntimeError(f"state last dim {state.shape[-1]}, expected 6 before the xy write")
    if scored.shape[-1] != 4 or source_logits.shape[-1] != 4 or target_logits.shape[-1] != 4:
        raise RuntimeError("xy is 4 and each logit head is 4")
    if scored.shape[0] != state.shape[0] or source_logits.shape[0] != state.shape[0]:
        raise RuntimeError("batch mismatch")
    out = state.new_zeros(*state.shape[:-1], 32)
    out[..., :6] = state[..., :6]
    mid = (scored.shape[0], *([1] * (out.ndim - 2)))
    out[..., 6:10] = scored.to(dtype=state.dtype, device=state.device).view(*mid, 4)
    out[..., 10:14] = source_logits.to(dtype=state.dtype, device=state.device).view(*mid, 4)
    out[..., 14:18] = target_logits.to(dtype=state.dtype, device=state.device).view(*mid, 4)
    return out


def load_color_classifier(path: Path):
    z = np.load(path)
    mean = np.asarray(z["mean"], np.float64)
    std = np.asarray(z["std"], np.float64)
    sw = np.asarray(z["source_weight"], np.float64)
    sb = np.asarray(z["source_bias"], np.float64).reshape(-1)
    tw = np.asarray(z["target_weight"], np.float64)
    tb = np.asarray(z["target_bias"], np.float64).reshape(-1)
    if mean.shape != (960,) or sw.shape != (4, 960) or tw.shape != (4, 960) or sb.shape != (4,) or tb.shape != (4,):
        raise RuntimeError(f"color classifier shapes mean {mean.shape} source {sw.shape} target {tw.shape}")
    std = std.copy()
    std[std == 0.0] = 1.0
    return mean, std, sw, sb, tw, tb


def eight_logits(notes: np.ndarray, pack) -> np.ndarray:
    mean, std, sw, sb, tw, tb = pack
    x = (np.asarray(notes, np.float64) - mean) / std
    src = x @ sw.T + sb
    tgt = x @ tw.T + tb
    return np.concatenate([src, tgt], axis=-1).astype(np.float32)


def self_check() -> None:
    if sentence("yellow", "blue") != "stack the yellow cube on the blue cube":
        raise SystemExit("held sentence")
    if sentence("blue", "yellow") != "stack the blue cube on the yellow cube":
        raise SystemExit("control sentence")
    if not forbidden(sentence("yellow", "blue")) or forbidden(sentence("blue", "yellow")):
        raise SystemExit("forbidden pairs")
    if not forbidden("stack the orange cube on the purple cube"):
        raise SystemExit("held-out words")
    if set(PAIRS) & {HELD}:
        raise SystemExit("held pair is in the training pairs")
    if len(PAIRS) != 11 or len({p[0] for p in PAIRS}) != 4 or len({p[1] for p in PAIRS}) != 4:
        raise SystemExit("eleven pairs")
    if not np.allclose(RGBA["orange"], [0.9, 0.4, 0.05, 1.0]):
        raise SystemExit("orange rgba")
    if not np.allclose(RGBA["purple"], [0.55, 0.15, 0.7, 1.0]):
        raise SystemExit("purple rgba")
    before = view.draw(np.random.default_rng(33000))
    body, colors = paint(33000, "yellow", "blue")
    after = view.draw(np.random.default_rng(33000))
    if not np.allclose(before[0], after[0]) or not np.allclose(before[1], after[1]):
        raise SystemExit("paint changed the layout draw")
    if set(colors) != {"red", "green"} or colors[body] != "yellow" or colors[other(body)] != "blue":
        raise SystemExit(f"paint {body} {colors}")
    import torch

    state = torch.zeros(2, 1, 6)
    state[..., 0] = 1
    scored = torch.tensor([[1.0, 2, 3, 4], [5, 6, 7, 8]])
    src = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    tgt = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]])
    out = write_color_state(state, scored, src, tgt)
    if out.shape != (2, 1, 32) or not torch.equal(out[:, 0, 6:10], scored):
        raise SystemExit("color state xy")
    if not torch.equal(out[:, 0, 10:14], src) or not torch.equal(out[:, 0, 14:18], tgt):
        raise SystemExit("color state logits")
    if float(out[:, :, 18:].abs().sum()) != 0 or not torch.equal(out[:, 0, :6], state[:, 0, :6]):
        raise SystemExit("color state pad or joints")
    print("layout self_check ok", flush=True)


if __name__ == "__main__":
    self_check()
