"""Frozen word-note classifier from sentence/probe.

The note matches the probe: mean of the masked word tokens after the 16th text
layer, before the final text norm. Width 960. The two logits are a pair whose
difference is the logistic decision function. Class 0 is red_box, class 1 is green_box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WIDTH = 960
FRONT = 64
WRIST = 64


def configure_fp32() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_classifier(path: Path):
    z = np.load(path)
    mean = np.asarray(z["mean"], np.float64)
    std = np.asarray(z["std"], np.float64)
    weight = np.asarray(z["weight"], np.float64)
    bias = float(np.asarray(z["bias"]).reshape(-1)[0])
    if mean.shape != (WIDTH,) or std.shape != (WIDTH,) or weight.shape != (WIDTH,):
        raise RuntimeError(f"classifier shapes mean {mean.shape} std {std.shape} weight {weight.shape}")
    std = std.copy()
    std[std == 0.0] = 1.0
    return mean, std, weight, bias


def two_logits(notes: np.ndarray, mean: np.ndarray, std: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """Return (N, 2) logits, red_box then green_box. Their difference is the log-odds of green_box."""
    score = (np.asarray(notes, np.float64) - mean) / std
    score = score @ weight + bias
    return np.stack((-0.5 * score, 0.5 * score), axis=-1).astype(np.float32)


def write_state(state, scored, logits):
    import torch

    if state.shape[-1] != 6:
        raise RuntimeError(f"state last dim {state.shape[-1]}, expected 6 before the xy write")
    if scored.shape[0] != state.shape[0] or logits.shape[0] != state.shape[0]:
        raise RuntimeError(f"batch state {state.shape[0]} xy {scored.shape[0]} logits {logits.shape[0]}")
    if scored.shape[-1] != 4 or logits.shape[-1] != 2:
        raise RuntimeError(f"xy {tuple(scored.shape)} logits {tuple(logits.shape)}")
    out = state.new_zeros(*state.shape[:-1], 32)
    out[..., :6] = state[..., :6]
    mid = (scored.shape[0], *([1] * (out.ndim - 2)))
    out[..., 6:10] = scored.to(dtype=state.dtype, device=state.device).view(*mid, 4)
    out[..., 10:12] = logits.to(dtype=state.dtype, device=state.device).view(*mid, 2)
    return out


def batch_notes(policy, batch) -> np.ndarray:
    """Word notes for a preprocessed batch. Shape (B, 960), float64. Does not build a graph."""
    import torch
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
        raise RuntimeError(f"batch has no language tokens: {sorted(batch)}")
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
            outputs, past = vlm.forward(
                attention_mask=att_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            hidden = outputs[0]
            del outputs, past
    finally:
        text_model.norm = saved_norm

    n_lang = int(lang_tokens.shape[-1])
    seq = int(hidden.shape[1])
    n_img = seq - n_lang - 1
    if hidden.shape[-1] != WIDTH or n_img != FRONT + WRIST or len(text_model.layers) != 16:
        raise RuntimeError(
            f"word span hidden {tuple(hidden.shape)} n_img {n_img} n_lang {n_lang} layers {len(text_model.layers)}"
        )
    if policy.config.add_image_special_tokens or int(policy.config.prefix_length) != 0:
        raise RuntimeError("prefix layout is not 64+64 word tokens then state")
    mask = lang_masks.bool()
    counts = mask.sum(dim=1)
    if int(counts.min()) < 1:
        raise RuntimeError("a row has an empty word span")
    words = hidden[:, n_img : n_img + n_lang, :]
    mask_f = mask.to(dtype=words.dtype)
    note = (words * mask_f.unsqueeze(-1)).sum(dim=1) / counts.to(dtype=words.dtype).unsqueeze(-1)
    if not batch_notes.checked:
        print(
            f"word notes {tuple(note.shape)} words/row {int(counts[0])} lang positions {n_lang}",
            flush=True,
        )
        batch_notes.checked = True
    return note.detach().float().cpu().numpy().astype(np.float64)


batch_notes.checked = False


def verify_saved_notes(clf_path: Path, notes_path: Path) -> None:
    """The reloaded classifier has to reproduce the probe's val accuracy."""
    mean, std, weight, bias = load_classifier(clf_path)
    z = np.load(notes_path)
    rows = []
    labels = []
    for split in ("train", "val"):
        red = np.asarray(z[f"{split}_red"], np.float64)
        green = np.asarray(z[f"{split}_green"], np.float64)
        for i in range(len(red)):
            rows.append(red[i])
            labels.append(0)
            rows.append(green[i])
            labels.append(1)
        if split == "val":
            val_x = np.stack(rows[-(2 * len(red)) :])
            val_y = np.array(labels[-(2 * len(red)) :])
    logits = two_logits(val_x, mean, std, weight, bias)
    pred = (logits[:, 1] >= logits[:, 0]).astype(np.int64)
    acc = float((pred == val_y).mean())
    if acc != 1.0:
        raise RuntimeError(f"reloaded classifier val accuracy {acc}, probe was 1.0")
    score = ((val_x - mean) / std) @ weight + bias
    if not np.allclose(logits[:, 1] - logits[:, 0], score, rtol=1.0e-4, atol=1.0e-4):
        raise RuntimeError("two logits do not differ by the decision function")
    print(f"reloaded classifier val accuracy {acc:.3f} on {len(val_y)} notes", flush=True)
