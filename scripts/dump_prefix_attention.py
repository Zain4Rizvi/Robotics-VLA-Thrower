"""SmolVLM prefix attention and one expert denoising step, for two sentences.

One cube layout. The sentences are the stack swap. Weights are the mean over
the 15 heads, stored as thousandths of the query row. Even expert layers store the 50 chunk keys (renormalized over this step and
earlier ones) and, separately, where the rest of that row sat on the prefix.
Odd expert layers are the prefix only. chunkShare is how much of the even-layer
row sat on the chunk before that split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trs_so_arm100"))

import view  # noqa: E402

BEST = ROOT / "trs_so_arm100" / "best" / "pretrained_model"
FALLBACK = (
    ROOT
    / "trs_so_arm100"
    / "sentence"
    / "cont"
    / "run1"
    / "checkpoints"
    / "checkpoints"
    / "008000"
    / "pretrained_model"
)
OUT = ROOT / "frontend" / "public" / "attention"
GRID = 8
SEED = 0
STEPS = [0, 7, 14, 21, 28, 35, 42, 49]
SENTENCES = (
    ("red", "stack the red cube on the green cube"),
    ("green", "stack the green cube on the red cube"),
)


def checkpoint() -> Path:
    if (BEST / "model.safetensors").is_file():
        return BEST
    if (FALLBACK / "model.safetensors").is_file():
        return FALLBACK
    raise SystemExit(f"no weights in {BEST} or {FALLBACK}")


def per_mille(weights: np.ndarray, keep: np.ndarray | None = None) -> list[int]:
    row = np.asarray(weights, np.float64).copy()
    if keep is not None:
        row = np.where(keep, row, 0.0)
    total = float(row.sum())
    if total <= 0:
        return [0] * int(row.shape[0])
    row /= total
    scaled = np.rint(row * 1000).astype(np.int64)
    if keep is not None:
        scaled = np.where(keep, scaled, 0)
    drift = 1000 - int(scaled.sum())
    if drift != 0:
        for index in np.argsort(-row):
            if keep is not None and not keep[index]:
                continue
            updated = int(scaled[index] + drift)
            if updated >= 0:
                scaled[index] = updated
                break
    return scaled.tolist()


def attention_probs(vlm, attention_mask, batch_size, head_dim, query_states, key_states):
    import torch

    heads = vlm.num_attention_heads
    kv_heads = vlm.num_key_value_heads
    groups = heads // kv_heads
    length = key_states.shape[1]
    keys = key_states[:, :, :, None, :].expand(batch_size, length, kv_heads, groups, head_dim)
    keys = keys.reshape(batch_size, length, heads, head_dim)
    queries = query_states.to(dtype=torch.float32).transpose(1, 2)
    keys = keys.to(dtype=torch.float32).transpose(1, 2)
    scores = torch.matmul(queries, keys.transpose(2, 3)) * (head_dim**-0.5)
    blocked = torch.finfo(scores.dtype).min
    masked = torch.where(attention_mask[:, None, :, :], scores, blocked)
    return torch.nn.functional.softmax(masked, dim=-1)


def install(vlm, caught: list[np.ndarray]) -> None:
    original = vlm.eager_attention_forward

    def wrapped(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        probs = attention_probs(vlm, attention_mask, batch_size, head_dim, query_states, key_states)
        caught.append(probs[0].mean(0).detach().float().cpu().numpy())
        return original(attention_mask, batch_size, head_dim, query_states, key_states, value_states)

    vlm.eager_attention_forward = wrapped


def piece_label(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    if text == "\n" or text.strip() == "":
        return "↵"
    return text.strip()


def main() -> None:
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    path = checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = view.StackSim()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    sim.reset(SEED)

    policy, pre, _post = view.load_policy(path, device)
    model = policy.model
    vlm = model.vlm_with_expert
    if model.config.add_image_special_tokens or int(model.config.prefix_length) > 0:
        raise SystemExit("prefix layout is not 64 + 64 + words + state")
    if int(model.config.chunk_size) != 50 or int(model.config.num_vlm_layers) != 16:
        raise SystemExit("expected chunk 50 and 16 text layers")

    tokenizer = None
    pending = list(getattr(pre, "steps", []))
    while pending:
        step = pending.pop(0)
        found = getattr(step, "input_tokenizer", None) or getattr(step, "tokenizer", None)
        if found is not None and hasattr(found, "decode"):
            tokenizer = found
            break
        pending.extend(getattr(step, "steps", []) or [])
    if tokenizer is None:
        raise SystemExit("preprocessor has no tokenizer")

    images = {}
    for name in ("camera1", "camera2"):
        renderer.update_scene(sim.data, camera=name)
        images[name] = renderer.render().copy()
    degrees = sim.degrees()
    xy = view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2])

    caught: list[np.ndarray] = []
    install(vlm, caught)

    OUT.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    saved = {}
    snapshots = []
    for name, sentence in SENTENCES:
        obs = {"observation.state": torch.from_numpy(degrees.astype(np.float32)), "task": sentence}
        for camera, frame in images.items():
            obs[f"observation.images.{camera}"] = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        batch = pre(obs)
        state = batch[OBS_STATE]
        scored = torch.as_tensor(xy, dtype=state.dtype, device=state.device).reshape(1, 4)
        batch[OBS_STATE] = view.write_xy(state, scored)

        img_tensors, img_masks = policy.prepare_images(batch)
        if len(img_tensors) != 2:
            raise SystemExit(f"expected 2 cameras, got {len(img_tensors)}")
        if name == "red":
            for camera, tensor in zip(("front", "wrist"), img_tensors, strict=True):
                seen = ((tensor[0].detach().float().cpu() + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                imageio.imwrite(OUT / f"{camera}.png", seen)
                saved[camera] = f"{camera}.png"

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        state_in = policy.prepare_state(batch)
        with torch.inference_mode():
            prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
                img_tensors, img_masks, lang_tokens, lang_masks, state=state_in
            )
            caught.clear()
            _outputs, past = vlm.forward(
                attention_mask=view_make_mask(prefix_pad, prefix_att),
                position_ids=torch.cumsum(prefix_pad, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            prefix_rows = list(caught)
            noise = model.sample_noise((1, model.config.chunk_size, model.config.max_action_dim), device)
            time = torch.ones(1, dtype=torch.float32, device=device)
            caught.clear()
            model.denoise_step(
                prefix_pad_masks=prefix_pad,
                past_key_values=past,
                x_t=noise,
                timestep=time,
            )
            expert_rows = list(caught)
            del past

        tokens, query_indexes, keep = token_tape(tokenizer, lang_tokens[0].detach().cpu(), lang_masks[0].detach().cpu())
        seq = len(tokens)
        if len(prefix_rows) != 16 or len(expert_rows) != 16:
            raise SystemExit(f"{name}: layers prefix {len(prefix_rows)} expert {len(expert_rows)}")
        if any(row.shape != (seq, seq) for row in prefix_rows):
            raise SystemExit(f"{name}: prefix shape {prefix_rows[0].shape}, tokens {seq}")

        vlm_layers = []
        for index in query_indexes:
            vlm_layers.append([per_mille(row[index], keep[index]) for row in prefix_rows])

        expert = []
        for step in STEPS:
            self_rows = []
            scene_rows = []
            cross_rows = []
            shares = []
            for layer, row in enumerate(expert_rows):
                if layer % 2 == 0:
                    if row.shape != (50, seq + 50):
                        raise SystemExit(f"{name} even layer {layer} shape {row.shape}")
                    suffix = row[step, seq:]
                    shares.append(int(round(float(suffix.sum()) * 1000)))
                    causal = np.arange(50) <= step
                    self_rows.append(per_mille(suffix, causal))
                    scene_rows.append(per_mille(row[step, :seq], keep[query_indexes[-1]]))
                else:
                    if row.shape[1] != seq:
                        raise SystemExit(f"{name} odd layer {layer} shape {row.shape}")
                    cross_rows.append(per_mille(row[step], keep[query_indexes[-1]]))
            expert.append(
                {
                    "step": step,
                    "self": self_rows,
                    "scene": scene_rows,
                    "cross": cross_rows,
                    "chunkShare": shares,
                }
            )

        snapshots.append(
            {
                "id": name,
                "instruction": sentence,
                "tokens": tokens,
                "vlmQueries": query_indexes,
                "vlm": vlm_layers,
                "expert": expert,
            }
        )
        word = [item["label"] for item in tokens if item["kind"] == "word"]
        share = expert[-1]["chunkShare"]
        print(f"{name} words {word} late-step chunk share {share}", flush=True)

    payload = {
        "grid": GRID,
        "seed": SEED,
        "steps": STEPS,
        "front": saved["front"],
        "wrist": saved["wrist"],
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "snapshots": snapshots,
    }
    (OUT / "prefix.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT / 'prefix.json'}", flush=True)
    renderer.close()


def view_make_mask(pad, att):
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    return make_att_2d_masks(pad, att)


def token_tape(tokenizer, token_ids, lang_mask) -> tuple[list[dict], list[int], list[np.ndarray]]:
    tokens: list[dict] = []
    for kind in ("front", "wrist"):
        for index in range(GRID * GRID):
            row, col = divmod(index, GRID)
            tokens.append({"kind": kind, "label": kind, "row": row, "col": col})
    ids = token_ids.tolist()
    mask = lang_mask.tolist()
    if len(ids) != len(mask):
        raise SystemExit(f"lang ids {len(ids)} mask {len(mask)}")
    word_indexes = []
    for index, (token_id, valid) in enumerate(zip(ids, mask, strict=True)):
        kind = "word" if valid else "pad"
        tokens.append({"kind": kind, "label": piece_label(tokenizer, int(token_id)) if valid else ""})
        if valid:
            word_indexes.append(len(tokens) - 1)
    tokens.append({"kind": "state", "label": "joints"})
    state_index = len(tokens) - 1
    if len(word_indexes) < 1:
        raise SystemExit("sentence produced no word tokens")

    allowed = np.ones(len(tokens), dtype=bool)
    for index, token in enumerate(tokens):
        if token["kind"] == "pad":
            allowed[index] = False
    query_indexes = word_indexes + [state_index]
    keep = []
    for index in range(len(tokens)):
        row = allowed.copy()
        if index != state_index:
            row[state_index] = False
        keep.append(row)
    return tokens, query_indexes, keep


if __name__ == "__main__":
    main()
