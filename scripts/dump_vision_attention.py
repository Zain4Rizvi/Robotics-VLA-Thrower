"""Two SigLIP snapshots: the patch on the red cube, and who it attends to.

Every vision layer, mean of the 12 heads. The frames are the 512x512 images
the tower actually sees. best/pretrained_model has no model.safetensors; the
vision tower is frozen, so the weights are read from the step-8000 copy.
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
GRID = 32
PATCH = 16
H, W = 480, 640


def checkpoint() -> Path:
    if (BEST / "model.safetensors").is_file():
        return BEST
    if (FALLBACK / "model.safetensors").is_file():
        return FALLBACK
    raise SystemExit(f"no vision weights in {BEST} or {FALLBACK}")


def render_top(sim: view.StackSim, renderer) -> np.ndarray:
    renderer.update_scene(sim.data, camera="camera1")
    return renderer.render().copy()


def resized(frame: np.ndarray, device: str):
    import torch
    from lerobot.policies.common.vla_utils import resize_with_pad

    img = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float().div_(255)
    canvas = resize_with_pad(img.unsqueeze(0).to(device), 512, 512, pad_value=0)
    seen = (canvas[0].detach().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return canvas * 2.0 - 1.0, seen


def query_patch(sim: view.StackSim) -> tuple[int, int]:
    """Patch that contains the red cube. The arm is orange, so a red-pixel count lands on the arm."""
    mujoco.mj_forward(sim.model, sim.data)
    cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera1")
    rot = sim.data.cam_xmat[cam].reshape(3, 3)
    local = rot.T @ (sim.red_pos() - sim.data.cam_xpos[cam])
    depth = -float(local[2])
    if depth <= 1e-6:
        raise RuntimeError("red cube is behind the camera")
    fy = (H / 2) / np.tan(np.deg2rad(float(sim.model.cam_fovy[cam])) / 2)
    u = W / 2 + fy * float(local[0]) / depth
    v = H / 2 - fy * float(local[1]) / depth
    ratio = max(W / 512, H / 512)
    resized_h = int(H / ratio)
    resized_w = int(W / ratio)
    uu = (512 - resized_w) + u * resized_w / W
    vv = (512 - resized_h) + v * resized_h / H
    col = int(np.clip(np.floor(uu), 0, 511)) // PATCH
    row = int(np.clip(np.floor(vv), 0, 511)) // PATCH
    return row, col


def pick_seeds(sim: view.StackSim) -> list[tuple[int, int, int]]:
    found = []
    for seed in range(40):
        sim.reset(seed)
        row, col = query_patch(sim)
        found.append((seed, row, col))
    best = None
    best_dist = -1
    for i, a in enumerate(found):
        for b in found[i + 1 :]:
            dist = abs(a[1] - b[1]) + abs(a[2] - b[2])
            if dist > best_dist:
                best_dist = dist
                best = (a, b)
    if best is None or best_dist < 4:
        raise SystemExit("could not find two layouts with the red cube in different patches")
    return list(best)


def attention_rows(policy, pixels, query: int) -> list[np.ndarray]:
    import torch

    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    vision.config._attn_implementation = "eager"
    caught: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(vision.encoder.layers):
        layer.self_attn.config._attn_implementation = "eager"

        def save(_module, _inputs, output, index=index):
            weights = output[1]
            if weights is None:
                raise RuntimeError("attention weights were not returned; eager attention is off")
            caught[index] = weights[0, :, query, :].float().mean(0).cpu()

        handles.append(layer.self_attn.register_forward_hook(save))
    try:
        with torch.inference_mode():
            vision(
                pixel_values=pixels.to(dtype=vision.dtype),
                patch_attention_mask=None,
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(caught) != len(vision.encoder.layers):
        raise RuntimeError(f"expected {len(vision.encoder.layers)} layers, got {sorted(caught)}")
    rows = []
    for index in range(len(caught)):
        row = caught[index].numpy()
        if row.shape != (GRID * GRID,):
            raise RuntimeError(f"layer {index + 1} row shape {row.shape}")
        rows.append(row)
    return rows


def main() -> None:
    import torch

    path = checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = view.StackSim()
    renderer = mujoco.Renderer(sim.model, height=H, width=W)
    chosen = pick_seeds(sim)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(path)).to(device).float().eval()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = []
    names = ("a", "b")
    for name, (seed, row, col) in zip(names, chosen, strict=True):
        sim.reset(seed)
        row, col = query_patch(sim)
        pixels, seen = resized(render_top(sim, renderer), device)
        query = row * GRID + col
        rows = attention_rows(policy, pixels, query)
        image = f"{name}.png"
        import imageio.v2 as imageio

        imageio.imwrite(OUT / image, seen)
        snapshots.append(
            {
                "id": name,
                "seed": seed,
                "image": image,
                "query": [row, col],
                "layers": [np.round(row_weights, 6).tolist() for row_weights in rows],
            }
        )
        peaks = ", ".join(f"{float(item.max()):.3f}" for item in rows)
        print(f"{name} seed {seed} query {row},{col} peaks {peaks}", flush=True)

    payload = {
        "grid": GRID,
        "patch": PATCH,
        "heads": "mean",
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "snapshots": snapshots,
    }
    (OUT / "red-patch.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
