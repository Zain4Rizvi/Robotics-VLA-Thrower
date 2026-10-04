"""Ask SmolVLM2, in words, where the named ball is on the val_head resets.

Two prompts, no training. Overhead headcam alone, then headcam plus both wrist
cameras. The reply is scored against the ball's projection into headcam, not
against the meter xy in the label file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from imageio.v2 import imwrite

from openarm_vla.config import EnvConfig
from openarm_vla.constants import COLOR_RGBA, COLORS, REPO_ROOT
from openarm_vla.env.throw_env import ThrowEnv

ART = REPO_ROOT / "artifacts" / "vision_xy"
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BALL_R = 0.03
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")

# Fixed before any reply is seen. The prompt says the photo is 256 pixels.
# Unit interval -> times 256. A 0-1000 answer (a number above 256) -> times 256/1000.
# Anything else is already a pixel. The error does not pick the scale.


def to_pixels(raw: list[str]) -> tuple[float, float, str] | None:
    if len(raw) != 2:
        return None
    a, b = float(raw[0]), float(raw[1])
    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and any("." in s for s in raw):
        return a * 256.0, b * 256.0, "unit"
    if 0.0 <= a <= 1000.0 and 0.0 <= b <= 1000.0 and (a > 256.0 or b > 256.0):
        return a * 256.0 / 1000.0, b * 256.0 / 1000.0, "thousand"
    return a, b, "pixel"


def project(model, data, cam_id, point, size: int) -> tuple[float, float, float]:
    """Pixel of a world point in a square MuJoCo camera. Origin is top-left."""
    rot = data.cam_xmat[cam_id].reshape(3, 3)
    local = rot.T @ (np.asarray(point, dtype=np.float64) - data.cam_xpos[cam_id])
    depth = -float(local[2])
    if depth <= 1e-6:
        raise RuntimeError(f"point is behind camera {cam_id}")
    fy = (size / 2) / np.tan(np.deg2rad(float(model.cam_fovy[cam_id])) / 2)
    u = size / 2 + fy * float(local[0]) / depth
    v = size / 2 - fy * float(local[1]) / depth
    return u, v, fy * BALL_R / depth


def nearest_color(img: np.ndarray, u: float, v: float, radius: float) -> str | None:
    h, w = img.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    r = max(2, int(round(radius)))
    best_name, best_d = None, 1e9
    palette = {c: np.array(COLOR_RGBA[c][:3]) * 255 for c in COLORS}
    for y in range(vi - r, vi + r + 1):
        for x in range(ui - r, ui + r + 1):
            if not (0 <= x < w and 0 <= y < h):
                continue
            if (x - u) ** 2 + (y - v) ** 2 > radius ** 2:
                continue
            pix = img[y, x].astype(np.float64)
            for name, rgb in palette.items():
                d = float(np.linalg.norm(pix - rgb))
                if d < best_d:
                    best_name, best_d = name, d
    if best_d > 90:
        return None
    return best_name


def mark(img: np.ndarray, points: list[tuple[float, float, tuple[int, int, int]]]) -> np.ndarray:
    out = np.array(img, copy=True)
    h, w = out.shape[:2]
    for u, v, color in points:
        x, y = int(round(u)), int(round(v))
        for d in range(-6, 7):
            for yy, xx in ((y + d, x), (y, x + d), (y + d, x + 1), (y + 1, x + d)):
                if 0 <= yy < h and 0 <= xx < w:
                    out[yy, xx] = (0, 0, 0) if abs(d) == 6 else color
    return out


def overhead_prompt(color: str) -> str:
    return (
        f"This is a 256 by 256 overhead photo of a table. Give the pixel coordinates "
        f"of the center of the {color} ball. Reply with only x y."
    )


def three_prompt(color: str) -> str:
    return (
        "Image 1 is the overhead headcam. Image 2 is the right wrist (camera_wrist_right). "
        "Image 3 is the left wrist (camera_wrist_left). "
        f"Give the pixel coordinates of the center of the {color} ball in image 1 only. "
        "Image 1 is a 256 by 256 photo. Origin is the top-left corner, x to the right, y down. "
        "Reply with only x y."
    )


def load_resets() -> list[dict]:
    labels = np.load(ART / "val_head.npz")
    front = np.memmap(ART / "val_head_front.u8", dtype=np.uint8, mode="r", shape=(len(labels["reset"]), 256, 256, 3))
    seeds = json.loads((REPO_ROOT / "data" / "datasets" / "val_head" / "openarm_seeds.json").read_text(encoding="utf-8"))
    idx = np.flatnonzero(labels["reset"])
    if len(idx) != 20 or len(seeds) != 20:
        raise SystemExit(f"expected 20 val resets, got {len(idx)} frames and {len(seeds)} seeds")

    env = ThrowEnv(EnvConfig.from_yaml(REPO_ROOT / "configs" / "env.yaml"), render_mode="rgb_array")
    cam_left = int(env.model.camera("camera_wrist_left").id)
    rows = []
    try:
        for i, (frame, seed) in enumerate(zip(idx, seeds)):
            obs, _ = env.reset(seed=int(seed["seed"]), options={"instruction": seed["instruction"]})
            color = COLORS[int(labels["color"][frame])]
            if env.task["ball_color"] != seed["ball_color"] or env.task["ball_color"] != color:
                raise SystemExit(f"reset {i}: color {env.task['ball_color']!r} != {color!r}")
            if env.task["instruction"] != seed["instruction"]:
                raise SystemExit(f"reset {i}: instruction drifted")
            body = env.task["ball_body"]
            xy = env.ball_pos(body)[:2]
            if float(np.linalg.norm(xy - labels["xy"][frame])) > 1e-4:
                raise SystemExit(f"reset {i}: ball xy does not match the label file")
            stored = np.asarray(front[frame])
            live = obs["image_front"]
            mae = float(np.abs(stored.astype(np.int16) - live.astype(np.int16)).mean())
            if mae > 1.0:
                raise SystemExit(f"reset {i}: stored headcam mae {mae:.2f} against a fresh render")
            u, v, radius = project(env.model, env.data, env._cam_front, env.ball_pos(body), 256)
            left = env._rgb(cam_left)
            rows.append(
                {
                    "i": i,
                    "seed": int(seed["seed"]),
                    "color": color,
                    "body": body,
                    "u": u,
                    "v": v,
                    "radius_px": radius,
                    "front": stored.copy(),
                    "wrist_right": obs["image_wrist"].copy(),
                    "wrist_left": left.copy(),
                    "front_mae": mae,
                    "hit_color": nearest_color(stored, u, v, radius),
                }
            )
            print(
                f"reset {i} {color:7} body {body:7} px ({u:6.1f},{v:6.1f}) "
                f"r {radius:4.1f} disk {rows[-1]['hit_color']} mae {mae:.3f}",
                flush=True,
            )
    finally:
        env.close()
    return rows


def chat(processor, model, images: list[np.ndarray], text: str, split: bool) -> str:
    import torch

    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    kwargs = {"text": prompt, "images": images, "return_tensors": "pt"}
    if not split:
        kwargs["do_image_splitting"] = False
        kwargs["size"] = {"longest_edge": 512}
        kwargs["max_image_size"] = {"longest_edge": 512}
    inputs = processor(**kwargs)
    n_in = int(inputs["input_ids"].shape[-1])
    inputs = inputs.to(model.device)
    if inputs["pixel_values"].dtype != torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].float()
    out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    return processor.decode(out[0, n_in:], skip_special_tokens=True).strip()


def score_reply(raw: str, row: dict) -> dict:
    found = NUM.findall(raw)
    parsed = to_pixels(found)
    rec = {"raw": raw, "parsed": parsed is not None}
    if parsed is None:
        rec["error_px"] = None
        rec["on_ball"] = False
        return rec
    x, y, scale = parsed
    err = float(np.hypot(x - row["u"], y - row["v"]))
    rec.update(
        {
            "x": x,
            "y": y,
            "scale": scale,
            "error_px": err,
            "on_ball": bool(err <= row["radius_px"]),
        }
    )
    return rec


def summarize(name: str, rows: list[dict], key: str) -> dict:
    scored = [r[key] for r in rows]
    errs = [r["error_px"] for r in scored if r["parsed"]]
    return {
        "prompt": overhead_prompt("<color>") if name == "overhead" else three_prompt("<color>"),
        "n": len(rows),
        "n_parsed": int(sum(bool(r["parsed"]) for r in scored)),
        "n_on_ball": int(sum(bool(r["on_ball"]) for r in scored)),
        "median_px": None if not errs else float(np.median(errs)),
        "median_radius_px": float(np.median([r["radius_px"] for r in rows])),
        "replies": scored,
    }


def sheet(rows: list[dict], path: Path) -> None:
    picks = [0, 4, 9, 14, 19]
    tiles = []
    for key, pred_color in (("overhead", (255, 40, 40)), ("three", (255, 180, 0))):
        band = []
        for i in picks:
            row = rows[i]
            pts = [(row["u"], row["v"], (40, 255, 80))]
            rec = row[key]
            if rec["parsed"]:
                pts.append((rec["x"], rec["y"], pred_color))
            band.append(mark(row["front"], pts))
        tiles.append(np.concatenate(band, axis=1))
    gap = np.full((6, tiles[0].shape[1], 3), 255, np.uint8)
    imwrite(path, np.concatenate([tiles[0], gap, tiles[1]], axis=0))


def overlays(rows: list[dict], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for key, color in (("overhead", (255, 40, 40)), ("three", (255, 180, 0))):
            pts = [(row["u"], row["v"], (40, 255, 80))]
            rec = row[key]
            if rec["parsed"]:
                pts.append((rec["x"], rec["y"], color))
            imwrite(path / f"{key}_{row['i']:02d}.png", mark(row["front"], pts))


def load_vlm():
    import gc

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = model.to("cuda").eval()
    n_layers = len(model.model.text_model.layers)
    if n_layers != 32:
        raise SystemExit(f"expected the 32-layer SmolVLM2, got {n_layers}")
    # The chat processor upscales to 2048 and tiles. Both tests use one setting.
    # Three images is the larger case; if that tile does not fit, neither test tiles.
    blank = [np.zeros((256, 256, 3), np.uint8) for _ in range(3)]
    try:
        chat(processor, model, blank, three_prompt("red"), True)
        return processor, model, True
    except torch.cuda.OutOfMemoryError:
        print("default image split does not fit; using one 512 image per camera", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        chat(processor, model, blank, three_prompt("red"), False)
        return processor, model, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry-only", action="store_true")
    args = p.parse_args()
    rows = load_resets()
    hits = sum(r["hit_color"] == r["color"] for r in rows)
    print(f"projection on named color {hits}/20", flush=True)
    if hits < 18:
        raise SystemExit("headcam projection missed the named ball; not asking the model")
    if args.geometry_only:
        path = ART / "vlm_point_geometry.png"
        imwrite(path, mark(rows[0]["front"], [(rows[0]["u"], rows[0]["v"], (40, 255, 80))]))
        print(f"wrote {path}", flush=True)
        return

    processor, model, split = load_vlm()
    print(f"image splitting {split}", flush=True)
    for row in rows:
        raw = chat(processor, model, [row["front"]], overhead_prompt(row["color"]), split)
        row["overhead"] = score_reply(raw, row)
        print(f"overhead {row['i']:02d} {row['overhead']}", flush=True)
    for row in rows:
        raw = chat(
            processor,
            model,
            [row["front"], row["wrist_right"], row["wrist_left"]],
            three_prompt(row["color"]),
            split,
        )
        row["three"] = score_reply(raw, row)
        print(f"three {row['i']:02d} {row['three']}", flush=True)

    result = {
        "model": MODEL_ID,
        "dtype": "float32",
        "layers": 32,
        "image_splitting": split,
        "n": 20,
        "ball_diameter_m": 0.06,
        "scale_rule": "fractional [0,1] times 256; any number above 256 with both in [0,1000] times 256/1000; else pixels",
        "projection_on_color": hits,
        "overhead": summarize("overhead", rows, "overhead"),
        "three": summarize("three", rows, "three"),
        "frames": [
            {
                "i": r["i"],
                "seed": r["seed"],
                "color": r["color"],
                "u": float(r["u"]),
                "v": float(r["v"]),
                "radius_px": float(r["radius_px"]),
                "front_mae": float(r["front_mae"]),
            }
            for r in rows
        ],
    }
    out = ART / "vlm_point.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    overlays(rows, ART / "vlm_point_overlays")
    sheet(rows, ART / "vlm_point_sheet.png")
    print(
        f"overhead median {result['overhead']['median_px']} parsed {result['overhead']['n_parsed']} "
        f"on_ball {result['overhead']['n_on_ball']}",
        flush=True,
    )
    print(
        f"three median {result['three']['median_px']} parsed {result['three']['n_parsed']} "
        f"on_ball {result['three']['n_on_ball']}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
