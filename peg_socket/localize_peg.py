"""Ask frozen SmolVLM2 for the peg pixel in tablecam. No training.

One overhead tablecam frame per val layout. The reply parse is the same rule
as scripts/vlm_point.py. Centimeters are the table-plane miss of that pixel.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_peg_expert import draw, prepare
from peg_expert import PEG_R, PegSim

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "data" / "datasets" / "peg_val" / "openarm_seeds.json"
OUT = Path(__file__).resolve().parent / "findings" / "localize"
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
INSTRUCTION = "place the peg in the hole"
IMG = 256
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")
# Fixed before any reply is seen. Same rule as scripts/vlm_point.py.
# Unit interval -> times 256. A 0-1000 answer (a number above 256) -> times 256/1000.
# Anything else is already a pixel. The error does not pick the scale.
# An unparsed reply stays in the median as a miss greater than the 1 cm gate.
UNPARSED_CM = 1.0e6


def to_pixels(raw: list[str]) -> tuple[float, float, str] | None:
    if len(raw) != 2:
        return None
    a, b = float(raw[0]), float(raw[1])
    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and any("." in s for s in raw):
        return a * 256.0, b * 256.0, "unit"
    if 0.0 <= a <= 1000.0 and 0.0 <= b <= 1000.0 and (a > 256.0 or b > 256.0):
        return a * 256.0 / 1000.0, b * 256.0 / 1000.0, "thousand"
    return a, b, "pixel"


def peg_prompt() -> str:
    return (
        "This is a 256 by 256 overhead photo of a table. Give the pixel coordinates "
        "of the center of the peg. Reply with only x y."
    )


def _cmdline(pid: int) -> str:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

    class PBI(ctypes.Structure):
        _fields_ = [
            ("ExitStatus", ctypes.c_ulong),
            ("PebBaseAddress", ctypes.c_void_p),
            ("AffinityMask", ctypes.c_void_p),
            ("BasePriority", ctypes.c_void_p),
            ("UniqueProcessId", ctypes.c_void_p),
            ("InheritedFromUniqueProcessId", ctypes.c_void_p),
        ]

    class USTR(ctypes.Structure):
        _fields_ = [
            ("Length", ctypes.c_ushort),
            ("MaximumLength", ctypes.c_ushort),
            ("_pad", ctypes.c_uint),
            ("Buffer", ctypes.c_void_p),
        ]

    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return ""
    info = PBI()
    retlen = ctypes.c_ulong()
    status = ntdll.NtQueryInformationProcess(
        handle, 0, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(retlen)
    )
    if status != 0:
        kernel32.CloseHandle(handle)
        return ""
    params = ctypes.c_void_p()
    read = ctypes.c_size_t()
    ok = kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(info.PebBaseAddress + 0x20),
        ctypes.byref(params),
        ctypes.sizeof(params),
        ctypes.byref(read),
    )
    if not ok or not params.value:
        kernel32.CloseHandle(handle)
        return ""
    us = USTR()
    ok = kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(params.value + 0x70),
        ctypes.byref(us),
        ctypes.sizeof(us),
        ctypes.byref(read),
    )
    if not ok or us.Length <= 0:
        kernel32.CloseHandle(handle)
        return ""
    buf = ctypes.create_unicode_buffer(us.Length // 2)
    ok = kernel32.ReadProcessMemory(handle, ctypes.c_void_p(us.Buffer), buf, us.Length, ctypes.byref(read))
    kernel32.CloseHandle(handle)
    return buf.value if ok else ""


def _python_pids() -> list[int]:
    """Process ids for python.exe. WMI and tasklist fail on this machine."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.Process32FirstW.restype = ctypes.c_int
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.Process32NextW.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    snap = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid = ctypes.c_void_p(-1).value
    if not snap or snap == invalid:
        raise SystemExit(f"process snapshot failed {ctypes.get_last_error()}")
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    pids = []
    ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
    while ok:
        if entry.szExeFile.lower() in ("python.exe", "pythonw.exe"):
            pids.append(int(entry.th32ProcessID))
        ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    kernel32.CloseHandle(snap)
    return pids


def assert_no_collector_or_train() -> None:
    """Stop if a demo collector or a SmolVLA train job is already running."""
    blocked = []
    for pid in _python_pids():
        if pid == os.getpid():
            continue
        text = _cmdline(pid).replace("\\", "/")
        low = text.lower()
        if "peg_socket/collect_peg_demos.py" in low or "scripts/train_smolvla.py" in low:
            blocked.append(f"pid {pid}: {text}")
    if blocked:
        raise SystemExit("blocked\n" + "\n".join(blocked))


def _camera(sim: PegSim) -> tuple[np.ndarray, np.ndarray, float]:
    cam = int(sim.model.camera("tablecam").id)
    pos = sim.data.cam_xpos[cam].copy()
    rot = sim.data.cam_xmat[cam].reshape(3, 3).copy()
    fovy = float(sim.model.cam_fovy[cam])
    return pos, rot, fovy


def project(pos, rot, fovy, point: np.ndarray) -> tuple[float, float, float]:
    """Pixel of a world point in tablecam. Origin is top-left. Third value is the peg radius in pixels."""
    local = rot.T @ (np.asarray(point, dtype=np.float64) - pos)
    depth = -float(local[2])
    if depth <= 1e-6:
        raise RuntimeError("peg is behind tablecam")
    fy = (IMG / 2) / np.tan(np.deg2rad(fovy) / 2)
    u = IMG / 2 + fy * float(local[0]) / depth
    v = IMG / 2 - fy * float(local[1]) / depth
    return float(u), float(v), float(fy * PEG_R / depth)


def ray_cm(pos, rot, fovy, u: float, v: float, peg: np.ndarray) -> float:
    """Horizontal centimeters from the peg center to the tablecam ray through (u, v) at the peg's z."""
    fy = (IMG / 2) / np.tan(np.deg2rad(fovy) / 2)
    direction = rot @ np.array(
        [(u - IMG / 2) / fy, -(v - IMG / 2) / fy, -1.0],
        dtype=np.float64,
    )
    if abs(float(direction[2])) < 1e-9:
        raise RuntimeError("camera ray is parallel to the peg plane")
    t = (float(peg[2]) - float(pos[2])) / float(direction[2])
    if t <= 0.0:
        raise RuntimeError("camera ray does not meet the peg plane")
    hit = pos + t * direction
    return float(np.hypot(hit[0] - peg[0], hit[1] - peg[1]) * 100.0)


def load_frames(sim: PegSim) -> list[dict]:
    rows = json.loads(SEEDS.read_text(encoding="utf-8"))
    if len(rows) != 20:
        raise SystemExit(f"expected 20 val seeds, got {len(rows)}")
    if sim.home_key != 0:
        raise SystemExit(f"home key id is {sim.home_key}, expected 0")
    renderer = mujoco.Renderer(sim.model, height=IMG, width=IMG)
    frames = []
    try:
        for i, row in enumerate(rows):
            seed = int(row["seed"])
            if row.get("instruction") != INSTRUCTION:
                raise SystemExit(f"seed {seed}: instruction {row.get('instruction')!r}")
            rng = np.random.default_rng(seed)
            peg_xy, sock_xy = draw(rng)
            if not np.allclose(peg_xy, row["peg_xy"], atol=5e-5) or not np.allclose(
                sock_xy, row["socket_xy"], atol=5e-5
            ):
                raise SystemExit(
                    f"seed {seed}: draw peg {np.round(peg_xy, 4)} socket {np.round(sock_xy, 4)} "
                    f"!= stored {row['peg_xy']} {row['socket_xy']}"
                )
            issue = prepare(sim, peg_xy, sock_xy)
            if issue:
                raise SystemExit(f"seed {seed}: prepare rejected {issue}")
            if int(sim.data.eq_active[sim.weld]) != 0:
                raise SystemExit(f"seed {seed}: grasp_right_peg is active")
            peg = sim.peg_pos()
            delta_mm = float(np.linalg.norm(peg[:2] - np.asarray(row["peg_xy"], dtype=np.float64)) * 1000.0)
            if delta_mm > 5.0:
                raise SystemExit(f"seed {seed}: settled peg xy is {delta_mm:.2f} mm from stored peg_xy")
            pos, rot, fovy = _camera(sim)
            u, v, radius = project(pos, rot, fovy, peg)
            true_cm = ray_cm(pos, rot, fovy, u, v, peg)
            if true_cm > 0.05:
                raise SystemExit(f"seed {seed}: true-pixel ray misses by {true_cm:.4f} cm")
            renderer.update_scene(sim.data, camera="tablecam")
            image = renderer.render().copy()
            if image.shape != (IMG, IMG, 3):
                raise SystemExit(f"seed {seed}: tablecam shape {image.shape}")
            frames.append(
                {
                    "i": i,
                    "seed": seed,
                    "peg_xy": [float(x) for x in row["peg_xy"]],
                    "live_peg": peg.copy(),
                    "cam_pos": pos,
                    "cam_rot": rot,
                    "fovy": fovy,
                    "settle_mm": round(delta_mm, 3),
                    "true_px": (u, v),
                    "radius_px": radius,
                    "true_ray_cm": true_cm,
                    "image": image,
                }
            )
            print(
                f"reset {i:02d} seed {seed} settle {delta_mm:.3f} mm "
                f"px ({u:.1f},{v:.1f}) r {radius:.2f} ray {true_cm:.4f} cm",
                flush=True,
            )
    finally:
        renderer.close()
    return frames


def chat(processor, model, image: np.ndarray, text: str, split: bool) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    kwargs = {"text": rendered, "images": [image], "return_tensors": "pt"}
    if not split:
        kwargs["do_image_splitting"] = False
        kwargs["size"] = {"longest_edge": 512}
        kwargs["max_image_size"] = {"longest_edge": 512}
    inputs = processor(**kwargs)
    n_in = int(inputs["input_ids"].shape[-1])
    inputs = inputs.to(model.device)
    if inputs["pixel_values"].dtype != torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].float()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    return processor.decode(out[0, n_in:], skip_special_tokens=True).strip()


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
    for param in model.parameters():
        param.requires_grad_(False)
    n_layers = len(model.model.text_model.layers)
    if n_layers != 32:
        raise SystemExit(f"expected the 32-layer SmolVLM2, got {n_layers}")
    if any(p.requires_grad for p in model.parameters()):
        raise SystemExit("model is not frozen")
    blank = np.zeros((IMG, IMG, 3), np.uint8)
    try:
        chat(processor, model, blank, peg_prompt(), True)
        return processor, model, True
    except torch.cuda.OutOfMemoryError:
        print("default image split does not fit; using one 512 image", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        for param in model.parameters():
            param.requires_grad_(False)
        chat(processor, model, blank, peg_prompt(), False)
        return processor, model, False


def score(raw: str, frame: dict) -> dict:
    parsed = to_pixels(NUM.findall(raw))
    rec = {
        "seed": frame["seed"],
        "reply": raw,
        "parsed_px": None,
        "true_px": [round(float(v), 4) for v in frame["true_px"]],
        "cm_error": None,
        "on_peg": False,
    }
    if parsed is None:
        return rec
    x, y, scale = parsed
    err_px = float(np.hypot(x - frame["true_px"][0], y - frame["true_px"][1]))
    rec["parsed_px"] = [round(x, 4), round(y, 4)]
    rec["scale"] = scale
    rec["cm_error"] = round(
        ray_cm(frame["cam_pos"], frame["cam_rot"], frame["fovy"], x, y, frame["live_peg"]),
        4,
    )
    rec["on_peg"] = bool(err_px <= frame["radius_px"])
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    assert_no_collector_or_train()
    sim = PegSim()
    frames = load_frames(sim)
    print(
        f"layouts 20 settle_mm max {max(f['settle_mm'] for f in frames):.3f} "
        f"true_ray_cm max {max(f['true_ray_cm'] for f in frames):.4f}",
        flush=True,
    )
    if args.geometry_only:
        return

    assert_no_collector_or_train()
    processor, model, split = load_vlm()
    import torch

    dtype = next(model.parameters()).dtype
    print(f"model {MODEL_ID} dtype {dtype} eval {not model.training} split {split}", flush=True)
    if dtype != torch.float32:
        raise SystemExit(f"expected fp32, got {dtype}")

    scored = []
    text = peg_prompt()
    for frame in frames:
        raw = chat(processor, model, frame["image"], text, split)
        rec = score(raw, frame)
        scored.append(rec)
        print(f"reply {frame['i']:02d} {rec}", flush=True)

    cm_values = [UNPARSED_CM if r["cm_error"] is None else r["cm_error"] for r in scored]
    median_cm = float(np.median(cm_values))
    n_on = int(sum(bool(r["on_peg"]) for r in scored))
    n_parsed = int(sum(r["parsed_px"] is not None for r in scored))
    passed = bool(median_cm < 1.0 and n_on >= 11)
    payload = {
        "model": MODEL_ID,
        "dtype": "float32",
        "frozen": True,
        "eval": True,
        "trained": False,
        "camera": "tablecam",
        "image_key": "observation.images.image_front",
        "instruction": INSTRUCTION,
        "prompt": text,
        "image_splitting": split,
        "n": 20,
        "peg_radius_m": PEG_R,
        "scale_rule": (
            "fractional [0,1] times 256; any number above 256 with both in [0,1000] "
            "times 256/1000; else pixels"
        ),
        "unparsed_counts_as_cm": UNPARSED_CM,
        "median_cm": round(median_cm, 4),
        "n_on_peg": n_on,
        "n_parsed": n_parsed,
        "gate": "pass" if passed else "fail",
        "seeds": [f["seed"] for f in frames],
        "frames": [
            {
                "seed": r["seed"],
                "reply": r["reply"],
                "parsed_px": r["parsed_px"],
                "scale": r.get("scale"),
                "true_px": r["true_px"],
                "cm_error": r["cm_error"],
                "on_peg": r["on_peg"],
            }
            for r in scored
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "points.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"median_cm {median_cm:.4f} n_on_peg {n_on}/20 n_parsed {n_parsed} gate {payload['gate']}",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
