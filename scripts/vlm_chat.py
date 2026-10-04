"""Chat with SmolVLM2. Each question is sent with the current overhead headcam.

Text from earlier turns stays in the chat. The photo is attached only to the
question just asked, so a 6 GB card is not holding a stack of frames.
"""

from __future__ import annotations

import os

import numpy as np
from imageio.v2 import imwrite

from openarm_vla.config import EnvConfig
from openarm_vla.constants import REPO_ROOT
from openarm_vla.env.throw_env import ThrowEnv

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
FRAME = REPO_ROOT / "artifacts" / "vision_xy" / "chat_overhead.png"
KEEP = 6  # ponytail: drop oldest turns if a long chat starts forgetting
# Stock headcam is (0.24, -0.30, 1.25). This chat only drops it to 32 cm above the balls.
CLOSE_CAM = np.array([0.24, -0.30, 0.75])


def messages(history: list[tuple[str, str]], question: str) -> list[dict]:
    out = []
    for user, answer in history[-KEEP:]:
        out.append({"role": "user", "content": [{"type": "text", "text": user}]})
        out.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    out.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    )
    return out


def _check() -> None:
    got = messages([("a", "b")], "c")
    images = [p for msg in got for p in msg["content"] if p["type"] == "image"]
    assert len(images) == 1 and got[-1]["content"][1]["text"] == "c"


def reply(processor, model, history, question, image) -> str:
    import torch

    prompt = processor.apply_chat_template(messages(history, question), add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    n_in = int(inputs["input_ids"].shape[-1])
    inputs = inputs.to(model.device)
    if inputs["pixel_values"].dtype != torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].float()
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.decode(out[0, n_in:], skip_special_tokens=True).strip()


def close_cam(env) -> None:
    import mujoco

    env.model.cam_pos[env._cam_front] = CLOSE_CAM
    mujoco.mj_forward(env.model, env.data)


def show(env, seed: int) -> tuple[np.ndarray, str, str]:
    env.reset(seed=seed)
    close_cam(env)
    frame = np.asarray(env.get_obs()["image_front"])
    FRAME.parent.mkdir(parents=True, exist_ok=True)
    path = FRAME
    try:
        imwrite(path, frame)
    except PermissionError:
        path = FRAME.with_name(f"chat_overhead_{seed}.png")
        imwrite(path, frame)
    try:
        os.startfile(path)
    except OSError as exc:
        print(f"could not open the photo ({exc})", flush=True)
    return frame, env.task["instruction"], str(path)


def main() -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("loading SmolVLM2 fp32...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        .to("cuda")
        .eval()
    )
    env = ThrowEnv(EnvConfig.from_yaml(REPO_ROOT / "configs" / "env.yaml"), render_mode="rgb_array")
    rng = np.random.default_rng()
    seed = int(rng.integers(0, 2**31 - 1))
    frame, instruction, path = show(env, seed)
    history: list[tuple[str, str]] = []
    print(
        "Type normally. The overhead photo is attached to whatever you send.\n"
        "Camera is 0.75 m up, over the table, closer than the policy headcam.\n"
        "r = new table and a cleared chat. q = quit.\n"
        f"seed {seed}\n{instruction}\n{path}",
        flush=True,
    )
    try:
        while True:
            try:
                question = input("you> ").strip()
            except EOFError:
                break
            if not question:
                continue
            if question in {"q", "quit"}:
                break
            if question in {"r", "reset"}:
                seed = int(rng.integers(0, 2**31 - 1))
                frame, instruction, path = show(env, seed)
                history.clear()
                print(f"seed {seed}\n{instruction}\n{path}", flush=True)
                continue
            close_cam(env)
            frame = np.asarray(env.get_obs()["image_front"])
            text = reply(processor, model, history, question, frame)
            history.append((question, text))
            print(f"vlm> {text}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    _check()
    main()
