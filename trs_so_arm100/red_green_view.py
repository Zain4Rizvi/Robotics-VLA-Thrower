"""MuJoCo viewer for the latest sentence/cont/run1 stacking checkpoint.

Type an instruction and Enter. r draws a new cube layout and homes the arm. q quits.
Cube xy is filled into state dims 6:10. Dims 10:12 are the frozen probe logits
from sentence/probe/classifier.npz. Dims 12:32 stay zero.

The next chunk is computed on a background thread while the arm keeps moving.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sentence"))

import view  # noqa: E402
from note import batch_notes, configure_fp32, load_classifier, two_logits, write_state  # noqa: E402

RUN = HERE / "sentence" / "cont" / "run1" / "checkpoints" / "checkpoints"
CLF_PATH = HERE / "sentence" / "probe" / "classifier.npz"


def latest_pretrained() -> Path:
    found = []
    if RUN.is_dir():
        for path in RUN.glob("[0-9]*"):
            model = path / "pretrained_model"
            if (model / "model.safetensors").is_file():
                found.append((int(path.name), model))
    if not found:
        raise SystemExit(f"missing pretrained_model under {RUN}")
    found.sort()
    return found[-1][1]


def predict(policy, pre, post, images, degrees, task, xy):
    import torch

    obs = {
        "observation.state": torch.from_numpy(degrees.astype(np.float32)),
        "task": task,
    }
    for name, img in images.items():
        obs[f"observation.images.{name}"] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    batch = pre(obs)
    state = batch["observation.state"]
    notes = batch_notes(policy, batch)
    logits = torch.from_numpy(two_logits(notes, *predict.clf))
    scored = torch.as_tensor(xy, dtype=state.dtype, device=state.device).reshape(1, 4)
    batch["observation.state"] = write_state(state, scored, logits)
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    return post(actions)[0].detach().cpu().numpy()


def main() -> None:
    configure_fp32()
    ckpt = latest_pretrained()
    if not CLF_PATH.is_file():
        raise SystemExit(f"missing {CLF_PATH}")
    predict.clf = load_classifier(CLF_PATH)
    view.predict = predict

    import mujoco
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {ckpt} on {device} ...", flush=True)
    policy, pre, post = view.load_policy(ckpt, device)
    sim = view.StackSim()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    rng = np.random.default_rng()
    pump = view.ChunkPump(policy, pre, post)

    def capture():
        images = {}
        for name in ("camera1", "camera2"):
            renderer.update_scene(sim.data, camera=name)
            images[name] = renderer.render().copy()
        xy = view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2])
        return images, sim.degrees(), xy

    def new_layout() -> None:
        seed = int(rng.integers(0, 2**31))
        sim.reset(seed)
        pump.interrupt()
        print(f"layout seed {seed}", flush=True)

    new_layout()
    print("type an instruction + Enter, 'r' = new layout, 'q' = quit", flush=True)
    print(f"instruction -> {view.INSTRUCTION!r}", flush=True)

    cmds: queue.Queue = queue.Queue()
    threading.Thread(target=view.read_stdin, args=(cmds,), daemon=True).start()
    task = view.INSTRUCTION
    period = 1.0 / view.CTRL_HZ
    next_tick = time.perf_counter()
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                cmd = None if cmds.empty() else cmds.get()
                if cmd == "q":
                    break
                if cmd == "r":
                    new_layout()
                elif cmd:
                    task = cmd
                    pump.interrupt()
                    print(f"instruction -> {task!r}", flush=True)

                action = pump.next_action(capture, task)
                if action is None:
                    sim.step(sim.data.qpos[:6].copy())
                else:
                    sim.step(view.degrees_to_ctrl(action, sim.limits))
                viewer.sync()
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
    finally:
        pump.close()
        renderer.close()


if __name__ == "__main__":
    main()
