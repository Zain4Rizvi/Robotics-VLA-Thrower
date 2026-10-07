"""Viewer for the color/swap checkpoint. Does not load trs_so_arm100/best.

Type an instruction and Enter. r draws a new layout. q quits.
Same two sentences as the swap training set:

    stack the red cube on the green cube
    stack the green cube on the red cube
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import view  # noqa: E402

HERE = Path(__file__).resolve().parent


def latest_checkpoint() -> Path:
    root = HERE / "checkpoints" / "checkpoints"
    found = [
        p / "pretrained_model"
        for p in root.glob("[0-9]*")
        if (p / "pretrained_model" / "model.safetensors").exists()
    ]
    if not found:
        raise SystemExit(f"no checkpoint under {root}")
    return max(found, key=lambda p: int(p.parent.name))


def main() -> None:
    import torch

    ckpt = latest_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {ckpt} on {device} ...", flush=True)
    policy, pre, post = view.load_policy(ckpt, device)
    sim = view.StackSim()
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    task = view.INSTRUCTION

    def new_layout() -> None:
        seed = int(np.random.default_rng().integers(0, 2**31))
        sim.reset(seed)
        policy.reset()
        print(f"layout seed {seed}", flush=True)

    new_layout()
    print("type an instruction + Enter, 'r' = new layout, 'q' = quit", flush=True)
    print(f"instruction -> {task!r}", flush=True)
    print("red on green: 'stack the red cube on the green cube'", flush=True)
    print("green on red: 'stack the green cube on the red cube'", flush=True)

    cmds: queue.Queue = queue.Queue()
    threading.Thread(target=view.read_stdin, args=(cmds,), daemon=True).start()
    buf, ptr = None, 0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            cmd = None if cmds.empty() else cmds.get()
            if cmd == "q":
                break
            if cmd == "r":
                new_layout()
                buf = None
            elif cmd:
                task = cmd
                policy.reset()
                buf = None
                print(f"instruction -> {task!r}", flush=True)

            if task and (buf is None or ptr >= view.REPLAN_EVERY or ptr >= len(buf)):
                images = {}
                for name in ("camera1", "camera2"):
                    renderer.update_scene(sim.data, camera=name)
                    images[name] = renderer.render().copy()
                xy = view.zscore_xy(sim.red_pos()[:2], sim.green_pos()[:2])
                buf = view.predict(policy, pre, post, images, sim.degrees(), task, xy)
                ptr = 0
            if buf is not None:
                ptr += 1
                sim.step(view.degrees_to_ctrl(buf[ptr - 1], sim.limits))
            viewer.sync()
            time.sleep(1 / view.CTRL_HZ)
    renderer.close()


if __name__ == "__main__":
    main()
