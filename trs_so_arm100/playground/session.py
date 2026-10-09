"""One stacking sim, driven by the shared 30 Hz loop in red_green_view."""

from __future__ import annotations

import queue
import subprocess
import threading
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

POLICY_EVERY = 3
FREE_EVERY = 2
FREE_W, FREE_H = 960, 540
POLICY_W, POLICY_H = 640, 480


def gpu_python_running() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any("python" in line.lower() for line in out.stdout.splitlines())


def _jpeg(rgb: np.ndarray, quality: int) -> bytes:
    import cv2

    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class Session:
    """Loads the stacking policy once and keeps a single sim running."""

    def __init__(self) -> None:
        self.commands: queue.Queue = queue.Queue()
        self._moves: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: list[bytes | None] = [None, None, None]
        self._seqs = [0, 0, 0]
        self._phase = "loading"
        self._instruction = ""
        self._seed: int | None = None
        self._computing = False
        self._message = "Loading policy"
        self._status = self._status_locked()
        self._pump = None
        self._tick = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._thread_main, name="stack-playground", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.commands.put("q")
        if self._thread is not None:
            self._thread.join(timeout=8)

    def submit(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "instruction":
            text = str(message.get("text", "")).strip()
            if not text:
                return
            with self._lock:
                self._instruction = text
                self._status = self._status_locked()
            self.commands.put(text)
            return
        if kind == "layout":
            self.commands.put("r")
            return
        if kind == "camera":
            with self._lock:
                ready = self._phase == "ready"
            if not ready:
                return
            action = message.get("action")
            if action not in ("orbit", "pan", "zoom"):
                return
            try:
                dx = float(message.get("dx", 0))
                dy = float(message.get("dy", 0))
            except (TypeError, ValueError):
                return
            self._moves.put((action, dx, dy))

    def snapshot(self) -> tuple[dict, list[tuple[int, bytes | None]]]:
        with self._lock:
            status = dict(self._status)
            frames = [(self._seqs[i], self._frames[i]) for i in range(3)]
        return status, frames

    def _status_locked(self) -> dict:
        return {
            "type": "status",
            "phase": self._phase,
            "instruction": self._instruction,
            "seed": self._seed,
            "computing": self._computing,
            "message": self._message,
        }

    def _fail(self, message: str) -> None:
        with self._lock:
            self._phase = "error"
            self._message = message
            self._computing = False
            self._status = self._status_locked()

    def _thread_main(self) -> None:
        try:
            self._run()
        except Exception as exc:
            traceback.print_exc()
            self._fail(str(exc))

    def _run(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "sentence"))

        if gpu_python_running():
            raise RuntimeError("A Python process is already using the GPU. Stop it before opening the playground.")

        import mujoco

        import red_green_view as rg
        import view

        rg.configure_fp32()
        ckpt = rg.latest_pretrained()
        if not rg.CLF_PATH.is_file():
            raise RuntimeError(f"missing {rg.CLF_PATH}")
        rg.predict.clf = rg.load_classifier(rg.CLF_PATH)
        view.predict = rg.predict

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {ckpt} on {device} ...", flush=True)
        policy, pre, post = view.load_policy(ckpt, device)
        sim = view.StackSim()
        vis = sim.model.vis.global_
        vis.offwidth = max(int(vis.offwidth), FREE_W)
        vis.offheight = max(int(vis.offheight), FREE_H)
        policy_renderer = mujoco.Renderer(sim.model, height=POLICY_H, width=POLICY_W)
        try:
            free_renderer = mujoco.Renderer(sim.model, height=FREE_H, width=FREE_W)
            shared_renderer = False
        except Exception as exc:
            print(f"free-camera size fell back to the policy frame: {exc}", flush=True)
            free_renderer = policy_renderer
            shared_renderer = True
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(sim.model, cam)
        cam.lookat[:] = (0.0, -0.18, 0.14)
        cam.distance = 0.72
        cam.azimuth = 132.0
        cam.elevation = -22.0

        rng = np.random.default_rng()
        pump = view.ChunkPump(policy, pre, post)
        self._pump = pump
        actions = {
            "orbit": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
            "pan": mujoco.mjtMouse.mjMOUSE_MOVE_V,
            "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
        }

        with self._lock:
            if not self._instruction:
                self._instruction = view.INSTRUCTION
            self._status = self._status_locked()

        def relayout() -> None:
            seed = int(rng.integers(0, 2**31))
            sim.reset(seed)
            pump.interrupt()
            print(f"layout seed {seed}", flush=True)
            with self._lock:
                self._seed = seed
                self._status = self._status_locked()

        def on_sync() -> None:
            moves: list[tuple[str, float, float]] = []
            while True:
                try:
                    moves.append(self._moves.get_nowait())
                except queue.Empty:
                    break
            self._tick += 1
            for name, dx, dy in moves:
                mujoco.mjv_moveCamera(sim.model, actions[name], float(dx), float(dy), cam)
            draw_free = bool(moves) or self._tick % FREE_EVERY == 0
            blobs: list[tuple[int, bytes]] = []
            if draw_free:
                free_renderer.update_scene(sim.data, camera=cam)
                blobs.append((2, _jpeg(free_renderer.render(), 82)))
            if self._tick % POLICY_EVERY == 0:
                for index, name in enumerate(("camera1", "camera2")):
                    policy_renderer.update_scene(sim.data, camera=name)
                    blobs.append((index, _jpeg(policy_renderer.render(), 75)))
            with self._lock:
                for index, blob in blobs:
                    self._seqs[index] += 1
                    self._frames[index] = blob
                self._phase = "ready"
                self._message = ""
                self._computing = pump._planner.busy()
                self._status = self._status_locked()

        relayout()
        try:
            rg.drive(
                sim,
                pump,
                policy_renderer,
                self.commands,
                view.INSTRUCTION,
                alive=lambda: not self._stop.is_set(),
                on_sync=on_sync,
                relayout=relayout,
            )
        finally:
            pump.close()
            policy_renderer.close()
            if not shared_renderer:
                free_renderer.close()
            self._pump = None
        if not self._stop.is_set():
            self._fail("The session stopped.")
