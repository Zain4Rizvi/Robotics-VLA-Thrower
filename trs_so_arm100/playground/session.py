"""One stacking sim, driven by the shared 30 Hz loop in red_green_view."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
ROBOTS = ("stack", "openarm", "peg")

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
        self._robot = "stack"
        self._status = self._status_locked()
        self._pump = None
        self._tick = 0
        self._mouse: dict = {}
        self._closers: list = []

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
        if kind == "robot":
            robot = message.get("id")
            if robot not in ROBOTS:
                return
            with self._lock:
                if self._robot == robot:
                    return
                self._robot = robot
                self._phase = "loading"
                self._message = "Switching"
                self._computing = False
                self._frames = [None, None, None]
                self._status = self._status_locked()
            self.commands.put("switch")
            return
        if kind == "instruction":
            text = str(message.get("text", "")).strip()
            if not text:
                return
            with self._lock:
                if self._robot != "stack":
                    return
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

    def _take(self) -> str | None:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def _drain_switches(self) -> None:
        held: list[str] = []
        while True:
            cmd = self._take()
            if cmd is None:
                break
            if cmd != "switch":
                held.append(cmd)
        for cmd in held:
            self.commands.put(cmd)

    def _begin_plan(self) -> None:
        with self._lock:
            self._computing = True
            if self._phase != "ready":
                self._message = "Planning"
            self._status = self._status_locked()

    def _note(self, robot: str, seed: int, instruction: str) -> bool:
        with self._lock:
            if self._robot != robot:
                return False
            self._seed = seed
            self._instruction = instruction
            self._computing = False
            self._status = self._status_locked()
        return True

    def _open_views(self, model, lookat, distance: float, azimuth: float, elevation: float):
        import mujoco

        vis = model.vis.global_
        vis.offwidth = max(int(vis.offwidth), FREE_W)
        vis.offheight = max(int(vis.offheight), FREE_H)
        policy = mujoco.Renderer(model, height=POLICY_H, width=POLICY_W)
        shared = False
        try:
            free = mujoco.Renderer(model, height=FREE_H, width=FREE_W)
        except Exception as exc:
            print(f"free-camera size fell back to the policy frame: {exc}", flush=True)
            free = policy
            shared = True
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation

        def close() -> None:
            policy.close()
            if not shared:
                free.close()

        self._closers.append(close)
        return policy, free, cam

    def _draw(self, robot: str, model, data, policy_renderer, free_renderer, cam, cameras: tuple[str, str]) -> None:
        import mujoco

        with self._lock:
            if self._robot != robot:
                return
        moves: list[tuple[str, float, float]] = []
        while True:
            try:
                moves.append(self._moves.get_nowait())
            except queue.Empty:
                break
        self._tick += 1
        for name, dx, dy in moves:
            mujoco.mjv_moveCamera(model, self._mouse[name], float(dx), float(dy), cam)
        blobs: list[tuple[int, bytes]] = []
        if moves or self._tick % FREE_EVERY == 0:
            free_renderer.update_scene(data, camera=cam)
            blobs.append((2, _jpeg(free_renderer.render(), 82)))
        if self._tick % POLICY_EVERY == 0:
            for index, name in enumerate(cameras):
                policy_renderer.update_scene(data, camera=name)
                blobs.append((index, _jpeg(policy_renderer.render(), 75)))
        with self._lock:
            if self._robot != robot:
                return
            for index, blob in blobs:
                self._seqs[index] += 1
                self._frames[index] = blob
            self._phase = "ready"
            self._message = ""
            self._computing = False
            self._status = self._status_locked()

    def _park(self) -> bool:
        """Wait after a failed plan. True means leave this robot."""
        while not self._stop.is_set():
            cmd = self._take()
            if cmd == "q":
                self._stop.set()
                return True
            if cmd == "switch":
                return True
            if cmd == "r":
                return False
            time.sleep(0.05)
        return True

    def _drive_expert(self, robot: str, model, data, policy_renderer, free_renderer, cam, cameras, step, relayout) -> None:
        period = 1.0 / 50.0
        while not self._stop.is_set():
            outcome = relayout()
            if outcome == "leave":
                return
            if outcome != "ok":
                if self._park():
                    return
                continue
            next_tick = time.perf_counter()
            while not self._stop.is_set():
                cmd = self._take()
                if cmd == "q":
                    self._stop.set()
                    return
                if cmd == "switch":
                    return
                if cmd == "r":
                    break
                if cmd is not None:
                    continue
                step()
                self._draw(robot, model, data, policy_renderer, free_renderer, cam, cameras)
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()

    def _run(self) -> None:
        import sys

        sys.path.insert(0, str(REPO / "peg_socket"))
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(ROOT / "sentence"))
        sys.path.insert(0, str(ROOT))

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
        policy_renderer, free_renderer, cam = self._open_views(
            sim.model, (0.0, -0.18, 0.14), 0.72, 132.0, -22.0
        )
        rng = np.random.default_rng()
        pump = view.ChunkPump(policy, pre, post)
        self._pump = pump
        self._mouse = {
            "orbit": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
            "pan": mujoco.mjtMouse.mjMOUSE_MOVE_V,
            "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
        }
        openarm: dict = {}
        peg: dict = {}

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
            with self._lock:
                if self._robot != "stack":
                    return
            moves: list[tuple[str, float, float]] = []
            while True:
                try:
                    moves.append(self._moves.get_nowait())
                except queue.Empty:
                    break
            self._tick += 1
            for name, dx, dy in moves:
                mujoco.mjv_moveCamera(sim.model, self._mouse[name], float(dx), float(dy), cam)
            blobs: list[tuple[int, bytes]] = []
            if moves or self._tick % FREE_EVERY == 0:
                free_renderer.update_scene(sim.data, camera=cam)
                blobs.append((2, _jpeg(free_renderer.render(), 82)))
            if self._tick % POLICY_EVERY == 0:
                for index, name in enumerate(("camera1", "camera2")):
                    policy_renderer.update_scene(sim.data, camera=name)
                    blobs.append((index, _jpeg(policy_renderer.render(), 75)))
            with self._lock:
                if self._robot != "stack":
                    return
                for index, blob in blobs:
                    self._seqs[index] += 1
                    self._frames[index] = blob
                self._phase = "ready"
                self._message = ""
                self._computing = pump._planner.busy()
                self._status = self._status_locked()

        def ensure_openarm() -> None:
            if openarm:
                return
            from openarm_vla.config import EnvConfig
            from openarm_vla.env.throw_env import ThrowEnv
            from openarm_vla.expert.throw_expert import ThrowExpert

            env = ThrowEnv(EnvConfig(render=False), render_mode=None)
            views = self._open_views(env.model, (0.4, -0.2, 0.35), 1.7, 130.0, -24.0)
            openarm.update(env=env, expert=ThrowExpert(), views=views)

        def ensure_peg() -> None:
            if peg:
                return
            from eval_peg_expert import draw, prepare
            from peg_expert import PegExpert, PegSim

            sim_peg = PegSim()
            views = self._open_views(sim_peg.model, (0.28, -0.28, 0.4), 1.35, 140.0, -28.0)
            peg.update(sim=sim_peg, expert=PegExpert(), draw=draw, prepare=prepare, views=views)

        def openarm_layout() -> str:
            with self._lock:
                if self._robot != "openarm":
                    return "leave"
            self._begin_plan()
            env = openarm["env"]
            expert = openarm["expert"]
            seed = int(rng.integers(0, 2**31))
            try:
                env.reset(seed=seed)
                expert.reset(env)
            except Exception as exc:
                traceback.print_exc()
                with self._lock:
                    self._phase = "error"
                    self._message = str(exc)
                    self._computing = False
                    self._status = self._status_locked()
                return "fail"
            print(f"throw seed {seed}", flush=True)
            if not self._note("openarm", seed, str(env.task["instruction"])):
                return "leave"
            return "ok"

        def peg_layout() -> str:
            with self._lock:
                if self._robot != "peg":
                    return "leave"
            self._begin_plan()
            sim_peg = peg["sim"]
            expert = peg["expert"]
            for _ in range(40):
                if self._stop.is_set():
                    return "leave"
                with self._lock:
                    if self._robot != "peg":
                        return "leave"
                seed = int(rng.integers(0, 2**31))
                peg_xy, sock_xy = peg["draw"](np.random.default_rng(seed))
                if peg["prepare"](sim_peg, peg_xy, sock_xy):
                    continue
                if not expert.reset(sim_peg):
                    continue
                print(f"peg seed {seed}", flush=True)
                if not self._note("peg", seed, "place the peg in the hole"):
                    return "leave"
                return "ok"
            with self._lock:
                self._phase = "error"
                self._message = "No reachable layout"
                self._computing = False
                self._status = self._status_locked()
            return "fail"

        try:
            while not self._stop.is_set():
                self._drain_switches()
                if self._stop.is_set():
                    break
                with self._lock:
                    robot = self._robot
                if robot == "stack":
                    if self._phase == "loading":
                        relayout()
                    rg.drive(
                        sim,
                        pump,
                        policy_renderer,
                        self.commands,
                        self._instruction or view.INSTRUCTION,
                        alive=lambda: not self._stop.is_set(),
                        on_sync=on_sync,
                        relayout=relayout,
                    )
                elif robot == "openarm":
                    ensure_openarm()
                    env = openarm["env"]
                    expert = openarm["expert"]
                    policy_r, free_r, cam_r = openarm["views"]
                    self._drive_expert(
                        "openarm",
                        env.model,
                        env.data,
                        policy_r,
                        free_r,
                        cam_r,
                        ("headcam", "camera_wrist_right"),
                        lambda: env.physics_step(expert.act(env)),
                        openarm_layout,
                    )
                else:
                    ensure_peg()
                    sim_peg = peg["sim"]
                    expert = peg["expert"]
                    policy_r, free_r, cam_r = peg["views"]
                    self._drive_expert(
                        "peg",
                        sim_peg.model,
                        sim_peg.data,
                        policy_r,
                        free_r,
                        cam_r,
                        ("tablecam", "camera_wrist_right"),
                        lambda: sim_peg.step(expert.act()),
                        peg_layout,
                    )
        finally:
            pump.close()
            self._pump = None
            for close in self._closers:
                close()
            self._closers.clear()
        if not self._stop.is_set():
            self._fail("The session stopped.")
