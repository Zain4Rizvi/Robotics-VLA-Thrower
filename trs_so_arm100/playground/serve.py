"""Serve the stacking playground: one sim, one site, on localhost."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "web" / "dist"
HOST = "127.0.0.1"
PORT = 8765


def pin_caches() -> None:
    """Keep caches off the C: system temp. Z: is used when it is mounted."""
    if Path("Z:/").exists():
        defaults = {
            "HF_HOME": r"Z:\hf_cache",
            "UV_CACHE_DIR": r"Z:\uv_cache",
            "TEMP": r"Z:\tmp",
            "TMP": r"Z:\tmp",
            "TORCH_HOME": r"Z:\hf_cache\torch",
            "XDG_CACHE_HOME": r"Z:\hf_cache\xdg",
        }
    else:
        print("Z: is not mounted. Using D: for caches and temp.", flush=True)
        defaults = {
            "HF_HOME": r"D:\hf_cache",
            "UV_CACHE_DIR": r"D:\uv_cache",
            "TEMP": r"D:\tmp",
            "TMP": r"D:\tmp",
            "TORCH_HOME": r"D:\torch",
            "XDG_CACHE_HOME": r"D:\xdg",
        }
    for key, value in defaults.items():
        current = os.environ.get(key, "")
        if key in ("TEMP", "TMP"):
            if not current.upper().startswith(("D:", "Z:")):
                os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    for key in ("HF_HOME", "TEMP", "TMP", "TORCH_HOME", "XDG_CACHE_HOME"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


pin_caches()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from session import Session

session = Session()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    session.start()
    print(f"playground http://{HOST}:{PORT}", flush=True)
    yield
    session.stop()


app = FastAPI(lifespan=lifespan)
_client: WebSocket | None = None


def _handle(raw: str) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        return
    if isinstance(message, dict):
        session.submit(message)


async def _push(websocket: WebSocket) -> None:
    seen = [-1, -1, -1]
    last = None
    try:
        while True:
            status, frames = session.snapshot()
            if status != last:
                await websocket.send_text(json.dumps(status))
                last = status
            for index, (seq, blob) in enumerate(frames):
                if blob is not None and seq != seen[index]:
                    await websocket.send_bytes(bytes([index]) + blob)
                    seen[index] = seq
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    global _client
    await websocket.accept()
    previous = _client
    _client = websocket
    if previous is not None and previous is not websocket:
        try:
            await previous.close(code=4000)
        except RuntimeError:
            pass
    sender = asyncio.create_task(_push(websocket))
    try:
        while True:
            _handle(await websocket.receive_text())
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        if _client is websocket:
            _client = None


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:

    @app.get("/")
    def missing_ui():
        return PlainTextResponse(
            "UI build is missing. Run npm run build in playground/web.",
            status_code=503,
        )


def main() -> None:
    import uvicorn

    if not DIST.is_dir():
        print(f"UI build is missing at {DIST}. Run npm run build in playground/web.", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
