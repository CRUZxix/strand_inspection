"""Streaming inspection server (Phase 3).

  uvicorn server.app:app --host 0.0.0.0 --port 8000

WebSocket /ws/inspect
  client -> server  binary : one JPEG frame, optionally prefixed by a 4-byte big-endian
                             sequence number (detected by the missing JPEG magic FF D8)
  client -> server  text   : {"cmd": "reset"}   forget the current packet, start over
                             {"cmd": "ping"}    -> {"type": "pong"}
  server -> client  text   : JSON status per PROCESSED frame (see Inspector.process), plus
                             "type":"result", "seq", "frame_size":[w,h], "dropped", "latency_ms"

Backpressure: only the NEWEST frame is kept. If the phone sends faster than the server can
infer, older frames are dropped (reported in "dropped") so latency never accumulates.

  client -> server  text   : {"cmd": "snapshot"}  the NEXT frame gets a one-shot full inspection
                             (no voting) -> {"type": "snapshot", ...same fields as a result}

GET /health -> model/engine info.   GET / -> the mobile web app (web/).
"""
import asyncio
import io
import json
import struct
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.infer import inspect_packet, load_classifier
from src.live import Inspector

MAX_FRAME_BYTES = 4 * 1024 * 1024
# One worker: Inspector state is per-connection, and ONNX/CPU inference already uses all cores.
_pool = ThreadPoolExecutor(max_workers=1)
_state = {}


@asynccontextmanager
async def lifespan(app):
    _state["clf"] = load_classifier()
    yield


app = FastAPI(title="Strand inspection", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "engine": type(_state["clf"]).__name__}


def decode(data):
    """-> (seq | None, PIL image). Raises ValueError on a bad frame."""
    seq = None
    if data[:2] != b"\xff\xd8":
        if len(data) < 6:
            raise ValueError("frame too short")
        seq = struct.unpack(">I", data[:4])[0]; data = data[4:]
    try:
        img = Image.open(io.BytesIO(data)); img.load()
    except Exception as e:
        raise ValueError(f"undecodable image: {e}")
    return seq, img


def snapshot_payload(img, clf):
    """One-shot inspection of a single frame (full pipeline, no temporal voting)."""
    try:
        r = inspect_packet(img, clf)
    except ValueError as e:
        raise ValueError(f"no packet found: {e}")
    boxes = [{"id": x["position"], "coords": list(x["px"]), "count": x["pred"],
              "conf": round(x["conf"], 3), "status": x["verdict"]} for x in r["results"]]
    st = {b["status"] for b in boxes}
    verdict = "FAIL" if "DEFECT" in st else "UNCERTAIN" if "UNCERTAIN" in st else "PASS"
    return {"state": "SNAPSHOT", "packet_verdict": verdict, "boxes": boxes, "hint": "Snapshot",
            "blur_var": None, "relocalized": True, "votes": 0}


def run_frame(inspector, data, snapshot=False):
    t0 = time.perf_counter()
    seq, img = decode(data)
    t1 = time.perf_counter()
    out = snapshot_payload(img, _state["clf"]) if snapshot else inspector.process(img)
    out.update(type="snapshot" if snapshot else "result", seq=seq, frame_size=[img.width, img.height],
               decode_ms=round((t1 - t0) * 1000, 1),
               latency_ms=round((time.perf_counter() - t0) * 1000, 1))   # decode + inference
    return out


@app.websocket("/ws/inspect")
async def ws_inspect(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    inspector = Inspector(_state["clf"])
    slot = {"data": None, "dropped": 0, "snap": False}
    ready = asyncio.Event()
    closed = asyncio.Event()

    async def reader():
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if len(msg["bytes"]) > MAX_FRAME_BYTES:
                        await ws.send_json({"type": "error", "error": "frame too large"})
                        continue
                    if slot["data"] is not None:
                        slot["dropped"] += 1          # replaced an unprocessed frame
                    slot["data"] = msg["bytes"]; ready.set()
                elif msg.get("text") is not None:
                    try:
                        cmd = json.loads(msg["text"]).get("cmd")
                    except (ValueError, AttributeError):
                        cmd = None
                    if cmd == "reset":
                        await loop.run_in_executor(_pool, inspector.reset)
                        await ws.send_json({"type": "reset"})
                    elif cmd == "snapshot":
                        slot["snap"] = True            # applies to the next frame processed
                    elif cmd == "ping":
                        await ws.send_json({"type": "pong"})
                    else:
                        await ws.send_json({"type": "error", "error": "unknown command"})
        except WebSocketDisconnect:
            pass
        finally:
            closed.set(); ready.set()

    async def worker():
        while True:
            await ready.wait()
            if closed.is_set():
                return
            data, dropped, snap = slot["data"], slot["dropped"], slot["snap"]
            slot["data"], slot["dropped"] = None, 0
            ready.clear()
            if data is None:
                continue
            slot["snap"] = False
            try:
                out = await loop.run_in_executor(_pool, run_frame, inspector, data, snap)
                out["dropped"] = dropped
                await ws.send_json(out)
            except ValueError as e:                    # bad frame: report, keep the connection
                await ws.send_json({"type": "error", "error": str(e)})
            except RuntimeError:                       # socket already closed
                return

    r, w = asyncio.create_task(reader()), asyncio.create_task(worker())
    await asyncio.wait({r, w}, return_when=asyncio.FIRST_COMPLETED)
    for t in (r, w):
        t.cancel()


# the mobile web app; mounted last so /health and /ws/* take precedence
_web = Path(__file__).resolve().parents[1] / "web"
if _web.exists():
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
