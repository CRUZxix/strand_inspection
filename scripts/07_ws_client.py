#!/usr/bin/env python
"""Phase 3 test client: stream a simulated handheld video to the server and profile it.

  uvicorn server.app:app --port 8000          # in another terminal
  python scripts/07_ws_client.py images/PKT_20261027_125031_EE17.jpg --fps 8 --frames 40

Sends JPEG frames (4-byte seq prefix) at --fps, logs state changes and the round trip
(send -> JSON received) per frame.
"""
import argparse, asyncio, io, json, random, struct, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import importlib
sim = importlib.import_module("06_simulate_stream")


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return b.getvalue()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/inspect")
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import websockets

    frames = [jpeg(im, a.quality) for im, _ in sim.make_stream(Path(a.image), random.Random(a.seed), a.frames)]
    print(f"{len(frames)} frames, avg {np.mean([len(f) for f in frames]) / 1024:.0f} KB each")
    sent, rtt, results = {}, [], []

    async with websockets.connect(a.url, max_size=2 ** 22) as ws:
        async def sender():
            for i, f in enumerate(frames):
                sent[i] = time.perf_counter()
                await ws.send(struct.pack(">I", i) + f)
                await asyncio.sleep(1 / a.fps)

        async def receiver():
            prev = None
            while True:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                except asyncio.TimeoutError:
                    return
                if msg.get("type") != "result":
                    print("msg:", msg); continue
                ms = (time.perf_counter() - sent[msg["seq"]]) * 1000
                rtt.append(ms); results.append(msg)
                key = (msg["state"], msg["packet_verdict"])
                if key != prev:
                    print(f"seq {msg['seq']:3d}  {msg['state']:10} {str(msg['packet_verdict']):9} "
                          f"{msg['hint']:26} server {msg['latency_ms']:6.1f} ms  rtt {ms:6.1f} ms  dropped {msg['dropped']}")
                    prev = key

        s = asyncio.create_task(sender()); r = asyncio.create_task(receiver())
        await s; await r

    srv = [m["latency_ms"] for m in results]
    print(f"\nprocessed {len(results)}/{len(frames)} frames "
          f"(dropped {sum(m['dropped'] for m in results)}), final verdict: {results[-1]['packet_verdict']}")
    print(f"server latency  median {np.median(srv):.1f} ms  p95 {np.percentile(srv, 95):.1f} ms")
    print(f"round trip      median {np.median(rtt):.1f} ms  p95 {np.percentile(rtt, 95):.1f} ms")
    cheap = [m["latency_ms"] for m in results if m["state"] in ("SEARCHING", "LATCHED")]
    if cheap:
        print(f"latched/idle frames only: server median {np.median(cheap):.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())
