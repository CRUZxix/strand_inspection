#!/usr/bin/env python
"""Phase 2 check: simulate a handheld video from still photos and compare a naive per-frame
verdict against the Inspector (quality gate + tracking + voting).

  python scripts/06_simulate_stream.py --calibrate     # blur-threshold calibration only
  python scripts/06_simulate_stream.py                 # full stream simulation

Frames: ~720px wide, random shift/scale (hand shake), brightness change, soft glare patches,
and motion-blurred frames. Uses CV packets only (never the frozen test split), so this
measures temporal STABILITY, not generalisation accuracy (the model trained on these).
"""
import argparse, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.infer import inspect_packet, load_classifier  # noqa: E402
from src.live import Inspector, frame_sharpness  # noqa: E402


def load_small(path, width=720):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def shake(im, rng, shift=0.02, scale=0.02):
    a = np.asarray(im); h, w = a.shape[:2]
    s = 1 + rng.uniform(-scale, scale)
    M = np.float32([[s, 0, rng.uniform(-shift, shift) * w + (1 - s) * w / 2],
                    [0, s, rng.uniform(-shift, shift) * h + (1 - s) * h / 2]])
    out = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out *= rng.uniform(0.88, 1.12)
    return out


def glare(a, rng):
    h, w = a.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(rng.uniform(.2, .8) * w), int(rng.uniform(.2, .8) * h)),
                (int(rng.uniform(.05, .15) * w), int(rng.uniform(.03, .08) * h)), rng.uniform(0, 180), 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 15)[..., None]
    return a * (1 - 0.6 * m) + 255 * 0.6 * m


def motion_blur(a, k):
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(a, -1, ker)


def to_pil(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_stream(path, rng, n=30, blur_frames=(0, 1, 2, 17), glare_frames=(8, 9, 21)):
    base = load_small(path)
    frames = []
    for i in range(n):
        a = shake(base, rng)
        if i in glare_frames:
            a = glare(a, rng)
        if i in blur_frames:
            a = motion_blur(a, 31)
        frames.append((to_pil(a), i in blur_frames))
    return frames


def calibrate(paths, rng):
    sharp, blur = [], []
    for p in paths:
        base = load_small(p)
        for _ in range(3):
            a = shake(base, rng)
            sharp.append(frame_sharpness(to_pil(a)))
            blur.append(frame_sharpness(to_pil(motion_blur(a, 31))))
    print(f"sharp   min {min(sharp):7.1f}  p5 {np.percentile(sharp, 5):7.1f}  median {np.median(sharp):7.1f}")
    print(f"blurred max {max(blur):7.1f}  p95 {np.percentile(blur, 95):7.1f}  median {np.median(blur):7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--packets", type=int, default=12)
    ap.add_argument("--blur-thr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    ds = pd.read_csv(ROOT / "data_build" / "dataset.csv")
    pk = ds[ds.split == "cv"].groupby("packet").agg(defect=("is_defect", "max"), fn=("image_filename", "first"))
    good = pk[pk.defect == 0]            # dataset has very few fully-good packets (1 in CV)
    bad = pk[pk.defect == 1].sample(a.packets - len(good), random_state=a.seed)
    sel = pd.concat([good, bad])
    paths = [ROOT / "images" / f for f in sel.fn]
    if a.calibrate:
        return calibrate(paths, rng)

    clf = load_classifier()
    kw = {} if a.blur_thr is None else {"blur_thr": a.blur_thr}
    print(f"{'packet':27} {'truth':5} | naive: flips  wrong% | live: verdict  latch@frame  changes  ok  tracked/full  ms/frame")
    tot = dict(flips=0, wrong=0, frames=0, ok=0, changes=0, lat=[])
    for (pkt, row), p in zip(sel.iterrows(), paths):
        truth = "FAIL" if row.defect else "PASS"
        frames = make_stream(p, rng)
        # naive: classify EVERY frame independently with the full Phase-1 pipeline
        nv = []
        for im, _ in frames:
            try:
                nv.append("PASS" if inspect_packet(im, clf)["passed"] else "FAIL")
            except ValueError:
                nv.append("NONE")
        flips = sum(x != y for x, y in zip(nv, nv[1:])); wrong = sum(v != truth for v in nv)
        # live engine
        ins = Inspector(clf, **kw); seq = []; full = 0; lat = None; times = []
        for i, (im, _) in enumerate(frames):
            t = time.perf_counter(); out = ins.process(im); times.append((time.perf_counter() - t) * 1000)
            seq.append(out["packet_verdict"]); full += out["relocalized"]
            if lat is None and out["state"] == "LATCHED":
                lat = i
        final = seq[-1]
        shown = [s for s in seq if s]
        changes = sum(x != y for x, y in zip(shown, shown[1:]))
        ok = final == truth
        print(f"{pkt:27} {truth:5} | {flips:11d}  {100 * wrong / len(nv):5.0f}% | {str(final):10} {str(lat):>10}  {changes:7d}  "
              f"{'Y' if ok else 'N'}  {len(frames) - full:3d}/{full:<3d}  {np.mean(times):7.0f}")
        tot["flips"] += flips; tot["wrong"] += wrong; tot["frames"] += len(nv); tot["ok"] += ok
        tot["changes"] += changes; tot["lat"].append(lat if lat is not None else len(frames))
    n = len(sel)
    print(f"\nnaive per-frame: {tot['flips']} verdict flips over {n} streams, {100 * tot['wrong'] / tot['frames']:.1f}% frames wrong")
    print(f"live engine    : {tot['changes']} verdict changes after first verdict, {tot['ok']}/{n} streams correct, "
          f"median latch frame {int(np.median(tot['lat']))}")


if __name__ == "__main__":
    main()
