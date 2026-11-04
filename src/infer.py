"""Local packet inference: photo in -> per-component results + annotated picture.

No network / API key needed. Used by scripts/04_predict.py and app_local.py.
"""
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import models

from src.dataset import GOOD_CLASS, NUM_CLASSES, make_transform
from src.localize import (box_to_pixels, crop_fixed_aspect, grid_components,
                          solid_background_present)
from src.preprocess import enhance_rgb

ROOT = Path(__file__).resolve().parents[1]
W, H = 576, 320
COL = {"GOOD": (40, 170, 70), "DEFECT": (220, 50, 50), "UNCERTAIN": (240, 170, 20)}
DEFAULT_MODELS = "runs/phase4/enh_val[0-4]/model.pt"


def default_model_paths():
    return sorted(glob.glob(str(ROOT / DEFAULT_MODELS)))


FINAL_MODEL = "runs/phase4/enh_final/model.pt"
FINAL_ONNX = "runs/phase4/enh_final/model_int8.onnx"
INPUT_HW = (160, 288)
_TF = make_transform(False, INPUT_HW)


def load_models(paths):
    nets = []
    for p in paths:
        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        nets.append(m.eval())
    return nets


class TorchClassifier:
    """One or more torch nets (ensemble = mean of softmax). probs(batch) -> (N, 6) numpy."""

    def __init__(self, nets):
        self.nets = nets if isinstance(nets, (list, tuple)) else [nets]

    def probs(self, batch):
        x = torch.from_numpy(batch)
        with torch.no_grad():   # ONE forward pass per net for the whole packet
            return torch.stack([torch.softmax(n(x), 1) for n in self.nets]).mean(0).numpy()


class OrtClassifier:
    """ONNX Runtime session (no PyTorch needed at inference time)."""

    def __init__(self, path, threads=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def probs(self, batch):
        logits = self.sess.run(None, {self.inp: batch})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def load_classifier(path=None):
    """.onnx -> OrtClassifier; .pt -> single torch net; None -> final .onnx/.pt if present,
    else the legacy 5-fold ensemble."""
    if path is None:
        for cand in (FINAL_ONNX, FINAL_MODEL):
            if (ROOT / cand).exists():
                path = ROOT / cand
                break
    if path is None:
        return TorchClassifier(load_models(default_model_paths()))
    path = Path(path)
    return OrtClassifier(path) if path.suffix == ".onnx" else TorchClassifier(load_models([path]))


def make_batch(crops):
    """List of PIL crops -> float32 (N, 3, H, W) batch tensor (same preprocessing as training)."""
    return np.stack([_TF(Image.fromarray(enhance_rgb(c))).numpy() for c in crops])


def verdict(pred, conf, thr):
    if conf < thr:
        return "UNCERTAIN"
    return "GOOD" if pred == GOOD_CLASS else "DEFECT"


def inspect_packet(image, nets, conf_thr=0.6):
    """image: PIL image. nets: classifier from load_classifier(), or a list of torch nets
    (legacy). Returns dict(results, crops, passed, table_mode, image, timing_ms)."""
    clf = nets if hasattr(nets, "probs") else TorchClassifier(nets)
    t0 = time.perf_counter()
    img = ImageOps.exif_transpose(image).convert("RGB")
    table = solid_background_present(img)   # yellow table behind packet -> mask it out
    comps, _ = grid_components(img, 2, 5, 0.72, split="mid", col_overlap=0.08, reject_solid=table)
    pxs = [box_to_pixels(c["bbox"], img.width, img.height) for c in comps]
    crops = [crop_fixed_aspect(img, px, W / H, 0.04).resize((W, H), Image.LANCZOS) for px in pxs]
    t1 = time.perf_counter()
    batch = make_batch(crops)                # (10, 3, 160, 288)
    t2 = time.perf_counter()
    prob = clf.probs(batch)                  # single forward pass for all components
    t3 = time.perf_counter()
    results = []
    for c, px, p in zip(comps, pxs, prob):
        pred = int(p.argmax()); conf = float(p[pred])
        results.append({**c, "px": px, "pred": pred, "conf": conf,
                        "verdict": verdict(pred, conf, conf_thr)})
    passed = all(r["verdict"] == "GOOD" for r in results)
    ms = lambda a, b: round((b - a) * 1000, 1)
    return {"results": results, "crops": crops, "passed": passed, "table_mode": table, "image": img,
            "timing_ms": {"localize_crop": ms(t0, t1), "preprocess": ms(t1, t2),
                          "model": ms(t2, t3), "total": ms(t0, t3)}}


def _font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(out):
    """Annotated photo + 5x2 result matrix side by side (PIL image)."""
    img, res, crops, passed = out["image"], out["results"], out["crops"], out["passed"]
    ann = img.copy(); d = ImageDraw.Draw(ann); f = _font(max(18, img.width // 45))
    for r in res:
        d.rectangle(r["px"], outline=COL[r["verdict"]], width=max(3, img.width // 300))
        d.text((r["px"][0] + 6, r["px"][1] + 4), f"{r['position']}: {r['pred']} ({r['conf']:.0%})",
               fill=COL[r["verdict"]], font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    ann.thumbnail((1400, 1400))

    cw, ch, gap, top = 300, 167, 8, 70
    mat = Image.new("RGB", (2 * cw + 3 * gap, top + 5 * (ch + gap) + gap), (30, 30, 30))
    md = ImageDraw.Draw(mat); fm = _font(18)
    md.text((gap, 18), f"PACKET {'PASS' if passed else 'FAIL'}",
            fill=(40, 200, 80) if passed else (240, 70, 70), font=_font(34))
    for r, crop in zip(res, crops):
        x0 = gap + (r["col"] - 1) * (cw + gap); y0 = top + (r["row"] - 1) * (ch + gap)
        mat.paste(crop.resize((cw, ch)), (x0, y0))
        md.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=COL[r["verdict"]], width=5)
        md.rectangle([x0, y0, x0 + cw, y0 + 26], fill=COL[r["verdict"]])
        md.text((x0 + 6, y0 + 3), f"{r['position']}  count={r['pred']}  {r['verdict']} {r['conf']:.0%}",
                fill=(255, 255, 255), font=fm)
    mat = mat.resize((int(mat.width * ann.height / mat.height), ann.height))
    canvas = Image.new("RGB", (ann.width + mat.width, ann.height))
    canvas.paste(ann, (0, 0)); canvas.paste(mat, (ann.width, 0))
    return canvas
