"""Local packet inference: photo in -> per-component results + annotated picture.

No network / API key needed. Used by scripts/04_predict.py and app_local.py.
"""
import glob
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import models

from src.dataset import GOOD_CLASS, NUM_CLASSES, make_transform
from src.localize import (box_to_pixels, crop_fixed_aspect, grid_components,
                          solid_background_present)
from src.preprocess import make_variants

ROOT = Path(__file__).resolve().parents[1]
W, H = 576, 320
COL = {"GOOD": (40, 170, 70), "DEFECT": (220, 50, 50), "UNCERTAIN": (240, 170, 20)}
DEFAULT_MODELS = "runs/phase4/enh_val[0-4]/model.pt"


def default_model_paths():
    return sorted(glob.glob(str(ROOT / DEFAULT_MODELS)))


def load_models(paths):
    nets = []
    for p in paths:
        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        nets.append(m.eval())
    return nets


def verdict(pred, conf, thr):
    if conf < thr:
        return "UNCERTAIN"
    return "GOOD" if pred == GOOD_CLASS else "DEFECT"


def inspect_packet(image, nets, conf_thr=0.6):
    """image: PIL image. Returns dict(results, crops, passed, table_mode, image)."""
    img = ImageOps.exif_transpose(image).convert("RGB")
    table = solid_background_present(img)   # yellow table behind packet -> mask it out
    comps, _ = grid_components(img, 2, 5, 0.72, split="mid", col_overlap=0.08, reject_solid=table)
    tf = make_transform(False)
    results, crops = [], []
    for c in comps:
        px = box_to_pixels(c["bbox"], img.width, img.height)
        crop = crop_fixed_aspect(img, px, W / H, 0.04).resize((W, H), Image.LANCZOS)
        x = tf(Image.fromarray(make_variants(crop)["enh"])).unsqueeze(0)
        with torch.no_grad():
            prob = torch.stack([torch.softmax(n(x), 1)[0] for n in nets]).mean(0)
        conf, pred = prob.max(0)
        results.append({**c, "px": px, "pred": int(pred), "conf": float(conf),
                        "verdict": verdict(int(pred), float(conf), conf_thr)})
        crops.append(crop)
    passed = all(r["verdict"] == "GOOD" for r in results)
    return {"results": results, "crops": crops, "passed": passed, "table_mode": table, "image": img}


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
