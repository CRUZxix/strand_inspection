#!/usr/bin/env python
r"""Inspect one NEW packet photo with the trained model(s) and save an annotated picture.

  python scripts/04_predict.py path\to\photo.jpg
  python scripts/04_predict.py photo.jpg --models runs/phase4/enh_val0/model.pt --open

Logic lives in src/infer.py (shared with app_local.py). Default model: runs/phase4/enh_final/model_int8.onnx
(else model.pt, else the legacy enh_val* fold ensemble). Several .pt --models are averaged. Component verdict: count 4 = GOOD, other
count = DEFECT, max-prob < --conf = UNCERTAIN (threshold NOT yet tuned). Packet PASSES
only if all 10 are GOOD.
"""
import argparse, os, sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.infer import (TorchClassifier, inspect_packet, load_classifier, load_models,  # noqa: E402
                       render)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--out", default="runs/predictions")
    ap.add_argument("--open", action="store_true", help="open the result image when done")
    a = ap.parse_args()

    if a.models and all(m.endswith(".pt") for m in a.models):
        clf = TorchClassifier(load_models(a.models))          # explicit .pt model(s): ensemble if >1
    else:
        clf = load_classifier(a.models[0] if a.models else None)  # .onnx, or final model, or legacy folds
    out = inspect_packet(Image.open(a.image), clf, a.conf)
    print("timing (ms):", out["timing_ms"])
    print("yellow background detected: table-rejecting mask" if out["table_mode"] else "standard localisation")

    print("\ncount matrix (rows x cols):")
    by = {(r["row"], r["col"]): r for r in out["results"]}
    for rr in range(1, 6):
        print("  " + "   ".join(f"{by[(rr, cc)]['pred']}/{by[(rr, cc)]['verdict'][0]}" for cc in (1, 2)))
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{Path(a.image).stem}_result.png"; render(out).save(fp)
    print(f"PACKET {'PASS' if out['passed'] else 'FAIL'}   saved: {fp}")
    if a.open:
        os.startfile(fp)


if __name__ == "__main__":
    main()
