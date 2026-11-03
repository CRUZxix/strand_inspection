#!/usr/bin/env python
r"""Inspect one NEW packet photo with the trained model(s) and save an annotated picture.

  python scripts/04_predict.py path\to\photo.jpg
  python scripts/04_predict.py photo.jpg --models runs/phase4/enh_val0/model.pt --open

Logic lives in src/infer.py (shared with app_local.py). With several --models (default:
all enh_val* fold models) softmax is averaged. Component verdict: count 4 = GOOD, other
count = DEFECT, max-prob < --conf = UNCERTAIN (threshold NOT yet tuned). Packet PASSES
only if all 10 are GOOD.
"""
import argparse, os, sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.infer import default_model_paths, inspect_packet, load_models, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--out", default="runs/predictions")
    ap.add_argument("--open", action="store_true", help="open the result image when done")
    a = ap.parse_args()

    paths = a.models or default_model_paths()
    if not paths:
        sys.exit("no model.pt found; train first")
    print(f"using {len(paths)} model(s)")
    out = inspect_packet(Image.open(a.image), load_models(paths), a.conf)
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
