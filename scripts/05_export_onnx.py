#!/usr/bin/env python
"""Phase 1 (live system): export the final model to ONNX, quantize, verify parity, benchmark.

  python scripts/05_export_onnx.py                       # runs/phase4/enh_final/model.pt
  python scripts/05_export_onnx.py --model path/model.pt

Writes next to the .pt:
  model_fp32.onnx   exact export, dynamic batch (N,3,160,288)
  model_int8.onnx   static INT8 (QDQ), calibrated on training-fold crops only
Then reports, for each variant: agreement with the PyTorch model on the packets in
--parity-split (default cv; never the frozen test split) and latency for a 10-crop batch.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import NUM_CLASSES, load_table, make_transform  # noqa: E402

H, W = 160, 288


def load_torch(path):
    m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu")); return m.eval()


def crops_array(df, n, seed=0):
    tf = make_transform(False, (H, W))
    df = df.sample(min(n, len(df)), random_state=seed)
    return np.stack([tf(Image.open(p).convert("RGB")).numpy() for p in df.path])


class Reader:
    """Calibration data for static quantization."""
    def __init__(self, arr, name):
        self.it = iter([{name: arr[i:i + 1]} for i in range(len(arr))])

    def get_next(self):
        return next(self.it, None)


def bench(fn, x, warm=10, iters=100):
    for _ in range(warm): fn(x)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(x); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/phase4/enh_final/model.pt")
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--parity-split", default="cv", choices=["cv", "test"])
    a = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    mp = Path(a.model); out = mp.parent
    net = load_torch(mp)
    fp32, pre, int8 = out / "model_fp32.onnx", out / "model_fp32_pre.onnx", out / "model_int8.onnx"

    torch.onnx.export(net, torch.zeros(10, 3, H, W), str(fp32), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17, dynamo=False)
    print("exported", fp32)

    df = load_table("enh")
    cv = df[df.split == "cv"]
    # calibrate on CV crops only (never test)
    calib = crops_array(cv, a.calib, seed=1)
    quant_pre_process(str(fp32), str(pre))
    class R(Reader, CalibrationDataReader): pass
    quantize_static(str(pre), str(int8), R(calib, "input"), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink(missing_ok=True)
    print("quantized", int8)

    ev = df[df.split == a.parity_split]
    xs = crops_array(ev, 400, seed=2)
    with torch.no_grad():
        ref = net(torch.from_numpy(xs)).softmax(1).numpy()
    ref_pred = ref.argmax(1)
    ref_good = ref_pred == 4

    report = {}
    x10 = xs[:10]
    with torch.no_grad():
        report["torch_fp32"] = dict(zip(("median_ms", "p95_ms"), bench(lambda x: net(torch.from_numpy(x)), x10, iters=50)))
    # legacy cost for context: 5-model ensemble, 10 sequential single-crop passes
    for name, path in (("onnx_fp32", fp32), ("onnx_int8", int8)):
        so = ort.SessionOptions()
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        run = lambda x: s.run(None, {"input": x})[0]
        lg = np.concatenate([run(xs[i:i + 50]) for i in range(0, len(xs), 50)])
        pr = (np.exp(lg - lg.max(1, keepdims=True)) / np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True))
        report[name] = {
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "argmax_agreement": float((pr.argmax(1) == ref_pred).mean()),
            "good_vs_defect_agreement": float(((pr.argmax(1) == 4) == ref_good).mean()),
            "max_abs_prob_diff": float(np.abs(pr - ref).max()),
            **dict(zip(("median_ms", "p95_ms"), bench(run, x10))),
        }
    report["n_parity_crops"] = int(len(xs)); report["parity_split"] = a.parity_split
    json.dump(report, open(out / "export_report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
