#!/usr/bin/env python
"""Phase 4: train ResNet18 on enh or cs3 crops. Model selection uses a CV validation fold only.

  python scripts/03_train.py --variant enh --val-fold 0 --epochs 15
  python scripts/03_train.py --variant enh --smoke
  python scripts/03_train.py --variant enh --final --epochs N   # train on all CV, evaluate frozen test ONCE
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import (GOOD_CLASS, NUM_CLASSES, StrandDataset, class_weights,
                         get_split, load_table)


def metrics(y, p):
    y, p = np.array(y), np.array(p)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1); rc = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rc > 0, 2 * prec * rc / np.maximum(prec + rc, 1e-9), 0.0)
    present = cm.sum(1) > 0
    rec = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(NUM_CLASSES)]
    d = y != GOOD_CLASS
    return {"accuracy": float((y == p).mean()),
            "macro_f1": float(f1[present].mean()),
            "per_class_recall": rec,
            "defect_recall": float((p[d] != GOOD_CLASS).mean()) if d.any() else None,
            "false_acceptance_rate": float((p[d] == GOOD_CLASS).mean()) if d.any() else None,  # defects passed as GOOD
            "false_rejection_rate": float((p[~d] != GOOD_CLASS).mean()) if (~d).any() else None,
            "confusion_matrix": cm.tolist(), "n": int(len(y))}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
    return metrics(ys, ps)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--variant", choices=["enh", "cs3"], default="enh")
    a.add_argument("--val-fold", type=int, default=0)
    a.add_argument("--epochs", type=int, default=15)
    a.add_argument("--bs", type=int, default=32)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--size", type=int, nargs=2, default=[160, 288], metavar=("H", "W"))
    a.add_argument("--final", action="store_true", help="train on all CV folds, evaluate test (once)")
    a.add_argument("--smoke", action="store_true")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="runs/phase4")
    a = a.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_table(a.variant)
    tr, va, te = get_split(df, a.val_fold, use_test=a.final)
    ev = te if a.final else va
    if a.smoke:
        tr = tr.groupby("label").head(4); ev = ev.groupby("label").head(3); a.epochs = 2
    assert not set(tr.packet) & set(ev.packet), "packet leakage"
    tag = f"{a.variant}_{'final' if a.final else 'val%d' % a.val_fold}{'_smoke' if a.smoke else ''}"
    out = Path(a.out) / tag; out.mkdir(parents=True, exist_ok=True)

    size = tuple(a.size)
    tl = DataLoader(StrandDataset(tr, True, size), a.bs, shuffle=True, drop_last=len(tr) > a.bs)
    el = DataLoader(StrandDataset(ev, False, size), a.bs)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(dev)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr.label.tolist()).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    best, hist = -1, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); tot += loss.item() * len(y)
        sched.step()
        m = evaluate(model, el, dev)
        hist.append({"epoch": ep, "loss": tot / len(tr), "macro_f1": m["macro_f1"],
                     "acc": m["accuracy"], "far": m["false_acceptance_rate"]})
        print(f"ep{ep} loss {tot/len(tr):.3f} f1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f} "
              f"defRecall {m['defect_recall']} FAR {m['false_acceptance_rate']} ({time.time()-t0:.0f}s)", flush=True)
        if a.final:   # never select on test: keep last epoch
            best_m = dict(m, epoch=ep); torch.save(model.state_dict(), out / "model.pt")
        elif m["macro_f1"] > best:   # selection on validation fold only
            best, best_m = m["macro_f1"], dict(m, epoch=ep); torch.save(model.state_dict(), out / "model.pt")
    json.dump({"args": vars(a), "history": hist, "result": best_m}, open(out / "metrics.json", "w"), indent=1)
    print("RESULT", json.dumps({k: v for k, v in best_m.items() if k != "confusion_matrix"}))
    print(np.array(best_m["confusion_matrix"]))


if __name__ == "__main__":
    main()
