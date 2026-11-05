#!/usr/bin/env python
r"""Compute and visualize the 2x2 PASS / FAIL Confusion Matrix for the tested dataset.

Usage:
    python scripts/plot_confusion_matrix.py
    python scripts/plot_confusion_matrix.py --variant enh --save confusion_matrix_2x2.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import GOOD_CLASS, NUM_CLASSES, StrandDataset, get_split, load_table


def evaluate_folds(variant="enh"):
    """Evaluate each cross-validation fold using its trained out-of-fold model."""
    df = load_table(variant)
    crop_records = []

    print(f"Loading models and evaluating 5 folds ({variant})...")
    for fold in range(5):
        tr, va, _ = get_split(df, fold)
        model_path = ROOT / f"runs/phase4/{variant}_val{fold}/model.pt"
        if not model_path.exists():
            print(f"Warning: {model_path} does not exist, skipping fold {fold}")
            continue

        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        loader = torch.utils.data.DataLoader(StrandDataset(va, False, (160, 288)), batch_size=32)
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                preds.extend(model(x).argmax(1).tolist())

        va_fold = va.copy()
        va_fold["pred_label"] = preds
        crop_records.append(va_fold)

    cv_crops = pd.concat(crop_records, ignore_index=True)
    return cv_crops


def draw_2x2_matrix(ax, cm, title, labels=("PASS", "FAIL"), cmap="Blues"):
    """Draw a clean, styled 2x2 confusion matrix on the given matplotlib axis."""
    total = cm.sum()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=11, fontweight="semibold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=11, fontweight="semibold")
    ax.set_xlabel("Predicted Verdict", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="bold", labelpad=8)

    # Color threshold for text readability
    thresh = cm.max() / 2.0

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            
            # Meaning tag for each cell
            if i == 0 and j == 0:
                tag = "True Pass (TP)"
            elif i == 0 and j == 1:
                tag = "False Rejection (FN)"
            elif i == 1 and j == 0:
                tag = "False Acceptance (FP)"
            else:
                tag = "True Defect (TN)"

            color = "white" if count > thresh else "black"
            text = f"{count}\n({pct:.1f}%)\n{tag}"
            ax.text(j, i, text, ha="center", va="center", color=color,
                    fontsize=10, fontweight="bold", linespacing=1.2)

    # Subtle grid lines
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)


def main():
    parser = argparse.ArgumentParser(description="Generate 2x2 PASS/FAIL Confusion Matrix Visualizations")
    parser.add_argument("--variant", choices=["enh", "cs3"], default="enh", help="Crop preprocessing variant")
    parser.add_argument("--save", default="confusion_matrix_2x2.png", help="Path to save the output image")
    args = parser.parse_args()

    crops_df = evaluate_folds(args.variant)

    # 1. Component (Crop) Level Evaluation
    crops_df["true_pass"] = crops_df["label"] == GOOD_CLASS
    crops_df["pred_pass"] = crops_df["pred_label"] == GOOD_CLASS

    tp_crop = int(((crops_df["true_pass"]) & (crops_df["pred_pass"])).sum())
    fn_crop = int(((crops_df["true_pass"]) & (~crops_df["pred_pass"])).sum())
    fp_crop = int(((~crops_df["true_pass"]) & (crops_df["pred_pass"])).sum())
    tn_crop = int(((~crops_df["true_pass"]) & (~crops_df["pred_pass"])).sum())
    cm_crop = np.array([[tp_crop, fn_crop],
                        [fp_crop, tn_crop]])

    # 2. Packet (Whole Image) Level Evaluation
    pkt_df = crops_df.groupby("packet").agg(
        true_packet_pass=("true_pass", "all"),
        pred_packet_pass=("pred_pass", "all")
    ).reset_index()

    tp_pkt = int(((pkt_df["true_packet_pass"]) & (pkt_df["pred_packet_pass"])).sum())
    fn_pkt = int(((pkt_df["true_packet_pass"]) & (~pkt_df["pred_packet_pass"])).sum())
    fp_pkt = int(((~pkt_df["true_packet_pass"]) & (pkt_df["pred_packet_pass"])).sum())
    tn_pkt = int(((~pkt_df["true_packet_pass"]) & (~pkt_df["pred_packet_pass"])).sum())
    cm_pkt = np.array([[tp_pkt, fn_pkt],
                       [fp_pkt, tn_pkt]])

    # Plot Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Left plot: Crop level
    draw_2x2_matrix(ax1, cm_crop, title=f"Component Level Matrix (N = {len(crops_df)} crops)", cmap="Blues")
    # Right plot: Packet level
    draw_2x2_matrix(ax2, cm_pkt, title=f"Packet Level Matrix (N = {len(pkt_df)} packet images)", cmap="GnBu")

    # Metrics summary text at bottom
    acc_crop = (tp_crop + tn_crop) / len(crops_df) * 100
    defect_recall_crop = tn_crop / (tn_crop + fp_crop) * 100
    far_crop = fp_crop / (tn_crop + fp_crop) * 100

    acc_pkt = (tp_pkt + tn_pkt) / len(pkt_df) * 100
    defect_recall_pkt = tn_pkt / (tn_pkt + fp_pkt) * 100

    footer = (
        f"Crop Metrics: Accuracy = {acc_crop:.1f}% | Defect Recall = {defect_recall_crop:.1f}% | False Acceptance Rate = {far_crop:.1f}%\n"
        f"Packet Metrics: Accuracy = {acc_pkt:.1f}% | Defect Packet Catch Rate = {defect_recall_pkt:.1f}%"
    )
    plt.figtext(0.5, 0.02, footer, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path = Path(args.save)
    plt.savefig(out_path, dpi=200)
    print(f"\nVisual confusion matrix saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

