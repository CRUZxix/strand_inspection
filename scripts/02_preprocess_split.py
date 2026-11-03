#!/usr/bin/env python
"""Phase 2: clean labels, make packet-level splits, preprocess every crop.

Run from the repo root after 01_make_crops.py:
    python scripts/02_preprocess_split.py

Reads   data_build/crops.csv
Writes  data_build/dataset.csv          crops + clean targets + split/fold columns
        data_build/splits.json          packet -> fold (frozen; do not regenerate casually)
        data_build/split_report.csv     class counts per fold
        data_build/label_issues.csv     suspicious labels to review by hand
        data_build/processed/<variant>/ preprocessed crops (enh, cs3)
        data_build/preview_preprocess.jpg  visual check of the preprocessing

Splitting is by PACKET, never by crop: neighbouring crops overlap, so a crop-level
split leaks. Fold 0 is the held-out TEST set; folds 1..K-1 are the CV folds.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import make_variants  # noqa: E402

NONE_LIKE = {"", "nan", "none", "good", "ok", "no", "null", "0", "false", "-"}


# ------------------------------------------------------------------ labels
def clean_labels(df, expected):
    df = df.copy()
    df["strand_count"] = pd.to_numeric(df["strand_count"], errors="coerce")
    # NOTE: blank cells are NaN; with recent pandas astype(str) keeps them NaN, so fill first
    if "defect_type" in df:
        df["defect_type"] = df["defect_type"].fillna("none").astype(str).str.strip().str.lower()
    else:
        df["defect_type"] = "none"
    df.loc[df["defect_type"].isin(NONE_LIKE), "defect_type"] = "none"
    if "status" in df:
        df["status"] = df["status"].fillna("").astype(str).str.strip().str.upper()
    else:
        df["status"] = np.where(df["defect_type"] == "none", "GOOD", "DEFECT")
    df["is_defect"] = ((df["status"] == "DEFECT") | (df["defect_type"] != "none")).astype(int)
    return df


def defect_mirrors_count(df):
    """True when every defect_type is just 'strand_count_<N>' (a restatement of the count)."""
    kinds = df.loc[df["defect_type"] != "none", "defect_type"]
    return len(kinds) > 0 and kinds.str.fullmatch(r"strand_count_\d+").all()


def label_issues(df, expected):
    """Rows whose labels contradict each other. Review these by hand."""
    rows = []
    for _, r in df.iterrows():
        why = []
        if pd.isna(r["strand_count"]):
            why.append("strand_count missing")
        else:
            if r["strand_count"] != expected and r["status"] == "GOOD":
                why.append(f"status GOOD but strand_count={int(r['strand_count'])}")
            if r["strand_count"] == expected and r["status"] == "DEFECT" and r["defect_type"] == "none":
                why.append("status DEFECT but count is as expected and no defect_type")
            if r["strand_count"] != expected and r["status"] == "DEFECT" and r["defect_type"] == "none":
                why.append("DEFECT with no defect_type")
            m = re.fullmatch(r"strand_count_(\d+)", r["defect_type"])
            if m and int(m.group(1)) != int(r["strand_count"]):
                why.append(f"defect_type says {r['defect_type']} but strand_count={int(r['strand_count'])}")
            if r["strand_count"] == expected and r["is_defect"] == 1 and not m:
                why.append("count is as expected but row is marked defective")
        if why:
            rows.append({"crop_path": r["crop_path"], "packet": r["packet"], "position": r["position"],
                         "strand_count": r["strand_count"], "status": r["status"],
                         "defect_type": r["defect_type"], "issue": "; ".join(why)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ splits
def class_keys(row, mirrored=False):
    """Classes used to balance the folds. When defect_type merely repeats the
    strand count it is not a separate class, so only the count is used."""
    keys = [f"count={int(row['strand_count'])}"]
    if row["defect_type"] != "none" and not (mirrored and str(row["defect_type"]).startswith("strand_count_")):
        keys.append(f"defect={row['defect_type']}")
    return keys


def assign_folds(df, k, seed, mirrored=False):
    """Greedy group-stratified assignment: spread every class (rare ones first)
    as evenly as possible over k folds while keeping each packet in one fold."""
    rng = np.random.default_rng(seed)
    per_packet = {}
    for pk, g in df.groupby("packet"):
        counts = {}
        for _, r in g.iterrows():
            for key in class_keys(r, mirrored):
                counts[key] = counts.get(key, 0) + 1
        per_packet[pk] = counts
    totals = {}
    for counts in per_packet.values():
        for key, v in counts.items():
            totals[key] = totals.get(key, 0) + v
    target = {key: v / k for key, v in totals.items()}
    # packets holding the rarest classes are placed first
    weight = {pk: sum(v / totals[key] for key, v in c.items()) for pk, c in per_packet.items()}
    order = sorted(per_packet, key=lambda p: (-weight[p], rng.random()))
    cur = [dict.fromkeys(totals, 0) for _ in range(k)]
    size = [0] * k
    avg_size = len(per_packet) / k
    fold_of = {}
    for pk in order:
        best, best_cost = None, None
        for f in rng.permutation(k):
            # convex cost: prefer the fold that holds the least of every class this
            # packet contains (rare classes weigh most because their target is small)
            cost = sum((2 * cur[f][key] * v + v * v) / (target[key] ** 2)
                       for key, v in per_packet[pk].items())
            cost += (2 * size[f] + 1) / (avg_size ** 2)
            if best_cost is None or cost < best_cost:
                best, best_cost = f, cost
        fold_of[pk] = int(best)
        size[best] += 1
        for key, v in per_packet[pk].items():
            cur[best][key] += v
    return fold_of


def split_report(df, mirrored=False):
    table = {}
    for fold, g in df.groupby("fold"):
        counts = {"packets": g["packet"].nunique(), "crops": len(g)}
        for _, r in g.iterrows():
            for key in class_keys(r, mirrored):
                counts[key] = counts.get(key, 0) + 1
        table[f"fold{fold}" + (" (TEST)" if fold == 0 else "")] = counts
    return pd.DataFrame(table).fillna(0).astype(int)


# ------------------------------------------------------------------ images
def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def save_variants(build, crop_path, variants):
    stem = Path(crop_path).stem
    out = make_variants(Image.open(build / crop_path))
    paths = {}
    for v in variants:
        d = build / "processed" / v
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{stem}.png"
        cv2.imwrite(str(p), cv2.cvtColor(out[v], cv2.COLOR_RGB2BGR))
        paths[f"path_{v}"] = f"processed/{v}/{stem}.png"
    return paths


def preview(build, df, n_per_group=2, max_rows=10, thumb=(288, 160)):
    """Raw | enhanced | CLAHE-L | edges | belt mask for a mix of classes."""
    picks = []
    for _, g in df.groupby(df["is_defect"] * 10 + df["strand_count"].fillna(-1).astype(int)):
        picks.append(g.sample(min(n_per_group, len(g)), random_state=0))
    picks = pd.concat(picks).head(max_rows)
    cols = ["raw", "enhanced", "CLAHE-L", "edge (Sobel-Feldman)", "belt mask"]
    font, head = load_font(14), 20
    W, H = thumb
    sheet = Image.new("RGB", (W * len(cols), (H + head) * len(picks) + head), "white")
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(cols):
        d.text((i * W + 4, 3), c, fill="black", font=font)
    for r, (_, row) in enumerate(picks.iterrows()):
        out = make_variants(Image.open(build / row["crop_path"]))
        tiles = [out["raw"], out["enh"], np.dstack([out["parts"]["clahe"]] * 3),
                 np.dstack([out["parts"]["edge"]] * 3), np.dstack([out["parts"]["mask"]] * 3)]
        y = head + r * (H + head)
        d.text((4, y + 2), f"{row['position']} {row['packet'][-9:]} count={int(row['strand_count'])} "
                           f"{row['status']} {row['defect_type']}", fill="black", font=font)
        for i, t in enumerate(tiles):
            sheet.paste(Image.fromarray(t).resize(thumb), (i * W, y + head))
    sheet.save(build / "preview_preprocess.jpg", quality=90)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--build", default="data_build")
    ap.add_argument("--expected", type=int, default=4, help="expected strands for a good component")
    ap.add_argument("--folds", type=int, default=6, help="fold 0 = test, the rest = CV folds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="+", default=["enh", "cs3"], choices=["enh", "cs3"])
    ap.add_argument("--regen-splits", action="store_true",
                    help="ignore an existing splits.json and make new splits (breaks old results!)")
    ap.add_argument("--skip-images", action="store_true", help="only labels + splits")
    args = ap.parse_args()

    build = Path(args.build)
    crops = pd.read_csv(build / "crops.csv")
    for col in ("strand_count",):
        if col not in crops:
            sys.exit(f"crops.csv has no '{col}' column. Columns: {list(crops.columns)}")
    n_all = len(crops)
    crops = crops[crops["has_label"].astype(str).str.lower() == "true"].copy()
    print(f"{len(crops)} labelled crops of {n_all} ({n_all - len(crops)} dropped: no label)")

    df = clean_labels(crops, args.expected)
    bad = df["strand_count"].isna()
    if bad.any():
        print(f"WARNING: dropping {bad.sum()} crops with no numeric strand_count")
        df = df[~bad].copy()
    issues = label_issues(df, args.expected)
    issues.to_csv(build / "label_issues.csv", index=False)
    print(f"Label issues to review: {len(issues)}  -> {build / 'label_issues.csv'}")

    print("\nstrand_count :", df["strand_count"].astype(int).value_counts().sort_index().to_dict())
    print("status       :", df["status"].value_counts().to_dict())
    print("defect_type  :", df["defect_type"].value_counts().to_dict())
    mirrored = defect_mirrors_count(df)
    if mirrored:
        print("NOTE: defect_type only restates strand_count, so a single strand-count "
              "head is enough; DEFECT = count != expected.")

    # ---- splits (frozen once written)
    split_path = build / "splits.json"
    if split_path.exists() and not args.regen_splits:
        fold_of = json.loads(split_path.read_text())
        missing = set(df["packet"]) - set(fold_of)
        if missing:
            sys.exit(f"splits.json has no entry for {len(missing)} packet(s), e.g. {sorted(missing)[:3]}. "
                     "Use --regen-splits only if you accept losing comparability with old runs.")
        print(f"\nUsing existing {split_path} (splits are frozen)")
    else:
        fold_of = assign_folds(df, args.folds, args.seed, mirrored)
        split_path.write_text(json.dumps(fold_of, indent=1, sort_keys=True))
        print(f"\nWrote new {split_path}")
    df["fold"] = df["packet"].map(fold_of)
    df["split"] = np.where(df["fold"] == 0, "test", "cv")
    df["cv_fold"] = np.where(df["fold"] == 0, -1, df["fold"] - 1)

    report = split_report(df, mirrored)
    report.to_csv(build / "split_report.csv")
    print("\nClass counts per fold:\n", report.to_string())
    test_rows = df[df["fold"] == 0]
    for key in report.index:
        if key.startswith("defect=") and report.loc[key].get("fold0 (TEST)", 0) == 0:
            print(f"WARNING: no '{key}' example in the test fold")
        if key.startswith(("defect=", "count=")) and report.loc[key].sum() < 10:
            print(f"NOTE: '{key}' has only {int(report.loc[key].sum())} examples in total")

    # ---- preprocessing
    if not args.skip_images:
        paths = []
        for i, p in enumerate(df["crop_path"], 1):
            paths.append(save_variants(build, p, args.variants))
            if i % 100 == 0 or i == len(df):
                print(f"preprocessed {i}/{len(df)}")
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(paths)], axis=1)
        preview(build, df)
        print(f"Preview: {build / 'preview_preprocess.jpg'}")

    df.to_csv(build / "dataset.csv", index=False)
    print(f"\nWrote {build / 'dataset.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
