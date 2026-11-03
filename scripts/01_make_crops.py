#!/usr/bin/env python
"""Phase 1: cut every packet photo into component crops and attach the labels.

Run from the repo root:
    python scripts/01_make_crops.py --images images --labels labels.csv

Outputs (under --out, default data_build/):
    crops/<packet>_R<r>C<c>.png     fixed-size crops (constant aspect ratio)
    crops.csv                       one row per crop + its label columns
    qc_report.csv                   one row per packet with QC flags
    contact/<packet>_overlay.jpg    photo with the grid boxes + labels drawn
    contact/<packet>_crops.jpg      the crops laid out in their R/C positions

Per-packet fixes go in overrides.json (see --help), then re-run with --only <packet>.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.localize import box_to_pixels, crop_fixed_aspect, grid_components  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
PACKET_CANDIDATES = ["image", "image_name", "image_id", "filename", "file_name", "file",
                     "packet", "packet_id", "packet_name", "name"]
ROW_CANDIDATES = ["row", "r", "row_no", "row_number"]
COL_CANDIDATES = ["col", "column", "c", "col_no", "column_number"]
SHOW_COLS = ["strand_count", "status", "defect_type"]   # shown on the QC sheets if present


# ----------------------------------------------------------------- labels
def find_col(df, candidates, override=None):
    lower = {c.lower().strip(): c for c in df.columns}
    if override:
        if override.lower() not in lower:
            sys.exit(f"Column '{override}' not found. Columns are: {list(df.columns)}")
        return lower[override.lower()]
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def packet_key(value):
    return Path(str(value).strip()).stem.lower()


def load_labels(path, packet_col, row_col, col_col):
    df = pd.read_csv(path)
    pk = find_col(df, PACKET_CANDIDATES, packet_col)
    if pk is None:
        sys.exit("Could not find the column holding the image/packet name in "
                 f"{path}. Columns are: {list(df.columns)}\nRe-run with --packet-col <name>.")
    rc, cc = find_col(df, ROW_CANDIDATES, row_col), find_col(df, COL_CANDIDATES, col_col)
    if rc is None or cc is None:
        pos = find_col(df, ["position", "pos"])
        if pos is None:
            sys.exit(f"Need row/column columns (or a 'position' like R2C1). Columns: {list(df.columns)}")
        parsed = df[pos].astype(str).str.extract(r"R(\d+)\s*C(\d+)", flags=re.I)
        df["_row"], df["_col"] = parsed[0].astype(int), parsed[1].astype(int)
    else:
        df["_row"], df["_col"] = df[rc].astype(int), df[cc].astype(int)
    df["_packet"] = df[pk].map(packet_key)
    return df, pk


# ----------------------------------------------------------------- image utils
def resize_fixed(img, size):
    """The crop already has the target aspect ratio, so this is a plain resize."""
    W, H = size
    arr = np.asarray(img.convert("RGB"))
    interp = cv2.INTER_AREA if arr.shape[1] > W else cv2.INTER_CUBIC
    return Image.fromarray(cv2.resize(arr, (W, H), interpolation=interp))


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def short_label(rec):
    if rec is None:
        return "no label"
    parts = []
    for c in SHOW_COLS:
        if c in rec and str(rec[c]).lower() not in ("nan", "", "none"):
            parts.append(f"{rec[c]}" if c == "strand_count" else str(rec[c]))
    return " | ".join(parts) if parts else "labelled"


def is_defect(rec):
    if rec is None:
        return None
    status = str(rec.get("status", "")).lower()
    dtype = str(rec.get("defect_type", "")).lower()
    return "defect" in status or dtype not in ("", "nan", "none", "good", "ok", "0")


def draw_overlay(img, comps, recs, out_path, max_side=1400):
    scale = min(1.0, max_side / max(img.size))
    over = img.resize((round(img.width * scale), round(img.height * scale)))
    draw = ImageDraw.Draw(over)
    font = load_font(max(14, over.height // 70))
    for comp in comps:
        px = box_to_pixels(comp["bbox"], over.width, over.height)
        rec = recs.get(comp["position"])
        flag = is_defect(rec)
        colour = "#9ca3af" if flag is None else ("#ef4444" if flag else "#10b981")
        draw.rectangle(px, outline=colour, width=3)
        text = f"{comp['position']} {short_label(rec)}"
        tb = draw.textbbox((px[0] + 4, px[1] + 4), text, font=font)
        draw.rectangle(tb, fill=colour)
        draw.text((px[0] + 4, px[1] + 4), text, fill="white", font=font)
    over.save(out_path, quality=88)


def draw_crop_montage(crops, recs, n_rows, n_cols, out_path, cell=(300, 188)):
    font = load_font(15)
    head = 22
    sheet = Image.new("RGB", (n_cols * cell[0], n_rows * (cell[1] + head)), "white")
    draw = ImageDraw.Draw(sheet)
    for (r, c), crop in crops.items():
        x, y = (c - 1) * cell[0], (r - 1) * (cell[1] + head)
        thumb = crop.copy()
        thumb.thumbnail(cell)
        sheet.paste(thumb, (x, y + head))
        pos = f"R{r}C{c}"
        draw.text((x + 4, y + 3), f"{pos} {short_label(recs.get(pos))}", fill="black", font=font)
    sheet.save(out_path, quality=88)


# ----------------------------------------------------------------- per-packet work
def process_packet(path, args, labels_for_packet, overrides, out_dir):
    stem = path.stem
    ov = overrides.get(stem, overrides.get(stem.lower(), {}))
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rotation = ov.get("rotation", args.rotation)
    if rotation:
        img = img.rotate(rotation, expand=True)
    n_cols, n_rows = ov.get("cols", args.cols), ov.get("rows", args.rows)
    comps, info = grid_components(img, n_cols, n_rows, ov.get("overlap", args.overlap),
                                  split=ov.get("split", args.split),
                                  col_overlap=ov.get("col_overlap", args.col_overlap),
                                  split_x=ov.get("split_x"), y_ranges=ov.get("y_ranges"),
                                  robust_y=not args.plain_extent)
    for pos, box in ov.get("boxes", {}).items():          # manual box overrides (0-1000)
        for comp in comps:
            if comp["position"] == pos:
                comp["bbox"] = [float(v) for v in box]

    recs = {f"R{r}C{c}": rec for (r, c), rec in labels_for_packet.items()}
    rows, crops = [], {}
    for comp in comps:
        px = box_to_pixels(comp["bbox"], img.width, img.height)
        raw = crop_fixed_aspect(img, px, args.width / args.height,
                                ov.get("padding", args.padding))
        fixed = resize_fixed(raw, (args.width, args.height))
        crop_name = f"{stem}_{comp['position']}.png"
        fixed.save(out_dir / "crops" / crop_name)
        crops[(comp["row"], comp["col"])] = fixed
        row = {"crop_path": f"crops/{crop_name}", "packet": stem, "row": comp["row"],
               "col": comp["col"], "position": comp["position"],
               "bbox_x1": round(comp["bbox"][0], 2), "bbox_y1": round(comp["bbox"][1], 2),
               "bbox_x2": round(comp["bbox"][2], 2), "bbox_y2": round(comp["bbox"][3], 2),
               "raw_crop_w": raw.width, "raw_crop_h": raw.height,
               "has_label": comp["position"] in recs}
        row.update(recs.get(comp["position"], {}))
        rows.append(row)

    draw_overlay(img, comps, recs, out_dir / "contact" / f"{stem}_overlay.jpg")
    draw_crop_montage(crops, recs, n_rows, n_cols, out_dir / "contact" / f"{stem}_crops.jpg")

    flags = []
    grid_positions = {c["position"] for c in comps}
    unlabelled = sorted(grid_positions - set(recs))
    extra = sorted(set(recs) - grid_positions)
    if not labels_for_packet:
        flags.append("no labels found for this packet")
    if unlabelled and labels_for_packet:
        flags.append(f"crops without labels: {','.join(unlabelled)}")
    if extra:
        flags.append(f"labels outside the grid: {','.join(extra)}")
    if info["belt_coverage"] < 0.03:
        flags.append("low belt coverage (check lighting/rotation)")
    if info["belt_width_frac"] < 0.35:
        flags.append("belt block unusually narrow")
    heights = info["column_heights"]
    if len(heights) > 1 and max(heights) > 0 and min(heights) / max(heights) < 0.8:
        flags.append("columns have very different heights")
    return rows, info, flags


def propose_range(healthy, out, ratio_of, med, tol, stray=60):
    """Fix an outlier column using its healthy neighbour. Trim only the end that sticks
    out beyond the neighbour by more than the normal stagger; if that does not give a
    normal height, fall back to the neighbour's own range. Units: 0-1000 of image height."""
    y0, y1 = out
    if healthy[0] - out[0] > stray:
        y0 = healthy[0]
    if out[1] - healthy[1] > stray:
        y1 = healthy[1]
    if abs(ratio_of(y0, y1) / med - 1) > tol:
        y0, y1 = healthy
    return [round(float(y0), 1), round(float(y1), 1)]


def flag_height_outliers(qc, infos, tol, out_dir):
    """Flag columns whose belt height / belt width ratio differs from the dataset median.

    The ratio is scale-free, so packets photographed closer or further away are not
    flagged. A too-tall column usually means stray yellow (table/background) was counted;
    a too-short one usually means an empty end slot. When exactly one column is an
    outlier and the other is healthy, a y_ranges fix is proposed in
    suggested_overrides.json (review the overlay, then copy into overrides.json)."""
    def column_ratios(info):
        col_w = info["belt_width_frac"] / max(len(info["column_y_auto"]), 1)
        return [((y1 - y0) / 1000) / col_w for y0, y1 in info["column_y_auto"]]

    ratios = [r for i in infos.values() if not i["y_overridden"] for r in column_ratios(i)]
    if len(ratios) < 6:
        return
    med = float(np.median(ratios))
    suggestions = {}
    for row in qc:
        info = infos.get(row["packet"])
        if info is None or info["y_overridden"]:
            continue
        rs = column_ratios(info)
        bad = [c for c, r in enumerate(rs) if abs(r / med - 1) > tol]
        if not bad:
            continue
        txt = ", ".join(f"col{c + 1} {rs[c] / med - 1:+.0%}" for c in bad)
        row["flags"] = "; ".join(f for f in (row["flags"], f"column height outlier ({txt} vs median)") if f)
        if len(bad) == 1 and len(rs) == 2:
            c, h = bad[0], 1 - bad[0]
            col_w = info["belt_width_frac"] / 2
            ratio_of = lambda a, b: ((b - a) / 1000) / col_w  # noqa: E731
            fix = propose_range(info["column_y_auto"][h], info["column_y_auto"][c], ratio_of, med, tol)
            suggestions[row["packet"]] = {"y_ranges": [fix if k == c else None for k in range(2)]}
    if suggestions:
        path = out_dir / "suggested_overrides.json"
        path.write_text(json.dumps(suggestions, indent=1))
        print(f"\n{len(suggestions)} packet(s) have one column with an unusual height; "
              f"proposed y_ranges in {path}")
    manual = [r["packet"] for r in qc if "outlier" in str(r["flags"]) and r["packet"] not in suggestions]
    if manual:
        print(f"{len(manual)} packet(s) need a manual y_ranges (both columns off): {manual}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", default="images", help="folder with the packet photos")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--out", default="data_build")
    ap.add_argument("--overrides", default="overrides.json",
                    help='optional JSON, e.g. {"PKT_001": {"overlap": 0.68, "rotation": 90,\n'
                         '"split_x": 0.5,\n"y_ranges": [null, [120, 760]], "boxes": {"R1C1": [x1, y1, x2, y2]}}}  (box coords 0-1000)')
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--overlap", type=float, default=0.72)
    ap.add_argument("--padding", type=float, default=0.04)
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--split", choices=["mid", "valley"], default="mid",
                    help="how the two columns are separated (mid = equal widths)")
    ap.add_argument("--col-overlap", type=float, default=0.08,
                    help="push inner column edges into the neighbour so folded ends are kept")
    ap.add_argument("--width", type=int, default=576, help="saved crop width")
    ap.add_argument("--height", type=int, default=320, help="saved crop height")
    ap.add_argument("--height-tol", type=float, default=0.15,
                    help="flag columns whose height/width ratio differs from the median by more than this fraction")
    ap.add_argument("--plain-extent", action="store_true",
                    help="use plain percentile extents instead of ignoring stray yellow blobs")
    ap.add_argument("--packet-col", default=None)
    ap.add_argument("--row-col", default=None)
    ap.add_argument("--col-col", default=None)
    ap.add_argument("--only", nargs="*", help="process only these packet names (no extension)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N packets")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "contact").mkdir(parents=True, exist_ok=True)

    overrides = {}
    if Path(args.overrides).exists():
        overrides = json.loads(Path(args.overrides).read_text())
        print(f"Loaded overrides for {len(overrides)} packet(s)")

    labels, packet_col = load_labels(args.labels, args.packet_col, args.row_col, args.col_col)
    skip = set(ROW_CANDIDATES + COL_CANDIDATES + ["position", "pos"])
    label_cols = [c for c in labels.columns
                  if c not in ("_packet", "_row", "_col", packet_col) and c.lower().strip() not in skip]
    by_packet = {}
    for rec in labels.to_dict("records"):
        by_packet.setdefault(rec["_packet"], {})[(rec["_row"], rec["_col"])] = {
            c: rec[c] for c in label_cols}

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT)
    if args.only:
        wanted = {packet_key(s) for s in args.only}
        images = [p for p in images if p.stem.lower() in wanted]
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"No images found in {args.images}")

    image_keys = {p.stem.lower() for p in images}
    print(f"{len(images)} image(s); {len(by_packet)} packet(s) in labels")
    missing_img = sorted(set(by_packet) - image_keys) if not (args.only or args.limit) else []
    if missing_img:
        print(f"WARNING: {len(missing_img)} labelled packet(s) have no image, e.g. {missing_img[:5]}")

    all_rows, qc, infos = [], [], {}
    for i, path in enumerate(images, 1):
        try:
            rows, info, flags = process_packet(path, args, by_packet.get(path.stem.lower(), {}),
                                               overrides, out_dir)
            all_rows += rows
            infos[path.stem] = info
            qc.append({"packet": path.stem, "status": "ok", "n_crops": len(rows),
                       "flags": "; ".join(flags), **info})
        except Exception as exc:
            qc.append({"packet": path.stem, "status": "FAILED", "n_crops": 0,
                       "flags": str(exc)})
        print(f"[{i}/{len(images)}] {path.name}: {qc[-1]['status']}"
              + (f"  ({qc[-1]['flags']})" if qc[-1]["flags"] else ""))

    flag_height_outliers(qc, infos, args.height_tol, out_dir)

    new_crops, new_qc = pd.DataFrame(all_rows), pd.DataFrame(qc)
    if args.only or args.limit:
        # partial re-run: keep the rows of packets that were not re-processed
        done = {p.stem for p in images}
        for name, new in (("crops.csv", new_crops), ("qc_report.csv", new_qc)):
            old_path = out_dir / name
            if old_path.exists():
                old = pd.read_csv(old_path)
                new = pd.concat([old[~old["packet"].isin(done)], new], ignore_index=True)
                new = new.sort_values("packet", kind="stable").reset_index(drop=True)
            if name == "crops.csv":
                new_crops = new
            else:
                new_qc = new
    new_crops.to_csv(out_dir / "crops.csv", index=False)
    new_qc.to_csv(out_dir / "qc_report.csv", index=False)

    qc_df = new_qc.copy()
    qc_df["flags"] = qc_df["flags"].fillna("")
    failed = (qc_df["status"] == "FAILED").sum()
    flagged = ((qc_df["status"] == "ok") & (qc_df["flags"] != "")).sum()
    print(f"\nDone: {len(new_crops)} crops from {len(qc_df) - failed} packet(s) in total. "
          f"{failed} failed, {flagged} flagged.")
    print(f"Open {out_dir}/contact/*_overlay.jpg and *_crops.jpg to check the grid, "
          f"then see {out_dir}/qc_report.csv.")


if __name__ == "__main__":
    main()
