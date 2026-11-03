"""Component localization shared by the dataset scripts and the Streamlit app.

Boxes are normalized 0-1000 [left, top, right, bottom] relative to the full image.
"""
import numpy as np
from PIL import Image


def belt_mask(image, max_side=1000, reject_solid=False, solid_frac=0.045):
    """Boolean mask of yellow/orange belt pixels on a downscaled copy of the photo.

    reject_solid: also drop large SOLID yellow regions (e.g. a wooden table with the same
    hue as the belts). Belts are thin rings, so a morphological opening wider than a belt
    stroke keeps only the solid areas, which are then removed. Off by default so the
    Phase 1 crops stay reproducible."""
    small = image.copy()
    small.thumbnail((max_side, max_side))
    hsv = np.asarray(small.convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= 18) & (h <= 50) & (s >= 70) & (v >= 70)
    if reject_solid:
        import cv2
        k = max(5, int(solid_frac * max(mask.shape)))
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k // 3),) * 2)
        filled = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kc)   # bridge wood grain
        solid = cv2.morphologyEx(filled, cv2.MORPH_OPEN, ker)
        solid = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_RECT, (k // 2 * 2 + 1,) * 2))
        mask = mask & ~solid.astype(bool)
    return mask


def solid_background_present(image, thresh=0.55):
    """True when most 'belt-coloured' pixels are really a solid yellow background (table).
    Training packets remove <=48% under reject_solid; the wooden-table photo removes 66%."""
    a, b = belt_mask(image), belt_mask(image, reject_solid=True)
    return 1 - b.sum() / max(a.sum(), 1) > thresh


def _extent(profile, lo=0.004, hi=0.996):
    total = profile.sum()
    if total <= 0:
        return 0, len(profile) - 1
    cum = np.cumsum(profile) / total
    return int(np.searchsorted(cum, lo)), int(np.searchsorted(cum, hi))


def _main_extent(profile, smooth_frac=0.015, thresh=0.05, gap_frac=0.03):
    """Vertical extent of the dominant block of belt pixels.

    Stray yellow (table, background objects) that is separated from the loops by a
    gap is ignored: contiguous segments of the smoothed row profile are found (gaps
    below `gap_frac` of the height are bridged), the one holding most belt pixels
    wins, and the usual percentile trimming is applied inside it."""
    n = len(profile)
    if profile.sum() <= 0:
        return 0, n - 1
    k = max(3, int(n * smooth_frac))
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    idx = np.flatnonzero(smooth >= thresh * smooth.max())
    if len(idx) == 0:
        return _extent(profile)
    gap = max(1, int(n * gap_frac))
    segments, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            segments.append((start, prev))
            start = i
        prev = i
    segments.append((start, prev))
    s0, s1 = max(segments, key=lambda seg: profile[seg[0]:seg[1] + 1].sum())
    lo, hi = _extent(profile[s0:s1 + 1])
    return int(s0 + lo), int(s0 + hi)


def grid_components(image, n_cols, n_rows, overlap, split="mid", col_overlap=0.08,
                    split_x=None, trim=0.01, y_ranges=None, robust_y=True, reject_solid=False):
    """Locate the belt block with a colour mask, then lay a regular grid over it.

    split: "mid" splits the belt block into equal-width columns (robust: the two
    staggered columns are about the same width); "valley" searches for the weakest
    point of the column profile (old behaviour, unreliable because folded ends of
    one column sit right next to the other).
    col_overlap: each inner column edge is pushed this fraction of a column width
    into its neighbour so the folded ends of staggered loops are not cut off.
    split_x: explicit split position for 2 columns, 0-1 of image width (overrides).
    y_ranges: optional per-column vertical extent [[top, bottom], None, ...] in 0-1000
    image-height units; None = detect from the colour mask (robust_y=True ignores stray
    yellow blobs separated from the loops by a gap). Use it when stray yellow
    (table, background) or an empty end slot makes the detected extent wrong.
    trim: fraction of belt pixels ignored at each side when finding the block edges.

    Returns (components, info). `info` holds diagnostics used for QC flags.
    Raises ValueError when the belts cannot be found.
    """
    mask = belt_mask(image, reject_solid=reject_solid)
    mh, mw = mask.shape
    coverage = float(mask.mean())
    if coverage < 0.01:
        raise ValueError("could not find yellow belts (check lighting/rotation)")
    col_profile = mask.sum(0).astype(float)
    x0, x1 = _extent(col_profile, trim, 1 - trim)
    if x1 - x0 < 50:
        raise ValueError("belt region too narrow to build a grid")
    edges = [x0 + (x1 - x0) * i / n_cols for i in range(n_cols + 1)]
    if n_cols == 2 and split_x is not None:
        edges[1] = float(split_x) * mw
    elif n_cols == 2 and split == "valley":
        smooth = np.convolve(col_profile, np.ones(15) / 15, mode="same")
        a, b = int(x0 + (x1 - x0) * 0.44), int(x0 + (x1 - x0) * 0.56)
        edges[1] = a + int(np.argmin(smooth[a:b]))
    base_edges = list(edges)
    components = []
    col_heights, auto_y = [], []
    for c in range(n_cols):
        cx0, cx1 = int(base_edges[c]), int(base_edges[c + 1])
        width = cx1 - cx0
        if c > 0:
            cx0 = max(0, int(cx0 - col_overlap * width))
        if c < n_cols - 1:
            cx1 = min(mw, int(cx1 + col_overlap * width))
        row_profile = mask[:, int(base_edges[c]):int(base_edges[c + 1])].sum(1).astype(float)
        y0, y1 = (_main_extent if robust_y and not reject_solid else _extent)(row_profile)
        auto_y.append([round(1000 * y0 / mh, 1), round(1000 * y1 / mh, 1)])
        if y_ranges and c < len(y_ranges) and y_ranges[c]:
            y0, y1 = int(y_ranges[c][0] * mh / 1000), int(y_ranges[c][1] * mh / 1000)
        col_heights.append((y1 - y0) / mh)
        height_total = max(1, y1 - y0)
        h = height_total / (1 + (n_rows - 1) * overlap)
        pitch = h * overlap
        for r in range(n_rows):
            top = y0 + r * pitch
            bbox = [1000 * cx0 / mw, 1000 * top / mh, 1000 * cx1 / mw, 1000 * (top + h) / mh]
            bbox = [max(0.0, min(1000.0, v)) for v in bbox]
            components.append({
                "bbox": bbox, "row": r + 1, "col": c + 1, "position": f"R{r + 1}C{c + 1}",
            })
    info = {
        "belt_coverage": round(coverage, 4),
        "belt_width_frac": round((x1 - x0) / mw, 3),
        "column_heights": [round(v, 3) for v in col_heights],
        "split_x": [round(e / mw, 3) for e in base_edges[1:-1]],
        "column_y_auto": auto_y,
        "y_overridden": bool(y_ranges and any(y_ranges)),
    }
    return components, info


def box_to_pixels(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    return (
        max(0, min(width, round(x1 * width / 1000))),
        max(0, min(height, round(y1 * height / 1000))),
        max(0, min(width, round(x2 * width / 1000))),
        max(0, min(height, round(y2 * height / 1000))),
    )


def crop_with_padding(image, px_box, padding=0.06):
    x1, y1, x2, y2 = px_box
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        raise ValueError("component box is too small")
    dx, dy = max(3, round(bw * padding)), max(3, round(bh * padding))
    crop = image.crop((max(0, x1 - dx), max(0, y1 - dy),
                       min(image.width, x2 + dx), min(image.height, y2 + dy)))
    if crop.width < 20 or crop.height < 20:
        raise ValueError("component crop is too small")
    return crop


def crop_fixed_aspect(image, px_box, aspect, padding=0.04):
    """Crop a window of constant width/height ratio centred on the box.

    The window grows (never shrinks) to contain the padded box, and is shifted
    to stay inside the image, so every crop has the same shape with real pixels
    (no replicated borders)."""
    x1, y1, x2, y2 = px_box
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("component box is too small")
    bw, bh = (x2 - x1) * (1 + 2 * padding), (y2 - y1) * (1 + 2 * padding)
    w = max(bw, bh * aspect)
    h = w / aspect
    if w > image.width:
        w, h = image.width, image.width / aspect
    if h > image.height:
        h, w = image.height, image.height * aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - w / 2, 0), image.width - w)
    top = min(max(cy - h / 2, 0), image.height - h)
    crop = image.crop((round(left), round(top), round(left + w), round(top + h)))
    if crop.width < 20 or crop.height < 20:
        raise ValueError("component crop is too small")
    return crop
