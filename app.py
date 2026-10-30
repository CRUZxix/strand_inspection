import base64
import hashlib
import io
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

st.set_page_config(page_title="Packet Strand Inspection", page_icon="🔎", layout="wide")

st.html("""
<style>
.matrix-container{width:100%;margin:16px 0 24px}
.matrix-grid{display:grid;gap:12px;width:100%}
.comp-card{border-radius:10px;padding:12px 8px;text-align:center;display:flex;
flex-direction:column;justify-content:space-between;align-items:center;min-height:112px;
box-sizing:border-box;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.card-id{font-size:13px;font-weight:700;letter-spacing:.4px}
.card-count{font-size:34px;font-weight:800;line-height:1.1;margin:5px 0}
.card-status{font-size:12px;font-weight:700;letter-spacing:.5px}
.card-votes{font-size:11px;opacity:.75}
.card-good{background:#e8f5e9;color:#1b5e20;border:1.5px solid #a5d6a7}
.card-defect{background:#ffebee;color:#b71c1c;border:1.5px solid #ef9a9a}
.card-uncertain{background:#fffde7;color:#8a5a00;border:1.5px solid #ffe082}
</style>
""")

st.title("🔎 Packet Strand Inspection")
st.caption("AI-assisted component localization and strand-count inspection")

try:
    API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    API_KEY = ""
if not API_KEY:
    st.error("Configure OPENAI_API_KEY in .streamlit/secrets.toml.")
    st.stop()
client = OpenAI(api_key=API_KEY)

DEFAULT_HINT = (
    "Each component is a loop made of several flat yellow belts nested together. "
    "From above, the bundle looks like one large outer loop with further curved yellow "
    "edges nested inside it, converging towards the folded end. Every distinct nested "
    "belt layer is one strand; the outermost enclosing loop counts as one strand. "
    "The photo is taken through a transparent plastic bag, so wrinkles and glare are NOT strands."
)
DEFECT_TYPES = {"broken_strand", "tangled", "missing_strand", "foreign_material", "other"}


# -------------------- IMAGE HELPERS --------------------
def image_to_data_url(image, max_side=2400, quality=95):
    img = image.copy().convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def enhance_for_vision(image, gamma=0.65, min_side=1100):
    """Brighten dark photos, stretch contrast, sharpen thin edges and upscale small crops."""
    img = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    if gamma < 1.0:
        arr = np.asarray(img).astype(np.float32) / 255.0
        img = Image.fromarray((np.power(arr, gamma) * 255).astype(np.uint8))
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    scale = min_side / max(img.size)
    if scale > 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.Resampling.LANCZOS)
    return img


def parse_json_response(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The vision model returned an empty response.")
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder, data = json.JSONDecoder(), None
        for match in re.finditer(r"\{", cleaned):
            try:
                data, _ = decoder.raw_decode(cleaned[match.start():])
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            raise ValueError("Model response did not contain valid JSON.")
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return data


def ask_vision(model_name, content, effort="default"):
    """content = list of Responses-API content parts (text + images)."""
    kwargs = {"model": model_name, "input": [{"role": "user", "content": content}]}
    if effort != "default":
        kwargs["reasoning"] = {"effort": effort}
    try:
        response = client.responses.create(**kwargs)
    except Exception as exc:
        # Some models do not accept the reasoning parameter; retry without it.
        if "reasoning" in kwargs and "reasoning" in str(exc).lower():
            kwargs.pop("reasoning")
            response = client.responses.create(**kwargs)
        else:
            raise
    return parse_json_response(response.output_text)


def text_part(text):
    return {"type": "input_text", "text": text}


def image_part(image, max_side=2400):
    return {"type": "input_image", "image_url": image_to_data_url(image, max_side),
            "detail": "high"}


# -------------------- GEOMETRY HELPERS --------------------
def valid_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return False
    return all(0 <= v <= 1000 for v in (x1, y1, x2, y2)) and x1 < x2 and y1 < y2


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
        raise ValueError("Detected component box is too small.")
    dx, dy = max(3, round(bw * padding)), max(3, round(bh * padding))
    crop = image.crop((max(0, x1 - dx), max(0, y1 - dy),
                       min(image.width, x2 + dx), min(image.height, y2 + dy)))
    if crop.width < 20 or crop.height < 20:
        raise ValueError("Component crop is too small for reliable inspection.")
    return crop


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union else 0.0


# -------------------- LOCALIZATION: COLOUR-MASK GRID --------------------
def belt_mask(image):
    """Boolean mask of yellow/orange belt pixels on a downscaled copy of the photo."""
    small = image.copy()
    small.thumbnail((1000, 1000))
    hsv = np.asarray(small.convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 18) & (h <= 50) & (s >= 70) & (v >= 70)


def _extent(profile, lo=0.004, hi=0.996):
    total = profile.sum()
    if total <= 0:
        return 0, len(profile) - 1
    cum = np.cumsum(profile) / total
    return int(np.searchsorted(cum, lo)), int(np.searchsorted(cum, hi))


def grid_components(image, n_cols, n_rows, overlap):
    """
    Locate the belt block with a colour mask, then lay a regular grid over it.
    Columns are split at the weakest point of the column profile near the middle;
    each column is divided vertically into n_rows loops that overlap by `overlap`
    (pitch / loop height), matching the staggered, overlapping layout of the packet.
    """
    mask = belt_mask(image)
    mh, mw = mask.shape
    if mask.mean() < 0.01:
        raise ValueError("Could not find the yellow belts. Check lighting or rotation.")
    col_profile = mask.sum(0).astype(float)
    x0, x1 = _extent(col_profile)
    if x1 - x0 < 50:
        raise ValueError("Belt region is too narrow to build a grid.")
    edges = [x0 + (x1 - x0) * i / n_cols for i in range(n_cols + 1)]
    if n_cols == 2:
        smooth = np.convolve(col_profile, np.ones(15) / 15, mode="same")
        a, b = int(x0 + (x1 - x0) * 0.44), int(x0 + (x1 - x0) * 0.56)
        edges[1] = a + int(np.argmin(smooth[a:b]))
    components = []
    for c in range(n_cols):
        cx0, cx1 = int(edges[c]), int(edges[c + 1])
        y0, y1 = _extent(mask[:, cx0:cx1].sum(1).astype(float))
        height_total = max(1, y1 - y0)
        h = height_total / (1 + (n_rows - 1) * overlap)
        pitch = h * overlap
        for r in range(n_rows):
            top = y0 + r * pitch
            bbox = [1000 * cx0 / mw, 1000 * top / mh, 1000 * cx1 / mw, 1000 * (top + h) / mh]
            bbox = [max(0.0, min(1000.0, v)) for v in bbox]
            components.append({
                "bbox": bbox, "detection_confidence": "high",
                "detection_note": "grid", "possible_duplicate": False,
                "row": r + 1, "col": c + 1, "position": f"R{r + 1}C{c + 1}",
            })
    return components, n_rows, n_cols, []


# -------------------- LOCALIZATION: LLM BOXES (OPTIONAL) --------------------
def detect_components(model_name, image, expected_count, n_cols, effort):
    prompt = f"""
Locate every separate loop-shaped belt bundle in this packet photograph.
There are approximately {expected_count} components arranged in about {n_cols} columns.
Neighbouring loops partly overlap, so box each loop's own full outline.
Do not invent objects to reach that number. Return ONLY valid JSON:
{{"components":[{{"bbox":[x1,y1,x2,y2],"confidence":"high","note":""}}],"uncertainty_note":""}}
Coordinates are normalized 0-1000 relative to the FULL image, [0,0] top-left,
[1000,1000] bottom-right, bbox order [left,top,right,bottom].
Do not box the plastic bag, table, shadows or reflections. Do not return a packet box.
Confidence must be high, medium or low.
"""
    data = ask_vision(model_name, [text_part(prompt), image_part(image)], effort)
    raw = data.get("components", [])
    if not isinstance(raw, list):
        raise ValueError("'components' in model response must be a list.")
    components, rejected = [], 0
    for item in raw:
        if not isinstance(item, dict) or not valid_box(item.get("bbox")):
            rejected += 1
            continue
        box = list(map(float, item["bbox"]))
        if box[2] - box[0] < 15 or box[3] - box[1] < 15:
            rejected += 1
            continue
        conf = str(item.get("confidence", "low")).strip().lower()
        components.append({
            "bbox": box,
            "detection_confidence": conf if conf in {"high", "medium", "low"} else "low",
            "detection_note": str(item.get("note", ""))[:200],
            "possible_duplicate": False,
        })
    for i, comp in enumerate(components):
        for previous in components[:i]:
            if box_iou(comp["bbox"], previous["bbox"]) > 0.65:
                comp["possible_duplicate"] = True
    return {"components": components, "notes": str(data.get("uncertainty_note", "")),
            "rejected_boxes": rejected}


def assign_positions(components, n_cols):
    """Column-first assignment (1-D k-means on x), then top-to-bottom within a column.
    Staggered columns break row-first clustering, so the known column count is used."""
    if not components:
        return [], 0, 0, []
    n_cols = max(1, min(n_cols, len(components)))
    cx = np.array([(c["bbox"][0] + c["bbox"][2]) / 2 for c in components])
    centers = np.linspace(cx.min(), cx.max(), n_cols) if n_cols > 1 else np.array([cx.mean()])
    labels = np.zeros(len(cx), dtype=int)
    for _ in range(20):
        labels = np.abs(cx[:, None] - centers[None, :]).argmin(1)
        new = np.array([cx[labels == k].mean() if (labels == k).any() else centers[k]
                        for k in range(n_cols)])
        if np.allclose(new, centers):
            break
        centers = new
    ordered, sizes = [], []
    for col_no, k in enumerate(np.argsort(centers), start=1):
        members = sorted((components[i] for i in np.where(labels == k)[0]),
                         key=lambda c: c["bbox"][1] + c["bbox"][3])
        sizes.append(len(members))
        for r, comp in enumerate(members, start=1):
            comp = dict(comp)
            comp.update({"row": r, "col": col_no, "position": f"R{r}C{col_no}"})
            ordered.append(comp)
    warnings = []
    if len(set(sizes)) > 1:
        warnings.append(f"Columns contain different component counts: {sizes}. Verify the boxes.")
    return ordered, max(sizes), n_cols, warnings


# -------------------- COUNTING --------------------
def normalize_count(value):
    if isinstance(value, bool):
        return "uncertain"
    if isinstance(value, int) and 0 <= value <= 5:
        return value
    if isinstance(value, float) and value.is_integer() and 0 <= value <= 5:
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "uncertain":
            return value
        if value.isdigit() and 0 <= int(value) <= 5:
            return int(value)
    return "uncertain"


def normalize_confidence(value):
    value = str(value or "low").strip().lower()
    return value if value in {"high", "medium", "low"} else "low"


COUNT_PROMPT = """
You are inspecting ONE component from a photograph of a packet, identified as {position}.

What a component looks like: {hint}

The crop was cut from a regular grid, so slivers of neighbouring components can appear
at its top, bottom or side edges. Inspect ONLY the component whose centre is closest to
the centre of the image. Ignore everything else.

Procedure:
1. Scan the whole component, esgitpecially where the nested layers converge at the fold. 
2. List every distinct strand from outermost to innermost, one short entry each, saying
   where you see it. Do not list the same curve twice.
3. Do not count bag wrinkles, glare, reflections, shadows, or neighbouring components.
4. If layers are hidden or cannot be separated, use "uncertain".
Valid counts are 0,1,2,3,4,5 or "uncertain". Do NOT assume any expected count.

Return ONLY valid JSON:
{{"strands":["outer loop, ...","inner arc 1, ..."],"strand_count":2,"confidence":"medium",
"visible_defects":[],"uncertainty_reason":""}}
strand_count must equal the number of entries in "strands". confidence is high, medium or low.
visible_defects: only values from broken_strand, tangled, missing_strand, foreign_material,
other, and only if clearly visible in this crop.
"""


def count_vote(model_name, crop, position, hint, refs, effort):
    """One independent counting attempt for one component crop."""
    content = [text_part(COUNT_PROMPT.format(position=position, hint=hint.strip()))]
    for ref_img, ref_count in refs:
        content.append(text_part(
            f"REFERENCE EXAMPLE: the component centred in this image has exactly {ref_count} strands."))
        content.append(image_part(ref_img, 1600))
    content.append(text_part("TARGET COMPONENT TO INSPECT:" if refs else "Component to inspect:"))
    content.append(image_part(crop, 1600))
    try:
        data = ask_vision(model_name, content, effort)
    except Exception as exc:
        return {"strand_count": "uncertain", "confidence": "low", "defects": [],
                "reason": f"API/parse error: {exc}"[:300], "error": True}
    declared = normalize_count(data.get("strand_count"))
    strands = data.get("strands")
    listed = len(strands) if isinstance(strands, list) and len(strands) <= 5 else None
    count = declared
    if listed is not None and declared != "uncertain" and listed != declared:
        count = "uncertain"            # the evidence list and the answer disagree
    defects = data.get("visible_defects", [])
    if not isinstance(defects, list):
        defects = []
    defects = [str(v).strip().lower() for v in defects if str(v).strip().lower() in DEFECT_TYPES]
    return {"strand_count": count, "confidence": normalize_confidence(data.get("confidence")),
            "defects": defects, "reason": str(data.get("uncertainty_reason", ""))[:300],
            "error": False}


def aggregate_votes(votes):
    """Majority vote; agreement drives confidence."""
    n = len(votes)
    numeric = [v["strand_count"] for v in votes if v["strand_count"] != "uncertain"]
    reasons = [v["reason"] for v in votes if v["reason"]]
    reason = reasons[0] if reasons else ""
    if not numeric:
        return {"strand_count": "uncertain", "confidence": "low", "defects": [], "reason": reason,
                "votes": ",".join(str(v["strand_count"])[0] for v in votes)}
    tally = Counter(numeric).most_common()
    top, top_n = tally[0]
    tied = len(tally) > 1 and tally[1][1] == top_n
    agreement = top_n / n
    if tied or agreement < 0.5:
        count, conf = "uncertain", "low"
    elif n == 1:
        count, conf = top, votes[0]["confidence"]
    elif agreement == 1.0 and all(v["confidence"] != "low" for v in votes):
        count, conf = top, "high"
    else:
        count, conf = top, "medium"
    defect_votes = Counter(d for v in votes for d in set(v["defects"]))
    defects = sorted(d for d, k in defect_votes.items() if k >= max(1, (n + 1) // 2))
    return {"strand_count": count, "confidence": conf, "defects": defects, "reason": reason,
            "votes": ",".join(str(v["strand_count"])[0] if v["strand_count"] == "uncertain"
                              else str(v["strand_count"]) for v in votes)}


def classify_result(count, confidence, detection_confidence, defects, expected):
    if count == "uncertain" or confidence == "low" or detection_confidence == "low":
        return "UNCERTAIN"
    if defects or count != expected:
        return "DEFECT"
    return "GOOD"


# -------------------- DRAWING --------------------
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


def draw_annotated_image(source_image, components, results=None):
    overlay = source_image.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    font = load_font(max(18, min(44, int(h * 0.016))))
    line_width = max(3, min(10, int(min(w, h) * 0.003)))
    colors = {
        "GOOD": ("#10b981", "#065f46"), "DEFECT": ("#ef4444", "#991b1b"),
        "UNCERTAIN": ("#f59e0b", "#92400e"), "DETECTED": ("#06b6d4", "#0e7490"),
    }
    for comp in components:
        px = box_to_pixels(comp["bbox"], w, h)
        pos = comp["position"]
        result = (results or {}).get(pos, {})
        outline, badge = colors.get(result.get("status", "DETECTED"), colors["DETECTED"])
        draw.rectangle(px, outline=outline, width=line_width)
        label = pos if not result else f"{pos}: {result.get('strand_count', '?')}"
        if comp.get("possible_duplicate"):
            label += " !"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        bx = max(0, min(w - tw - 12, px[0]))
        by = max(0, px[1] - th - 12)
        draw.rounded_rectangle((bx, by, min(w, bx + tw + 12), min(h, by + th + 10)),
                               radius=4, fill=badge)
        draw.text((bx + 6, by + 3), label, fill="white", font=font)
    return overlay


def render_matrix(components, results, n_cols):
    if not components or not results or n_cols < 1:
        return
    cards = []
    for comp in sorted(components, key=lambda c: (c["row"], c["col"])):
        pos = comp["position"]
        result = results.get(pos, {})
        status = result.get("status", "UNCERTAIN")
        css = {"GOOD": "card-good", "DEFECT": "card-defect"}.get(status, "card-uncertain")
        count = result.get("strand_count", "—")
        votes = result.get("votes", "")
        sub = f'<div class="card-votes">votes: {votes}</div>' if votes else ""
        cards.append(
            f'<div class="comp-card {css}"><div class="card-id">{pos}</div>'
            f'<div class="card-count">{count}</div><div class="card-status">{status}</div>{sub}</div>'
        )
    st.html('<div class="matrix-container"><div class="matrix-grid" '
            f'style="grid-template-columns:repeat({n_cols}, minmax(110px, 1fr));">'
            + "".join(cards) + "</div></div>")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Inspection Configuration")
    model_name = st.text_input("Vision model", value="gpt-5.4-mini",
                               help="Use the strongest vision model your account allows for counting.")
    effort = st.selectbox("Reasoning effort", ["default", "low", "medium", "high"], index=2,
                          help="Higher effort helps counting. Ignored if the model does not support it.")
    st.subheader("Packet layout")
    expected_strands = st.number_input("Expected strand count", 0, 5, 4, 1)
    n_cols_cfg = st.number_input("Columns", 1, 6, 2, 1)
    n_rows_cfg = st.number_input("Components per column", 1, 15, 5, 1)
    expected_components = int(n_cols_cfg) * int(n_rows_cfg)
    loc_mode = st.radio("Localization", ["Colour-mask grid (recommended)", "LLM boxes"])
    overlap = st.slider("Vertical overlap factor", 0.50, 1.00, 0.72, 0.01,
                        help="Loop pitch ÷ loop height. Lower = taller boxes that overlap more.")
    st.subheader("Counting")
    n_votes = st.slider("Votes per component", 1, 5, 3,
                        help="Each component is counted several times; disagreement → UNCERTAIN.")
    workers = st.slider("Parallel requests", 1, 10, 5)
    crop_padding = st.slider("Crop padding (%)", 0, 15, 4)
    gamma = st.slider("Brighten dark photos (gamma)", 0.40, 1.00, 0.65, 0.05,
                      help="Lower = brighter. 1.0 disables.")
    hint = st.text_area("Component description for the model", value=DEFAULT_HINT, height=170,
                        help="Describe what one strand looks like. Edit this to match your product.")
    with st.expander("Reference examples (optional, strongly recommended)"):
        st.caption("Upload cropped examples with known strand counts, e.g. one GOOD and one DEFECT.")
        ref_files = st.file_uploader("Reference crops", type=["jpg", "jpeg", "png", "webp"],
                                    accept_multiple_files=True, key="refs")
        ref_counts = [st.number_input(f"Strands in {f.name}", 0, 5, int(expected_strands), 1,
                                      key=f"refcount_{f.name}") for f in ref_files[:4]]
    rotation = st.selectbox("Additional image rotation", [0, 90, 180, 270],
                            format_func=lambda x: f"{x}°")
    show_debug = st.checkbox("Show diagnostic details", False)

uploaded = st.file_uploader("Upload packet photograph", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Upload a packet image to begin.")
    st.stop()

raw_bytes = uploaded.getvalue()
image_hash = hashlib.sha256(raw_bytes + str(rotation).encode()).hexdigest()
try:
    source_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    if rotation:
        source_img = source_img.rotate(rotation, expand=True)
except Exception as exc:
    st.error(f"Could not load image: {exc}")
    st.stop()

STATE_KEYS = ("components", "inspection_results", "num_rows", "num_cols", "detection_notes",
              "layout_warnings", "debug_info", "crops")
if st.session_state.get("active_image_hash") != image_hash:
    for key in STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["active_image_hash"] = image_hash

st.caption(f"{uploaded.name} · {source_img.width} × {source_img.height} px")
col_a, col_b = st.columns(2)
with col_a:
    run_full = st.button("🚀 Run Full Inspection", type="primary", use_container_width=True)
with col_b:
    run_detection = st.button("🔍 Detect Components Only", use_container_width=True)

if run_full or run_detection:
    for key in STATE_KEYS:
        st.session_state.pop(key, None)
    try:
        with st.spinner("Step 1/2: Locating components..."):
            notes, rejected = "", 0
            if loc_mode.startswith("Colour"):
                ordered, n_rows, n_cols, layout_warnings = grid_components(
                    source_img, int(n_cols_cfg), int(n_rows_cfg), float(overlap))
            else:
                det = detect_components(model_name, source_img, expected_components,
                                        int(n_cols_cfg), effort)
                if not det["components"]:
                    st.error("No valid component boxes were returned.")
                    st.stop()
                ordered, n_rows, n_cols, layout_warnings = assign_positions(
                    det["components"], int(n_cols_cfg))
                notes, rejected = det["notes"], det["rejected_boxes"]
            st.session_state.update({
                "components": ordered, "num_rows": n_rows, "num_cols": n_cols,
                "detection_notes": notes, "layout_warnings": layout_warnings,
                "debug_info": {"image_dimensions": source_img.size, "detected_count": len(ordered),
                               "rejected_boxes": rejected, "localization": loc_mode},
            })
    except Exception as exc:
        st.error(f"Component localization failed: {exc}")
        st.stop()

    if run_full:
        # Reference examples (few-shot)
        refs = []
        for f, cnt in zip(ref_files[:4], ref_counts):
            try:
                refs.append((enhance_for_vision(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                                                gamma, 900), int(cnt)))
            except Exception:
                pass

        # Prepare crops in the main thread
        crops, results, tasks = {}, {}, []
        for comp in ordered:
            pos = comp["position"]
            try:
                px = box_to_pixels(comp["bbox"], source_img.width, source_img.height)
                crops[pos] = enhance_for_vision(
                    crop_with_padding(source_img, px, crop_padding / 100.0), gamma)
                tasks += [(comp, pos) for _ in range(n_votes)]
            except Exception as exc:
                results[pos] = {
                    "position": pos, "row": comp["row"], "col": comp["col"],
                    "strand_count": "uncertain", "expected_count": int(expected_strands),
                    "status": "UNCERTAIN", "defects": [], "confidence": "low", "votes": "",
                    "detection_confidence": comp["detection_confidence"],
                    "reason": f"Crop error: {exc}"}
        st.session_state["crops"] = crops

        # Run all votes in parallel (no Streamlit calls inside worker threads)
        votes_by_pos = {pos: [] for pos in crops}
        progress = st.progress(0.0)
        progress_text = st.empty()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(count_vote, model_name, crops[pos], pos, hint, refs, effort): pos
                       for _, pos in tasks}
            for fut in as_completed(futures):
                votes_by_pos[futures[fut]].append(fut.result())
                done += 1
                progress.progress(done / max(1, len(futures)))
                progress_text.write(f"Completed {done} of {len(futures)} counting passes")
        progress.empty()
        progress_text.empty()

        for comp in ordered:
            pos = comp["position"]
            if pos in results:
                continue
            agg = aggregate_votes(votes_by_pos[pos])
            status = classify_result(agg["strand_count"], agg["confidence"],
                                     comp["detection_confidence"], agg["defects"],
                                     int(expected_strands))
            results[pos] = {
                "position": pos, "row": comp["row"], "col": comp["col"],
                "strand_count": agg["strand_count"], "expected_count": int(expected_strands),
                "status": status, "defects": agg["defects"], "confidence": agg["confidence"],
                "votes": agg["votes"], "detection_confidence": comp["detection_confidence"],
                "reason": agg["reason"], "crop_dimensions": crops[pos].size}
        st.session_state["inspection_results"] = results

components = st.session_state.get("components")
results = st.session_state.get("inspection_results")
n_rows = st.session_state.get("num_rows", 0)
n_cols = st.session_state.get("num_cols", 0)

if components:
    st.subheader("1. Annotated Packet Image")
    st.image(draw_annotated_image(source_img, components, results), use_container_width=True)

    if len(components) != expected_components:
        st.warning(f"Found {len(components)} components; expected {expected_components}. "
                   "Review the boxes before trusting the result.")
    duplicates = sum(bool(c.get("possible_duplicate")) for c in components)
    if duplicates:
        st.warning(f"{duplicates} possible duplicate/overlapping boxes flagged for review.")
    for warning in st.session_state.get("layout_warnings", []):
        st.warning(warning)
    if st.session_state.get("detection_notes"):
        st.info(st.session_state["detection_notes"])

    if results:
        st.subheader("2. Final Inspection Matrix")
        statuses = [v["status"] for v in results.values()]
        if "DEFECT" in statuses:
            st.error("❌ Overall result: DEFECT DETECTED")
        elif "UNCERTAIN" in statuses:
            st.warning("⚠️ Overall result: REVIEW REQUIRED")
        else:
            st.success("✅ Overall result: ALL COMPONENTS GOOD")
        render_matrix(components, results, n_cols)

        df = pd.DataFrame(list(results.values()))
        report = [
            "# Packet Strand Inspection Report",
            f"- Filename: {uploaded.name}",
            f"- Image dimensions: {source_img.width} × {source_img.height}",
            f"- Expected strand count: {expected_strands}",
            f"- Detected components: {len(components)}", "",
            "| Position | Count | Votes | Expected | Status | Defects | Confidence |",
            "|---|---:|---|---:|---|---|---|",
        ]
        for r in sorted(results.values(), key=lambda r: (r["row"], r["col"])):
            report.append(f"| {r['position']} | {r['strand_count']} | {r.get('votes', '')} | "
                          f"{r['expected_count']} | {r['status']} | "
                          f"{', '.join(r['defects']) or 'none'} | {r['confidence']} |")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download Results CSV", df.to_csv(index=False).encode("utf-8"),
                               f"inspection_{uploaded.name}.csv", "text/csv",
                               use_container_width=True)
        with c2:
            st.download_button("📥 Download Markdown Report", "\n".join(report),
                               f"inspection_{uploaded.name}.md", "text/markdown",
                               use_container_width=True)
    else:
        st.info("Components located. Click Run Full Inspection to count strands.")

    if show_debug:
        st.subheader("Diagnostic details")
        st.json(st.session_state.get("debug_info", {}))
        with st.expander("Detected coordinates"):
            st.dataframe(pd.DataFrame([{
                "position": c["position"], "bbox_normalized": c["bbox"],
                "detection_confidence": c["detection_confidence"],
                "possible_duplicate": c["possible_duplicate"],
            } for c in components]), use_container_width=True)
        crops_dbg = st.session_state.get("crops")
        if crops_dbg:
            with st.expander("Crops sent to the model"):
                cols = st.columns(n_cols or 1)
                for comp in sorted(components, key=lambda c: (c["col"], c["row"])):
                    if comp["position"] in crops_dbg:
                        cols[(comp["col"] - 1) % len(cols)].image(
                            crops_dbg[comp["position"]], caption=comp["position"])
        if results:
            with st.expander("Counting diagnostics"):
                st.dataframe(pd.DataFrame(list(results.values())), use_container_width=True)