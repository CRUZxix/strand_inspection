import base64
import hashlib
import io
import json
import re
from statistics import median

import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps

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


def image_to_data_url(image, max_side=2400, quality=96):
    img = image.copy().convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


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


def ask_vision(model_name, prompt, image):
    response = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
        ]}],
    )
    return parse_json_response(response.output_text)


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
    crop = image.crop((max(0, x1-dx), max(0, y1-dy),
                       min(image.width, x2+dx), min(image.height, y2+dy)))
    if crop.width < 20 or crop.height < 20:
        raise ValueError("Component crop is too small for reliable inspection.")
    return crop


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


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def detect_components(model_name, image, expected_count):
    prompt = f"""
Locate every separate physical loop-shaped component in this packet photograph.
There are approximately {expected_count} components, but do not invent objects
to reach that number. Return ONLY valid JSON:
{{
 "components":[{{"bbox":[x1,y1,x2,y2],"confidence":"high","note":""}}],
 "uncertainty_note":""
}}
Coordinates are normalized 0-1000 relative to the FULL image, with [0,0] top-left
and [1000,1000] bottom-right. bbox order is [left,top,right,bottom].
Each box must tightly enclose exactly one complete visible component. Do not box
packet edges, empty areas, shadows or reflections. Do not merge neighboring parts.
Do not cut off visible parts. Confidence must be high, medium or low.
If two components cannot be distinguished, explain this in uncertainty_note.
Do not return a packet box. Do not create boxes just to satisfy the expected count.
"""
    data = ask_vision(model_name, prompt, image)
    raw = data.get("components", [])
    if not isinstance(raw, list):
        raise ValueError("'components' in model response must be a list.")
    components, rejected = [], 0
    for item in raw:
        if not isinstance(item, dict) or not valid_box(item.get("bbox")):
            rejected += 1
            continue
        box = list(map(float, item["bbox"]))
        if box[2]-box[0] < 15 or box[3]-box[1] < 15:
            rejected += 1
            continue
        components.append({
            "bbox": box,
            "detection_confidence": normalize_confidence(item.get("confidence")),
            "detection_note": str(item.get("note", ""))[:200],
            "possible_duplicate": False,
        })
    # Flag likely duplicate boxes; keep them visible for human review.
    for i, comp in enumerate(components):
        for previous in components[:i]:
            if box_iou(comp["bbox"], previous["bbox"]) > 0.65:
                comp["possible_duplicate"] = True
    return {
        "components": components,
        "notes": str(data.get("uncertainty_note", "")),
        "rejected_boxes": rejected,
    }


def assign_grid_positions(components):
    """Adaptive heuristic row grouping; warnings expose ambiguous layouts."""
    if not components:
        return [], 0, 0, []
    items, heights = [], []
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        heights.append(y2-y1)
        items.append({"comp": comp, "cx": (x1+x2)/2, "cy": (y1+y2)/2})
    tolerance = max(12.0, median(heights) * 0.42)
    items.sort(key=lambda x: x["cy"])
    rows = []
    for item in items:
        if not rows:
            rows.append([item])
            continue
        center = median([entry["cy"] for entry in rows[-1]])
        if abs(item["cy"]-center) <= tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])
    ordered = []
    row_sizes = []
    for r, row in enumerate(rows, start=1):
        row.sort(key=lambda x: x["cx"])
        row_sizes.append(len(row))
        for c, entry in enumerate(row, start=1):
            comp = dict(entry["comp"])
            comp.update({"row": r, "col": c, "position": f"R{r}C{c}"})
            ordered.append(comp)
    warnings = []
    if len(set(row_sizes)) > 1:
        warnings.append(f"Rows contain different component counts: {row_sizes}. Verify row assignment.")
    return ordered, len(rows), max(row_sizes), warnings


def count_single_component(model_name, crop, position):
    prompt = f"""
Inspect ONE isolated component, identified as {position}. Count distinct physical
strands forming this loop. Valid counts are 0,1,2,3,4,5 or "uncertain".
Five is valid. Do not assume the expected count.
Return ONLY valid JSON:
{{
 "strand_count":4,
 "confidence":"medium",
 "visible_defects":[],
 "uncertainty_reason":""
}}
Count physical strands, not shadows, highlights, reflections or edges. Do not
count one curved strand twice. If strands are obscured or cannot be separated,
return uncertain. confidence must be high, medium or low.
visible_defects must be a list containing only applicable values from:
broken_strand, tangled, missing_strand, foreign_material, other.
Only report defects clearly visible in this crop. Do not infer hidden defects.
"""
    data = ask_vision(model_name, prompt, crop)
    count = normalize_count(data.get("strand_count"))
    confidence = normalize_confidence(data.get("confidence"))
    defects = data.get("visible_defects", [])
    allowed = {"broken_strand", "tangled", "missing_strand", "foreign_material", "other"}
    if not isinstance(defects, list):
        defects = []
    defects = [str(v).strip().lower() for v in defects if str(v).strip().lower() in allowed]
    return {
        "strand_count": count,
        "count_confidence": confidence,
        "visible_defects": defects,
        "uncertainty_reason": str(data.get("uncertainty_reason", ""))[:300],
    }


def classify_result(count, confidence, detection_confidence, defects, expected):
    if count == "uncertain" or confidence == "low" or detection_confidence == "low":
        return "UNCERTAIN"
    if defects or count != expected:
        return "DEFECT"
    return "GOOD"


def draw_annotated_image(source_image, components, results=None):
    overlay = source_image.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    font_size = max(16, min(34, int(h*0.018)))
    line_width = max(3, min(10, int(min(w, h)*0.003)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    colors = {
        "GOOD": ("#10b981", "#065f46"),
        "DEFECT": ("#ef4444", "#991b1b"),
        "UNCERTAIN": ("#f59e0b", "#92400e"),
        "DETECTED": ("#06b6d4", "#0e7490"),
    }
    for comp in components:
        px = box_to_pixels(comp["bbox"], w, h)
        pos = comp["position"]
        result = (results or {}).get(pos, {})
        status = result.get("status", "DETECTED")
        outline, badge = colors.get(status, colors["DETECTED"])
        draw.rectangle(px, outline=outline, width=line_width)
        label = pos if not result else f"{pos}: {result.get('strand_count', '?')}"
        if comp.get("possible_duplicate"):
            label += " !"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        bx = max(0, min(w-tw-12, px[0]))
        by = max(0, px[1]-th-12)
        draw.rounded_rectangle((bx, by, min(w, bx+tw+12), min(h, by+th+10)),
                               radius=4, fill=badge)
        draw.text((bx+6, by+3), label, fill="white", font=font)
    return overlay


def render_matrix(components, results, n_cols):
    if not components or not results or n_cols < 1:
        return
    cards = []
    for comp in components:
        pos = comp["position"]
        result = results.get(pos, {})
        status = result.get("status", "UNCERTAIN")
        css = {"GOOD": "card-good", "DEFECT": "card-defect"}.get(status, "card-uncertain")
        count = result.get("strand_count", "—")
        cards.append(
            f'<div class="comp-card {css}"><div class="card-id">{pos}</div>'
            f'<div class="card-count">{count}</div><div class="card-status">{status}</div></div>'
        )
    st.html('<div class="matrix-container"><div class="matrix-grid" '
            f'style="grid-template-columns:repeat({n_cols}, minmax(110px, 1fr));">'
            + "".join(cards) + "</div></div>")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Inspection Configuration")
    model_name = st.text_input("Vision model", value="gpt-5.4-mini",
                               help="Use a vision-capable model enabled for your API account.")
    expected_strands = st.number_input("Expected strand count", 0, 5, 4, 1)
    expected_components = st.number_input("Expected total components", 1, 30, 10, 1)
    rotation = st.selectbox("Additional image rotation", [0, 90, 180, 270],
                            format_func=lambda x: f"{x}°")
    crop_padding = st.slider("Crop padding (%)", 0, 15, 6)
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

if st.session_state.get("active_image_hash") != image_hash:
    for key in ("components", "inspection_results", "num_rows", "num_cols",
                "detection_notes", "layout_warnings", "debug_info"):
        st.session_state.pop(key, None)
    st.session_state["active_image_hash"] = image_hash

st.caption(f"{uploaded.name} · {source_img.width} × {source_img.height} px")
col_a, col_b = st.columns(2)
with col_a:
    run_full = st.button("🚀 Run Full Inspection", type="primary", use_container_width=True)
with col_b:
    run_detection = st.button("🔍 Detect Components Only", use_container_width=True)

if run_full or run_detection:
    for key in ("components", "inspection_results", "num_rows", "num_cols",
                "detection_notes", "layout_warnings", "debug_info"):
        st.session_state.pop(key, None)
    try:
        with st.spinner("Step 1/2: Detecting components..."):
            det = detect_components(model_name, source_img, int(expected_components))
            if not det["components"]:
                st.error("No valid component boxes were returned. Check image quality and API response.")
                st.stop()
            ordered, n_rows, n_cols, layout_warnings = assign_grid_positions(det["components"])
            st.session_state["components"] = ordered
            st.session_state["num_rows"] = n_rows
            st.session_state["num_cols"] = n_cols
            st.session_state["detection_notes"] = det["notes"]
            st.session_state["layout_warnings"] = layout_warnings
            st.session_state["debug_info"] = {
                "image_dimensions": source_img.size,
                "detected_count": len(ordered),
                "rejected_boxes": det["rejected_boxes"],
                "row_sizes": [sum(c["row"] == r for c in ordered) for r in range(1, n_rows+1)],
            }
    except Exception as exc:
        st.error(f"Component detection failed: {exc}")
        st.stop()

    if run_full:
        results = {}
        progress = st.progress(0.0)
        progress_text = st.empty()
        for i, comp in enumerate(ordered):
            pos = comp["position"]
            try:
                px = box_to_pixels(comp["bbox"], source_img.width, source_img.height)
                crop = crop_with_padding(source_img, px, crop_padding/100.0)
                counted = count_single_component(model_name, crop, pos)
                count, conf, defects = (counted["strand_count"],
                                        counted["count_confidence"],
                                        counted["visible_defects"])
                status = classify_result(count, conf, comp["detection_confidence"],
                                         defects, int(expected_strands))
                results[pos] = {
                    "position": pos, "row": comp["row"], "col": comp["col"],
                    "strand_count": count, "expected_count": int(expected_strands),
                    "status": status, "defects": defects, "confidence": conf,
                    "detection_confidence": comp["detection_confidence"],
                    "reason": counted["uncertainty_reason"], "crop_dimensions": crop.size,
                }
            except Exception as exc:
                results[pos] = {
                    "position": pos, "row": comp["row"], "col": comp["col"],
                    "strand_count": "uncertain", "expected_count": int(expected_strands),
                    "status": "UNCERTAIN", "defects": [], "confidence": "low",
                    "detection_confidence": comp["detection_confidence"],
                    "reason": f"Inspection error: {exc}",
                }
            progress.progress((i+1)/len(ordered))
            progress_text.write(f"Inspected {i+1} of {len(ordered)} components")
        progress.empty()
        progress_text.empty()
        st.session_state["inspection_results"] = results

components = st.session_state.get("components")
results = st.session_state.get("inspection_results")
n_rows = st.session_state.get("num_rows", 0)
n_cols = st.session_state.get("num_cols", 0)

if components:
    st.subheader("1. Annotated Packet Image")
    st.image(draw_annotated_image(source_img, components, results), use_container_width=True)

    if len(components) != int(expected_components):
        st.warning(f"Detected {len(components)} components; expected {int(expected_components)}. "
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
        if "UNCERTAIN" in statuses:
            st.warning("⚠️ Overall result: REVIEW REQUIRED")
        elif "DEFECT" in statuses:
            st.error("❌ Overall result: DEFECT DETECTED")
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
            "| Position | Count | Expected | Status | Defects | Confidence |",
            "|---|---:|---:|---|---|---|",
        ]
        for r in results.values():
            report.append(f"| {r['position']} | {r['strand_count']} | {r['expected_count']} | "
                          f"{r['status']} | {', '.join(r['defects']) or 'none'} | {r['confidence']} |")
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
        st.info("Components detected. Click Run Full Inspection to count strands.")

    if show_debug:
        st.subheader("Diagnostic details")
        st.json(st.session_state.get("debug_info", {}))
        with st.expander("Detected coordinates"):
            st.dataframe(pd.DataFrame([{
                "position": c["position"], "bbox_normalized": c["bbox"],
                "detection_confidence": c["detection_confidence"],
                "possible_duplicate": c["possible_duplicate"],
            } for c in components]), use_container_width=True)
        if results:
            with st.expander("Counting diagnostics"):
                st.dataframe(pd.DataFrame(list(results.values())), use_container_width=True)
