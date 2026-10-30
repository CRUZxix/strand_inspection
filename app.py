import base64
import io
import json
import re

import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Page configuration
st.set_page_config(
    page_title="Packet Strand Inspection",
    page_icon="🔎",
    layout="wide",
)

# Custom styles for clean, modern cards and layout
st.html(
    """
    <style>
    .matrix-container {
        width: 100%;
        max-width: 520px;
        margin: 16px 0 24px 0;
    }
    .matrix-grid {
        display: grid;
        gap: 12px;
        width: 100%;
    }
    .comp-card {
        border-radius: 8px;
        padding: 12px 8px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        height: 115px;
        box-sizing: border-box;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        transition: transform 0.15s ease-in-out;
    }
    .comp-card:hover {
        transform: translateY(-2px);
    }
    .card-id {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        line-height: 1.2;
    }
    .card-count {
        font-size: 36px;
        font-weight: 800;
        line-height: 1;
        margin: 4px 0;
    }
    .card-status {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        line-height: 1.2;
    }
    /* GOOD: light green bg, dark green text */
    .card-good {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        border: 1.5px solid #a5d6a7 !important;
    }
    /* DEFECT: light red bg, dark red text */
    .card-defect {
        background-color: #ffebee !important;
        color: #b71c1c !important;
        border: 1.5px solid #ef9a9a !important;
    }
    /* UNCERTAIN: light yellow bg, dark yellow text */
    .card-uncertain {
        background-color: #fffde7 !important;
        color: #f57f17 !important;
        border: 1.5px solid #fff59d !important;
    }
    </style>
    """
)

st.title("🔎 Packet Strand Inspection")
st.caption("AI-assisted visual component detection and strand-count quality inspection")

# -------------------- API KEY --------------------
try:
    API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    API_KEY = ""

if not API_KEY or API_KEY == "PASTE_YOUR_API_KEY_HERE":
    st.error("Configure OPENAI_API_KEY in .streamlit/secrets.toml, then restart the app.")
    st.stop()

client = OpenAI(api_key=API_KEY)


# -------------------- HELPERS --------------------
def image_data_url(image, max_side=1600):
    img = image.copy().convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("Model response did not contain valid JSON.")


def ask_vision(model_name, prompt, image):
    response = client.responses.create(
        model=model_name,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": image_data_url(image),
                    "detail": "high",
                },
            ],
        }],
    )
    return parse_json(response.output_text)


def valid_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def box_to_pixels(box, width, height):
    x1, y1, x2, y2 = box
    return (
        max(0, int(x1 * width / 1000)),
        max(0, int(y1 * height / 1000)),
        min(width, int(x2 * width / 1000)),
        min(height, int(y2 * height / 1000)),
    )


def crop_with_padding(image, px_box, padding=0.10):
    x1, y1, x2, y2 = px_box
    dx = int((x2 - x1) * padding)
    dy = int((y2 - y1) * padding)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(image.width, x2 + dx)
    y2 = min(image.height, y2 + dy)
    return image.crop((x1, y1, x2, y2))


def normalize_count(value):
    if isinstance(value, bool):
        return "uncertain"
    if isinstance(value, int) and 0 <= value <= 5:
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "uncertain":
            return "uncertain"
        if v.isdigit() and 0 <= int(v) <= 5:
            return int(v)
    return "uncertain"


def assign_grid_positions(components):
    """
    Sorts detected components into rows (top to bottom) and columns (left to right)
    without using fixed coordinates or manual margins.
    """
    enriched = []
    for idx, c in enumerate(components):
        b = c["bbox"]
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        h = b[3] - b[1]
        enriched.append({"item": c, "cx": cx, "cy": cy, "h": h, "orig_idx": idx})

    if not enriched:
        return [], 0, 0

    enriched.sort(key=lambda x: x["cy"])

    avg_h = sum(x["h"] for x in enriched) / len(enriched)
    row_threshold = avg_h * 0.45

    rows = []
    current_row = [enriched[0]]
    for item in enriched[1:]:
        if abs(item["cy"] - current_row[-1]["cy"]) < row_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    ordered = []
    for r_idx, row in enumerate(rows, start=1):
        row.sort(key=lambda x: x["cx"])
        for c_idx, item in enumerate(row, start=1):
            comp = dict(item["item"])
            comp["row"] = r_idx
            comp["col"] = c_idx
            comp["position"] = f"R{r_idx}C{c_idx}"
            ordered.append(comp)

    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 0
    return ordered, num_rows, num_cols


def detect_components(model_name, image, expected_count=10):
    prompt = f"""
You are performing visual localization on an industrial inspection photograph of a transparent plastic packet containing individual loop-shaped components.

The packet contains approximately {expected_count} components arranged in rows and columns.
Locate EVERY individual physical component inside the packet.

Return ONLY a JSON object:
{{
  "packet_bbox": [x1, y1, x2, y2],
  "components": [
    {{
      "bbox": [x1, y1, x2, y2],
      "confidence": "high"
    }}
  ],
  "uncertainty_note": ""
}}

BOX COORDINATES:
- Use normalized coordinates [x1, y1, x2, y2] from 0 to 1000.
- [0, 0] is top-left, [1000, 1000] is bottom-right.
- Each box must tightly enclose exactly one component (the entire loop including all strands).
- Avoid unnecessary overlap, empty space, or capturing neighboring components.
- Do NOT use hardcoded grid coordinates; locate the actual physical components.
- Confidence must be "high", "medium", or "low".
"""
    data = ask_vision(model_name, prompt, image)
    packet_box = data.get("packet_bbox")
    if not valid_box(packet_box):
        packet_box = None

    components = []
    for item in data.get("components", []):
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not valid_box(box):
            continue
        conf = str(item.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        components.append({
            "bbox": [float(v) for v in box],
            "detection_confidence": conf,
        })

    return {
        "packet_bbox": packet_box,
        "components": components,
        "notes": str(data.get("uncertainty_note", "")),
    }


def count_single_component(model_name, crop, pos_label):
    prompt = f"""
Inspect this cropped photograph of component {pos_label} from the packet.
Count how many physical strands (wires/filaments) form this loop.

Valid strand counts are integers from 0 to 5. Five strands is a valid possible result.
If the strands are obscured, overlapping ambiguously, or cannot be distinguished reliably, return "uncertain".

Return ONLY JSON:
{{
  "strand_count": 4,
  "visible_defects": "none",
  "confidence": "high",
  "uncertainty_reason": ""
}}

Rules:
- strand_count must be an integer 0-5 or "uncertain".
- Describe visible defects only (e.g. broken strand, tangle, foreign material, or "none").
- confidence must be "high", "medium", or "low".
- Never invent counts or guess when strands cannot be distinguished.
"""
    data = ask_vision(model_name, prompt, crop)
    count = normalize_count(data.get("strand_count", "uncertain"))
    conf = str(data.get("confidence", "low")).lower()
    if conf not in ("high", "medium", "low"):
        conf = "low"
    defects = str(data.get("visible_defects", "none")).strip()
    reason = str(data.get("uncertainty_reason", "")).strip()

    return {
        "strand_count": count,
        "visible_defects": defects,
        "count_confidence": conf,
        "uncertainty_reason": reason,
    }


def draw_annotated_image(source_image, components, results_dict=None):
    """
    Draws tight bounding boxes and compact badges on the image.
    Color codes boxes:
      - Green: GOOD
      - Red: DEFECT
      - Yellow: UNCERTAIN
      - Cyan: Detection only (before counting)
    """
    overlay = source_image.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = source_image.size

    # Choose font size proportional to image height
    badge_h = max(24, int(h * 0.022))
    font_size = max(14, int(badge_h * 0.7))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    line_w = max(4, int(w * 0.003))

    color_map = {
        "GOOD": ("#10b981", "#065f46"),       # Green outline, dark bg
        "DEFECT": ("#ef4444", "#991b1b"),     # Red outline, dark bg
        "UNCERTAIN": ("#f59e0b", "#92400e"),  # Amber outline, dark bg
        "DETECTED": ("#06b6d4", "#0e7490"),   # Cyan outline, dark bg
    }

    for comp in components:
        px_box = box_to_pixels(comp["bbox"], w, h)
        pos = comp.get("position", "")

        status = "DETECTED"
        if results_dict and pos in results_dict:
            status = results_dict[pos].get("status", "UNCERTAIN")

        outline_color, badge_bg = color_map.get(status, ("#06b6d4", "#0e7490"))

        # Draw bounding box
        draw.rectangle(px_box, outline=outline_color, width=line_w)

        # Draw compact label badge at top-left
        label_text = f" {pos} " if pos else " C "
        x1, y1 = px_box[0], px_box[1]

        # Ensure label fits within image boundaries
        text_w = len(label_text) * (font_size * 0.65)
        badge_box = [x1, max(0, y1 - badge_h), x1 + text_w, max(badge_h, y1)]
        draw.rectangle(badge_box, fill=badge_bg)
        draw.text((badge_box[0] + 4, badge_box[1] + 2), label_text, fill="white", font=font)

    return overlay


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Inspection Configuration")
    model_name = st.selectbox(
        "Vision model",
        options=["gpt-5.4-mini", "gpt-5.6-sol"],
        index=0,
    )
    expected_strands = st.number_input(
        "Expected strand count per component",
        min_value=0,
        max_value=5,
        value=4,
        step=1,
        help="Target strand count. Components with a different count will be marked as DEFECT.",
    )
    expected_components = st.number_input(
        "Expected total components",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="Normal arrangement is 10 components (5 rows × 2 columns).",
    )
    rotation = st.selectbox(
        "Manual image rotation",
        options=[0, 90, 180, 270],
        index=0,
        format_func=lambda x: f"{x}°",
        help="EXIF orientation is applied automatically. Use this if the packet is still tilted.",
    )


# -------------------- MAIN INTERFACE --------------------
uploaded = st.file_uploader(
    "Upload packet photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an inspection photo of the packet.",
)

if not uploaded:
    st.info("Upload an image above to begin component detection and strand counting.")
    st.stop()

# 1. Load image and apply EXIF transpose + optional manual rotation
try:
    source_img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    if rotation:
        source_img = source_img.rotate(rotation, expand=True)
except Exception as exc:
    st.error(f"Could not load image: {exc}")
    st.stop()

# Action buttons
col_act1, col_act2 = st.columns([1, 1])
with col_act1:
    btn_inspect = st.button("🚀 Run Full Inspection", type="primary", width="stretch")
with col_act2:
    btn_detect_only = st.button("🔍 Detect Components Only", width="stretch")

# Run detection and inspection workflow
if btn_inspect or btn_detect_only:
    # Clear prior results
    st.session_state.pop("inspection_results", None)
    st.session_state.pop("components", None)
    st.session_state.pop("num_rows", None)
    st.session_state.pop("num_cols", None)

    with st.spinner("Step 1/2: Detecting individual components..."):
        try:
            det_res = detect_components(model_name, source_img, expected_count=expected_components)
            raw_comps = det_res["components"]
            if not raw_comps:
                st.error("No components could be located in the image. Please verify lighting, framing, and rotation.")
                st.stop()

            # Assign dynamic row and column positions (e.g. R1C1, R1C2...)
            ordered_comps, n_rows, n_cols = assign_grid_positions(raw_comps)
            st.session_state["components"] = ordered_comps
            st.session_state["num_rows"] = n_rows
            st.session_state["num_cols"] = n_cols
            st.session_state["detection_notes"] = det_res.get("notes", "")

        except Exception as exc:
            st.error(f"Component detection failed: {exc}")
            st.stop()

    if btn_inspect:
        with st.spinner("Step 2/2: Counting strands for each detected component..."):
            progress_bar = st.progress(0.0)
            res_dict = {}
            w, h = source_img.size

            for i, comp in enumerate(ordered_comps):
                pos = comp["position"]
                px_box = box_to_pixels(comp["bbox"], w, h)
                crop = crop_with_padding(source_img, px_box, padding=0.10)

                try:
                    c_res = count_single_component(model_name, crop, pos)
                except Exception as exc:
                    c_res = {
                        "strand_count": "uncertain",
                        "visible_defects": f"API Error: {exc}",
                        "count_confidence": "low",
                        "uncertainty_reason": str(exc),
                    }

                pred_count = c_res["strand_count"]
                confidence = c_res["count_confidence"]
                det_conf = comp["detection_confidence"]
                defects = c_res["visible_defects"]

                # Inspection status determination
                if pred_count == "uncertain" or confidence == "low" or det_conf == "low":
                    status = "UNCERTAIN"
                elif defects.lower() not in ("none", "none visible", "no defects", "n/a", ""):
                    status = "DEFECT"
                elif pred_count == expected_strands:
                    status = "GOOD"
                else:
                    status = "DEFECT"

                res_dict[pos] = {
                    "position": pos,
                    "row": comp["row"],
                    "col": comp["col"],
                    "strand_count": pred_count,
                    "expected_count": expected_strands,
                    "status": status,
                    "defects": defects,
                    "confidence": confidence,
                    "detection_confidence": det_conf,
                    "reason": c_res["uncertainty_reason"],
                }
                progress_bar.progress((i + 1) / len(ordered_comps))

            progress_bar.empty()
            st.session_state["inspection_results"] = res_dict


# -------------------- DISPLAY SECTION --------------------
comps = st.session_state.get("components")
res_dict = st.session_state.get("inspection_results")
n_rows = st.session_state.get("num_rows", 0)
n_cols = st.session_state.get("num_cols", 0)

if comps:
    # 1. Annotated full packet image
    st.subheader("1. Annotated Packet Image")
    annotated = draw_annotated_image(source_img, comps, results_dict=res_dict)
    st.image(annotated, width="stretch")

    # Detections validation check
    det_count = len(comps)
    if det_count != expected_components:
        st.warning(
            f"Detected {det_count} components (expected {expected_components}). "
            "Please check bounding boxes for any missed or duplicate components."
        )

    # 2. Final Results Display: Colour-Coded Matrix
    if res_dict:
        st.subheader("2. Final Inspection Matrix")

        # Determine overall status
        statuses = [item["status"] for item in res_dict.values()]
        has_uncertain = "UNCERTAIN" in statuses
        has_defect = "DEFECT" in statuses

        if has_uncertain:
            st.warning("⚠️ Overall Result: **REVIEW REQUIRED** — One or more components have ambiguous strand counts or low confidence.")
        elif has_defect:
            st.error("❌ Overall Result: **DEFECT DETECTED** — One or more components do not meet the expected strand count or have visible defects.")
        else:
            st.success("✅ Overall Result: **ALL COMPONENTS GOOD** — All components match expected strand criteria.")

        # Build clean CSS grid of cards matching rows and columns
        card_items = []
        for comp in comps:
            pos = comp["position"]
            data = res_dict.get(pos, {})
            count_val = data.get("strand_count", "—")
            status = data.get("status", "UNCERTAIN")

            # Card CSS class
            if status == "GOOD":
                card_class = "comp-card card-good"
            elif status == "DEFECT":
                card_class = "comp-card card-defect"
            else:
                card_class = "comp-card card-uncertain"

            count_display = str(count_val).capitalize() if str(count_val) == "uncertain" else str(count_val)

            card_items.append(
                f'<div class="{card_class}">'
                f'<div class="card-id">{pos}</div>'
                f'<div class="card-count">{count_display}</div>'
                f'<div class="card-status">{status}</div>'
                f'</div>'
            )

        matrix_html = (
            f'<div class="matrix-container">'
            f'<div class="matrix-grid" style="grid-template-columns: repeat({n_cols}, minmax(130px, 1fr));">'
            + "".join(card_items)
            + '</div></div>'
        )
        st.html(matrix_html)

        # 3. Downloads (CSV and Markdown)
        records = list(res_dict.values())
        df = pd.DataFrame(records)

        # Format markdown table manually to avoid tabulate dependency
        md_lines = [
            f"# Packet Strand Inspection Report",
            f"- **Filename:** {uploaded.name}",
            f"- **Expected Strand Count:** {expected_strands}",
            f"- **Overall Result:** {'REVIEW REQUIRED' if has_uncertain else ('DEFECT' if has_defect else 'GOOD')}",
            "",
            "| Position | Strand Count | Status | Defects | Confidence |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in records:
            md_lines.append(
                f"| {r['position']} | {r['strand_count']} | {r['status']} | {r['defects']} | {r['confidence']} |"
            )
        report_md = "\n".join(md_lines)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "📥 Download Results CSV",
                data=df.to_csv(index=False),
                file_name=f"inspection_{uploaded.name}.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_d2:
            st.download_button(
                "📥 Download Markdown Report",
                data=report_md,
                file_name=f"inspection_{uploaded.name}.md",
                mime="text/markdown",
                width="stretch",
            )

    elif not btn_inspect and comps:
        st.info("Components located. Click **🚀 Run Full Inspection** to count strands and generate the final matrix.")
