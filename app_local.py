"""Local Streamlit app: upload a packet photo, get the 5x2 inspection result.

Runs the trained ResNet18 locally. No API key, no internet.
    python -m streamlit run app_local.py
"""
import io

import pandas as pd
import streamlit as st
from PIL import Image

from src.infer import default_model_paths, inspect_packet, load_models, render

st.set_page_config(page_title="Strand Inspection (local)", layout="wide")
st.title("Strand Inspection (local model)")


@st.cache_resource
def get_models(paths):
    return load_models(list(paths))


paths = tuple(default_model_paths())
if not paths:
    st.error("No trained model found in runs/phase4/. Run scripts/03_train.py first.")
    st.stop()

with st.sidebar:
    st.caption(f"{len(paths)} model(s) loaded (predictions averaged)")
    conf = st.slider("UNCERTAIN below confidence", 0.3, 0.95, 0.6, 0.05,
                     help="Placeholder threshold, not yet tuned.")
    st.caption("Expected strand count: 4 (GOOD). Other counts = DEFECT.")
    st.warning("Assumes an upright 2-column x 5-row packet; always reports 10 components. "
               "Check the boxes sit on the components before trusting the counts.")

files = st.file_uploader("Packet photo(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
nets = get_models(paths)

for f in files:
    st.divider()
    with st.spinner(f"Inspecting {f.name}..."):
        out = inspect_packet(Image.open(f), nets, conf)
    (st.success if out["passed"] else st.error)(f"{f.name}: PACKET {'PASS' if out['passed'] else 'FAIL'}")
    if out["table_mode"]:
        st.caption("Yellow background detected: table-rejecting mask used.")
    st.image(render(out), use_container_width=True)

    res = out["results"]
    grid = pd.DataFrame([[next(r for r in res if r["row"] == i and r["col"] == j) for j in (1, 2)]
                         for i in range(1, 6)], index=[f"Row {i}" for i in range(1, 6)], columns=["Col 1", "Col 2"])
    st.dataframe(grid.map(lambda r: f"{r['pred']}  {r['verdict']}  ({r['conf']:.0%})"),
                 use_container_width=True)
    buf = io.BytesIO(); render(out).save(buf, "PNG")
    st.download_button("Download result image", buf.getvalue(),
                       file_name=f"{f.name.rsplit('.', 1)[0]}_result.png", mime="image/png", key=f"dl_{f.name}")
