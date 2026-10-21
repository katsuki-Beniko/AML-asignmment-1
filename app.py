"""Streamlit web app - CUB-200 bird species classifier (COS30082 Assignment 1).

Upload a bird photo (or pick an example), choose one of 6 trained models, and get the top-5 species with
confidences plus a Grad-CAM heat-map showing which image regions drove the prediction.

Run locally:   streamlit run app.py        (see README.md)
"""
import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image

import birdmodel as bm

st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦", layout="wide")


@st.cache_resource(max_entries=1, show_spinner="Loading model...")   # keep only one model in memory
def get_net(key):
    return bm.load_net(next(m for m in bm.MODELS if m["key"] == key))


@st.cache_data(max_entries=32, show_spinner=False)
def run(image_bytes, key):
    """Predict for (image, model); cached so reruns of the page are instant."""
    import io
    m = next(m for m in bm.MODELS if m["key"] == key)
    return bm.predict(get_net(key), Image.open(io.BytesIO(image_bytes)), m)


st.title("🐦 Bird Species Classifier")
st.write("Upload a photo of a bird (200 North-American species, CUB-200), **choose a model**, and see its top-5 "
         "predictions. The heat-map (Grad-CAM) shows where the network looked. Compare the from-scratch, "
         "fine-tuned and final ResNet50 / ConvNeXt-Tiny models.")

# ----- sidebar: model choice -----
with st.sidebar:
    st.header("Model")
    label = st.selectbox("Choose a model", [m["label"] for m in bm.MODELS],
                         index=[m["label"] for m in bm.MODELS].index(bm.DEFAULT))
    m = bm.BY_LABEL[label]
    st.markdown(f"{m['description']}")
    c1, c2 = st.columns(2)
    c1.metric("Top-1 (test)", f"{m['top1']}%")
    c2.metric("Avg / class", f"{m['avg']}%")
    st.caption(f"Reported on the 1,204 provided test images. Input {m['size']} px"
               f"{', flip averaging' if m['flip'] else ''}.")

# ----- input: upload or example -----
examples = sorted(Path(__file__).parent.joinpath("examples").glob("*.jpg"))
left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("Bird photo")
    up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    st.caption("Or click an example from the test set:")
    cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        col.image(str(ex), width="stretch")
        if col.button("Use", key=f"ex{i}", width="stretch"):
            st.session_state["example"] = str(ex)
    if up is not None:
        st.session_state["example"] = None            # an uploaded file takes priority over an example
    img_bytes = up.getvalue() if up is not None else (
        Path(st.session_state["example"]).read_bytes() if st.session_state.get("example") else None)
    if img_bytes:
        st.image(img_bytes, caption="Selected photo", width="stretch")

# ----- results -----
with right:
    st.subheader("Prediction")
    if not img_bytes:
        st.info("Upload a photo or pick an example to see the prediction.")
    else:
        with st.spinner("Classifying..."):
            top5, overlay = run(img_bytes, m["key"])
        for name, p in top5:
            st.progress(min(max(p, 0.0), 1.0), text=f"{name} - {p * 100:.1f}%")
        st.subheader("Grad-CAM")
        st.image(overlay, caption="Regions that drove the prediction (red = most influential)",
                 width="stretch")

st.divider()
st.caption("Data: Caltech-UCSD Birds 200 (Wah et al., 2011). Models: ResNet50 and ConvNeXt-Tiny, pretrained on "
           "ImageNet-1k only. COS30082 Assignment 1.")
