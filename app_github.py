"""
Fabric Classification App
Classifies textile images into 5 fabric categories using ConvNeXt-Small.
"""

import io

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_small
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIG
# ==============================================================================
CATEGORIES = ["jerseyknit", "plainweave", "ribknit", "satinweave", "twillweave"]
NUM_CLASSES = len(CATEGORIES)
DROPOUT_RATE = 0.5
IMG_HEIGHT = 224
IMG_WIDTH = 224

CATEGORY_DESCRIPTIONS = {
    "jerseyknit": "Jersey Knit — single-weft knit, soft & stretchy (e.g. T-shirts)",
    "plainweave":  "Plain Weave — simplest over-under interlacing (e.g. muslin, canvas)",
    "ribknit":     "Rib Knit — vertical columns of knit & purl stitches (e.g. cuffs)",
    "satinweave":  "Satin Weave — floating warps for a smooth lustrous face (e.g. satin)",
    "twillweave":  "Twill Weave — diagonal rib pattern (e.g. denim, gabardine)",
}

CATEGORY_COLORS = {
    "jerseyknit": "#4C72B0",
    "plainweave":  "#DD8452",
    "ribknit":     "#55A868",
    "satinweave":  "#C44E52",
    "twillweave":  "#8172B2",
}

# ==============================================================================
# MODEL
# ==============================================================================
def build_convnext_small(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Reconstruct the exact architecture used during training."""
    model = convnext_small(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(in_features, num_classes),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_model_from_bytes(file_bytes: bytes) -> nn.Module:
    buf = io.BytesIO(file_bytes)
    state = torch.load(buf, map_location="cpu", weights_only=True)
    model = build_convnext_small()
    model.load_state_dict(state)
    model.eval()
    return model


# ==============================================================================
# INFERENCE
# ==============================================================================
def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(model: nn.Module, image: Image.Image):
    tensor = get_val_transform()(image).unsqueeze(0)  # (1, 3, 224, 224)
    logits = model(tensor)                            # (1, 5)
    probs = torch.softmax(logits, dim=1).squeeze()    # (5,)
    pred_idx = int(probs.argmax())
    return probs.numpy(), pred_idx


# ==============================================================================
# PLOTS
# ==============================================================================
def make_bar_chart(probs: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = [CATEGORY_COLORS[c] for c in CATEGORIES]
    bars = ax.barh(CATEGORIES, probs * 100, color=colors, height=0.55, edgecolor="white")

    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob * 100 + 1.2, 101),
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Class Probabilities", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# ==============================================================================
# PAGE LAYOUT
# ==============================================================================
st.set_page_config(
    page_title="Fabric Classifier",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.title("🧵 Fabric Classifier")
    st.markdown(
        "Classify textile images into **5 fabric types** "
        "using a fine-tuned ConvNeXt-Small."
    )
    st.divider()

    st.subheader("1. Load Model")
    model_file = st.file_uploader(
        "Upload model checkpoint (.pth)",
        type=["pth", "pt"],
        help="Upload the ConvNeXt-Small_fold1_best.pth file.",
    )

    st.divider()
    st.subheader("Fabric Classes")
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        color = CATEGORY_COLORS[cat]
        st.markdown(
            f"<span style='color:{color}; font-weight:bold'>&#9632;</span> {desc}",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Model: ConvNeXt-Small | Pretrained: ImageNet V1 | Fine-tuned on fabric dataset")

# --- Model loading -----------------------------------------------------------
if model_file is None:
    st.info("Upload your **model checkpoint** (.pth) in the sidebar to get started.")
    st.stop()

with st.spinner("Loading model weights..."):
    try:
        model = load_model_from_bytes(model_file.getvalue())
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

st.sidebar.success(f"Model ready: `{model_file.name}`")

# --- Main area ---------------------------------------------------------------
st.header("Fabric Type Classification")
st.markdown(
    "Upload a **microscopy or macro image** of a fabric sample. "
    "The model will predict which of the 5 weave/knit structures it belongs to."
)

image_file = st.file_uploader(
    "2. Upload a fabric image",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
)

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True, caption=image_file.name)

    with col_res:
        st.subheader("Prediction")
        with st.spinner("Running inference..."):
            probs, pred_idx = predict(model, image)

        pred_label = CATEGORIES[pred_idx]
        confidence = probs[pred_idx]
        color = CATEGORY_COLORS[pred_label]

        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                border-left:5px solid {color};
                padding:14px 18px;
                border-radius:6px;
                margin-bottom:18px;
            ">
                <div style="font-size:0.85rem;color:#666;margin-bottom:4px;">Predicted Class</div>
                <div style="font-size:1.6rem;font-weight:700;color:{color};">
                    {pred_label.upper()}
                </div>
                <div style="font-size:1rem;color:#444;margin-top:4px;">
                    {CATEGORY_DESCRIPTIONS[pred_label]}
                </div>
                <div style="font-size:1.1rem;font-weight:600;color:{color};margin-top:8px;">
                    Confidence: {confidence * 100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(make_bar_chart(probs), use_container_width=True)

    with st.expander("Detailed probability table"):
        for cat, p in sorted(zip(CATEGORIES, probs), key=lambda x: x[1], reverse=True):
            st.progress(float(p), text=f"**{cat}** — {p * 100:.2f}%")

else:
    st.info("Upload a fabric image above to classify it.")
