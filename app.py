"""
Conv-Tex Fabric Classifier
==========================
Two views controlled by URL query parameter:
  ?view=mobile  — phone upload page (simple, clean)
  (default)     — desktop display page (auto-refreshes every 3s, runs classification)

How it works:
  1. Phone opens  https://your-app.streamlit.app/?view=mobile
  2. Phone uploads a fabric image -> saved to /tmp on the server
  3. Desktop polls /tmp every 3 seconds, detects new image, classifies it
"""

import io
import os
import time
import socket
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import convnext_small
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import qrcode
from huggingface_hub import hf_hub_download

# ==============================================================================
# CONFIG
# ==============================================================================
CATEGORIES = ["jerseyknit", "plainweave", "ribknit", "satinweave", "twillweave"]
NUM_CLASSES = len(CATEGORIES)
DROPOUT_RATE = 0.5
IMG_HEIGHT = 224
IMG_WIDTH = 224
TOP_N = 30

HF_REPO_ID = "yh84ian/Conv_Tex"
HF_FILENAME = "ConvNeXt-Small_focal_fold1_best.pth"

# Shared file locations on the server (/tmp persists within a Streamlit Cloud instance)
SHARED_IMAGE_PATH    = Path("/tmp/latest_fabric.jpg")
SHARED_TIMESTAMP_PATH = Path("/tmp/latest_fabric_ts.txt")

REFRESH_INTERVAL = 3   # seconds between desktop polls

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

LAYER_OPTIONS = {
    "Stage 1  (96 ch,  56x56)":        1,
    "Stage 2  (192 ch, 28x28)":        3,
    "Stage 3  (384 ch, 14x14)":        5,
    "Stage 4  (768 ch,  7x7) — final": 7,
}

# Update this after deploying to Streamlit Cloud
APP_BASE_URL = "https://your-app-name.streamlit.app"


# ==============================================================================
# QR CODE
# ==============================================================================
def make_qr_image(url: str) -> bytes:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=3,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ==============================================================================
# MODEL
# ==============================================================================
def build_convnext_small(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = convnext_small(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(in_features, num_classes),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_model_from_hub(repo_id: str, filename: str) -> nn.Module:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_convnext_small()
    model.load_state_dict(state)
    model.eval()
    return model


# ==============================================================================
# PREPROCESSING
# ==============================================================================
def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.squeeze(0).cpu() * std + mean
    img  = img.permute(1, 2, 0).clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)


# ==============================================================================
# INFERENCE
# ==============================================================================
@torch.no_grad()
def predict(model: nn.Module, image: Image.Image):
    tensor = get_val_transform()(image).unsqueeze(0)
    probs  = torch.softmax(model(tensor), dim=1).squeeze()
    pred_idx = int(probs.argmax())
    return probs.numpy(), pred_idx, tensor


# ==============================================================================
# GRAD-CAM
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._activations = None
        self._gradients   = None
        self._handles = [
            target_layer.register_forward_hook(self._fwd_hook),
            target_layer.register_full_backward_hook(self._bwd_hook),
        ]

    def _fwd_hook(self, module, inp, out):
        self._activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations.detach()).sum(dim=1))
        cam = F.interpolate(
            cam.unsqueeze(0), size=(tensor.shape[2], tensor.shape[3]),
            mode="bilinear", align_corners=False,
        ).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.numpy()

    def remove(self):
        for h in self._handles:
            h.remove()


# ==============================================================================
# FIGURES  — return plt.Figure, rendered with st.pyplot() (avoids media file bug)
# ==============================================================================
def make_gradcam_figure(image_rgb, cam, pred_label, confidence) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(image_rgb)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title(
        f"Grad-CAM -> {pred_label.upper()} ({confidence*100:.1f}%)",
        fontsize=12, fontweight="bold", color=CATEGORY_COLORS[pred_label],
    )
    axes[1].axis("off")
    sm = plt.cm.ScalarMappable(cmap="jet", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04, label="Activation")
    fig.suptitle("Grad-CAM: regions driving the prediction", fontsize=13)
    fig.tight_layout()
    return fig


def make_feature_map_figure(fmaps, layer_name) -> plt.Figure:
    n_channels    = fmaps.shape[0]
    channel_means = fmaps.mean(dim=(1, 2)).numpy()
    top_indices   = np.argsort(channel_means)[::-1][:TOP_N].copy()
    ncols, nrows  = 6, 5
    fig = plt.figure(figsize=(ncols * 2.2, nrows * 2.2 + 1.0))
    fig.suptitle(
        f"Top {TOP_N} Feature Maps - {layer_name}\n"
        f"(ranked by mean activation, {n_channels} total channels)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.25)
    for plot_i, ch_idx in enumerate(top_indices):
        row, col = divmod(plot_i, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(fmaps[ch_idx].numpy(), cmap="viridis", interpolation="nearest")
        ax.set_title(f"Ch {ch_idx}\nmu={channel_means[ch_idx]:.2f}", fontsize=7, pad=2)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_feature_overlay_figure(fmaps, image_rgb, layer_name) -> plt.Figure:
    channel_means = fmaps.mean(dim=(1, 2)).numpy()
    top_indices   = np.argsort(channel_means)[::-1][:TOP_N].copy()
    composite     = fmaps[top_indices].mean(dim=0).numpy()
    composite_t   = torch.from_numpy(composite).unsqueeze(0).unsqueeze(0)
    composite_r   = F.interpolate(
        composite_t, size=(IMG_HEIGHT, IMG_WIDTH),
        mode="bilinear", align_corners=False,
    ).squeeze().numpy()
    composite_r = (composite_r - composite_r.min()) / (composite_r.max() + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(image_rgb)
    axes[1].imshow(composite_r, cmap="plasma", alpha=0.5)
    axes[1].set_title(f"Top-{TOP_N} Feature Map Overlay\n{layer_name}",
                      fontsize=12, fontweight="bold")
    axes[1].axis("off")
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04, label="Activation")
    fig.suptitle(f"Mean of top-{TOP_N} most activated channels", fontsize=13)
    fig.tight_layout()
    return fig


def make_bar_chart(probs: np.ndarray) -> plt.Figure:
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
    return fig


def extract_feature_maps(model, tensor, layer_idx):
    captured = {}

    def hook(module, inp, out):
        captured["fmaps"] = out.detach()

    handle = model.features[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(tensor)
    handle.remove()
    return captured["fmaps"].squeeze(0)


# ==============================================================================
# SHARED: RUN CLASSIFICATION + VISUALIZATIONS
# ==============================================================================
def run_classification(model, image, layer_idx, layer_label, source_label=""):
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Input Image")
        st.image(image, caption=source_label)

    with col_res:
        st.subheader("Prediction")
        with st.spinner("Running inference..."):
            probs, pred_idx, input_tensor = predict(model, image)

        pred_label = CATEGORIES[pred_idx]
        confidence = probs[pred_idx]
        color      = CATEGORY_COLORS[pred_label]

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
        st.pyplot(make_bar_chart(probs))

    with st.expander("Detailed probability table"):
        for cat, p in sorted(zip(CATEGORIES, probs), key=lambda x: x[1], reverse=True):
            st.progress(float(p), text=f"**{cat}** — {p * 100:.2f}%")

    st.divider()
    st.header("Model Visualizations")
    tab_gradcam, tab_fmaps = st.tabs(["Grad-CAM", f"Top-{TOP_N} Feature Maps"])
    image_rgb = tensor_to_display(input_tensor)

    with tab_gradcam:
        st.markdown(
            "**Grad-CAM** highlights the most discriminative regions "
            "for the predicted class."
        )
        with st.spinner("Computing Grad-CAM..."):
            gcam    = GradCAM(model, model.features[7])
            cam_map = gcam.generate(input_tensor.clone(), pred_idx)
            gcam.remove()
            model.zero_grad()
        st.pyplot(make_gradcam_figure(image_rgb, cam_map, pred_label, confidence))
        st.caption("Hot (red) = high contribution; cool (blue) = low contribution.")

    with tab_fmaps:
        st.markdown(
            f"**Top {TOP_N} feature maps** from **{layer_label}**, "
            "ranked by mean activation."
        )
        with st.spinner(f"Extracting feature maps from {layer_label}..."):
            fmaps = extract_feature_maps(model, input_tensor, layer_idx)

        st.subheader("Composite Overlay")
        st.pyplot(make_feature_overlay_figure(fmaps, image_rgb, layer_label))
        st.caption(f"Average of the top-{TOP_N} most activated channels.")

        st.subheader(f"Top-{TOP_N} Individual Feature Maps")
        st.pyplot(make_feature_map_figure(fmaps, layer_label))
        st.caption("Brighter (yellow) = stronger activation.")


# ==============================================================================
# PAGE CONFIG  (must come before any st.* calls that render UI)
# ==============================================================================
st.set_page_config(
    page_title="Conv-Tex Fabric Classifier",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# ROUTE BY ?view= QUERY PARAMETER
# ==============================================================================
view = st.query_params.get("view", "desktop")


# ██████████████████████████████████████████████████████████████████████████████
#  MOBILE VIEW   (?view=mobile)
#  Phone opens this URL, uploads an image -> saved to /tmp on the server
# ██████████████████████████████████████████████████████████████████████████████
if view == "mobile":

    st.title("📱 Upload Fabric Image")
    st.markdown(
        "Take a photo or choose a file.  \n"
        "The desktop screen will update automatically."
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        img_bytes = uploaded.read()

        # Save to shared location
        SHARED_IMAGE_PATH.write_bytes(img_bytes)
        SHARED_TIMESTAMP_PATH.write_text(str(time.time()))

        # Confirm on phone
        st.image(img_bytes, caption="Uploaded successfully")
        st.success("Image sent to desktop!")
        st.markdown(
            "<p style='color:#888; font-size:0.85rem; margin-top:20px;'>"
            "Upload another image to replace the current one.</p>",
            unsafe_allow_html=True,
        )


# ██████████████████████████████████████████████████████████████████████████████
#  DESKTOP VIEW  (default / ?view=desktop)
#  Polls /tmp every REFRESH_INTERVAL seconds for a new image from the phone
# ██████████████████████████████████████████████████████████████████████████████
else:

    # ---- Sidebar -------------------------------------------------------------
    with st.sidebar:
        st.title("🧵 Conv-Tex")
        st.markdown(
            "Classify fabric images into **5 textile categories** "
            "using **Conv-Tex**, a fine-tuned ConvNeXt-Small."
        )
        st.divider()

        st.subheader("Model")
        st.markdown(
            f"`{HF_REPO_ID}`  \n`{HF_FILENAME}`  \n\n"
            "Downloaded from HuggingFace Hub automatically."
        )

        st.divider()
        st.subheader("Visualization Settings")
        layer_label = st.selectbox(
            "Feature map layer",
            options=list(LAYER_OPTIONS.keys()),
            index=3,
        )
        layer_idx = LAYER_OPTIONS[layer_label]

        st.divider()
        st.subheader("Fabric Classes")
        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            color = CATEGORY_COLORS[cat]
            st.markdown(
                f"<span style='color:{color}; font-weight:bold'>&#9632;</span> {desc}",
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption(
            "Model: ConvNeXt-Small  \n"
            "Pretrained: ImageNet-1K  \n"
            "Fine-tuned: WK-25 fabric dataset  \n"
            "Test accuracy: 92.29% (3-fold CV)"
        )

    # ---- Model loading -------------------------------------------------------
    if "model" not in st.session_state:
        with st.spinner("Loading Conv-Tex model from HuggingFace Hub..."):
            try:
                st.session_state["model"] = load_model_from_hub(HF_REPO_ID, HF_FILENAME)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()

    model = st.session_state["model"]

    # ---- Header + QR code ---------------------------------------------------
    mobile_url = f"{APP_BASE_URL}?view=mobile"

    st.title("🧵 Conv-Tex Fabric Classifier")

    col_instructions, col_qr = st.columns([3, 1])
    with col_instructions:
        st.markdown(
            f"""
            **How to use with your phone:**
            1. Scan the QR code (or open `{mobile_url}`)
            2. Upload a fabric photo on your phone
            3. This desktop screen refreshes automatically every {REFRESH_INTERVAL}s
            """
        )
    with col_qr:
        st.image(make_qr_image(mobile_url), width=150,
                 caption="Phone upload page")

    st.divider()

    # ---- Tabs ---------------------------------------------------------------
    tab_live, tab_upload = st.tabs(
        ["📲 Live — from phone", "📂 Upload — from this computer"]
    )

    # ── Live tab ──────────────────────────────────────────────────────────────
    with tab_live:

        if "last_seen_ts" not in st.session_state:
            st.session_state["last_seen_ts"] = 0.0

        has_image    = SHARED_IMAGE_PATH.exists()
        has_ts_file  = SHARED_TIMESTAMP_PATH.exists()

        if has_image and has_ts_file:
            server_ts = float(SHARED_TIMESTAMP_PATH.read_text().strip())
            img_bytes = SHARED_IMAGE_PATH.read_bytes()
            image_live = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            if server_ts > st.session_state["last_seen_ts"]:
                # New image — classify it
                st.session_state["last_seen_ts"] = server_ts
                st.success("New image received from phone — classifying...")
                run_classification(model, image_live, layer_idx, layer_label,
                                   source_label="Phone Upload")
            else:
                # Same image as before — show without re-running
                st.info("Waiting for a new image from your phone...")
                st.image(image_live, caption="Last received image", width=420)

        else:
            st.info(
                "No image received yet.  \n"
                "Scan the QR code above and upload a photo from your phone."
            )

        # Auto-refresh countdown
        placeholder = st.empty()
        for remaining in range(REFRESH_INTERVAL, 0, -1):
            placeholder.caption(f"Refreshing in {remaining}s...")
            time.sleep(1)
        placeholder.caption("Refreshing now...")
        st.rerun()

    # ── Local upload tab ──────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload a fabric image directly from this computer.")
        image_file = st.file_uploader(
            "Choose a fabric image",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
        )
        if image_file is None:
            st.info("Upload a fabric image above to get started.")
        else:
            try:
                image = Image.open(image_file).convert("RGB")
            except Exception as e:
                st.error(f"Could not open image: {e}")
                st.stop()
            run_classification(model, image, layer_idx, layer_label,
                               source_label=image_file.name)
