"""
Conv-Tex Fabric Classifier
==========================
A Streamlit web app that classifies fabric images into 5 textile categories
using a fine-tuned ConvNeXt-Small model (Conv-Tex).

Model weights are loaded automatically from HuggingFace Hub.
Includes Grad-CAM and Top-30 Feature Map visualizations.
Supports both image upload and phone camera capture.
"""

import io
import socket

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


# ==============================================================================
# QR CODE
# ==============================================================================
def make_qr_image(url: str) -> bytes:
    """Generate a QR code PNG pointing to `url`."""
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
    """Download weights from HuggingFace Hub and return a ready model."""
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
    """Undo ImageNet normalisation → uint8 HWC array."""
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
    """Grad-CAM for ConvNeXt (B, C, H, W) feature maps."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
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

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations.detach()).sum(dim=1))
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(tensor.shape[2], tensor.shape[3]),
            mode="bilinear", align_corners=False,
        ).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.numpy()

    def remove(self):
        for h in self._handles:
            h.remove()


def make_gradcam_figure(image_rgb, cam, pred_label, confidence) -> bytes:
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
    return _fig_to_bytes(fig)


# ==============================================================================
# FEATURE MAPS
# ==============================================================================
def extract_feature_maps(model, tensor, layer_idx):
    captured = {}

    def hook(module, inp, out):
        captured["fmaps"] = out.detach()

    handle = model.features[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(tensor)
    handle.remove()
    return captured["fmaps"].squeeze(0)   # (C, H, W)


def make_feature_map_figure(fmaps, image_rgb, layer_name) -> bytes:
    n_channels    = fmaps.shape[0]
    channel_means = fmaps.mean(dim=(1, 2)).numpy()
    top_indices   = np.argsort(channel_means)[::-1][:TOP_N].copy()
    ncols, nrows  = 6, 5
    fig = plt.figure(figsize=(ncols * 2.2, nrows * 2.2 + 1.0))
    fig.suptitle(
        f"Top {TOP_N} Feature Maps — {layer_name}\n"
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
    return _fig_to_bytes(fig)


def make_feature_overlay_figure(fmaps, image_rgb, layer_name) -> bytes:
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
    return _fig_to_bytes(fig)


# ==============================================================================
# BAR CHART
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
    return _fig_to_bytes(fig)


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# ==============================================================================
# SHARED: RUN CLASSIFICATION + VISUALIZATIONS
# ==============================================================================
def run_classification(model, image, layer_idx, layer_label, source_label=""):
    """Run inference and render all results in the current Streamlit container."""
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
        st.image(make_bar_chart(probs))

    with st.expander("Detailed probability table"):
        for cat, p in sorted(zip(CATEGORIES, probs), key=lambda x: x[1], reverse=True):
            st.progress(float(p), text=f"**{cat}** — {p * 100:.2f}%")

    # ---- Visualizations -----------------------------------------------------
    st.divider()
    st.header("Model Visualizations")
    tab_gradcam, tab_fmaps = st.tabs(["Grad-CAM", f"Top-{TOP_N} Feature Maps"])
    image_rgb = tensor_to_display(input_tensor)

    with tab_gradcam:
        st.markdown(
            "**Grad-CAM** uses the gradient of the predicted class score with respect to "
            "the final convolutional stage to highlight the most discriminative regions."
        )
        with st.spinner("Computing Grad-CAM..."):
            gcam    = GradCAM(model, model.features[7])
            cam_map = gcam.generate(input_tensor.clone(), pred_idx)
            gcam.remove()
            model.zero_grad()
        st.image(
            make_gradcam_figure(image_rgb, cam_map, pred_label, confidence),
            
        )
        st.caption("Hot (red) = high contribution to prediction; cool (blue) = low contribution.")

    with tab_fmaps:
        st.markdown(
            f"**Top {TOP_N} feature maps** from **{layer_label}**, "
            "ranked by mean activation. Each cell is one channel of the feature tensor."
        )
        with st.spinner(f"Extracting feature maps from {layer_label}..."):
            fmaps = extract_feature_maps(model, input_tensor, layer_idx)

        st.subheader("Composite Overlay")
        st.image(
            make_feature_overlay_figure(fmaps, image_rgb, layer_label),
            
        )
        st.caption(f"Average of the top-{TOP_N} most activated channels overlaid on the input.")

        st.subheader(f"Top-{TOP_N} Individual Feature Maps")
        st.image(
            make_feature_map_figure(fmaps, image_rgb, layer_label),
            
        )
        st.caption("Each cell = one feature map channel. Brighter (yellow) = stronger activation.")


# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Conv-Tex Fabric Classifier",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🧵 Conv-Tex")
    st.markdown(
        "Classify fabric images into **5 textile categories** "
        "using **Conv-Tex**, a fine-tuned ConvNeXt-Small."
    )
    st.divider()

    st.subheader("Model")
    st.markdown(
        f"**HuggingFace Hub**  \n"
        f"`{HF_REPO_ID}`  \n"
        f"`{HF_FILENAME}`  \n\n"
        "Weights are downloaded and cached automatically on first run."
    )

    st.divider()
    st.subheader("Visualization Settings")
    layer_label = st.selectbox(
        "Feature map layer",
        options=list(LAYER_OPTIONS.keys()),
        index=3,
        help="Which ConvNeXt stage to visualize feature maps from.",
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

# ==============================================================================
# MODEL LOADING
# ==============================================================================
if "model" not in st.session_state:
    with st.spinner("Loading Conv-Tex model from HuggingFace Hub..."):
        try:
            st.session_state["model"] = load_model_from_hub(HF_REPO_ID, HF_FILENAME)
            st.sidebar.success("Model ready.")
        except Exception as e:
            st.error(
                f"**Model load failed.**  \n\n"
                f"Repo: `{HF_REPO_ID}`  \n"
                f"File: `{HF_FILENAME}`  \n\n"
                f"Error: `{e}`"
            )
            st.stop()

model = st.session_state["model"]

# ==============================================================================
# MAIN TABS
# ==============================================================================
st.header("Conv-Tex Fabric Classifier")
st.markdown(
    "Classify fabric microscopy images into **5 weave/knit structures** "
    "using a fine-tuned ConvNeXt-Small with Focal Loss."
)

tab_upload, tab_phone = st.tabs(["📂 Upload Image", "📱 Phone Camera"])


# ── Tab 1: Upload Image ────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        "Upload a **microscopy or macro photograph** of a fabric sample. "
        "Supported formats: JPG, PNG, TIF, BMP."
    )
    image_file = st.file_uploader(
        "Choose a fabric image",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
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


# ── Tab 2: Phone Camera ────────────────────────────────────────────────────────
with tab_phone:
    # Build the public URL from the browser's location (works on Streamlit Cloud)
    # st.query_params gives us the current page URL components
    app_url = "https://your-app-name.streamlit.app"  # <-- update after deployment

    col_qr, col_info = st.columns([1, 2], gap="large")

    with col_qr:
        st.image(
            make_qr_image(app_url),
            caption=f"Scan to open on phone",
            width=220,
        )

    with col_info:
        st.markdown(
            f"""
#### Use your phone as a camera

This app is deployed on **Streamlit Community Cloud** —
your phone can access it from **any network**, no Wi-Fi pairing needed.

1. **Scan the QR code** with your phone camera,  
   or open this URL directly in your phone browser:  
   `{app_url}`
2. Tap the **📱 Phone Camera** tab on your phone.
3. Tap **Take Photo**, aim at your fabric sample, and capture.
4. The classification result and Grad-CAM appear immediately.

> After deploying, replace the placeholder URL in `app.py` with your actual
> Streamlit Cloud link (e.g. `https://conv-tex.streamlit.app`).
            """
        )

    st.divider()
    st.subheader("Take a Photo")
    st.markdown(
        "Once this page is open on your phone, use the camera button below."
    )

    camera_image = st.camera_input(
        "Point your camera at the fabric and press the shutter button"
    )

    if camera_image is not None:
        try:
            image_phone = Image.open(camera_image).convert("RGB")
        except Exception as e:
            st.error(f"Could not read camera image: {e}")
            st.stop()
        st.success("Photo captured — running classification...")
        run_classification(model, image_phone, layer_idx, layer_label,
                           source_label="Phone Camera")
