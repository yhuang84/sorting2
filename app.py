"""
Conv-Tex Fabric Classifier
==========================
Two views controlled by URL query parameter:
  ?view=mobile  — phone upload page (simple, clean)
  (default)     — desktop page with Upload tab (drag & drop) + Phone tab (QR code)
"""

import io
import time
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

# Update this after deploying to Streamlit Cloud
APP_BASE_URL = "https://your-app-name.streamlit.app"

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
# DRAG-AND-DROP STYLES
# Makes the st.file_uploader area visually larger and more prominent
# ==============================================================================
DRAG_DROP_CSS = """
<style>
/* Target the file uploader drop zone */
[data-testid="stFileUploader"] {
    width: 100%;
}
[data-testid="stFileUploader"] section {
    border: 2.5px dashed #4C72B0;
    border-radius: 16px;
    padding: 48px 24px;
    background: linear-gradient(135deg, #f0f4ff 0%, #fafafa 100%);
    text-align: center;
    transition: border-color 0.2s, background 0.2s;
    cursor: pointer;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #2a50a0;
    background: linear-gradient(135deg, #e4eaff 0%, #f5f5f5 100%);
}
[data-testid="stFileUploader"] section > div {
    font-size: 1.05rem;
    color: #444;
}
/* Browse files button */
[data-testid="stFileUploader"] section button {
    background-color: #4C72B0;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.95rem;
    cursor: pointer;
    margin-top: 8px;
}
[data-testid="stFileUploader"] section button:hover {
    background-color: #2a50a0;
}
</style>
"""


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
# FIGURES — return plt.Figure, rendered with st.pyplot() (avoids media file bug)
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
        st.image(image, caption=source_label, output_format="PNG")

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
# PAGE CONFIG
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
# ██████████████████████████████████████████████████████████████████████████████
if view == "mobile":

    st.title("📱 Upload Fabric Image")
    st.markdown(
        "Take a photo or choose a file from your phone.  \n"
        "Open the result on this screen."
    )

    # Inject drag-drop styles on mobile too
    st.markdown(DRAG_DROP_CSS, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop image here or tap to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        label_visibility="visible",
    )

    if uploaded is not None:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption="Your uploaded image")
        st.success("Done! You can upload another image to classify again.")

        # Also run classification directly on mobile if model is available
        # (optional — load model on mobile view too for self-contained use)
        if "model" not in st.session_state:
            with st.spinner("Loading model..."):
                try:
                    st.session_state["model"] = load_model_from_hub(
                        HF_REPO_ID, HF_FILENAME
                    )
                except Exception as e:
                    st.error(f"Model load failed: {e}")
                    st.stop()

        image_mobile = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Minimal sidebar for mobile
        with st.sidebar:
            layer_label_m = st.selectbox(
                "Feature map layer",
                options=list(LAYER_OPTIONS.keys()),
                index=3,
            )
            layer_idx_m = LAYER_OPTIONS[layer_label_m]

        run_classification(
            st.session_state["model"], image_mobile,
            layer_idx_m, layer_label_m,
            source_label="Phone Upload",
        )


# ██████████████████████████████████████████████████████████████████████████████
#  DESKTOP VIEW  (default)
# ██████████████████████████████████████████████████████████████████████████████
else:

    # Inject drag-drop styles
    st.markdown(DRAG_DROP_CSS, unsafe_allow_html=True)

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

    # ---- Header --------------------------------------------------------------
    st.title("🧵 Conv-Tex Fabric Classifier")
    st.markdown(
        "Fine-grained classification of 5 fabric structures "
        "using ConvNeXt-Small + Focal Loss."
    )
    st.divider()

    # ---- Sample images (click to classify) -----------------------------------
    SAMPLE_IMAGES = [
        ("SW_SC_183_Face_s512_p03.jpg", "1"),
        ("TW_HL_800__s512_p01.jpg",     "2"),
        ("RK_TK_16_Back_s512_p02.jpg",  "3"),
        ("PW_HL_803_s512_p00.jpg",      "4"),
        ("JK_HS_52_Face_s512_p06.jpg",  "5"),
    ]

    st.markdown("#### Sample Images — click one to classify")
    sample_cols = st.columns(len(SAMPLE_IMAGES))
    clicked_sample = None

    for col, (fname, label) in zip(sample_cols, SAMPLE_IMAGES):
        sample_path = Path(fname)
        if sample_path.exists():
            with col:
                st.image(str(sample_path), caption=label, width=130, output_format="PNG")

        else:
            with col:
                st.markdown(
                    f"<div style=\'color:#aaa;font-size:0.78rem;text-align:center\'>"
                    f"{label}<br><i style=\'font-size:0.7rem\'>{fname}</i></div>",
                    unsafe_allow_html=True,
                )

    if clicked_sample is not None:
        sample_path, sample_label = clicked_sample
        sample_img = Image.open(sample_path).convert("RGB")
        st.success(f"Sample selected: **{sample_label}**")
        run_classification(model, sample_img, layer_idx, layer_label,
                           source_label=f"Sample — {sample_label}")

    st.divider()

    # ---- Two tabs ------------------------------------------------------------
    tab_upload, tab_phone = st.tabs(
        ["📂 Upload from this computer", "📱 Use phone camera"]
    )

    # ── Tab 1: Desktop upload with drag-and-drop ──────────────────────────────
    with tab_upload:

        st.markdown(
            "**Drag and drop** a fabric image below, or click **Browse files**."
        )

        image_file = st.file_uploader(
            "Drop your fabric image here",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
            label_visibility="collapsed",   # hide label; instruction above is enough
        )

        if image_file is None:
            # Friendly placeholder shown inside the (now large) drop zone area
            st.markdown(
                """
                <div style="
                    text-align:center;
                    color:#888;
                    font-size:0.9rem;
                    margin-top:12px;
                ">
                    Supported formats: JPG · PNG · WEBP · TIF · BMP
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                image = Image.open(image_file).convert("RGB")
            except Exception as e:
                st.error(f"Could not open image: {e}")
                st.stop()
            run_classification(model, image, layer_idx, layer_label,
                               source_label=image_file.name)

    # ── Tab 2: Phone QR code ──────────────────────────────────────────────────
    with tab_phone:

        mobile_url = f"{APP_BASE_URL}?view=mobile"

        col_qr, col_info = st.columns([1, 2], gap="large")

        with col_qr:
            st.image(make_qr_image(mobile_url), width=200,
                     caption="Scan with your phone")

        with col_info:
            st.markdown(
                f"""
#### How to use your phone

1. **Scan the QR code** with your phone camera, or open this link:  
   `{mobile_url}`
2. Upload or take a photo of your fabric on the phone page.
3. Classification result + Grad-CAM appear on your phone screen instantly.

> After deploying, update `APP_BASE_URL` in the code to your actual
> Streamlit Cloud link (e.g. `https://conv-tex.streamlit.app`).
                """
            )
