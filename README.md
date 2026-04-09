# 🧵 Conv-Tex Fabric Classifier

A web application for fine-grained fabric classification using **Conv-Tex**, a fine-tuned ConvNeXt-Small model trained on the WK-25 scanner dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## Demo

| Feature | Description |
|---|---|
| 📂 **Upload Image** | Classify any fabric microscopy or macro image |
| 📱 **Phone Camera** | Scan QR code → open on phone → snap photo → instant result |
| 🔥 **Grad-CAM** | Visualize which regions drove the prediction |
| 🗺️ **Feature Maps** | Inspect top-30 activated channels at any ConvNeXt stage |

---

## Fabric Classes

| Class | Description |
|---|---|
| Jersey Knit | Single-weft knit, soft & stretchy (e.g. T-shirts) |
| Plain Weave | Simplest over-under interlacing (e.g. muslin, canvas) |
| Rib Knit | Vertical columns of knit & purl stitches (e.g. cuffs) |
| Satin Weave | Floating warps for a smooth lustrous face (e.g. satin) |
| Twill Weave | Diagonal rib pattern (e.g. denim, gabardine) |

---

## Model

| Property | Value |
|---|---|
| Architecture | ConvNeXt-Small + Focal Loss (γ=2.0) |
| Pretraining | ImageNet-1K |
| Dataset | WK-25 (self-collected scanner images, 5 classes) |
| Validation accuracy | 94.51% (mean, 3-fold CV) |
| Test accuracy | 92.29% |
| Weights | [HuggingFace Hub — yh84ian/Conv_Tex](https://huggingface.co/yh84ian/Conv_Tex) |

Weights are downloaded automatically on first run and cached locally.

---

## Project Structure

```
fabric-classifier/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme and server settings
└── README.md
```

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/fabric-classifier.git
cd fabric-classifier
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.  
Model weights (~50 MB) are downloaded from HuggingFace Hub on the first run.

---

## Deploy to Streamlit Community Cloud (Free)

1. **Fork or push this repository to your GitHub account.**

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. Click **New app** → select your repository → set main file to `app.py` → click **Deploy**.

4. After deployment, copy your public URL (e.g. `https://conv-tex.streamlit.app`).

5. **Update the URL in `app.py`** (line marked with `# <-- update after deployment`):
   ```python
   app_url = "https://conv-tex.streamlit.app"   # your actual URL
   ```
   Commit and push — Streamlit Cloud redeploys automatically.

6. **Phone camera access**: your phone can now open the app from any network — no shared Wi-Fi required. Scan the QR code in the **📱 Phone Camera** tab.

---

## Usage

### Upload Image
1. Click **📂 Upload Image** tab.
2. Upload a fabric image (JPG, PNG, TIF, BMP).
3. The predicted class, confidence score, and visualizations appear automatically.

### Phone Camera
1. Click **📱 Phone Camera** tab on your desktop.
2. Scan the QR code with your phone — the app opens in your phone browser.
3. On your phone, tap **📱 Phone Camera** tab → tap **Take Photo**.
4. Results appear instantly on your phone screen.

### Visualization Settings
Use the **sidebar** to select which ConvNeXt stage to inspect for feature maps:
- Stage 1 (96 channels, 56×56) — low-level textures
- Stage 2 (192 channels, 28×28)
- Stage 3 (384 channels, 14×14)
- Stage 4 (768 channels, 7×7) — high-level semantics (default)

---

## Citation

If you use this work, please cite:

```bibtex
@article{huang2025convtex,
  title   = {Conv-Tex: Fine-Grained Fabric Classification with ConvNeXt-Small and Focal Loss},
  author  = {Huang, Yihan and Quddus, Md Abdul},
  journal = {Under Review},
  year    = {2025}
}
```

---

## License

This project is released for research and educational purposes.  
Model weights are hosted on HuggingFace Hub under the same terms.

---

*Department of Textile and Apparel Technology and Management, NC State University*
