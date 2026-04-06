# Fabric Type Classifier — Streamlit App

A web interface for classifying fabric textile images into 5 weave/knit categories using a fine-tuned **ConvNeXt-Small** model.

## Classes

| Label | Description |
|---|---|
| `jerseyknit` | Single-weft knit, soft & stretchy (e.g. T-shirts) |
| `plainweave` | Simplest over-under interlacing (e.g. muslin, canvas) |
| `ribknit` | Vertical columns of knit & purl stitches (e.g. cuffs) |
| `satinweave` | Floating warps for a smooth lustrous face (e.g. satin) |
| `twillweave` | Diagonal rib pattern (e.g. denim, gabardine) |

## Setup

```bash
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Model

- Architecture: **ConvNeXt-Small** (ImageNet V1 pretrained, fine-tuned)
- Input: 224×224 RGB images
- Output: 5-class softmax probabilities

The model checkpoint (`.pth` file) is **not included** in this repository due to file size. Set the path to your local checkpoint in the sidebar when running the app.

## Project structure

```
user_interface/
├── app.py            # Streamlit application
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```
