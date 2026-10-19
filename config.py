"""Shared paths and settings for the dyslexia handwriting project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Raw Kaggle dataset (208k small images, extracted from the password-protected RAR).
# Kept outside OneDrive so the images are not synced; override with DYSLEXIA_RAW_DIR.
RAW_DIR = Path(os.environ.get("DYSLEXIA_RAW_DIR", Path.home() / "Datasets" / "dyslexia_handwriting"))
PROCESSED_FILE = ROOT / "data" / "processed" / "handwriting_32px.npz"
INDEX_FILE = ROOT / "data" / "processed" / "image_index.csv"

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"
KNOWLEDGE_DIR = ROOT / "knowledge_base"

for d in (PROCESSED_FILE.parent, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Class order is fixed so every script and the app agree on label ids.
CLASSES = ["Normal", "Reversal", "Corrected"]
IMG_SIZE = 32
SEED = 42
VAL_FRACTION = 0.15  # share of the official Train folder held out for validation
