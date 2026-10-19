"""Extra analysis: how many test letters have a near-identical image in the training split?

01_prepare_data.py already removes exact duplicates. The Corrected class, however,
contains many augmented copies (rotated / shifted versions of the same letter), which
are not byte-identical. For a random sample of test images this script finds the
closest training image (mean absolute pixel difference) and reports how many are
near-copies, together with the CNN accuracy on "novel" vs "near-copy" test images.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import load_splits, predict_proba
from config import CLASSES, FIGURES_DIR, RESULTS_DIR, SEED

SAMPLE = 3000
NEAR_COPY = 6.0  # mean absolute difference (0-255 scale) below which two letters are near-identical


def main():
    data = load_splits()
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(data["X_test"]), SAMPLE, replace=False)
    Xs, ys = data["X_test"][idx], data["y_test"][idx]

    train = torch.from_numpy(data["X_train"].reshape(len(data["X_train"]), -1)).float()
    nearest = np.empty(SAMPLE)
    for i in range(0, SAMPLE, 250):
        q = torch.from_numpy(Xs[i: i + 250].reshape(-1, 1024)).float()
        # Mean absolute difference = L1 distance / 1024.
        nearest[i: i + 250] = torch.cdist(q, train, p=1).min(1).values.numpy() / 1024

    from common import DEVICE, DyslexiaCNN
    from config import MODELS_DIR
    ckpt = torch.load(MODELS_DIR / "dyslexia_cnn.pt", map_location=DEVICE)
    model = DyslexiaCNN(ckpt["config"]["width"], ckpt["config"]["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    correct = predict_proba(model, Xs).argmax(1) == ys

    near = nearest < NEAR_COPY
    out = {"sample": SAMPLE, "threshold_mad": NEAR_COPY, "near_copy_share": float(near.mean()),
           "acc_near_copies": float(correct[near].mean()) if near.any() else None,
           "acc_novel": float(correct[~near].mean()) if (~near).any() else None,
           "near_copy_share_by_class": {c: float(near[ys == k].mean()) for k, c in enumerate(CLASSES)}}
    (RESULTS_DIR / "leakage_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, c in enumerate(CLASSES):
        ax.hist(nearest[ys == k], bins=50, range=(0, 40), alpha=0.6, label=c)
    ax.axvline(NEAR_COPY, color="black", ls="--", lw=1, label="near-copy threshold")
    ax.set_xlabel("Distance to closest training image (mean |pixel difference|)")
    ax.set_ylabel("Test images")
    ax.set_title(f"Test-to-train similarity ({SAMPLE} sampled test images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "leakage_check.png", dpi=150)


if __name__ == "__main__":
    main()
