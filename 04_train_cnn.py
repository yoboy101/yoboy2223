"""Step 4 + 6: train the final CNN on the full training split with the selected
hyperparameters, keep the epoch with the best validation macro-F1, and plot the
training curves (loss, accuracy, macro-F1, learning rate)."""
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch

from common import load_splits, train_model
from config import CLASSES, FIGURES_DIR, IMG_SIZE, MODELS_DIR, RESULTS_DIR

DEFAULT = {"width": 32, "dropout": 0.3, "lr": 1e-3, "batch_size": 256, "use_augmentation": True}


def main():
    cfg_file = RESULTS_DIR / "best_config.json"
    cfg = DEFAULT
    if cfg_file.exists():
        saved = json.loads(cfg_file.read_text())
        cfg = {k: saved[k] for k in DEFAULT}
    print("Training with", cfg)

    data = load_splits()
    model, hist = train_model(data, max_epochs=6, patience=3, **cfg)

    torch.save({"state_dict": model.state_dict(), "config": cfg, "classes": CLASSES, "img_size": IMG_SIZE},
               MODELS_DIR / "dyslexia_cnn.pt")
    h = pd.DataFrame(hist)
    h.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    best = h.loc[h["val_macro_f1"].idxmax()]
    print(f"Best epoch {int(best.epoch)}: val acc {best.val_acc:.4f}, val macro-F1 {best.val_macro_f1:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(h.epoch, h.train_loss, "o-", label="train")
    axes[0].plot(h.epoch, h.val_loss, "s-", label="validation")
    axes[0].set_title("Weighted cross-entropy loss")
    axes[1].plot(h.epoch, h.train_acc, "o-", label="train")
    axes[1].plot(h.epoch, h.val_acc, "s-", label="validation")
    axes[1].plot(h.epoch, h.val_macro_f1, "^-", label="validation macro-F1")
    axes[1].set_title("Accuracy / macro-F1")
    axes[2].plot(h.epoch, h.lr, "o-", color="gray")
    axes[2].set_title("Learning rate (cosine decay)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.axvline(best.epoch, ls="--", color="#e07a1f", lw=1)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("CNN training curves (dashed line = epoch kept)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
