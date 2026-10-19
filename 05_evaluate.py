"""Step 7: evaluate the final CNN (and the classical baselines) on the held-out test set.

Produces: test metrics JSON, classification report, confusion matrices, one-vs-rest ROC
curves, a model comparison chart, example images per class and a grid of mistakes.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, f1_score,
                             roc_curve)

from common import DEVICE, DyslexiaCNN, load_splits, predict_proba
from config import CLASSES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED

COLORS = ["#2a6f97", "#e07a1f", "#4f9d69"]


def load_cnn():
    ckpt = torch.load(MODELS_DIR / "dyslexia_cnn.pt", map_location=DEVICE)
    model = DyslexiaCNN(ckpt["config"]["width"], ckpt["config"]["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model


def plot_confusion(cm, title, path, normalise):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    shown = cm / cm.sum(1, keepdims=True) if normalise else cm
    im = ax.imshow(shown, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            txt = f"{shown[i, j]:.1%}" if normalise else f"{cm[i, j]:,}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if shown[i, j] > shown.max() / 2 else "black")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def image_grid(X, labels, title, path, ncols=10):
    n = len(X)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.35))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, lab in zip(np.ravel(axes), X, labels):
        ax.imshow(img, cmap="gray")
        ax.set_title(lab, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    data = load_splits()
    X_test, y_test = data["X_test"], data["y_test"]
    rng = np.random.RandomState(SEED)

    model = load_cnn()
    proba = predict_proba(model, X_test)
    pred = proba.argmax(1)

    metrics = {
        "model": "CNN",
        "test_images": int(len(y_test)),
        "test_acc": accuracy_score(y_test, pred),
        "test_macro_f1": f1_score(y_test, pred, average="macro"),
        "per_class_auc": {},
    }
    report = classification_report(y_test, pred, target_names=CLASSES, digits=4, output_dict=True)
    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=CLASSES, digits=4))
    print(classification_report(y_test, pred, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y_test, pred)
    np.savetxt(RESULTS_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, "CNN confusion matrix (test, counts)", FIGURES_DIR / "confusion_matrix_counts.png", False)
    plot_confusion(cm, "CNN confusion matrix (test, row %)", FIGURES_DIR / "confusion_matrix_normalised.png", True)

    # One-vs-rest ROC curves.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for k, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test == k, proba[:, k])
        a = auc(fpr, tpr)
        metrics["per_class_auc"][cls] = a
        ax.plot(fpr, tpr, color=COLORS[k], label=f"{cls} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # "Dyslexia-indicative" view: Reversal or Corrected vs Normal, used by the screening app.
    risk_true = (y_test != 0).astype(int)
    risk_score = 1 - proba[:, 0]
    fpr, tpr, _ = roc_curve(risk_true, risk_score)
    metrics["binary_indicative_vs_normal"] = {
        "auc": auc(fpr, tpr),
        "accuracy": accuracy_score(risk_true, (risk_score >= 0.5).astype(int)),
    }

    # Baselines on the same test set.
    comparison = {"CNN": {"test_acc": metrics["test_acc"], "test_macro_f1": metrics["test_macro_f1"]}}
    flat = X_test.reshape(len(X_test), -1) / 255.0
    for name in ("logreg_pca100", "random_forest"):
        path = MODELS_DIR / f"baseline_{name}.joblib"
        if path.exists():
            bp = joblib.load(path).predict(flat)
            comparison[name] = {"test_acc": accuracy_score(y_test, bp), "test_macro_f1": f1_score(y_test, bp, average="macro")}
    metrics["comparison"] = comparison
    metrics["per_class"] = {c: report[c] for c in CLASSES}
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(comparison, indent=2))

    names = list(comparison)
    labels = {"CNN": "CNN (ours)", "logreg_pca100": "LogReg + PCA", "random_forest": "Random forest"}
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.18, [comparison[n]["test_acc"] for n in names], 0.36, label="Accuracy", color=COLORS[0])
    ax.bar(x + 0.18, [comparison[n]["test_macro_f1"] for n in names], 0.36, label="Macro-F1", color=COLORS[1])
    for i, n in enumerate(names):
        ax.text(i - 0.18, comparison[n]["test_acc"] + 0.01, f"{comparison[n]['test_acc']:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.18, comparison[n]["test_macro_f1"] + 0.01, f"{comparison[n]['test_macro_f1']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [labels[n] for n in names])
    ax.set_ylim(0, 1.08)
    ax.set_title("Test-set comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Class distribution of the three splits.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, split in enumerate(("train", "val", "test")):
        counts = np.bincount(data[f"y_{split}"], minlength=3)
        ax.bar(np.arange(3) + (k - 1) * 0.27, counts, 0.27, label=split, color=COLORS[k])
    ax.set_xticks(range(3), CLASSES)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution per split (after cleaning)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # Example inputs per class (after preprocessing).
    ex, ex_lab = [], []
    for k, cls in enumerate(CLASSES):
        idx = rng.choice(np.where(y_test == k)[0], 10, replace=False)
        ex += list(X_test[idx])
        ex_lab += [cls] * 10
    image_grid(ex, ex_lab, "Preprocessed test images (32x32) by class", FIGURES_DIR / "sample_images.png")

    # Most confident mistakes.
    wrong = np.where(pred != y_test)[0]
    wrong = wrong[np.argsort(-proba[wrong, pred[wrong]])][:30]
    image_grid(X_test[wrong], [f"T:{CLASSES[y_test[i]][:3]} P:{CLASSES[pred[i]][:3]}" for i in wrong],
               "Most confident test errors (T = true, P = predicted)", FIGURES_DIR / "misclassified.png")


if __name__ == "__main__":
    main()
