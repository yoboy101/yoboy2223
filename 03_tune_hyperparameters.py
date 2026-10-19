"""Step 5: hyperparameter experiments for the CNN.

Each configuration is trained for a short budget (3 epochs on 15% of the training
split) and scored on the validation split. The best configuration by validation
macro-F1 is written to outputs/results/best_config.json for 04_train_cnn.py.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

from common import load_splits, train_model
from config import FIGURES_DIR, RESULTS_DIR

# One-factor-at-a-time around a sensible default, plus the augmentation ablation.
DEFAULT = {"width": 32, "dropout": 0.3, "lr": 1e-3, "batch_size": 256, "use_augmentation": True}
EXPERIMENTS = [
    ("default", {}),
    ("lr=3e-3", {"lr": 3e-3}),
    ("lr=3e-4", {"lr": 3e-4}),
    ("dropout=0.5", {"dropout": 0.5}),
    ("width=16", {"width": 16}),
    ("no augmentation", {"use_augmentation": False}),
]


def main():
    data = load_splits()
    rows = []
    for name, change in EXPERIMENTS:
        cfg = {**DEFAULT, **change}
        print(f"\n=== {name}: {cfg}")
        _, hist = train_model(data, max_epochs=3, patience=10, train_subset=0.15, **cfg)
        best = max(hist, key=lambda r: r["val_macro_f1"])
        rows.append({"experiment": name, **cfg, "val_acc": best["val_acc"], "val_macro_f1": best["val_macro_f1"],
                     "val_loss": best["val_loss"], "seconds": sum(r["seconds"] for r in hist)})
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "tuning_results.csv", index=False)  # save progress

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)
    print(df.to_string(index=False))

    best = df.sort_values("val_macro_f1", ascending=False).iloc[0]
    best_cfg = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in DEFAULT}
    (RESULTS_DIR / "best_config.json").write_text(json.dumps({"chosen_from": best["experiment"], **best_cfg}, indent=2))
    print("Best:", best["experiment"], best_cfg)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = df.sort_values("val_macro_f1")
    colors = ["#2a6f97" if e != best["experiment"] else "#e07a1f" for e in order["experiment"]]
    ax.barh(order["experiment"], order["val_macro_f1"], color=colors)
    lo = order["val_macro_f1"].min()
    ax.set_xlim(max(0, lo - 0.05), 1.0)
    for y, v in enumerate(order["val_macro_f1"]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Validation macro-F1 (3 epochs, 15% of training data)")
    ax.set_title("Hyperparameter experiments (orange = selected)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tuning_results.png", dpi=150)


if __name__ == "__main__":
    main()
