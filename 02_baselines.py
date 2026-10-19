"""Step 4 (comparison): classical machine-learning baselines on the same splits.

  * Logistic regression on 100 PCA components of the raw pixels
  * Random forest (200 trees) on the raw pixels

These give a reference point for how much the CNN adds. Fitted on the training
split, scored on the validation split (the test split is only used in 05_evaluate.py).
"""
import json
import time

import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_splits
from config import MODELS_DIR, RESULTS_DIR, SEED


def main():
    data = load_splits()
    X_tr = data["X_train"].reshape(len(data["X_train"]), -1) / 255.0
    X_val = data["X_val"].reshape(len(data["X_val"]), -1) / 255.0
    y_tr, y_val = data["y_train"], data["y_val"]

    models = {
        "logreg_pca100": make_pipeline(
            StandardScaler(), PCA(n_components=100, random_state=SEED),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced", n_jobs=6, random_state=SEED
        ),
    }
    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        results[name] = {
            "val_acc": accuracy_score(y_val, pred),
            "val_macro_f1": f1_score(y_val, pred, average="macro"),
            "fit_seconds": time.time() - t0,
        }
        print(name, json.dumps(results[name]))
        joblib.dump(model, MODELS_DIR / f"baseline_{name}.joblib", compress=3)

    (RESULTS_DIR / "baselines_val.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
