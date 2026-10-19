"""Step 1-3: index the Kaggle Dyslexia Handwriting Dataset, preprocess every image
and build stratified train / validation / test splits.

Dataset: https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset
Layout after extracting the RAR (password is in the file name):
    Gambo/Train/{Normal,Reversal,Corrected}/*.png
    Gambo/Test/{Normal,Reversal,Corrected}/*.png

Preprocessing (applied identically to every image):
  1. convert to 8-bit grayscale (files mix L, RGB and 1-bit modes)
  2. fix polarity so ink is always white on black (a few files are inverted)
  3. crop to the ink bounding box and pad to a square (removes the class-revealing
     differences in canvas size: Normal files are 28x28, Corrected ones 29-31 px)
  4. resize to 32x32
  5. drop exact duplicates, and any Train image that also appears in Test

Output: data/processed/handwriting_32px.npz and data/processed/image_index.csv
"""
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from common import preprocess_array
from config import CLASSES, INDEX_FILE, PROCESSED_FILE, RAW_DIR, SEED, VAL_FRACTION


def find_dataset_root():
    candidates = [p for p in RAW_DIR.rglob("Train") if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No 'Train' folder found under {RAW_DIR}. Extract the Kaggle RAR there first.")
    return candidates[0].parent


def preprocess(path, retries=3):
    """Load and preprocess one file; returns None if it cannot be read."""
    for attempt in range(retries):
        try:
            with Image.open(path) as im:
                return preprocess_array(np.asarray(im.convert("L"), dtype=np.uint8))
        except OSError:
            time.sleep(0.2 * (attempt + 1))  # transient Windows read errors (antivirus locks)
    return None


def main():
    root = find_dataset_root()
    print(f"Dataset root: {root}")
    rows = []
    for split in ("Train", "Test"):
        for label, cls in enumerate(CLASSES):
            for f in sorted((root / split / cls).glob("*.png")):
                rows.append({"path": str(f), "official_split": split, "label": label, "class": cls})
    index = pd.DataFrame(rows)
    print(index.groupby(["official_split", "class"]).size().unstack())

    print(f"Preprocessing {len(index):,} images ...")
    with ProcessPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(preprocess, index["path"], chunksize=1000))
    unreadable = np.array([im is None for im in loaded])
    if unreadable.any():
        print(f"Skipping {unreadable.sum()} unreadable files: {index.loc[unreadable, 'path'].tolist()[:5]}")
    index = index[~unreadable].reset_index(drop=True)
    images = np.stack([im for im in loaded if im is not None])

    # Duplicate handling on the preprocessed pixels.
    index["hash"] = [hashlib.md5(im.tobytes()).hexdigest() for im in images]
    test_hashes = set(index.loc[index.official_split == "Test", "hash"])
    leak = (index.official_split == "Train") & index["hash"].isin(test_hashes)
    dup = index.duplicated(subset=["official_split", "hash"])
    conflicting = index.groupby("hash")["label"].nunique()
    conflicting = index["hash"].isin(conflicting[conflicting > 1].index)
    keep = ~(leak | dup | conflicting)
    stats = {
        "raw_images": len(index) + int(unreadable.sum()),
        "unreadable_files": int(unreadable.sum()),
        "duplicates_within_split": int(dup.sum()),
        "train_images_also_in_test": int(leak.sum()),
        "images_with_conflicting_labels": int(conflicting.sum()),
    }
    index, images = index[keep].reset_index(drop=True), images[keep.values]
    stats["kept_images"] = len(index)
    print(stats)

    # Official Test folder = held-out test set. Train folder -> train + validation (stratified).
    train_idx = index.index[index.official_split == "Train"].to_numpy()
    test_idx = index.index[index.official_split == "Test"].to_numpy()
    tr_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_FRACTION, random_state=SEED, stratify=index.loc[train_idx, "label"]
    )
    index["split"] = "test"
    index.loc[tr_idx, "split"] = "train"
    index.loc[val_idx, "split"] = "val"

    y = index["label"].to_numpy(dtype=np.int64)
    np.savez_compressed(
        PROCESSED_FILE,
        X_train=images[tr_idx], y_train=y[tr_idx],
        X_val=images[val_idx], y_val=y[val_idx],
        X_test=images[test_idx], y_test=y[test_idx],
    )
    index.drop(columns=["hash"]).to_csv(INDEX_FILE, index=False)
    split_counts = index.groupby(["split", "class"]).size().unstack()[CLASSES]
    print(split_counts)
    pd.Series(stats).to_csv(PROCESSED_FILE.parent / "cleaning_stats.csv", header=["count"])
    split_counts.to_csv(PROCESSED_FILE.parent / "split_counts.csv")
    print(f"Saved {PROCESSED_FILE}")


if __name__ == "__main__":
    main()
