"""Model definition, data loading and the training loop shared by the training scripts."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from config import CLASSES, IMG_SIZE, PROCESSED_FILE, SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits():
    d = np.load(PROCESSED_FILE)
    return {k: d[k] for k in d.files}


def preprocess_array(img):
    """Grayscale uint8 letter image of any size -> 32x32 uint8, white ink on black.

    Same steps for training data and for letters cut from a photo in the app:
    fix polarity, crop to the ink, pad to a square, resize.
    """
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    if np.median(border) > 127:
        img = 255 - img
    ys, xs = np.where(img > 40)
    if len(xs):
        img = img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = img.shape
    side = max(h, w) + 4  # 2 px margin on the long side
    canvas = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top: top + h, left: left + w] = img
    out = Image.fromarray(canvas).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(out, dtype=np.uint8)


def to_tensor(X):
    """uint8 (N, 32, 32) -> float tensor (N, 1, 32, 32) scaled to [0, 1]."""
    return torch.from_numpy(X).float().unsqueeze(1) / 255.0


class DyslexiaCNN(nn.Module):
    """Three convolutional blocks (conv-BN-ReLU x2 + max-pool + dropout) and a dense head.

    Input 1x32x32 -> 32x16x16 -> 64x8x8 -> 128x4x4 -> 256 -> 3 classes.
    `width` scales the number of filters, `dropout` the regularisation.
    """

    def __init__(self, width=32, dropout=0.3, n_classes=len(CLASSES)):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(dropout / 2),
            )

        w = width
        self.features = nn.Sequential(block(1, w), block(w, 2 * w), block(2 * w, 4 * w))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(4 * w * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x, max_rot_deg=10.0, max_shift=0.08, max_scale=0.10):
    """Random small rotation, shift and zoom applied per image on the GPU/CPU tensor.

    Horizontal or vertical flips are deliberately NOT used: flipping a normal letter
    produces a reversed letter, which would corrupt the labels of this task.
    """
    n = x.shape[0]
    ang = (torch.rand(n) * 2 - 1) * max_rot_deg * np.pi / 180
    scale = 1 + (torch.rand(n) * 2 - 1) * max_scale
    tx, ty = [(torch.rand(n) * 2 - 1) * max_shift for _ in range(2)]
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1).to(x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


@torch.no_grad()
def predict_proba(model, X, batch_size=2048):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = to_tensor(X[i: i + batch_size]).to(DEVICE)
        out.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()


def class_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES))
    w = len(y) / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_model(data, width=32, dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=256,
                max_epochs=15, patience=3, use_augmentation=True, train_subset=None, verbose=True):
    """Train with AdamW + cosine LR decay, weighted cross-entropy and early stopping on
    validation macro-F1. Returns the best model (by validation macro-F1) and its history."""
    set_seed()
    X_tr, y_tr = data["X_train"], data["y_train"]
    if train_subset:
        idx = np.random.RandomState(SEED).choice(len(X_tr), size=int(train_subset * len(X_tr)), replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    X_val, y_val = data["X_val"], data["y_val"]

    model = DyslexiaCNN(width, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = int(np.ceil(len(X_tr) / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr).to(DEVICE))
    Xt, yt = to_tensor(X_tr), torch.from_numpy(y_tr)

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss, correct = 0.0, 0
        for i in range(0, len(Xt), batch_size):
            b = perm[i: i + batch_size]
            xb, yb = Xt[b].to(DEVICE), yt[b].to(DEVICE)
            if use_augmentation:
                xb = augment(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(b)
            correct += (logits.argmax(1) == yb).sum().item()

        val_proba = predict_proba(model, X_val)
        with torch.no_grad():
            val_loss = loss_fn(torch.from_numpy(np.log(val_proba + 1e-9)), torch.from_numpy(y_val)).item()
        val_pred = val_proba.argmax(1)
        rec = {
            "epoch": epoch,
            "train_loss": total_loss / len(Xt),
            "train_acc": correct / len(Xt),
            "val_loss": val_loss,
            "val_acc": accuracy_score(y_val, val_pred),
            "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(rec)
        if verbose:
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}))
        if rec["val_macro_f1"] > best_f1:
            best_f1, bad_epochs = rec["val_macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping after epoch {epoch} (no val macro-F1 gain for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history
