"""Step 8 demo: AI dyslexia screening and intervention dashboard.

    streamlit run app.py

1. Input: a photo or scan of a learner's handwritten letters, single letter images,
   or a simulated learner built from held-out test images.
2. The photo is split into letters (connected components). Each letter gets the same
   preprocessing as the training data and is classified by the trained CNN.
3. The share of letters classed as Reversal or Corrected gives a Green/Yellow/Red flag.
4. The RAG reading tutor retrieves matching intervention guidance and writes activities.
5. Screenings are kept in a progress tracker for the session.

This is a screening aid trained on a public research dataset. It is not a diagnosis.
"""
import io

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from scipy import ndimage

from common import DEVICE, DyslexiaCNN, load_splits, predict_proba, preprocess_array
from config import CLASSES, MODELS_DIR, RESULTS_DIR
from rag_tutor import tutor

LANGUAGES = ["English", "isiZulu", "isiXhosa", "Afrikaans", "Sepedi", "Setswana", "Sesotho",
             "Xitsonga", "siSwati", "Tshivenda", "isiNdebele"]
# Demonstration thresholds on the share of dyslexia-indicative letters. They are NOT clinically validated.
YELLOW_AT, RED_AT = 0.15, 0.30

st.set_page_config(page_title="AI Dyslexia Screening", layout="wide")


@st.cache_resource
def load_model():
    ckpt = torch.load(MODELS_DIR / "dyslexia_cnn.pt", map_location=DEVICE)
    model = DyslexiaCNN(ckpt["config"]["width"], ckpt["config"]["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model


@st.cache_data
def load_test_set():
    d = load_splits()
    return d["X_test"], d["y_test"]


@st.cache_data
def test_metrics():
    path = RESULTS_DIR / "test_metrics.json"
    return pd.read_json(path, typ="series") if path.exists() else None


def segment_letters(gray, min_area=30):
    """Split a page photo into letter crops using connected components."""
    img = np.asarray(gray, dtype=np.uint8)
    if np.median(img) > 127:  # dark ink on light paper -> invert
        img = 255 - img
    mask = img > max(60, img.mean() + img.std())
    mask = ndimage.binary_closing(mask, iterations=1)
    labels, n = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h * w >= min_area and max(h, w) >= 8:
            boxes.append(sl)
    if not boxes:
        return [], []
    # Reading order: group into rows by vertical centre, then left to right.
    med_h = np.median([b[0].stop - b[0].start for b in boxes])
    boxes.sort(key=lambda b: (round(((b[0].start + b[0].stop) / 2) / med_h), b[1].start))
    crops = [img[b] for b in boxes]
    return [preprocess_array(c) for c in crops], boxes


def flag_for(rate):
    if rate >= RED_AT:
        return "RED", "High"
    if rate >= YELLOW_AT:
        return "YELLOW", "Moderate"
    return "GREEN", "Low"


model = load_model()
st.title("AI-Driven Dyslexia Screening & Intervention")
st.caption("Multilingual South Africa | CNN on handwriting + RAG reading tutor")
st.warning("Screening aid only. A model prediction is NOT a diagnosis. Results must be reviewed by a teacher, "
           "the school-based support team and, where needed, an educational psychologist or speech-language therapist.")

m = test_metrics()
if m is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "CNN (PyTorch)")
    c2.metric("Test accuracy", f"{m['test_acc']:.1%}")
    c3.metric("Test macro-F1", f"{m['test_macro_f1']:.3f}")
    c4.metric("Test letters", f"{int(m['test_images']):,}")

with st.sidebar:
    st.header("Learner")
    learner_id = st.text_input("Learner code (no names)", "L-001")
    grade = st.selectbox("Grade", [1, 2, 3, 4, 5, 6, 7], index=1)
    language = st.selectbox("Home language", LANGUAGES, index=1)
    st.header("Input")
    mode = st.radio("Handwriting source", ["Simulated learner (test set)", "Upload page photo", "Upload letter images"])

letters = []
if mode == "Simulated learner (test set)":
    X_test, y_test = load_test_set()
    n = st.sidebar.slider("Letters in sample", 20, 120, 60)
    share = st.sidebar.slider("Share of reversed/corrected letters", 0.0, 1.0, 0.3, 0.05)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 7)
    rng = np.random.RandomState(seed)
    n_bad = int(round(n * share))
    idx = np.concatenate([rng.choice(np.where(y_test == 0)[0], n - n_bad, replace=False),
                          rng.choice(np.where(y_test != 0)[0], n_bad, replace=False)])
    rng.shuffle(idx)
    letters = list(X_test[idx])
    st.info(f"Simulated learner: {n} held-out test letters, true share of reversed/corrected = {n_bad / n:.0%}.")
elif mode == "Upload page photo":
    up = st.file_uploader("Photo or scan of handwritten letters (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up:
        gray = Image.open(io.BytesIO(up.read())).convert("L")
        if max(gray.size) > 1600:
            gray.thumbnail((1600, 1600))
        letters, _ = segment_letters(gray)
        st.image(gray, caption=f"{len(letters)} letters found", width=500)
else:
    ups = st.file_uploader("Single-letter images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    letters = [preprocess_array(np.asarray(Image.open(u).convert("L"), dtype=np.uint8)) for u in ups or []]

# ?demo=1 runs one screening automatically (used for presentation screenshots).
auto_demo = st.query_params.get("demo") == "1" and "last" not in st.session_state
clicked = st.button("Run screening", type="primary", use_container_width=True)
if letters and (clicked or auto_demo):
    proba = predict_proba(model, np.stack(letters))
    pred = proba.argmax(1)
    counts = {c: int((pred == k).sum()) for k, c in enumerate(CLASSES)}
    reversal_rate = counts["Reversal"] / len(pred)
    corrected_rate = counts["Corrected"] / len(pred)
    rate = reversal_rate + corrected_rate
    flag, risk = flag_for(rate)
    st.session_state["last"] = {"pred": pred, "proba": proba, "letters": letters, "counts": counts,
                                "reversal_rate": reversal_rate, "corrected_rate": corrected_rate,
                                "rate": rate, "flag": flag, "risk": risk}
    st.session_state.setdefault("history", []).append(
        {"learner": learner_id, "screening": len(st.session_state.get("history", [])) + 1,
         "letters": len(pred), "indicative_share": round(rate, 3), "flag": flag})

res = st.session_state.get("last")
if res:
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Risk flag", res["flag"])
    r2.metric("Dyslexia-indicative letters", f"{res['rate']:.0%}")
    r3.metric("Reversed", f"{res['reversal_rate']:.0%}")
    r4.metric("Corrected", f"{res['corrected_rate']:.0%}")
    st.caption(f"Flag thresholds (demo, not validated): Yellow from {YELLOW_AT:.0%}, Red from {RED_AT:.0%}.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Letters by predicted class")
        st.bar_chart(pd.Series(res["counts"]))
        st.subheader("Classified letters")
        shown = range(min(40, len(res["pred"])))
        st.image([res["letters"][i] for i in shown], width=56,
                 caption=[f"{CLASSES[res['pred'][i]][:3]} {res['proba'][i, res['pred'][i]]:.0%}" for i in shown])
    with right:
        st.subheader("Personalised activities (RAG tutor)")
        out = tutor({"reversal_rate": res["reversal_rate"], "corrected_rate": res["corrected_rate"],
                     "grade": grade, "language": language, "risk": res["risk"]})
        st.caption(f"Generator: {out['engine']}. Retrieved: "
                   + ", ".join(sorted({p['source'] for p in out['passages']})))
        st.text(out["text"])

    st.subheader("Progress tracker")
    hist = pd.DataFrame(st.session_state.get("history", []))
    st.dataframe(hist, use_container_width=True, hide_index=True)
    mine = hist[hist.learner == learner_id]
    if len(mine) > 1:
        st.line_chart(mine.set_index("screening")["indicative_share"])
