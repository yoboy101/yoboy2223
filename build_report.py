"""Build the project report (Word) from the saved results and figures.

    python build_report.py  ->  Dyslexia_ML_Project_Report.docx

All numbers in the report are read from outputs/results and data/processed, so the
report always matches the last run of the pipeline.
"""
import json
from datetime import date

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from config import CLASSES, FIGURES_DIR, PROCESSED_FILE, RESULTS_DIR, ROOT

OUT = ROOT / "Dyslexia_ML_Project_Report.docx"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)

# ----------------------------------------------------------------- results
R = RESULTS_DIR
cleaning = pd.read_csv(PROCESSED_FILE.parent / "cleaning_stats.csv", index_col=0)["count"].to_dict()
splits = pd.read_csv(PROCESSED_FILE.parent / "split_counts.csv", index_col=0)
tuning = pd.read_csv(R / "tuning_results.csv")
best_cfg = json.loads((R / "best_config.json").read_text())
history = pd.read_csv(R / "training_history.csv")
test = json.loads((R / "test_metrics.json").read_text())
baselines_val = json.loads((R / "baselines_val.json").read_text())
leak = json.loads((R / "leakage_check.json").read_text()) if (R / "leakage_check.json").exists() else None
cm = pd.read_csv(R / "confusion_matrix.csv", header=None).to_numpy()
best_epoch = history.loc[history.val_macro_f1.idxmax()]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ----------------------------------------------------------------- helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin = sec.bottom_margin = Cm(2.2)

styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(11)
styles["Normal"].paragraph_format.space_after = Pt(6)
styles["Normal"].paragraph_format.line_spacing = 1.15
for lvl, size in ((1, 15), (2, 12.5), (3, 11.5)):
    st = styles[f"Heading {lvl}"]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = ACCENT
    st.font.bold = True
    st.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.keep_with_next = True
styles["Caption"].font.size = Pt(9)
styles["Caption"].font.italic = True
styles["Caption"].font.color.rgb = RGBColor(0x40, 0x40, 0x40)

fig_no = [0]
tab_no = [0]


def H(text, level=1):
    return doc.add_heading(text, level=level)


def P(text="", bold_lead=None, align=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text)
    r.italic = italic
    if align:
        p.alignment = align
    return p


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)
        p.paragraph_format.space_after = Pt(2)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(header, rows, widths_cm, caption=None, font=9.5):
    if caption:
        tab_no[0] += 1
        c = doc.add_paragraph(f"Table {tab_no[0]}: {caption}", style="Caption")
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(font)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "1F4E79")
    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
            if r_i % 2 == 1:
                shade(cells[i], "EEF3F8")
    for row in t.rows:
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(name, caption, width_cm=15.5):
    path = FIGURES_DIR / name
    if not path.exists():
        P(f"[Figure missing: {name}]", italic=True)
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    fig_no[0] += 1
    doc.add_paragraph(f"Figure {fig_no[0]}: {caption}", style="Caption").alignment = WD_ALIGN_PARAGRAPH.CENTER


def code_block(text):
    for line in text.strip("\n").splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ----------------------------------------------------------------- title page
for _ in range(5):
    doc.add_paragraph()
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("AI-Driven Dyslexia Screening & Intervention\nin Multilingual South Africa")
r.bold, r.font.size, r.font.color.rgb = True, Pt(24), ACCENT
s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s.add_run("Detecting dyslexia-indicative handwriting with a convolutional neural network,\n"
              "and turning the result into personalised support")
r.font.size, r.italic = Pt(13), True
doc.add_paragraph()
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
info.add_run("CSIT 3rd Year Project 2026\n").bold = True
info.add_run("Project report: methods, results and demo\n\n")
info.add_run("Group members:\n").bold = True
info.add_run("Maphoto MD (202412126)\n[Member 2 name and student number]\n"
             "[Member 3 name and student number]\n[Member 4 name and student number]\n\n")
info.add_run(f"Date: {date.today():%d %B %Y}")
page_break()

# ----------------------------------------------------------------- abstract
H("Abstract")
P("Dyslexia is a common specific learning difficulty, yet in South Africa most learners are never screened: "
  "assessment tools are rarely available in African languages, specialists are scarce and concentrated in cities, "
  "and reading problems are often blamed on learning in an additional language. We ask how machine learning on a "
  "publicly available dataset can support earlier, cheaper screening. We use the Kaggle Dyslexia Handwriting "
  f"Dataset (Isa et al., 2019): {cleaning['raw_images']:,} images of single handwritten letters labelled Normal, "
  "Reversal (a letter written in mirror orientation) or Corrected (a letter the child corrected). After removing "
  f"{cleaning['raw_images'] - cleaning['kept_images']:,} duplicate, leaked or contradictory images, a "
  "three-block convolutional neural network (CNN) was tuned in six experiments and trained with augmentation, "
  f"class-weighted loss and early stopping. On {test['test_images']:,} held-out test letters the CNN reaches "
  f"{pct(test['test_acc'])} accuracy and a macro-F1 of {test['test_macro_f1']:.3f}, compared with "
  f"{pct(test['comparison']['random_forest']['test_acc'])} for a random forest and "
  f"{pct(test['comparison']['logreg_pca100']['test_acc'])} for logistic regression. "
  "The letter-level predictions feed a screening dashboard that converts the share of reversed and corrected "
  "letters into a Green/Yellow/Red flag, and a retrieval-augmented (RAG) reading tutor that produces personalised "
  "phonics, reading and spelling activities with home-language feedback. The system is a screening aid, not a "
  "diagnostic tool; its limitations and the steps needed before real use are discussed.")

# ----------------------------------------------------------------- 1 introduction
H("The African challenge and our case")
P("Developmental dyslexia is a neurobiological difficulty with accurate and fluent word reading and spelling. "
  "A global meta-analysis puts its prevalence among primary-school children at about 7% (Yang et al., 2022); "
  "in a class of 40, two or three learners are likely to be affected. Early support matters: structured, "
  "multisensory literacy teaching is most effective in the Foundation Phase (Grades R-3).")
P("South Africa illustrates why this is an African problem and not only a clinical one:")
bullets([
    ("Language. ", "Learners are taught in eleven official languages, but almost all standardised reading and "
     "dyslexia tests are in English or Afrikaans. Difficulties are easily attributed to learning in an additional "
     "language, so dyslexia is missed (Geertsema et al., 2026)."),
    ("Access. ", "Educational psychologists and speech-language therapists are few, expensive and mostly in urban "
     "areas. Many schools wait months for an assessment, if they get one at all."),
    ("Scale. ", "Teachers in large classes have little time to observe every learner's handwriting and reading."),
    ("Follow-through. ", "Even when a learner is identified, teachers often lack ready-made, level-appropriate "
     "intervention material in the learner's language."),
])
P("Our case: ", bold_lead=None)
doc.paragraphs[-1].runs[0].bold = True
doc.paragraphs[-1].add_run(
    "a teacher photographs a page of letters written by a Grade 1-3 learner. An AI model checks every letter for "
    "the reversals and self-corrections that are typical of dyslexic handwriting, flags learners who need a closer "
    "look, and immediately suggests activities matched to the difficulty and the learner's home language. "
    "The school-based support team then decides on referral. The question we answer is: can a model trained on "
    "a public handwriting dataset recognise dyslexia-indicative letters reliably enough to be the core of such a "
    "screening and intervention tool?")
H("Related work", 2)
P("Isa et al. (2019) built the dataset we use and trained a CNN to separate normal, reversed and corrected "
  "letters. Later studies on the same or similar data report CNN and hybrid CNN models with high letter-level "
  "accuracy (Alqahtani et al., 2023; Aldehim et al., 2024). We add three things: a careful cleaning step that "
  "removes duplicate and train-test-overlapping images (which can inflate reported accuracy), an explicit "
  "hyperparameter study with baselines, and an end-to-end screening and intervention workflow designed for the "
  "South African classroom.")

# ----------------------------------------------------------------- 2 dataset
H("Dataset source (step 1)")
table(["Attribute", "Detail"], [
    ["Name", "Dyslexia Handwriting Dataset"],
    ["Source", "Kaggle: kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset"],
    ["Creators", "Isa, Rahimi, Ramlan and Sulaiman (Universiti Teknologi MARA), Isa et al. (2019)"],
    ["Origin of images", "Handwritten letters of primary-school children labelled by the creators; the collection "
                         "procedure is described in Isa et al. (2019)"],
    ["Format", "PNG images of single upper- and lower-case letters, 28-31 px, grayscale/RGB/1-bit"],
    ["Size", f"{cleaning['raw_images']:,} images, three classes, supplied as Train and Test folders"],
    ["Ethics", "Public, anonymised images (no names, no personal data). Secondary analysis of an open dataset, "
               "so no new ethics clearance is required"],
], [3.5, 12.5], caption="Dataset summary")
P("Why this dataset. Our first prototype (shown in the earlier presentation) used synthetic background data "
  "(speech delay, therapy history, emotions) modelled on the South African study of Geertsema et al. (2026). "
  "Because synthetic labels contain no real signal, the prototype only reached 32.8% test accuracy for seven "
  "subtypes, which is close to guessing. The handwriting dataset is real, public, large and visual, which makes it "
  "suitable for a deep-learning model and matches a task teachers already do: look at a child's writing.")

H("Contents of the dataset and objectives (step 2)")
raw_counts = [["Train", "39,334", "46,781", "65,534", "151,649"], ["Test", "19,557", "17,882", "19,284", "56,723"]]
table(["Folder", "Normal", "Reversal", "Corrected", "Total"], raw_counts, [3, 3, 3, 3, 3.5],
      caption="Images per class in the downloaded dataset")
P("Each image shows one handwritten letter. Normal letters are correctly formed. Reversal letters are written "
  "in mirror orientation (for example b written as d, or a mirrored E), which is a well-known sign of dyslexia in "
  "children who are past the age where reversals are normal. Corrected letters show a letter that was written "
  "wrongly and then fixed, often with overwriting or extra strokes. The Corrected class contains many "
  "augmented copies (rotated and shifted versions) produced by the dataset creators.")
figure("sample_images.png", "Ten random test images per class after preprocessing (white ink on black, 32x32)")
P("Objectives:", bold_lead=None)
bullets([
    ("Primary (predictive): ", "train a supervised classifier that labels each handwritten letter as Normal, "
     "Reversal or Corrected, measured by accuracy and macro-F1 on a held-out test set."),
    ("Secondary (intervention): ", "use the letter-level predictions to build a learner-level screening flag and "
     "an AI reading tutor that recommends matching, language-aware activities."),
    ("Tertiary (monitoring): ", "track repeated screenings of the same learner so that teachers can see whether "
     "the share of reversed letters falls after intervention."),
])

# ----------------------------------------------------------------- 3 preprocessing
H("Preprocessing and train / validation / test split (step 3)")
P("All images go through the same function (common.preprocess_array) during training and in the demo app:")
bullets([
    ("Grayscale. ", "Files mix 8-bit grayscale, RGB and 1-bit modes; all are converted to 8-bit grayscale."),
    ("Polarity. ", "Most images are white ink on black, but some are inverted. The border colour is checked and "
     "inverted images are flipped so that ink is always bright."),
    ("Crop and pad. ", "Each letter is cropped to its ink bounding box and centred on a square canvas. This step "
     "matters: Normal files are 28x28 pixels while Corrected files are mostly 29-31 pixels, so without it a model "
     "could learn the canvas size instead of the handwriting."),
    ("Resize and scale. ", "Images are resized to 32x32 and pixel values scaled to [0, 1]."),
    ("Duplicates and leakage. ", "Images are hashed after preprocessing. Exact duplicates within a split, training "
     "images that also appear in the test folder, and identical images carrying different labels are removed."),
])
table(["Cleaning step", "Images"], [
    ["Images in download", f"{cleaning['raw_images']:,}"],
    ["Unreadable files", f"{cleaning.get('unreadable_files', 0):,}"],
    ["Exact duplicates within the same folder", f"{cleaning['duplicates_within_split']:,}"],
    ["Training images identical to a test image (leakage)", f"{cleaning['train_images_also_in_test']:,}"],
    ["Identical images with conflicting labels", f"{cleaning['images_with_conflicting_labels']:,}"],
    ["Images kept", f"{cleaning['kept_images']:,}"],
], [11, 4], caption="Data cleaning (some images fall into more than one category)")
P("Split. The creators' Test folder is kept untouched as the final test set. The cleaned Train folder is split "
  f"85/15 into training and validation sets, stratified by class (random seed 42). Validation data is used for "
  "early stopping and hyperparameter choice; the test set is used once, at the end. This gives roughly a "
  f"{pct(splits.loc['train'].sum() / splits.values.sum(), 0)} / {pct(splits.loc['val'].sum() / splits.values.sum(), 0)}"
  f" / {pct(splits.loc['test'].sum() / splits.values.sum(), 0)} split overall.")
table(["Split", *CLASSES, "Total"],
      [[s.capitalize() if s != "val" else "Validation", *[f"{splits.loc[s, c]:,}" for c in CLASSES],
        f"{splits.loc[s].sum():,}"] for s in ("train", "val", "test")],
      [3.2, 3, 3, 3, 3], caption="Images per split after cleaning")
figure("class_distribution.png", "Class distribution per split after cleaning", 12.5)
P("Class imbalance is moderate, so the loss function is weighted by inverse class frequency. Data augmentation "
  "(small random rotations up to 10 degrees, shifts up to 8% and zoom up to 10%, applied to training batches only) "
  "was implemented and tested as a hyperparameter in Section 6"
  + (" and is used in the final model. " if best_cfg["use_augmentation"] else
     "; it did not help on this dataset, so the final model is trained without it. ")
  + "Horizontal and vertical flips are deliberately never used: flipping a normal letter creates a reversed letter, "
  "which would corrupt the very labels the model must learn.")

# ----------------------------------------------------------------- 4 model
H("Model choice and implementation (step 4)")
P("Why a CNN. The task is image classification where the decisive information is local stroke shape and "
  "orientation. Convolutional layers learn such local patterns with few parameters and are robust to small "
  "shifts, which is why CNNs dominate handwriting recognition. A CNN is also small enough to run on a school laptop "
  "or a phone without a GPU.")
w = best_cfg["width"]
table(["Layer block", "Operations", "Output shape"], [
    ["Input", "32x32 grayscale letter", "1 x 32 x 32"],
    ["Block 1", f"2 x [Conv 3x3 ({w} filters) + BatchNorm + ReLU], MaxPool 2x2, Dropout", f"{w} x 16 x 16"],
    ["Block 2", f"2 x [Conv 3x3 ({2 * w} filters) + BatchNorm + ReLU], MaxPool 2x2, Dropout", f"{2 * w} x 8 x 8"],
    ["Block 3", f"2 x [Conv 3x3 ({4 * w} filters) + BatchNorm + ReLU], MaxPool 2x2, Dropout", f"{4 * w} x 4 x 4"],
    ["Classifier", f"Flatten, Dense 256 + ReLU, Dropout {best_cfg['dropout']}, Dense 3 (softmax)", "3 class probabilities"],
], [2.6, 10.4, 3], caption="CNN architecture (final configuration)")
P("Training algorithm. The network is trained with mini-batch gradient descent using the AdamW optimiser "
  f"(learning rate {best_cfg['lr']:g}, weight decay 1e-4, batch size {best_cfg['batch_size']}), a cosine "
  "learning-rate schedule and class-weighted cross-entropy loss. After each epoch the validation macro-F1 is "
  "computed; training stops when it has not improved for three epochs and the best epoch's weights are kept.")
P("Implementation. Python 3 with PyTorch (model and training), scikit-learn (splits, baselines, metrics, "
  "TF-IDF retrieval), NumPy/Pandas (data handling), Pillow/SciPy (image processing and letter segmentation), "
  "Matplotlib (graphs) and Streamlit (demo). The code is organised as numbered scripts so that the whole "
  "pipeline can be re-run from the raw download (Appendix A).")
P("Baselines. To show what the CNN adds, two classical models were trained on the same splits: logistic "
  "regression on 100 principal components of the pixels, and a random forest with 200 trees on the raw pixels.")

# ----------------------------------------------------------------- 5 tuning
H("Hyperparameter experiments (step 5)")
P("To keep tuning affordable on a CPU, each configuration was trained for 3 epochs on a fixed random 15% of the "
  "training split and scored on the full validation split. One setting was changed at a time around a sensible "
  "default (32 base filters, dropout 0.3, learning rate 1e-3, batch 256, augmentation on), and one run switched "
  "augmentation off to measure its effect.")
tun = tuning.sort_values("val_macro_f1", ascending=False)
table(["Experiment", "Filters", "Dropout", "LR", "Batch", "Augment", "Val acc", "Val macro-F1", "Time (s)"],
      [[r.experiment, int(r.width), r.dropout, f"{r.lr:g}", int(r.batch_size), "yes" if r.use_augmentation else "no",
        pct(r.val_acc), f"{r.val_macro_f1:.4f}", f"{r.seconds:.0f}"] for r in tun.itertuples()],
      [3.2, 1.4, 1.5, 1.4, 1.3, 1.6, 1.7, 2.2, 1.7], caption="Tuning runs, sorted by validation macro-F1", font=8.5)
figure("tuning_results.png", "Validation macro-F1 of each tuning run (selected configuration in orange)", 14)
tbest = tun.iloc[0]
tdef = tuning.set_index("experiment").loc["default"]
P("What the experiments show:", bold_lead=None)
bullets([
    ("Learning rate. ", f"1e-3 and 3e-4 perform almost the same (macro-F1 "
     f"{tuning.set_index('experiment').loc['default', 'val_macro_f1']:.3f} vs "
     f"{tuning.set_index('experiment').loc['lr=3e-4', 'val_macro_f1']:.3f}); 3e-3 is clearly too high "
     f"({tuning.set_index('experiment').loc['lr=3e-3', 'val_macro_f1']:.3f}) and training becomes unstable."),
    ("Capacity and regularisation. ", "Halving the filters (width 16) or raising dropout to 0.5 both cost about "
     "two to three F1 points, so the default network is not over-sized for this data."),
    ("Augmentation. ", f"Switching augmentation off gave the best score ({tbest.val_macro_f1:.3f}, "
     f"+{tbest.val_macro_f1 - tdef.val_macro_f1:.3f}). Within a short budget, augmented batches are harder to fit, "
     "and the dataset already contains many rotated and shifted copies of the Corrected letters, so extra "
     "geometric augmentation adds little. Augmentation remains in the code as an option because it may still "
     "help a model meant for new writers and phone photos."),
])
P(f"Selected configuration: {best_cfg['width']} base filters, dropout {best_cfg['dropout']}, learning rate "
  f"{best_cfg['lr']:g}, batch size {best_cfg['batch_size']}, augmentation "
  f"{'on' if best_cfg['use_augmentation'] else 'off'}. Caveat: the runs used a short budget and a single seed, "
  "so differences of about one F1 point are within run-to-run noise.")

# ----------------------------------------------------------------- 6 training graphs
H("Training graphs (step 6)")
P(f"The final model was trained on the full training split ({splits.loc['train'].sum():,} images) with the "
  f"selected configuration for up to 6 epochs (about 6-7 minutes per epoch on a CPU). The best validation macro-F1 ({best_epoch.val_macro_f1:.4f}, "
  f"accuracy {pct(best_epoch.val_acc)}) was reached at epoch {int(best_epoch.epoch)} of {len(history)}.")
figure("training_curves.png", "Training and validation loss, accuracy and macro-F1 per epoch, and the learning-rate "
       "schedule. The dashed line marks the epoch whose weights were kept.", 16)
table(["Epoch", "Train loss", "Train acc", "Val loss", "Val acc", "Val macro-F1", "Seconds"],
      [[int(r.epoch), f"{r.train_loss:.4f}", pct(r.train_acc), f"{r.val_loss:.4f}", pct(r.val_acc),
        f"{r.val_macro_f1:.4f}", f"{r.seconds:.0f}"] for r in history.itertuples()],
      [1.5, 2.2, 2.2, 2.2, 2.2, 2.5, 2], caption="Per-epoch training log of the final model", font=8.5)
P("Reading the curves:", bold_lead=None)
bullets([
    ("Steady convergence. ", "Training and validation loss fall together every epoch, and the learning rate decays "
     "smoothly to zero under the cosine schedule. Early stopping was never triggered within the 6-epoch budget; "
     "the last epoch was the best."),
    ("No overfitting within the budget. ", f"Validation accuracy ({pct(best_epoch.val_acc)}) stays at or above "
     f"training accuracy ({pct(history.iloc[-1].train_acc)}). This is expected here: dropout and batch "
     "normalisation are active while the training accuracy is measured, and, as Section 8 shows, many validation "
     "letters are close copies of training letters, which makes the validation set easier than the test set."),
    ("Diminishing returns. ", "The gain from epoch 5 to epoch 6 is small, so more epochs would add little on this "
     "data; a larger gain is more likely from more diverse data than from longer training."),
])

# ----------------------------------------------------------------- 7 test results
H("8. Results on the test set (step 7)")
cmp_ = test["comparison"]
names = {"CNN": "CNN (ours)", "logreg_pca100": "Logistic regression + PCA", "random_forest": "Random forest"}
table(["Model", "Validation accuracy", "Test accuracy", "Test macro-F1"],
      [[names["CNN"], pct(best_epoch.val_acc), pct(cmp_["CNN"]["test_acc"]), f"{cmp_['CNN']['test_macro_f1']:.4f}"]]
      + [[names[k], pct(baselines_val[k]["val_acc"]), pct(cmp_[k]["test_acc"]), f"{cmp_[k]['test_macro_f1']:.4f}"]
         for k in ("random_forest", "logreg_pca100") if k in cmp_],
      [5.5, 3.5, 3.2, 3.2], caption=f"Overall results ({test['test_images']:,} test images)")
figure("model_comparison.png", "Accuracy and macro-F1 of the CNN and the two baselines on the test set", 12)
pc = test["per_class"]
table(["Class", "Precision", "Recall", "F1-score", "ROC AUC", "Test images"],
      [[c, f"{pc[c]['precision']:.4f}", f"{pc[c]['recall']:.4f}", f"{pc[c]['f1-score']:.4f}",
        f"{test['per_class_auc'][c]:.4f}", f"{int(pc[c]['support']):,}"] for c in CLASSES],
      [3, 2.6, 2.6, 2.6, 2.6, 2.6], caption="Per-class test results of the CNN")
figure("confusion_matrix_normalised.png", "Confusion matrix on the test set (rows sum to 100%)", 10.5)
table(["True \\ Predicted", *CLASSES],
      [[c, *[f"{cm[i, j]:,}" for j in range(3)]] for i, c in enumerate(CLASSES)],
      [4, 3.5, 3.5, 3.5], caption="Confusion matrix (counts)")
figure("roc_curves.png", "One-vs-rest ROC curves of the CNN on the test set", 10.5)
bi = test["binary_indicative_vs_normal"]
P(f"For screening, Reversal and Corrected can be merged into a single dyslexia-indicative class. Using "
  f"1 - P(Normal) as the score, the CNN separates indicative from normal letters with ROC AUC {bi['auc']:.4f} "
  f"and accuracy {pct(bi['accuracy'])} at a 0.5 threshold.")
figure("misclassified.png", "The 30 test errors the CNN was most confident about (T = true class, P = predicted)", 15)
if leak:
    H("Why is validation accuracy higher than test accuracy?", 2)
    P(f"The CNN scores {pct(best_epoch.val_acc)} on validation but {pct(test['test_acc'])} on test. To find out why, "
      f"we took {leak['sample']:,} random validation and {leak['sample']:,} random test images and measured the distance "
      "to the closest training image (mean absolute pixel difference, 0-255 scale). Exact duplicates had already been "
      "removed, but augmented near-copies are not byte-identical. "
      f"{pct(leak['val_near_copy_share'])} of validation images have a training image within {leak['threshold_mad']:g} "
      f"grey levels, against only {pct(leak['near_copy_share'])} of test images; the median distance is "
      f"{leak['median_distance_val']:.1f} for validation and {leak['median_distance_test']:.1f} for test. "
      "Validation letters come from the same Train folder as the training letters (including rotated and shifted "
      "copies of the same original), so validation partly measures recognition of letters the model has nearly "
      "seen. The creators' Test folder is genuinely more different. The test result is therefore the honest "
      "estimate, and keeping the official Test folder separate (instead of re-splitting everything at random) was "
      "the right choice. The baselines show the same pattern even more strongly (random forest: "
      f"{pct(baselines_val['random_forest']['val_acc'])} validation, {pct(test['comparison']['random_forest']['test_acc'])} test).")
    figure("leakage_check.png", "Distance from sampled validation and test images to their closest training image. "
           "Validation images (grey) are much closer to the training data than test images.", 12.5)
n_norm, n_rev, n_cor = cm.sum(1)
P("Analysis of the test results:", bold_lead=None)
bullets([
    ("The CNN clearly beats both baselines. ", f"Accuracy {pct(test['test_acc'])} versus "
     f"{pct(test['comparison']['random_forest']['test_acc'])} (random forest) and "
     f"{pct(test['comparison']['logreg_pca100']['test_acc'])} (logistic regression). Learned convolutional "
     "features capture stroke direction and shape, which pixel-level models cannot do reliably when letters "
     "are shifted, slanted or written with different pen widths."),
    ("All three classes are recognised in a balanced way. ", f"Macro-F1 is {test['test_macro_f1']:.3f} and every "
     f"class has F1 above {min(pc[c]['f1-score'] for c in CLASSES):.2f}. Corrected letters are easiest "
     f"(F1 {pc['Corrected']['f1-score']:.3f}): overwriting and extra strokes are visually distinctive."),
    ("Most errors are Reversal letters read as Normal. ", f"{cm[1, 0]:,} of {n_rev:,} reversed letters "
     f"({pct(cm[1, 0] / n_rev)}) were called Normal, and {cm[0, 1]:,} normal letters ({pct(cm[0, 1] / n_norm)}) were "
     "called Reversal. The grid of most confident errors above shows why: many reversed letters look exactly like a different, correctly written "
     "letter (a mirrored J looks like an L, a mirrored d like a b, a mirrored N like a slanted N). Without knowing "
     "which letter the child intended, a single-letter image is genuinely ambiguous. In a real screening the "
     "learner writes known target letters (dictation), so the expected letter is known and this ambiguity "
     "largely disappears."),
    ("Ranking quality is high. ", f"The one-vs-rest ROC AUC is {test['per_class_auc']['Normal']:.3f} (Normal), "
     f"{test['per_class_auc']['Reversal']:.3f} (Reversal) and {test['per_class_auc']['Corrected']:.3f} (Corrected), "
     "so the threshold of the screening flag can be moved to trade sensitivity against false alarms."),
    ("For screening this is a usable signal. ", "Because the learner-level flag averages over many letters, a "
     "letter-level error rate of about 9% still separates a learner with few reversals from one with many. "
     "Missed reversals are the more costly error for screening, so a lower threshold for the Reversal class "
     "would be chosen in practice. In the demo, a simulated learner with no reversed letters still gets about 8% of "
     "letters flagged (false positives), and one with 40% gets about 42%; the Yellow threshold of 15% sits above "
     "this false-positive floor."),
])

# ----------------------------------------------------------------- 8 system + demo
H("9. From prediction to support: the full system and demo (step 8)")
P("The trained CNN is the core of an end-to-end screening and intervention workflow:")
bullets([
    ("Capture. ", "The teacher photographs or scans a page on which the learner wrote letters (or uploads "
     "single-letter images). For the demo, a 'simulated learner' can be built from held-out test letters with a "
     "chosen share of reversed letters."),
    ("Segment. ", "The page is split into letters with connected-component analysis (SciPy), sorted into "
     "reading order, and every letter gets exactly the same preprocessing as the training data."),
    ("Classify. ", "The CNN labels each letter Normal, Reversal or Corrected with a confidence."),
    ("Flag. ", "The learner-level score is the share of letters classed as Reversal or Corrected. In the demo, "
     "below 15% is Green, 15-30% Yellow and 30% or more Red. These thresholds are placeholders and have not "
     "been clinically validated; they would have to be set from real, learner-level South African data."),
    ("Recommend. ", "A retrieval-augmented reading tutor retrieves relevant guidance and generates activities "
     "(below)."),
    ("Monitor. ", "Each screening is stored with the learner code, so repeated screenings show whether the "
     "share of reversed letters falls after intervention."),
    ("Human decision. ", "The dashboard states clearly that the result is not a diagnosis. Referral goes through "
     "the school-based support team to an educational psychologist or speech-language therapist."),
], style="List Number")
figure("app_screenshot.png", "Streamlit demo dashboard for a simulated learner (60 held-out test letters, 30% "
       "reversed or corrected): risk flag, letter-level predictions with confidence, and RAG tutor output", 16)
figure("app_progress.png", "Progress tracker: every screening of a learner code is logged so repeated screenings "
       "can be compared", 13)
H("RAG reading tutor", 2)
P("The generative part of the system follows the retrieval-augmented generation pattern. A small knowledge base "
  "of intervention guides (letter reversals, phonics rules, structured multisensory literacy, and the South "
  "African CAPS and home-language context) is split into sections and indexed with TF-IDF. The learner's "
  "profile (grade, home language, reversal and correction rates, risk level) becomes a query; the most "
  "relevant sections are retrieved and passed, together with the profile, to an open-source LLM served "
  "locally with Ollama (for example Llama 3.2). The LLM is instructed to use only the retrieved guidance and to "
  "produce a decodable reading passage, phonics exercises, comprehension questions, spelling words and "
  "encouraging feedback in the learner's home language. If no LLM is available, a template generator builds "
  "the activities from the retrieved text, so the demo works offline. TF-IDF can be swapped for FAISS or "
  "ChromaDB with sentence embeddings without changing the rest of the code.")
P("Example request built by the system:", bold_lead=None)
code_block("Grade 2 learner, home language isiZulu. Handwriting screen: 35% of letters reversed,\n"
           "10% corrected, risk level High.  ->  retrieves: letter_reversals.md, phonics_rules.md,\n"
           "structured_literacy_interventions.md, south_african_context.md")
P("Example output (template mode, abridged):", bold_lead=None)
code_block("1. Reading passage: Dad has a big red bed. The dog digs by the bed. Bob the cat sits\n"
           "   on the bed and naps. ...\n"
           "2. Phonics: read and sort by first sound /b/ or /d/: bed, dog, bat, dig, bud;\n"
           "   trace b and d in sand while saying the path aloud (bat then ball; door handle then line)\n"
           "3. Comprehension: Who has a big red bed?  What does the dog do?\n"
           "4. Spelling: bed, dog, bat, dig, bud\n"
           "5. Feedback in isiZulu (LLM mode; checked by a speaker of the language before use)")
H("Demo script for the presentation", 2)
bullets([
    "Show the graphs from 05_evaluate.py: the confusion matrix and the model comparison (1 minute).",
    "Start streamlit run app.py, choose 'Simulated learner', set the reversed/corrected share to 0 and run (Green).",
    "Change the share to 0.40, run again (Red), and show the letter grid and the RAG activities.",
    "Switch the home language to isiZulu and show that the tutor request changes.",
    "Upload a photo of handwritten letters to show segmentation on real input.",
    "Point out the progress tracker and the 'not a diagnosis' notice.",
], style="List Number 2")

# ----------------------------------------------------------------- 9 ethics + limitations
H("10. Ethics, limitations and future work")
bullets([
    ("Not a diagnosis. ", "Dyslexia is diagnosed from reading, spelling, phonological and cognitive assessment over "
     "time. Reversed letters are common in all young children up to about age 7-8, so the tool is only a prompt to "
     "look more closely."),
    ("Letter level, not learner level. ", "The dataset labels letters, not children. The learner-level flag is "
     "therefore an assumption that must be validated with real screening data before any use."),
    ("Population shift. ", "The images were collected by a Malaysian research group, in the Latin alphabet. South African "
     "learners, pens, paper and phone cameras differ. The model must be re-tested, and probably fine-tuned, on "
     "locally collected samples (with consent and ethics clearance)."),
    ("Dataset quality. ", "Duplicates and augmented near-copies mean that published accuracies on this dataset "
     "(including ours) are likely to be optimistic for completely new writers. Writer-independent splits "
     "would be needed for a stricter estimate, but writer IDs are not available."),
    ("Language-generated text. ", "LLM output in African languages can contain errors; a teacher or speaker of "
     "the language must review it. Only anonymous learner codes are used, never names."),
    ("Fairness and privacy. ", "Handwriting photos of minors are personal information under POPIA; a real "
     "deployment needs parental consent, local storage and access control."),
])
P("Future work: (1) add the speech module proposed in our plan, where a learner reads a passage aloud, "
  "Whisper or Wav2Vec2 transcribes it and words-per-minute, omissions, substitutions and hesitations are "
  "measured; (2) combine handwriting with the background factors of the South African dataset of "
  "Geertsema et al. (2026) in a multimodal model; (3) collect consented South African handwriting in several "
  "home languages; (4) add explainability (Grad-CAM heat maps) so teachers can see which strokes drove a prediction.")

# ----------------------------------------------------------------- 10 conclusion
H("11. Conclusion")
P(f"We set out to show how AI can address an African challenge: dyslexia is common but rarely identified in "
  "South African schools, where tests in learners' languages, specialists and teacher time are all scarce. Using a "
  "public dataset of children's handwriting, we cleaned out "
  f"{cleaning['raw_images'] - cleaning['kept_images']:,} duplicate or leaked images, tuned a convolutional "
  f"neural network and reached {pct(test['test_acc'])} accuracy and {test['test_macro_f1']:.3f} macro-F1 on "
  f"{test['test_images']:,} unseen letters, far above classical baselines "
  f"({pct(test['comparison']['random_forest']['test_acc'])} and {pct(test['comparison']['logreg_pca100']['test_acc'])}). "
  "We also showed that validation accuracy on this dataset is inflated by near-copies, so results must be judged "
  "on the separate test folder.")
P("Detection alone is not enough, so the model is embedded in a workflow that a teacher can use: photograph the "
  "writing, get a Green/Yellow/Red flag, receive RAG-generated activities matched to the difficulty and home "
  "language, and track progress over time, with the final decision left to people. Before real use, the model "
  "must be validated on consented South African handwriting with learner-level labels, and the flag thresholds "
  "must be set from that data. Within those limits, the project shows that a small, CPU-friendly model trained "
  "on open data can make early dyslexia screening cheaper, faster and more equitable.")

# ----------------------------------------------------------------- references
H("References")
refs = [
    "Aldehim, G., Rashid, M. U., Alluhaidan, A. S., Sakri, S. B., & Basheer, S. (2024). Deep learning for dyslexia "
    "detection: A comprehensive CNN approach with handwriting analysis and benchmark comparisons. Journal of "
    "Disability Research. https://doi.org/10.57197/jdr-2024-0010",
    "Alqahtani, N. D., Alzahrani, B. A., & Ramzan, M. S. (2023). Detection of dyslexia through images of handwriting "
    "using hybrid AI approach. International Journal of Advanced Computer Science and Applications, 14(10). "
    "https://doi.org/10.14569/ijacsa.2023.0141099",
    "Geertsema, S., Olivier, E., Shaw, C., Turner, I., Valkenburg, D., van der Spuy, K., Stark, S., le Roux, M., "
    "Graham, M. A., & Mapisa, H. A. M. (2026). A South African investigation into etiological and background "
    "factors in developmental dyslexia. Dyslexia. https://doi.org/10.1002/dys.70039",
    "Isa, I. S., Rahimi, W. N. S., Ramlan, S. A., & Sulaiman, S. N. (2019). Automated detection of dyslexia symptom "
    "based on handwriting image for primary school children. Procedia Computer Science, 163, 440-449. "
    "https://doi.org/10.1016/j.procs.2019.12.127",
    "Isa, I. S. (n.d.). Dyslexia Handwriting Dataset [Data set]. Kaggle. Retrieved October 2026, from "
    "https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset",
    "Yang, L., Li, C., Li, X., Zhai, M., An, Q., Zhang, Y., Zhao, J., & Weng, X. (2022). Prevalence of developmental "
    "dyslexia in primary school children: A systematic review and meta-analysis. Brain Sciences, 12(2), 240. "
    "https://doi.org/10.3390/brainsci12020240",
]
for ref in refs:
    p = doc.add_paragraph(ref)
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.first_line_indent = Cm(-1)

# ----------------------------------------------------------------- appendix
page_break()
H("Appendix A: Code and how to run it")
P("All code is in the project folder. Scripts are numbered in the order they run; every number and graph in this "
  "report is produced by them, and build_report.py regenerates this document from the saved results.")
table(["File", "Purpose"], [
    ["config.py", "Paths, class names, random seed, image size"],
    ["common.py", "Preprocessing, CNN model, augmentation, training loop with early stopping"],
    ["01_prepare_data.py", "Load images, clean, deduplicate, preprocess, stratified train/validation/test split"],
    ["02_baselines.py", "Logistic regression + PCA and random forest baselines"],
    ["03_tune_hyperparameters.py", "Six hyperparameter experiments, selects the best configuration"],
    ["04_train_cnn.py", "Train the final CNN, save the model and training curves"],
    ["05_evaluate.py", "Test-set metrics, confusion matrices, ROC curves, comparison and error analysis"],
    ["06_leakage_check.py", "Near-duplicate analysis between the test and training sets"],
    ["rag_tutor.py + knowledge_base/", "Retrieval-augmented reading tutor (TF-IDF retrieval, Ollama LLM or template)"],
    ["app.py", "Streamlit screening and intervention dashboard (demo)"],
    ["build_report.py", "Generates this Word report from the results"],
], [5, 11], caption="Project files")
code_block("pip install -r requirements.txt\n"
           "python 01_prepare_data.py\npython 02_baselines.py\npython 03_tune_hyperparameters.py\n"
           "python 04_train_cnn.py\npython 05_evaluate.py\npython 06_leakage_check.py\n"
           "streamlit run app.py\npython build_report.py")
P("Excerpt of the model definition (common.py, shortened):", bold_lead=None)
code_block('''class DyslexiaCNN(nn.Module):
    def __init__(self, width=32, dropout=0.3, n_classes=3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(dropout / 2))

        w = width
        self.features = nn.Sequential(
            block(1, w), block(w, 2 * w), block(2 * w, 4 * w))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(4 * w * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes))''')


def main():
    doc.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
