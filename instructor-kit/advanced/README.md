# Advanced Machine Learning: Instructor Kit ("Course-in-a-Box")

A ready-to-teach package that lets another instructor deliver **Advanced Machine Learning**, the
second half of the course **Python Machine Learning**, without building it from scratch. The live
course site (<https://rays-python-machine-learning.netlify.app/>) is the learner's textbook: every
lesson has real output under every code block, a guided project, exercises, a quiz and a journal
prompt. This kit is everything the instructor needs around Modules 7 to 13 of that site: explaining
models and checking fairness, clustering and anomaly detection, dimensionality reduction, deep
learning with PyTorch, text and recommendations, MLOps, a churn case study and a capstone.

**Prerequisite:** the companion kit **Machine Learning Foundations** (Modules 0 to 6, 11 sessions)
or equivalent experience: scikit-learn pipelines, train/test splits and cross-validation, metrics
and thresholds, trees and gradient boosting. Session 1 opens with an ungraded refresher on exactly
those skills, and `labs/module-01/l7_0-orientation/diagnostic_starter.py` gives anyone who needs it
a structured catch-up. The two kits are sold and taught separately; together they are 24 sessions
(72 class hours).

**Format:** 13 sessions × 3 hours (**39 class hours**), two lessons per session for Sessions 1 to
11; Sessions 12 and 13 are the capstone studio (lesson 13.2, 480 minutes for a solo learner). Run it
weekly (13 weeks), twice a week (about 7 weeks), or as intensive days (two sessions a day). In the
file layout a session is called a "module": `source/module-07/` is Session 7. In prose, "Module 10"
always means a course module (deep learning).

Learners start each lesson project in the room and finish it at home, so "leave with" below means
"have started in class and finish before the next session".

| Session | Lessons | Learners leave with |
|---------|---------|---------------------|
| S1 · Interpreting Models and SHAP | 7.1–7.2 (+ welcome, environment check incl. PyTorch, Foundations refresher) | A READY environment (images downloaded at home with `--download`); `churn_drivers.csv` (grouped permutation importance and partial-dependence turning points); `call_list_reasons.csv` (SHAP reason codes an agent can read aloud) |
| S2 · Local Explanations, Fairness and K-Means | 7.3–8.1 | `fairness_audit.csv` (LIME fidelity and stability, recall by age group with a permutation test); `customer_groups.csv` (named K-Means groups) |
| S3 · Choosing k and Hierarchical Clustering | 8.2–8.3 | `customer_segments.csv` (five named segments, k defended with four kinds of evidence); `product_families.csv` (a correlation-distance family tree) |
| S4 · Density-Based Clustering and Gaussian Mixtures | 8.4–8.5 | `customer_density_segments.csv` (HDBSCAN core segments and misfits); `customer_soft_segments.csv` (mixture probabilities, borderline and atypical customers) |
| S5 · Anomaly Detection and PCA | 8.6–9.1 | A fraud review queue (Isolation Forest, LOF and a mixture against boosting at a 165-review budget, in dollars); `seasonal_scores.csv` (PCA fingerprints of weekly demand) |
| S6 · LDA, t-SNE and UMAP | 9.2–9.3 + **Checkpoint 1** (30 min) | `tier_map.csv` (LDA axes); `customer_map.csv` (t-SNE and UMAP maps read by the rules); a scored checkpoint on Modules 7 to 9 |
| S7 · Neural Networks and PyTorch | 10.1–10.2 (+ Checkpoint 1 debrief) | `churn_net.npz` (a network written in NumPy); `churn_mlp.pt` + `churn_preprocess.joblib` (the same model in PyTorch, compared honestly with boosting); `training.py` |
| S8 · CNNs and Sequence Models | 10.3–10.4 | `fashion_cnn.pt` (a Fashion-MNIST CNN, test accuracy about 0.89); `sales_lstm.pt` (a calendar-aware LSTM forecast against boosting on lag features) |
| S9 · Transfer Learning and Text Classification | 10.5–11.1 | `landcover_resnet18.pt` (EuroSAT, frozen features then fine-tuning, test about 0.93); `review_follow_up.csv` (a TF-IDF bigram triage model) |
| S10 · Recommendations and Reproducibility | 11.2–12.1 | `picked_for_you.csv` (item-kNN against best sellers, hit rate at 10); `churn_final.skops` + `churn_final.json` + `mlflow.db` (a tracked, fingerprinted, reproduced model) |
| S11 · Serving, Monitoring and the Churn Case Study | 12.2–13.1 + capstone choice | `serve.py`, `schemas.py`, `monitor.py` and `drift_report.csv`; `campaign_list.csv` and a retention memo in dollars; a chosen capstone problem (A to E) and a one-paragraph plan |
| S12 · Capstone Kickoff and Studio I | 13.2 | `capstone.ipynb` through Milestone 4: plan, data checks, split, baselines, a validation harness and a first pipeline; the test set still unopened |
| S13 · Capstone Studio II and Showcase | 13.2 | The test set opened once, an interval, error analysis, a model card and a showcase talk. Submit within a week |

**Why this shape (and the timing trade-off):** the lessons were written and timed honestly for a
solo learner at 90 to 120 minutes each (the capstone at 480), so two of them come to 185 to 235 solo
minutes against about 150 room minutes once the warm-up, break and wrap-up are out. They do not fit
whole, and the kit does not pretend they do. The room protects what is hardest alone: the **mental
model**, a **live-coding demo** learners type along with, the **guided lab** (new practice on the
same data, not the lesson's project), and **starting** each lesson project together. Home gets the
rest: finishing the two projects, the lesson's own exercises, quiz and journal, about two to three
hours between sessions (the capstone sessions ask about the same, plus the final write-up). Every
facilitator guide says, for its session, what is done in the room, what goes home, what to protect
and what to cut first if you run long.

Every session runs 0:00 to 3:00: warm-up (five retrieval questions and a homework check, 10 min;
Session 1 opens with a 30-minute welcome, environment check and refresher instead, Session 7 with a
15-minute warm-up that includes the Checkpoint 1 debrief), Lesson A (frame, demo, guided lab,
project start), a 10-minute break, Lesson B, and a 10-minute wrap-up (exit ticket, homework, the next
session's full bring list). Every wrap-up's bring list was checked against the next session's
warm-up and "Before the session" list, including every download and every file a later session
reuses.

### Primary documents: the combined books (start here)

| File | Audience | Contents |
|------|----------|----------|
| `participant-workbook.{html,pdf}` | Learner | The complete workbook: all 13 sessions as chapters |
| `facilitator-guide.{html,pdf}` | Instructor | The complete teaching manual: all 13 sessions |
| `answer-key.{html,pdf}` | Instructor | Every self-check answered with reasons, lab solution notes, the lesson pages' quiz answers, warm-up, exit-ticket and checkpoint answers (confidential) |

(`render_pdfs_ml.py` prints each book's page count when it renders the PDFs.)

**Self-check items:** 12 per session, 13 × 12 = **156**, every one answered in the answer key with
why each wrong option is wrong. The workbook quiz, the answer key and the guide's compact answers
were generated from one item list per session and are checked to match word for word. Every code
output in the guides, workbooks, keys and decks comes from a real run.

The editable per-session docs live in `source/module-NN/`. **If you edit a session's doc, run
`python3 build-combined.py` to regenerate the three books, then re-render their PDFs** (see
"Editing and building").

### Kit-wide documents

| File | Audience | Purpose |
|------|----------|---------|
| `setup-guide.html` | Learner | Before-you-begin sheet: an "am I ready?" list of Foundations skills; Python 3.12+, a `.venv`, the CPU build of PyTorch first on Linux, `python -m pip install -r requirements.txt`, the data zip, the image downloads with sizes (`check_environment.py --download`, about 175 MB), cache locations, GPU note, Windows/macOS/Linux commands, common problems, the USB-stick fallback. Send it a week before Session 1 with `labs/module-01/l7_0-orientation/check_environment.py` attached |
| `final-assessment.html` | Instructor | The capstone (lesson 13.2): the brief, where the 480 minutes go, the rubric, scoring guidance and sheet, submission rules, and answer notes for each of the five problems A to E with the real reference numbers (confidential) |
| `sell-sheet.html` | Prospective instructors | Marketing one-pager: what's inside, who it's for, how it pairs with Foundations, license tiers |
| `README.md` · `LICENSE.md` | — | This overview and the tiered license template (with the third-party data and model terms) |

Slide decks: one per session in `slides/` (`module-NN-<slug>.html`). At v1.0 the 13 decks hold 401 slides.
Labs: 50 starter/solution pairs across `labs/module-01/` … `labs/module-13/`, plus 18 `debug_*.py`
"find the bug" handouts, two `prepare_*.py` fixture builders, the Session 1 environment check and
refresher diagnostic, and the Session 6 checkpoint.

The 13-session **syllabus** lives in the course root, two levels up, so it stays public:
`syllabus-advanced.html` (screen, light/dark), `syllabus-advanced-print.html` (print-first), and
`Advanced-Machine-Learning-Syllabus-13-sessions.pdf`. They are generated (see "Editing and building").

**Still to do before selling:** the `LICENSE.md` is a plain-language **template**; have it reviewed
by a lawyer before commercial distribution (including its third-party section on the image datasets
and the ImageNet-trained weights). Pricing in the sell sheet and the license is a suggested anchor
(the bracketed prices are placeholders set at point of sale).

---

## Folder layout

```
instructor-kit/advanced/
├─ participant-workbook.{html,pdf}   ← canonical hand-out books (all 13 sessions)
├─ facilitator-guide.{html,pdf}
├─ answer-key.{html,pdf}
├─ setup-guide / final-assessment / sell-sheet  (.html + .pdf)
├─ slides/            one self-contained deck per session: module-NN-<slug>.html (NN = 01 … 13)
├─ labs/              labs_paths.py, check_labs.py, README.md, and module-01/ … module-13/:
│                     per lesson a folder lM_N-<slug>/ with *_starter.py + *_solution.py pairs,
│                     debug_*.py handouts, prepare_*.py fixture builders, and _fixtures/
│                     (module-08: two small Fashion-MNIST models; module-09: EuroSAT features)
├─ source/            module-01/ … module-13/ editable per-session docs (build the books)
├─ kit.css            shared print-first stylesheet
├─ build-combined.py  regenerate the 3 books from source/
└─ README.md · LICENSE.md
```

Every document is print-ready (a **Print / Save as PDF** button and a tuned print layout, US
Letter). Slide decks are single files and need no server (arrow keys / space to advance).

The whole `instructor-kit/` folder is kept **off the live site**: the course root's `_redirects`
returns 404 for `/instructor-kit/*`. It lives in the repository only.

### How to teach from it
1. Send learners the **setup guide** and `check_environment.py` a week before Session 1; ask each
   learner to reply with its last line. `READY for Advanced Machine Learning` means everything,
   images included; `READY for Sessions 1 to 4` means the images still have to be downloaded at home
   (`python labs/module-01/l7_0-orientation/check_environment.py --download`, about 175 MB, once)
   before Session 5. Learners who took Foundations already have the environment and only need the
   download.
2. Before each session, read that session's **facilitator guide**: "Before the session" (files,
   downloads and caches, known environment traps, and for Sessions 7 to 9 a CPU-time plan), the
   0:00–3:00 agenda, each lesson's live-demo script (the exact code to type, all of it run), the
   three mistakes to expect, and "If you run long or short".
3. Present from the session's deck in **`slides/`**; live-code in JupyterLab or VS Code.
4. Hand out the `labs/` folder **without the `*_solution.py` files** (keep `labs_paths.py` and every
   `_fixtures/` folder). Learners keep it inside their `pinecrest-ml` folder next to `data/`, so every
   lab finds the data by itself (`labs_paths.py`; or set `PINECREST_ML_DATA`).
5. Learners work in the **participant workbook** and keep a **learning journal**.
6. Check self-checks, warm-ups and exit tickets with the **answer key**; score Checkpoint 1 and the
   capstone with their rubrics.

## The data

Everything runs on **Pinecrest Outfitters**, a fictional outdoor-gear retailer. Every person,
product and figure is invented, so learners can dig into "customer records" with no privacy concerns.

- `data/pinecrest-ml-data.zip` (on the course site, 1.4 MB): `customers_churn.csv` (6,000
  customers), `weekly_demand.csv`, `transactions.csv` (50,000 payments, about 1% fraud),
  `reviews.csv` (4,000), `products_ml.csv`, `interactions.csv` and `daily_store_sales.csv`.
  `data/README.md` documents every file and column; `data/make_ml_data.py` rebuilds identical files.
- **Two public image datasets and two pretrained networks**, downloaded by torchvision on each
  learner's machine (never shipped in this kit): Fashion-MNIST (about 30 MB download, Lesson 9.3 and
  Module 10), EuroSAT RGB (94 MB, Lesson 10.5), and the ImageNet-trained ResNet-18 (45 MB) and
  MobileNetV3-small (10 MB) weights. About 175 MB in all, about 330 MB on disk. Their licenses and
  the ImageNet terms are in `LICENSE.md`, "Third-party data and models".

## Deep learning on a CPU (Sessions 6 to 9)

Nothing in the course needs a GPU, and every number in the kit comes from the CPU code at 4 threads
(the same settings as the lesson pages). Measured on the build machine:

| Session | What is slow | Plan in the guide |
|---------|--------------|-------------------|
| S6 | t-SNE / UMAP on Fashion-MNIST samples (the image lab about 30 s); first `import umap` | Sample 1,000 to 2,000 rows, PCA to 50 first; warm the import before class |
| S7 | Nothing over a few seconds (NumPy and PyTorch churn networks, early stopping) | Everything trains live; the rehearsal's `churn_net.pt` is the fallback |
| S8 | SmallCNN 5 epochs about 22 s; the project CNN about 45 s (up to 2 min on older laptops) | Launch the project run before the break; `labs/module-08/_fixtures/` holds the two models the labs load, so no lab trains a CNN from scratch |
| S9 | A from-scratch CNN on EuroSAT about 37 s, ResNet-18 features about 26 s, fine-tuning 15 to 30 s per epoch | Run All on the demo notebook during the warm-up; the lab loads `labs/module-09/_fixtures/eurosat_resnet18_feats.npz` (2.4 MB) and runs in about 5 s; stage 2 fine-tuning goes home |

Every lab solution runs in under 90 seconds at 4 threads. On another machine PyTorch results may
differ in the last digits; the docs label those outputs "yours may vary slightly" and say why. A
learner with an NVIDIA GPU may try `.to("cuda")` at home; in the room keep everyone on the CPU so
outputs match the pages and the neighbors. Rebuild a fixture with
`python labs/module-08/l10_3-cnn/prepare_fashion_checkpoints.py` or
`python labs/module-09/l10_5-transfer-learning/prepare_eurosat_features.py` (needs the image data).

## Teaching offline

The venue's internet is only needed by learners who have not installed yet; installation and the
image downloads are homework. Put these on a **USB stick** for the room: `requirements.txt`,
`pinecrest-ml-data.zip`, the `labs/` folder without solutions, the `FashionMNIST/` and `eurosat/`
folders (copy them into a learner's `data/`), and the two `.pth` weight files (copy them into
`~/.cache/torch/hub/checkpoints/`; Windows `%USERPROFILE%\.cache\torch\hub\checkpoints`). The setup
guide tells learners the same. With `LABS_OFFLINE=1` every lab refuses to download and stops with a
clear message instead (`LABS_OFFLINE=1 python <lab>.py`; PowerShell:
`$env:LABS_OFFLINE = "1"; python <lab>.py`). Session 11's API runs on `127.0.0.1`, no internet.

## Checkpoint 1 and the capstone

- **Checkpoint 1 (Session 6, 2:20, 30 minutes):** an individual, open-notebook task on Modules 7 to
  9 (`labs/module-06/checkpoint-1/`): permutation importance on held-out rows, a Gaussian mixture
  segmentation, PCA against LDA. Five-row rubric and model answer in the Session 6 answer key; the
  debrief opens Session 7.
- **Capstone (lesson 13.2):** in the **Session 11** wrap-up each learner chooses one of five
  problems (A demand forecasting, B fraud review queue, C review triage, D recommendations, E
  customer segments; or their own, approved in writing) and writes a one-paragraph plan.
  **Session 12** is the kickoff and Studio I (plan clinic, data checks, baselines, a harness, a
  first pipeline) and **Session 13** is Studio II (the test set opened once, an interval, error
  analysis, the model card) and the showcase. Learners submit a notebook that reruns from a fresh
  kernel, a model card, figures, a skops model, `results.json` and `requirements.txt` within a week.
  `final-assessment.html` holds the rubric and coaching notes per problem, with the honest negative
  results a good project may report.

## Requirements
- **Python 3.12 or newer** on Windows, macOS or Linux, with a virtual environment, and the course
  `requirements.txt` (NumPy 2.5, pandas 3.0, scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, SHAP 0.53,
  LIME 0.2, umap-learn 0.5, PyTorch 2.14 + torchvision 0.29 (CPU build), MLflow 3.17, skops 0.16,
  FastAPI 0.143, uvicorn, httpx, JupyterLab 4.6).
- **JupyterLab or VS Code**. A laptop from the last five or so years with 8 GB of RAM; no GPU.
- About 3 GB of disk (the environment about 2 GB, image data about 330 MB). No paid tools, accounts
  or API keys.

## Editing and building

```bash
cd instructor-kit/advanced
python3 build-combined.py                 # regenerate the 3 books from source/ (skips missing sessions)
python3 labs/check_labs.py                # run every starter and solution offline (LABS_OFFLINE=1),
                                          # 4 threads, in temp copies; 100 files, about 5 minutes
python3 labs/check_labs.py module-08      # one session
```

`check_labs.py` uses `--python`, `$LABS_PYTHON` or the course venv; it fails a lab that crashes,
prints a traceback, runs past 150 s, downloads anything or writes outside its temp copy.

The author's tools in `~/projects/_kit_tools/python_ml/` (not shipped with a licensed kit):

```bash
python3 check_kit_ml.py advanced --all    # per-session consistency: one sheet/foot, agenda sums to 180
                                          # with one Break (and the Session 6 checkpoint), quiz identical
                                          # across workbook/key/guide, lab paths exist, titles match the
                                          # course, decks fit 1280x720, nothing wider than 390 px
python3 kit_generators/B07/gen_B07.py     # regenerate one session's 3 docs + deck from its generator
                                          # (every code block rerun or read from its cache);
                                          # never hand-edit the generated HTML
python3 gen_syllabus_ml.py                # rebuild ../../syllabus-advanced.html + -print.html from the
                                          # agendas in source/module-NN/facilitator-guide.html
python3 render_pdfs_ml.py advanced        # render every kit PDF + the syllabus PDF, print page counts,
                                          # and phone-check every page at 390 px
```

Or open any document and use **Print / Save as PDF** (Letter, 0.5 in margins, background graphics on).

---

## License

See `LICENSE.md`. Short version: licensed to a **single instructor or organization** to teach the
course; **not** for resale; the facilitator guide, answer key and final assessment are **not** for
learner distribution. The Pinecrest dataset is part of the licensed materials. Fashion-MNIST,
EuroSAT and the pretrained weights are third-party and are **not** redistributed by the kit:
learners download them from their sources, and the ImageNet-trained weights come with ImageNet's
non-commercial terms, which a commercial training provider should review. Tiers (Solo /
Organization / White-label, plus an optional updates add-on) are set at point of sale.

© 2026 Ray de la Paz. Python Machine Learning · Advanced Machine Learning. All rights reserved.
