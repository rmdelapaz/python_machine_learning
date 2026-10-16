# Machine Learning Foundations: Instructor Kit ("Course-in-a-Box")

A ready-to-teach package that lets another instructor deliver **Machine Learning Foundations**, the
first half of the course **Python Machine Learning**, without building it from scratch. The live
course site (<https://rays-python-machine-learning.netlify.app/>) is the learner's textbook: every
lesson has real output under each code block, a guided project, exercises, a quiz and a journal
prompt. This kit is everything the instructor needs around it.

**Scope:** course Modules 0 to 6, 21 lessons: the scikit-learn workflow and honest evaluation,
preparing data, linear models, classic classifiers, trees and ensembles, model selection and tuning.
It ends with a final project. The second kit, **Advanced Machine Learning** (`../advanced/`,
Modules 7 to 13, 13 sessions), starts where this one stops; each kit is sold and taught on its own.
Learners should already know Python and pandas (the course follows **Python Data Science**, at
<https://rays-home.netlify.app/>; Session 1 opens with a short diagnostic on exactly those skills).

**Format:** 11 sessions × 3 hours (33 class hours), two lessons per session (Session 11 has one, then
a final-project studio and showcase). Run it weekly (11 weeks), twice a week (about 6 weeks), or as
intensive days (two sessions a day). In the file layout a session is called a "module":
`source/module-07/` is Session 7. In prose, "Module 5" always means a course module (trees and ensembles).

Learners start each lesson project in the room and finish it at home, so "leave with" below means
"have started in class and finish before the next session".

| Session | Lessons | Learners leave with |
|---------|---------|---------------------|
| S1 · Your ML Toolkit and the Estimator API | 0.1 Your ML Toolkit · 1.1 The scikit-learn Estimator API (+ welcome, environment check, 6-item diagnostic) | A working `.venv` (`check_environment.py` says READY); a data card of the seven Pinecrest files; a churn-risk shortlist (`churn_shortlist.csv`: the forest finds 41 real churners in its top 50) |
| S2 · Splits, Leakage and Cross-Validation | 1.2 Splits, Baselines and Leakage · 1.3 Cross-Validation | An honest evaluation report with a sealed test set (`evaluation_report.txt`); a cross-validated bake-off (`cv_leaderboard.csv`) |
| S3 · Metrics and Preprocessing | 1.4 Metrics for Classification and Regression · 2.1 Impute, Scale and Encode | A retention-campaign plan with a profit-tuned threshold (`retention_calls.csv`); a model-ready customer table built with a `ColumnTransformer` |
| S4 · Pipelines and Creating Features | 2.2 Pipelines · 2.3 Creating Features | A saved churn pipeline (`churn_pipeline.joblib`); a demand-feature scoreboard (`demand_scoreboard.csv`) |
| S5 · Feature Selection and Imbalanced Classes | 2.4 Feature Selection · 2.5 Imbalanced Classes | A lean six-column churn model (`lean_features.txt`); a cost-tuned fraud review queue (`fraud_review_queue.csv`) |
| S6 · Linear Models and Regularization | 3.1 Linear and Logistic Regression · 3.2 Regularization and the Bias-Variance Trade-off + **Checkpoint 1** (30 min) | A price-response model; a regularized weekly demand forecast (`demand_forecast_2025.csv`); a checkpoint scored with its rubric |
| S7 · Nearest Neighbors and Naive Bayes | 4.1 k-Nearest Neighbors · 4.2 Naive Bayes | A lookalike churn shortlist (`lookalike_shortlist.csv`); a review-triage list from text (`triage_three_star.csv`) |
| S8 · Support Vector Machines and Decision Trees | 4.3 Support Vector Machines · 5.1 Decision Trees | An SVM retention list (`svm_shortlist.csv`); a churn rule book a care team can read (`churn_rules.txt`) |
| S9 · Random Forests and Gradient Boosting | 5.2 Bagging and Random Forests · 5.3 Gradient Boosting | A forest call list (`call_list.csv`); a boosted weekly demand forecaster |
| S10 · Ensembles and Model Selection | 5.4 Voting and Stacking · 6.1 Model Selection and Learning Curves + final project chosen | An honest verdict on ensembling; a fair bake-off (`bakeoff_results.csv`); a chosen final project with a one-paragraph plan and milestone 1 |
| S11 · Hyperparameter Tuning and Final Project Studio | 6.2 Hyperparameter Tuning + studio + showcase | A tuned retention model (`best_params.json`); a final project in progress and a two-minute lightning talk. Submit within a week |

**Why this shape (and the timing trade-off):** the lessons were written and timed honestly for a solo
learner at 90 to 105 minutes each, so two of them come to about 180 to 205 solo minutes against about
150 room minutes once the warm-up, break and wrap-up are out. They do not fit whole, and the kit does
not pretend they do. The room protects what is hardest alone: the **mental model**, a **live-coding
demo** learners type along with, the **guided lab** (new practice on the same data, not the lesson's
project), and **starting** each lesson project together. Home gets the rest: finishing the two
projects, the lesson's exercises, quiz and journal, about two to three hours between sessions. Every
facilitator guide says, for its session, what is done in the room, what goes home, what to protect and
what to cut first if you run long.

Every session runs 0:00 to 3:00: warm-up (five retrieval questions and a homework check, 10 min;
Session 1 opens with a 30-minute welcome, environment check and diagnostic instead), Lesson A (frame,
demo, guided lab, project start), a 10-minute break, Lesson B, and a 10-minute wrap-up (exit ticket,
homework, the next session's bring list). Session 6 fits Checkpoint 1 in at 0:10.

### Primary documents: the combined books (start here)

| File | Audience | Contents |
|------|----------|----------|
| `participant-workbook.{html,pdf}` | Learner | The complete workbook: all 11 sessions as chapters (goals, key ideas, a new worked example per lesson, guided-lab steps, predict-then-run items, a 12-item self-check, homework checklist, glossary) |
| `facilitator-guide.{html,pdf}` | Instructor | The complete teaching manual: per session, Before the Session (prep checklist, files, known environment traps), the 0:00–3:00 agenda, warm-up with answers, lesson plans with live-demo scripts (every line of code run, with its real output), the three mistakes to expect, checks for understanding, closing (exit ticket, homework, next bring list) and "If you run long or short" |
| `answer-key.{html,pdf}` | Instructor | Every self-check item answered with reasons (including why each wrong option is wrong), lab solution notes with the solutions' real output, the lesson pages' own quiz answers, warm-up, diagnostic, exit-ticket and checkpoint answers (confidential) |

**Self-check items:** 12 per session, 11 × 12 = **132**, every one answered in the answer key. The
workbook quiz, the answer key and the guide's compact answers are generated from one item list per
session and are checked to match word for word, with the correct letters balanced across A to D.

The editable per-session docs live in `source/module-NN/`. **If you edit a session's doc, run
`python3 build-combined.py` to regenerate the three books, then re-render their PDFs** (see
"Editing and building"). The page counts of the books are printed by the PDF step.

### Kit-wide documents

| File | Audience | Purpose |
|------|----------|---------|
| `setup-guide.html` | Learner (+ a facilitator page) | Before-you-begin sheet: an "am I ready?" list of Python and pandas skills; Python 3.12+, a `pinecrest-ml` folder with a `.venv`, the CPU build of PyTorch first on Linux, `python -m pip install -r requirements.txt`, `pinecrest-ml-data.zip`, where the `labs/` folder goes, and `check_environment.py`; Windows/macOS/Linux commands; common install problems; a room-setup checklist for the facilitator. Send it a week before Session 1 with `labs/module-01/l0_1-orientation/check_environment.py` attached |
| `final-assessment.html` | Instructor | The final project (churn, weekly demand or fraud on the Pinecrest data): the brief, the seven-row rubric, and instructor answer notes with real reference numbers for all three problems (confidential) |
| `sell-sheet.html` | Prospective instructors | One-page overview: who it is for, outcomes, format, what is included, prerequisites, license tiers |
| `README.md` · `LICENSE.md` | — | This overview and the tiered license template |

**Slide decks:** one per session in `slides/` (`module-NN-<slug>.html`), 362 slides in all (28 to 39
per session). Each deck is one self-contained file: open it in a browser, arrow keys or space to
advance. Every code slide shows real output and fits a 1280×720 projector without scrolling.

**Labs:** 45 starter/solution pairs plus 14 `debug_*.py` "find the bug" handouts across
`labs/module-01/` … `labs/module-11/`, each session with its own `README.md` (files, what each lab
practices, timing, what "done" looks like). Session 1 adds `check_environment.py` and the diagnostic
pair; Session 6 the Checkpoint 1 pair (`labs/module-06/checkpoint-1/`); Session 11 the final-project
starter (`labs/module-11/l6_2-hyperparameter-tuning/final_project_starter.py`). Starters run as handed
out (TODOs and all) and print what "done" should look like; solutions print the answers.

The public **syllabus** for this kit lives in the course root, one level up from `instructor-kit/`, so
it stays public: `syllabus-foundations.html` (screen, light/dark), `syllabus-foundations-print.html`
(print-first) and `Machine-Learning-Foundations-Syllabus-11-sessions.pdf`. They are generated (see
"Editing and building").

**Still to do before selling:** `LICENSE.md` is a plain-language **template**: have it reviewed by a
lawyer before commercial distribution. Pricing in the sell sheet and the license is a **suggested
anchor**: the bracketed prices are placeholders set at point of sale.

---

## Folder layout

```
instructor-kit/foundations/
├─ participant-workbook.{html,pdf}   ← canonical hand-out books (all 11 sessions)
├─ facilitator-guide.{html,pdf}
├─ answer-key.{html,pdf}
├─ setup-guide / final-assessment / sell-sheet  (.html + .pdf)
├─ slides/            one self-contained deck per session: module-NN-<slug>.html (NN = 01 … 11)
├─ labs/              labs_paths.py, check_labs.py, README.md, and module-01/ … module-11/:
│                     per lesson a folder lM_N-<slug>/ with *_starter.py + *_solution.py pairs
│                     and debug_*.py handouts; module-01/l0_1-orientation/ holds check_environment.py
│                     and the diagnostic; module-06/checkpoint-1/ the checkpoint
├─ source/            module-01/ … module-11/ editable per-session docs (they build the books)
├─ kit.css            shared print-first stylesheet
├─ build-combined.py  regenerate the 3 books from source/
└─ README.md · LICENSE.md
```

Every document is print-ready (a **Print / Save as PDF** button and a tuned print layout, US Letter)
and readable on a phone.

The whole `instructor-kit/` folder is kept **off the live site**: the course root's `_redirects`
returns 404 for `/instructor-kit/*`. It lives in the repository only.

## How to teach from it

1. **A week before Session 1**, send learners the **setup guide** with
   `labs/module-01/l0_1-orientation/check_environment.py` attached; ask each learner to reply with its
   last line (`READY for Python Machine Learning`). No reply means an install risk: chase it.
2. **Hand out** `requirements.txt` and `pinecrest-ml-data.zip` (also on a USB stick, in case the venue
   network fails) and the `labs/` folder **without the `*_solution.py` files** (keep `labs_paths.py`:
   every lab imports it). Learners put `labs/` inside their `pinecrest-ml` folder, next to `data/`, so
   every lab finds the data by itself (or they set `PINECREST_ML_DATA` to the data folder).
3. **Before each session**, read that session's facilitator guide: Before the Session (files, timings
   of slow cells, known environment traps), the agenda, each lesson's live-demo script, the three
   mistakes to expect, and "If you run long or short". Run `python3 labs/check_labs.py module-NN` once
   on your own machine.
4. **Present** from the session's deck in `slides/`; live-code in JupyterLab or VS Code from your own
   `pinecrest-ml` folder.
5. Learners work in the **participant workbook** and keep a **learning journal** (`journal.md`).
6. Check self-checks, warm-ups and exit tickets with the **answer key**; score Checkpoint 1 and the
   final project with their rubrics.

## The data

Everything runs on **Pinecrest Outfitters**, a fictional outdoor-gear retailer. Every customer,
product, review and payment is invented, so learners can study "customer records" with no privacy
concerns. `pinecrest-ml-data.zip` (1.4 MB, on the course site) holds seven CSV files:
`customers_churn.csv` (6,000 customers), `weekly_demand.csv`, `transactions.csv` (50,000 payments,
0.88% fraud), `reviews.csv`, `products_ml.csv`, `interactions.csv` and `daily_store_sales.csv`, plus
`README.md` (every file and column, the planted findings and honest caveats) and `make_ml_data.py`
(the seeded generator that rebuilds identical files).

**No downloads during the Foundations kit.** The image datasets and pretrained networks (about 175 MB)
are only needed in the Advanced kit. The kit runs without internet once the environment is installed.

## Checkpoint and the final project

- **Checkpoint 1 (Session 6, 0:10, 30 minutes):** an individual, open-notebook task on Modules 1 and 2:
  audit a colleague's leaky churn notebook, rebuild an honest model in a pipeline against a baseline,
  choose who gets a voucher from the prices without touching the test set, report three sentences.
  Starter in `labs/module-06/checkpoint-1/`; five-row rubric and model answer in the answer key.
- **Final project:** in the **Session 10** wrap-up each learner chooses one of three supervised
  problems on the Pinecrest data (A churn: who gets a retention offer; B weekly demand: how many units
  to stock; C fraud: which payments to review) and writes a one-paragraph plan; milestone 1 (split,
  bar to beat, a first cross-validated model) is homework with `final_project_starter.py`.
  **Session 11** is lesson 6.2, then a studio with milestone check-ins and a coaching guide per
  problem, then a showcase (lightning talks or a gallery walk). Learners submit a notebook that reruns
  from top to bottom, a one-page `report.md` and a README within a week. `final-assessment.html` holds
  the brief, the rubric and the instructor answer notes.

## Requirements

- **Python 3.12 or newer** (64-bit) on Windows, macOS or Linux, with a virtual environment and the
  course `requirements.txt` (NumPy 2.5, pandas 3.0, Matplotlib 3.11, scikit-learn 1.9, XGBoost 3.4,
  LightGBM 4.7, imbalanced-learn 0.14, Optuna 5.0, JupyterLab 4.6, and the Advanced kit's packages:
  SHAP, LIME, UMAP, PyTorch 2.14 CPU, MLflow, FastAPI). One environment serves both kits;
  `check_environment.py` checks all of it.
- **JupyterLab or VS Code.** About 2 GB of disk. An ordinary laptop processor: no GPU, no accounts, no
  paid tools, no API keys.

## Editing and building

```bash
cd instructor-kit/foundations
python3 build-combined.py                 # regenerate the 3 books from source/ (skips missing sessions)
python3 labs/check_labs.py                # run every starter and solution offline (LABS_OFFLINE=1), 4 threads
python3 labs/check_labs.py module-07      # one session
```

The author's tools in `~/projects/_kit_tools/python_ml/` (not shipped with a licensed kit):

```bash
python3 check_kit_ml.py foundations --all # per-session consistency: one sheet/foot, agenda sums to 180 with
                                          # contiguous clocks and one Break, lesson titles match the course,
                                          # quiz identical across workbook/key/guide, lab paths exist,
                                          # decks fit 1280x720, nothing wider than 390 px, no console errors
python3 gen_syllabus_ml.py foundations    # rebuild ../../syllabus-foundations{,-print}.html from each
                                          # session's agenda in source/module-NN/facilitator-guide.html
python3 render_pdfs_ml.py foundations     # render the six kit PDFs + the syllabus PDF, print page counts,
                                          # and phone-check every page at 390 px
python3 verify_all.py                     # the course itself: every lesson's code run and compared
                                          # with its printed output
```

Each session's three documents and deck are **generated**: the generator for Session N is
`~/projects/_kit_tools/python_ml/kit_generators/A0N/gen_A0N.py` (A10, A11 for Sessions 10 and 11),
with every code block it shows in `blocks_A0N.py`, run for real in the course environment. To change a
session, edit its generator and rerun it, then rebuild the books; hand edits to the generated HTML
are lost the next time the generator runs.

Or open any document and use **Print / Save as PDF** (Letter, 0.5 in margins, background graphics on).

---

## License

See `LICENSE.md`. Short version: licensed to a **single instructor or organization** to teach the
kit; **not** for resale; the facilitator guide, answer key and final assessment are **not** for
learner distribution. The Pinecrest dataset is part of the licensed materials. Tiers (Solo /
Organization / White-label, plus an optional updates add-on) are set at point of sale.

© 2026 Ray de la Paz. Machine Learning Foundations (Python Machine Learning). All rights reserved.
