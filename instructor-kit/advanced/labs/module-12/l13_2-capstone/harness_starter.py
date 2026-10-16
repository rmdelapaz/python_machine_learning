# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Capstone lab 2: an evaluation harness and a first pipeline (Milestones 3 and 4).

Milestone 3: ONE function that scores any candidate on the TRAINING data only, the way your problem's
split works. Milestone 4: the simplest sensible pipeline, scored by that function, beating the baseline.
The test set is never touched here. Set PROBLEM, then do the TODOs for your problem.
"""
# %% Step 0: the lesson's template (capstone_start.py next to this file)
sys.path.insert(0, str(pathlib.Path(globals().get("__file__", "lab.py")).resolve().parent))
import numpy as np
import pandas as pd

from capstone_start import STARTERS

PROBLEM = "reviews"        # TODO 0: your problem
problem = STARTERS[PROBLEM]()
train = problem["train"]
print(f"{PROBLEM}: {len(train):,} training rows; the test set is not used in this lab")

# %% Step 1: the harness (Milestone 3): pick the one for your problem
# demand   evaluate(make_model): sort the 2024 rows by week_start, TimeSeriesSplit(n_splits=4); fit on each
#          past block, MAE on the next; return the mean and the spread. Wrap the per-product mean as a tiny
#          class with fit/predict so the baseline goes through the same harness.
# fraud    evaluate(model): fit on January to April of the training rows, average precision on May.
# reviews  evaluate(model): cross_validate with StratifiedKFold(5, shuffle=True, random_state=42) and two
#          scorers: accuracy, and recall_score(..., pos_label=0) for the negative reviews.
# recs     evaluate(scores): hide each customer's LATEST TRAINING purchase (groupby(...).tail(1) again),
#          build R from the rest, mask owned products, hit rate at 10 on the hidden ones.
# segments evaluate(k): silhouette (sample_size=3000, random_state=42) AND the adjusted Rand index between
#          KMeans(k, n_init=10) fitted with random_state=42 and with random_state=7.
evaluate = None
if evaluate is None:
    print("TODO 1: write evaluate() for your problem and score the baseline with it")

# %% Step 2: the first pipeline (Milestone 4)
# TODO 2: the simplest sensible model as a Pipeline: one-hot + Ridge (demand), scaled numbers + one-hot
#         category + LogisticRegression (fraud), TfidfVectorizer + LogisticRegression (reviews),
#         item-kNN R @ cosine_similarity(R.T) (recs), KMeans on scaled features for k = 4, 5, 6 (segments).
#         Score it with evaluate(). It should beat the baseline; if not, look for a bug first.
print("TODO 2: the first pipeline, scored by the same harness")

print("\nExpected when done (training data only): demand rolling folds: per-product mean MAE 13.924, ridge")
print("13.128; fraud on May: no model AP 0.008, logistic regression 0.539; reviews CV: always positive 0.745")
print("(negative recall 0.000), tf-idf + LR 0.857 (0.577); recs validation: popularity 0.209, item-kNN 0.286,")
print("SVD k=5 0.285; segments: tiers 0.079, k-means k=5 silhouette 0.470 with seed ARI 1.000.")
