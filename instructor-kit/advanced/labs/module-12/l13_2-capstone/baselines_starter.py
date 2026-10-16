# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Capstone lab 1: a second honest baseline for YOUR problem (Milestones 1 to 3).

The starter template gives each problem ONE baseline. Often there is a second, stronger "no machine
learning" answer: what a sensible person would do today with a spreadsheet. If your model cannot beat
THAT, it is not worth deploying. Set PROBLEM, run the file, then do the TODO for your problem only.
"""
# %% Step 1: the template's baseline (the lesson's five functions, in capstone_start.py next to this file)
sys.path.insert(0, str(pathlib.Path(globals().get("__file__", "lab.py")).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, recall_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from capstone_start import STARTERS

PROBLEM = "fraud"          # TODO 0: your problem: "demand", "fraud", "reviews", "recs" or "segments"
problem = STARTERS[PROBLEM]()
label, value = problem["baseline"]
print(f"{PROBLEM}: {problem['metric']}, template baseline ({label}) = {value:.3f}")

# %% Step 2: one more baseline, on the SAME split and metric
# demand   TODO: the mean per product AND store; then "same product, store and ISO week in 2024"
#                (group by week_of_year and average: 2024 has two week-1 rows). MAE on the 2025 rows.
# fraud    TODO: a one-line rule: rank June payments by velocity_24h (ties broken by amount / 1e5) and
#                compute its average precision; how many frauds are in its top 1% (83 payments)?
# reviews  TODO: a keyword rule: negative if the text contains "not", "return", "returned" or "regret"
#                (str.contains with word boundaries). Accuracy, and the recall of negative reviews.
# recs     TODO: ten random products the customer does not own (np.random.default_rng(42)): hit rate.
# segments TODO: five random groups (rng.integers(0, 5, n)): the silhouette of "no structure at all".
print("TODO: your problem's second baseline")

# %% Step 3: write the bar into your plan
# TODO: one sentence for your capstone.ipynb plan cell: "A model must beat ____ (template) and ____
#       (second baseline) by at least ____ on ____, or I will recommend not deploying it."
print("TODO: the 'do not deploy' sentence for your plan")

print("\nExpected when done (one line per problem): demand per product x store 12.953 and same week last")
print("year 13.810 (template 13.419); fraud rule AP 0.250, 19 of 64 frauds in its top 1% (template 0.008);")
print("reviews keyword rule accuracy 0.788, negative recall 0.477 (template 0.745, recall 0); recs random")
print("0.053 (template 0.209); segments five random groups -0.023 (template 0.079).")
