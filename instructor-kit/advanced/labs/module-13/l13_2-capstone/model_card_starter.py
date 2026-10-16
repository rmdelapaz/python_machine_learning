# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Capstone lab 4: results.json, model.skops and a model card that cannot disagree (Milestones 8-9).

The most common reproducibility failure in a capstone is a model card whose numbers no longer match the
notebook. This lab runs the hand-in steps on a small review-triage project: choose with CV, test once,
save the results as data, save the model with skops and reload it, write the card FROM the results,
and check every number in the card against results.json. Step 1 is done for you; TODO 1 to 4.
"""
# %% Step 1: two candidates, chosen on the training data (accuracy AND the recall of negative reviews)
sys.path.insert(0, str(pathlib.Path(globals().get("__file__", "lab.py")).resolve().parent))
import json
import re
from importlib.metadata import version

import numpy as np
import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

from capstone_start import start_reviews

rv = start_reviews()
train, test = rv["train"], rv["test"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "neg_recall": lambda est, X, y: recall_score(y, est.predict(X), pos_label=0)}
candidates = {
    "tfidf-1-2gram-lr": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000)),
    "tfidf-1-2gram-lr-balanced": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                                               LogisticRegression(max_iter=1000, class_weight="balanced")),
}
log = {}
for name, pipe in candidates.items():
    r = cross_validate(pipe, train["text"], train["positive"], cv=cv, scoring=scoring)
    log[name] = {"cv_accuracy": round(float(r["test_accuracy"].mean()), 3),
                 "cv_neg_recall": round(float(r["test_neg_recall"].mean()), 3)}
    print(name, log[name])
# A missed complaint is the costly error (the brief), so choose on negative recall, as long as accuracy holds.
chosen = max(log, key=lambda n: log[n]["cv_neg_recall"])
print("chosen:", chosen)

# %% Step 2: test once, with an interval, next to the baseline
# TODO 1: fit candidates[chosen] on the training text, predict the test text ONCE, bootstrap the accuracy
#         (1,000 resamples, rng = np.random.default_rng(42)), and build a dict `results` with the problem,
#         the model name, test_rows, the baseline (name, accuracy, neg_recall 0.0), the test accuracy, its
#         interval and the negative recall, the CV log, the seed and the package versions. Save it to
#         OUTPUT / "results.json".
results = None
if results is None:
    print("TODO 1: test once and write results.json")

# %% Step 3: save with skops, review the types, reload
# TODO 2: sio.dump the fitted pipeline to OUTPUT / "model.skops", print sio.get_untrusted_types(file=...),
#         write the reviewed list as TRUSTED = [...], reload with trusted=TRUSTED and check the reloaded model
#         predicts exactly the same labels. Write OUTPUT / "requirements.txt" with pinned versions.
print("TODO 2: skops round trip and requirements.txt")

# %% Step 4: the card, written FROM results
# TODO 3: an f-string card (Recommendation, Performance, Limitations) whose every number comes from
#         `results`; save it to OUTPUT / "model_card.md".
# TODO 4: untraceable(card, results): find the decimals and percentages in the card (re.findall) and list
#         any that do not appear in json.dumps(results). Then replace one number by hand and run it again.
print("TODO 3-4: the card and its number check")

print("\nExpected when done: CV picks tfidf-1-2gram-lr-balanced (CV accuracy 0.875, negative recall 0.849 vs")
print("0.86 / 0.57); test accuracy 0.873 (95% CI 0.85 to 0.895) vs 0.745 always positive, negative recall 0.873;")
print("skops lists no types to review and the reload predicts the same (True); 6 numbers in the card, none")
print("untraceable; a hand-typed 0.877 is caught.")
