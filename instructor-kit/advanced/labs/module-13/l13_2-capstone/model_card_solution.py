# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Capstone lab 4 (SOLUTION): results.json, model.skops and a model card that cannot disagree (Milestones 8-9).

The most common reproducibility failure in a capstone is a model card whose numbers no longer match the
notebook. This lab runs the hand-in steps on a small review-triage project: choose with CV, test once,
save the results as data, save the model with skops and reload it, write the card FROM the results,
and check every number in the card against results.json.
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
final = candidates[chosen].fit(train["text"], train["positive"])
pred = final.predict(test["text"])
y = test["positive"].to_numpy()
rng = np.random.default_rng(42)
boot = [accuracy_score(y[r], pred[r]) for r in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
results = {
    "problem": "review triage", "model": chosen, "test_rows": int(len(y)),
    "baseline": {"name": rv["baseline"][0], "accuracy": round(rv["baseline"][1], 3), "neg_recall": 0.0},
    "test": {"accuracy": round(accuracy_score(y, pred), 3), "accuracy_ci": [round(float(v), 3) for v in np.percentile(boot, [2.5, 97.5])],
             "neg_recall": round(recall_score(y, pred, pos_label=0), 3)},
    "cv_log": log, "seed": 42,
    "versions": {p: version(p) for p in ["scikit-learn", "pandas", "numpy", "skops"]},
}
(OUTPUT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results["test"]), "baseline", results["baseline"]["accuracy"])

# %% Step 3: save with skops, review the types, reload, prove the reload predicts the same
path = OUTPUT / "model.skops"
sio.dump(final, path)
print("types to review:", sio.get_untrusted_types(file=path))
TRUSTED = []                     # reviewed: a TF-IDF + logistic regression pipeline needs no extra types
reloaded = sio.load(path, trusted=TRUSTED)
print("reloaded model predicts the same:", bool((reloaded.predict(test["text"]) == pred).all()))
(OUTPUT / "requirements.txt").write_text("".join(f"{p}=={v}\n" for p, v in results["versions"].items()))

# %% Step 4: the card, written FROM results (prose is yours; numbers come from the dictionary)
t, b = results["test"], results["baseline"]
card = f"""# Model card: review triage, version 1

## Recommendation
Use it to sort incoming 1-2 and 4-5 star reviews so the product team reads likely complaints first.

## Performance
On {results['test_rows']} held-out reviews: accuracy {t['accuracy']} (95% CI {t['accuracy_ci'][0]} to {t['accuracy_ci'][1]})
against {b['accuracy']} for "{b['name']}"; it finds {t['neg_recall']:.0%} of the negative reviews (the baseline finds none).

## Limitations
Trained without 3-star reviews; their scores are not validated. Star ratings that contradict the text count as errors.
"""
(OUTPUT / "model_card.md").write_text(card)
print(card)

# %% Step 5: every number in the card must appear in results.json
def untraceable(card_text, results_dict):
    """Decimals and percentages in the card that cannot be found in the results (a % is checked as a share)."""
    flat = json.dumps(results_dict)
    numbers = re.findall(r"\d+\.\d+|\d+%", card_text)
    return numbers, [n for n in numbers if n.rstrip("%") not in flat and str(float(n.rstrip("%")) / 100) not in flat]


numbers, missing = untraceable(card, results)
print(f"{len(numbers)} numbers in the card; not traceable to results.json: {missing}")
typed_by_hand = card.replace(str(t["accuracy"]), "0.877", 1)       # a number copied from an earlier run
print("a card typed from memory:", untraceable(typed_by_hand, results)[1])
