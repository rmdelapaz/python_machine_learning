# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 11.1: a review-sentiment script with FOUR bugs (deliberately broken handout).

Two bugs stop the script with an error; two let it run and print numbers you should not trust.
Find and fix all four. When everything is fixed it prints test accuracy 0.877, a threshold chosen
WITHOUT the test set (0.2), and P(negative) [0.05 0.3 0.13] for the three new emails (the empty one
included). Then look hard at the third email's score.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

reviews = pd.read_csv(DATA / "reviews.csv")
labeled = reviews[reviews["rating"] != 3].copy()
labeled["positive"] = (labeled["rating"] >= 4).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    labeled[["text"]], labeled["positive"], test_size=0.25, stratify=labeled["positive"], random_state=42)

model = make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression())
model.fit(X_train, y_train)
print(f"test accuracy: {model.score(X_test, y_test):.3f}")

# The care team wants to catch at least 85% of the negative reviews: pick the highest threshold that does.
p_negative = 1 - model.predict_proba(X_test)[:, 1]
is_negative = y_test.to_numpy() == 0
for threshold in [0.5, 0.4, 0.3, 0.2]:
    if ((p_negative >= threshold) & is_negative).sum() / is_negative.sum() >= 0.85:
        break
print("chosen threshold:", threshold)

emails = pd.Series(["Love it, so warm.", None, "It leaked on day one and I want a refund."])
print("P(negative) for new emails:", (1 - model.predict_proba(emails)[:, 1]).round(2))
