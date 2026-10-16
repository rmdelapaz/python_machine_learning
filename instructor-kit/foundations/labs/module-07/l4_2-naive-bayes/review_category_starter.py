# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.2 B: which product is this review about?

Lesson 4.2 sorted reviews into happy and unhappy. The merchandising team asks a different question of the
same text: which of the ten product categories is a review about? Ten classes, all 4,000 reviews (the
3-star ones too), and the same tools: CountVectorizer, the Naive Bayes variants, a baseline. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1 (done for you): all reviews, a stratified split on the category
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

reviews = pd.read_csv(DATA / "reviews.csv")
train, test = train_test_split(reviews, test_size=0.25, stratify=reviews["category"], random_state=42)
print(len(train), "train,", len(test), "test;", reviews["category"].nunique(), "categories")
print(reviews["category"].value_counts().head(3).to_dict())

# %% Step 2: baseline and three text models
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline

# TODO 1: fit four models on train["text"] -> train["category"]: DummyClassifier(strategy="most_frequent"),
#         CountVectorizer + MultinomialNB, CountVectorizer + ComplementNB, CountVectorizer +
#         LogisticRegression(max_iter=2000). Print each test accuracy. Keep the MultinomialNB pipeline as `nb`.
nb = None
print("TODO 1: four models, test accuracy")

# %% Step 3: smoothing by cross-validation
from sklearn.model_selection import cross_val_score

# TODO 2: 5-fold CV accuracy on the training rows for MultinomialNB(alpha=...) with alpha 0.1, 0.3, 1.0, 3.0.
print("TODO 2: CV accuracy by alpha")

# %% Step 4: what did it count?
# TODO 3: with vocab = nb[0].get_feature_names_out() and logp = nb[-1].feature_log_prob_ (one row per class),
#         compute lift = logp - logp.mean(axis=0) and print the five words with the highest lift for Tents,
#         Paddling and Footwear.
print("TODO 3: the most telling words for three categories")

# %% Step 5: where it goes wrong
from sklearn.metrics import confusion_matrix

# TODO 4: build the confusion matrix (labels=nb[-1].classes_) as a DataFrame and print the three most frequent
#         off-diagonal mix-ups. Why would Apparel and Paddling be confused?
# TODO 5: print the three categories with the lowest recall.
print("TODO 4-5: mix-ups and recall")

print("\nExpected when done: baseline 0.237, MultinomialNB 0.586, ComplementNB 0.592, logistic regression 0.605;")
print("CV accuracy 0.583, 0.582, 0.579, 0.580 for alpha 0.1 to 3.0; Tents words start with canopy, camp, tent;")
print("top mix-ups Camp Kitchen -> Accessories 20, Apparel -> Paddling 20, Paddling -> Apparel 19; lowest recall")
print("Packs 0.39, Camp Kitchen 0.43, Footwear 0.49.")
