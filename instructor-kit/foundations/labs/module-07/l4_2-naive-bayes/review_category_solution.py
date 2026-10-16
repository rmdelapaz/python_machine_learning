# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.2 B (SOLUTION): which product is this review about?

Lesson 4.2 sorted reviews into happy and unhappy. The merchandising team asks a different question of the
same text: which of the ten product categories is a review about? Ten classes, all 4,000 reviews (the
3-star ones too), and the same tools: CountVectorizer, the Naive Bayes variants, a baseline.
"""
# %% Step 1: all reviews, a stratified split on the category
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

models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "MultinomialNB": make_pipeline(CountVectorizer(), MultinomialNB()),
    "ComplementNB": make_pipeline(CountVectorizer(), ComplementNB()),
    "logistic regression": make_pipeline(CountVectorizer(), LogisticRegression(max_iter=2000)),
}
for name, model in models.items():
    model.fit(train["text"], train["category"])
    print(f"{name:20s} accuracy {model.score(test['text'], test['category']):.3f}")

# %% Step 3: smoothing by cross-validation (training rows only)
from sklearn.model_selection import cross_val_score

for alpha in [0.1, 0.3, 1.0, 3.0]:
    s = cross_val_score(make_pipeline(CountVectorizer(), MultinomialNB(alpha=alpha)),
                        train["text"], train["category"], cv=5, scoring="accuracy")
    print(f"alpha={alpha:<4} CV accuracy {s.mean():.3f}")

# %% Step 4: what did it count? The most telling words for three categories
nb = models["MultinomialNB"]
vocab = nb[0].get_feature_names_out()
logp = nb[-1].feature_log_prob_
lift = logp - logp.mean(axis=0)                 # how much more likely a word is in this class than on average
for cat in ["Tents", "Paddling", "Footwear"]:
    row = lift[list(nb[-1].classes_).index(cat)]
    print(f"{cat:9s} {list(vocab[np.argsort(row)[::-1][:5]])}")

# %% Step 5: where it goes wrong
from sklearn.metrics import confusion_matrix

pred = nb.predict(test["text"])
cm = pd.DataFrame(confusion_matrix(test["category"], pred, labels=nb[-1].classes_),
                  index=nb[-1].classes_, columns=nb[-1].classes_)
pairs = cm.where(~np.eye(len(cm), dtype=bool)).stack().sort_values(ascending=False).head(3)
print("\nmost frequent mix-ups (true -> predicted):")
print(pairs.astype(int).to_string())
print("recall by category:", (np.diag(cm) / cm.sum(axis=1)).round(2).sort_values().head(3).to_dict())
