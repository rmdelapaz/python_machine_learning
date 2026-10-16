# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.1 B (SOLUTION): route a message to the right product team.

Customer emails arrive without a product attached. Pinecrest has one specialist per product category,
so the care team wants each message routed to a category from its words alone. Ten classes instead of
two: the same TF-IDF pipeline, read with per-class recall, a confusion pair and the words behind one class.
"""
# %% Step 1: the task and its baseline
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

reviews = pd.read_csv(DATA / "reviews.csv")
print(reviews["category"].value_counts().to_dict())
train, test = train_test_split(reviews, test_size=0.25, stratify=reviews["category"], random_state=42)
baseline = DummyClassifier(strategy="most_frequent").fit(train["text"], train["category"])
print(f"baseline (always '{baseline.classes_[np.argmax(baseline.class_prior_)]}'): "
      f"test accuracy {baseline.score(test['text'], test['category']):.3f}")

# %% Step 2: a ten-class TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

router = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000))
router.fit(train["text"], train["category"])
pred = router.predict(test["text"])
print(f"test accuracy {accuracy_score(test['category'], pred):.3f}, macro F1 {f1_score(test['category'], pred, average='macro'):.3f}")

# %% Step 3: which teams would get lost messages?
recall = pd.Series({c: (pred[test["category"] == c] == c).mean() for c in router.classes_}).sort_values()
print("lowest recall:", recall.head(3).round(3).to_dict())
print("highest recall:", recall.tail(2).round(3).to_dict())
pairs = pd.crosstab(test["category"], pred)
off = pairs.where(~np.eye(len(pairs), dtype=bool)).stack().sort_values(ascending=False)
print("most frequent wrong routes:", {f"{t} -> {p}": int(n) for (t, p), n in off.head(3).items()})

# %% Step 4: the words behind one class
vec = router.named_steps["tfidfvectorizer"]
clf = router.named_steps["logisticregression"]
tents = pd.Series(clf.coef_[list(clf.classes_).index("Tents")], index=vec.get_feature_names_out())
print("strongest words for Tents:", tents.nlargest(6).round(2).index.tolist())

# %% Step 5: messages that never name the product
messages = ["It kept us dry through a storm and the poles went up fast.",
            "Runs half a size small, my toes hit the front on descents.",
            "Arrived quickly. Great purchase."]
for text, cat, p in zip(messages, router.predict(messages), router.predict_proba(messages).max(axis=1)):
    print(f"{cat:12s} ({p:.2f})  {text}")

# %% Step 6: route only when sure: the rest goes to a general inbox
confidence = router.predict_proba(test["text"]).max(axis=1)
sure = confidence >= 0.5
print(f"confidence >= 0.5: {sure.mean():.1%} of messages routed automatically, accuracy "
      f"{(pred[sure] == test['category'].to_numpy()[sure]).mean():.3f}; the rest go to the general inbox")

# %% Stretch: give the small teams a louder voice
balanced = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))
bpred = balanced.fit(train["text"], train["category"]).predict(test["text"])
brecall = pd.Series({c: (bpred[test["category"] == c] == c).mean() for c in balanced.classes_})
print(f"class_weight='balanced': accuracy {accuracy_score(test['category'], bpred):.3f}, "
      f"macro F1 {f1_score(test['category'], bpred, average='macro'):.3f}, "
      f"Tents recall {brecall['Tents']:.3f}, Apparel recall {brecall['Apparel']:.3f}")
