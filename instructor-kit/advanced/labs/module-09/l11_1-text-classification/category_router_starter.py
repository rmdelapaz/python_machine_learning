# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.1 B: route a message to the right product team.

Customer emails arrive without a product attached. Pinecrest has one specialist per product category,
so the care team wants each message routed to a category from its words alone. Ten classes instead of
two. Work through TODO 1 to 6; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the task and its baseline
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

reviews = pd.read_csv(DATA / "reviews.csv")
print(reviews["category"].value_counts().to_dict())
# TODO 1: split 75/25 stratified on `category`, random_state=42 (whole rows), and print the test
#         accuracy of DummyClassifier(strategy="most_frequent") fitted on train["text"].
train = test = None
if train is None:
    print("TODO 1: split and baseline")

# %% Step 2: a ten-class TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

# TODO 2: router = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000)).
#         Fit on the training text and categories; print test accuracy and macro F1 (average="macro").
router = None
if router is None:
    print("TODO 2: fit the router, accuracy and macro F1")

# %% Step 3: which teams would get lost messages?
# TODO 3: recall per category ((pred[test["category"] == c] == c).mean()), the three lowest and two
#         highest; then the three most frequent wrong routes from pd.crosstab(test["category"], pred).
#         Predict first: which small team suffers most? ______
print("TODO 3: per-category recall and wrong routes")

# %% Step 4: the words behind one class
# TODO 4: coef_ has one row per class (in clf.classes_ order). Print the six strongest terms for "Tents".
print("TODO 4: the words behind Tents")

# %% Step 5: messages that never name the product
messages = ["It kept us dry through a storm and the poles went up fast.",
            "Runs half a size small, my toes hit the front on descents.",
            "Arrived quickly. Great purchase."]
# TODO 5: print the predicted category and its probability for each message. Are they right?
print("TODO 5: three new messages")

# %% Step 6: route only when sure
# TODO 6: confidence = router.predict_proba(test["text"]).max(axis=1). With a cut of 0.5: what share of
#         messages is routed automatically, and how accurate are those routes?
print("TODO 6: route only when sure")

# %% Stretch: LogisticRegression(max_iter=1000, class_weight="balanced"): accuracy, macro F1, Tents and Apparel recall.

print("\nExpected when done: baseline (Apparel) 0.237; test accuracy 0.615, macro F1 0.585; lowest recall")
print("Tents 0.333, Camp Kitchen 0.377, Footwear 0.408; most frequent wrong route Paddling -> Apparel (63);")
print("Tents words start 'camp canopy', 'canopy', 'camp'; all three messages go to Apparel or Accessories with")
print("probability 0.31 or less; at 0.5, 42.7% routed with accuracy 0.932. Stretch: 0.580, macro F1 0.553,")
print("Tents recall 0.452, Apparel 0.616.")
