# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.1 A: would this customer recommend the product?

Every review form ends with a checkbox, "I would recommend this product" (the `recommended` column).
The marketing team wants to read that intent from text that arrives WITHOUT the checkbox (emails,
chat). A new target on the same reviews, including the mixed 3-star ones the lesson left out.
Work through TODO 1 to 6; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: target, split, baseline
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

reviews = pd.read_csv(DATA / "reviews.csv")
print(reviews.shape, f"recommended share {reviews['recommended'].mean():.3f}")
print("recommended share by star rating:", reviews.groupby("rating")["recommended"].mean().round(3).to_dict())

# TODO 1: split ALL 4,000 reviews 75/25, stratified on `recommended`, random_state=42 (keep whole rows:
#         train, test = train_test_split(reviews, ...)). Fit DummyClassifier(strategy="most_frequent")
#         on train["text"] and print its test accuracy.
train = test = None
if train is None:
    print("TODO 1: split and baseline")

# %% Step 2: three vectorizer settings, compared with cross-validation on the training reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO 2: 5-fold CV accuracy (mean ± std) of make_pipeline(vectorizer, LogisticRegression()) for
#         TfidfVectorizer(), TfidfVectorizer(stop_words="english") and
#         TfidfVectorizer(ngram_range=(1, 2), min_df=2). Use the training reviews only.
print("TODO 2: three settings, cross-validated")

# %% Step 3: fit the bigram model, test once, read the "not recommended" row
from sklearn.metrics import classification_report

# TODO 3: fit the bigram pipeline on the training text, call it `model`, predict the test text and print
#         classification_report(..., target_names=["not recommended", "recommended"], digits=3).
#         Which row matters for marketing, and what is its recall?
model = None
if model is None:
    print("TODO 3: fit, test once, classification report")

# %% Step 4: the words with the strongest pull
# TODO 4: a Series of model.named_steps["logisticregression"].coef_[0] indexed by the vectorizer's
#         get_feature_names_out(), sorted; print the six most negative and six most positive terms.
print("TODO 4: the strongest terms")

# %% Step 5: where does the model struggle?
# TODO 5: add the test predictions as a column and print the test accuracy within each star rating
#         (groupby("rating")). Predict first: which rating is hardest? ______
print("TODO 5: accuracy by star rating")

# %% Step 6: probe it with sentences of your own
probes = ["Would buy again.", "Would not buy again.", "Not bad at all, but it runs small.",
          "It leaked on the first night.", "Great value. Though the zipper broke."]
# TODO 6: print P(recommended) for each probe (model.predict_proba(probes)[:, 1]). Which one is wrong?
print("TODO 6: probe sentences")

print("\nExpected when done: baseline 0.655; CV accuracy 0.726 (unigrams), 0.719 (stop words removed),")
print("0.732 (bigrams); test accuracy 0.755, 'not recommended' recall 0.481 and precision 0.716; strongest")
print("terms 'not' (-2.81) and 'highly recommend' (1.58); accuracy by stars 1: 0.794, 2: 0.543, 3: 0.464,")
print("4: 0.845, 5: 0.963; 'It leaked on the first night.' gets P(recommended) 0.66.")
