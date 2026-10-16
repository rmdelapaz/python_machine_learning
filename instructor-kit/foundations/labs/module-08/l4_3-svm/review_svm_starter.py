# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.3 A: a support vector machine for Pinecrest's product reviews.

Lesson 4.3 says SVMs shine on data with many features, such as text. Lesson 4.2 gave the care team a
Naive Bayes review model. Here the same reviews and the same split meet a linear SVM: does it rank
negative reviews better, what does "scaling" mean for word counts, and how wide is the street?
Work through TODO 1 to 6; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the reviews, labeled and split exactly as in Lesson 4.2 (done for you)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

reviews = pd.read_csv(DATA / "reviews.csv")
labeled = reviews[reviews["rating"] != 3].copy()
labeled["positive"] = (labeled["rating"] >= 4).astype(int)
train, test = train_test_split(labeled, test_size=0.25, stratify=labeled["positive"], random_state=42)
X_train, y_train = train["text"], train["positive"]
X_test, y_test = test["text"], test["positive"]
print(len(train), "training reviews,", len(test), "test reviews; first test id", test["review_id"].iloc[0])

nb = make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.3)).fit(X_train, y_train)
print(f"Naive Bayes (Lesson 4.2)     accuracy {nb.score(X_test, y_test):.3f}  "
      f"ROC AUC {roc_auc_score(y_test, nb.predict_proba(X_test)[:, 1]):.3f}")

# %% Step 2: a linear SVM on raw word counts
# TODO 1: make_pipeline(CountVectorizer(ngram_range=(1, 2)), LinearSVC()), fit it, and print its test
#         accuracy and ROC AUC. An SVM has no predict_proba: pass decision_function(X_test) to roc_auc_score.
print("TODO 1: LinearSVC on raw counts: accuracy and ROC AUC")

# %% Step 3: "scale" the text
# TF-IDF divides each review by its length and down-weights words that appear in almost every review:
# for word counts it plays the part StandardScaler plays for numbers.
# TODO 2: the same pipeline with TfidfVectorizer(ngram_range=(1, 2)) instead of CountVectorizer. Call it
#         svm, keep svm_scores = svm.decision_function(X_test), print accuracy, ROC AUC and
#         len(svm[0].vocabulary_). Before you run it, predict: better or worse than raw counts? ______
svm = None
svm_scores = None
if svm is None:
    print("TODO 2: LinearSVC on TF-IDF")

# %% Step 4: choose C by cross-validation on the training reviews only
from sklearn.model_selection import cross_val_score

# TODO 3: print the parameter key that ends in "__C" (list the keys of svm.get_params()). Then for C in
#         [0.01, 0.1, 1, 10]: svm.set_params(<that key>=C) and print the 5-fold CV ROC AUC on the
#         training reviews. Finish with svm.set_params(<key>=1).fit(X_train, y_train).
print("TODO 3: C by cross-validation")

# %% Step 5: a linear SVM has one weight per word
# TODO 4: weights = pd.Series(svm[-1].coef_[0], index=svm[0].get_feature_names_out()).sort_values();
#         print the five most negative and five most positive (head(5), tail(5)).
#         One positive "word pair" looks odd. Read a few reviews that contain it: where does it come from?
print("TODO 4: the strongest weights")

# %% Step 6: how many test reviews sit inside the street?
# TODO 5: count the test reviews with |svm_scores| < 1 (inside the margin) and those whose sign
#         disagrees with y_test (on the wrong side of the center line).
print("TODO 5: inside the street, wrong side")

# %% Step 7: probabilities, when the team asks for them
from sklearn.calibration import CalibratedClassifierCV

# TODO 6: make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), CalibratedClassifierCV(LinearSVC(), ensemble=False)),
#         fit, and compare the average predicted P(positive) with the real share y_test.mean(). Print its AUC.
print("TODO 6: calibrated probabilities")

# %% Stretch: an RBF SVC (from sklearn.svm import SVC) on TF-IDF. Does a curved boundary help on text?

print("\nExpected when done: raw counts accuracy 0.859, AUC 0.921 (below Naive Bayes' 0.868 / 0.934);")
print("TF-IDF accuracy 0.884, AUC 0.936, 1,856 features; key linearsvc__C, CV AUC 0.9077 / 0.9253 /")
print("0.9269 / 0.9066 for C = 0.01 / 0.1 / 1 / 10; most negative 'not' (-2.51), most positive")
print("'would buy' (1.93); 445 of 774 reviews inside the street, 90 on the wrong side; calibrated")
print("average 0.734 vs 0.745, AUC 0.936. Stretch: RBF accuracy 0.876, AUC 0.928 (no gain).")
