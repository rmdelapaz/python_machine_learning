# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.3 A (SOLUTION): a support vector machine for Pinecrest's product reviews.

Lesson 4.3 says SVMs shine on data with many features, such as text. Lesson 4.2 gave the care team a
Naive Bayes review model. Here the same reviews and the same split meet a linear SVM: does it rank
negative reviews better, what does "scaling" mean for word counts, and how wide is the street?
"""
# %% Step 1: the reviews, labeled and split exactly as in Lesson 4.2
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

# %% Step 2: a linear SVM on raw word counts. Rank with decision_function: an SVM has no predict_proba
counts_svm = make_pipeline(CountVectorizer(ngram_range=(1, 2)), LinearSVC()).fit(X_train, y_train)
print(f"LinearSVC on raw counts      accuracy {counts_svm.score(X_test, y_test):.3f}  "
      f"ROC AUC {roc_auc_score(y_test, counts_svm.decision_function(X_test)):.3f}")

# %% Step 3: "scale" the text. TF-IDF divides each review by its length and down-weights words that
# appear in almost every review: for word counts it plays the part StandardScaler plays for numbers.
svm = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC()).fit(X_train, y_train)
svm_scores = svm.decision_function(X_test)
print(f"LinearSVC on TF-IDF          accuracy {svm.score(X_test, y_test):.3f}  "
      f"ROC AUC {roc_auc_score(y_test, svm_scores):.3f}")
print("features (words and word pairs):", len(svm[0].vocabulary_))

# %% Step 4: choose C by cross-validation on the training reviews only
from sklearn.model_selection import cross_val_score

print("\nparameter key:", [key for key in svm.get_params() if key.endswith("__C")])
for C in [0.01, 0.1, 1, 10]:
    svm.set_params(linearsvc__C=C)
    auc = cross_val_score(svm, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"C={C:<5} CV ROC AUC {auc:.4f}")
svm.set_params(linearsvc__C=1).fit(X_train, y_train)

# %% Step 5: a linear SVM has one weight per word: read the strongest ones
weights = pd.Series(svm[-1].coef_[0], index=svm[0].get_feature_names_out()).sort_values()
print("\nmost negative:", weights.head(5).round(2).to_dict())
print("most positive:", weights.tail(5).round(2).to_dict())

# %% Step 6: how many test reviews sit inside the street?
inside = np.abs(svm_scores) < 1
wrong_side = (svm_scores > 0) != (y_test.to_numpy() == 1)
print(f"\ntest reviews inside the street (|score| < 1): {inside.sum()} of {len(svm_scores)}")
print(f"test reviews on the wrong side of the center line: {wrong_side.sum()}")

# %% Step 7: probabilities, when the team asks for them
from sklearn.calibration import CalibratedClassifierCV

calibrated = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                           CalibratedClassifierCV(LinearSVC(), ensemble=False)).fit(X_train, y_train)
p_positive = calibrated.predict_proba(X_test)[:, 1]
print(f"\ncalibrated: average P(positive) {p_positive.mean():.3f} vs real share {y_test.mean():.3f}; "
      f"ROC AUC {roc_auc_score(y_test, p_positive):.3f}")

# %% Stretch: does a curved (RBF) boundary help on text?
from sklearn.svm import SVC

rbf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), SVC()).fit(X_train, y_train)
print(f"SVC (RBF) on TF-IDF          accuracy {rbf.score(X_test, y_test):.3f}  "
      f"ROC AUC {roc_auc_score(y_test, rbf.decision_function(X_test)):.3f}  "
      f"support vectors {rbf[-1].n_support_.sum()} of {len(X_train)}")
