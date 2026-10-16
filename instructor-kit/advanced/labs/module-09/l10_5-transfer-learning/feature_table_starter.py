# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root, use_cached_weights, fixtures

"""Lab 10.5 A: pretrained features are just a table.

The trip-planning team will not label every tile, and a wrong land-cover label on a trail map is
embarrassing. Using frozen ResNet-18 features of 3,000 NEW EuroSAT tiles (not the lesson's), treat the
512 numbers per tile like any scikit-learn table. Work through TODO 1 to 5; run the file after each one.
It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: load the cached features (done for you; computed once by prepare_eurosat_features.py)
import numpy as np
import pandas as pd

path = fixtures(__file__) / "eurosat_resnet18_feats.npz"
if not path.is_file():
    raise SystemExit("Run prepare_eurosat_features.py in this folder once (about 30 seconds), then run this lab again.")
data = np.load(path)
X_train, y_train = data["X_train"].astype("float32"), data["y_train"].astype(int)
X_test, y_test = data["X_test"].astype("float32"), data["y_test"].astype(int)
classes = data["classes"].tolist()
print(X_train.shape, X_test.shape)
print(classes)

# TODO 1: the do-nothing baseline. Find the most frequent class in y_train (np.bincount(...).argmax())
#         and print its name and the test accuracy of always predicting it.
print("TODO 1: majority-class baseline")

# %% Step 2: a cross-validated head (Module 1 habits on deep-learning features)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO 2: 5-fold CV accuracy (mean ± std) for LogisticRegression(max_iter=2000), with and without a
#         StandardScaler in front (make_pipeline). Does scaling help? Then fit the plain one on all of
#         X_train, call it `head`, and print its test accuracy ONCE.
head = None
if head is None:
    print("TODO 2: cross-validate two heads, then one test score")

# %% Step 3: no training at all: k-nearest neighbors on the features
from sklearn.neighbors import KNeighborsClassifier

# TODO 3: test accuracy of KNeighborsClassifier(n_neighbors=k, metric="cosine") for k = 1, 5, 15.
print("TODO 3: k-nearest neighbors for k = 1, 5, 15")

# %% Step 4: which land types are hard?
# TODO 4: with pred = head.predict(X_test), compute the accuracy per class (pred[y_test == i] == i).mean()
#         and print the three weakest. Then a pd.crosstab of true vs predicted class names: which
#         wrong pair is the most common? Before you run it, predict: ______________
print("TODO 4: per-class accuracy and the most common mix-up")

# %% Step 5: a "send to a human" queue for unsure tiles
# TODO 5: confidence = head.predict_proba(X_test).max(axis=1). For cuts 0.6 and 0.8, print the share of
#         tiles below the cut (they go to a human), the accuracy inside that queue and on the rest.
print("TODO 5: the human-review queue")

# %% Stretch: how many of the 512 numbers does the head need?
# make_pipeline(PCA(n_components=n, random_state=42), LogisticRegression(max_iter=2000)) for n = 10, 50, 200.

print("\nExpected when done: baseline SeaLake 0.126; CV accuracy 0.899 ± 0.011 plain, 0.896 ± 0.009 scaled;")
print("test accuracy 0.903; kNN 0.850 / 0.866 / 0.861 for k = 1 / 5 / 15; weakest River 0.759, Highway 0.835,")
print("AnnualCrop 0.86; most common mix-up River -> Highway (11 tiles); below 0.6: 3.8% of tiles, accuracy")
print("0.474 inside vs 0.920 outside; below 0.8: 10.5%, 0.571 vs 0.942. Stretch: PCA 10/50/200 -> 0.808/0.870/0.896.")
