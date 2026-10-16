# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.2 A: can the payment details tell you what was bought?

Lesson 9.2 separated churners and loyalty tiers. Here the classes are the ten product categories of
card payments, and the features are the nine numeric payment columns. How many discriminant axes are
there, what do they lean on, and how much separation is there really? Work through TODO 1 to 5.
"""
# %% Step 1: ten categories, nine features, a stratified split (done for you)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

pd.set_option("display.width", 120)
tx = pd.read_csv(DATA / "transactions.csv")
numeric = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, c_train, c_test = train_test_split(tx[numeric], tx["category"], test_size=0.25,
                                                    stratify=tx["category"], random_state=42)
print(X_train.shape, "classes:", c_train.nunique())

# %% Step 2: axes and separation
# TODO 1: lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(X_train, c_train).
#         Predict first: how many axes can ten classes and nine features give? Then print the number of
#         columns of lda.transform(X_train) and lda[-1].explained_variance_ratio_.round(3).
lda = None
if lda is None:
    print("TODO 1: fit the LDA pipeline; number of axes; separation per axis")

# %% Step 3: what does LD1 lean on?
# TODO 2: pd.Series(lda[-1].scalings_[:, 0], index=numeric).round(2).sort_values(). Which two columns
#         matter, and with which signs? Put LD1 into words.
print("TODO 2: LD1 weights")

# %% Step 4: LD1 per category
# TODO 3: on the test rows, the median LD1 and the median amount per category, sorted by LD1.
print("TODO 3: median LD1 and amount per category")

# %% Step 5: as a classifier
# TODO 4: baseline (most_frequent) accuracy vs LDA accuracy on the test rows.
# TODO 5: recall per category: (pred == c_test).groupby(c_test).mean(). Which categories does LDA never predict?
print("TODO 4-5: accuracy and recall by category")

print("\nExpected when done: 9 axes, but the first carries 0.995 of the separation; LD1 = amount (1.42) against")
print("items (-0.59): price per item; median LD1 runs from Accessories (-0.93, $65) to Tents (1.41, $350);")
print("accuracy 0.290 against a baseline of 0.219; six categories have recall 0.0, Apparel 0.86.")
