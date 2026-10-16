# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.2 A (SOLUTION): can the payment details tell you what was bought?

Lesson 9.2 separated churners and loyalty tiers. Here the classes are the ten product categories of
card payments, and the features are the nine numeric payment columns. How many discriminant axes are
there, what do they lean on, and how much separation is there really? An honest weak result is a result.
"""
# %% Step 1: ten categories, nine features, a stratified split
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

# %% Step 2: how many axes, and how much separation does each carry?
lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(X_train, c_train)
step = lda[-1]
print("axes:", step.transform(lda[0].transform(X_train[:1])).shape[1])
print("separation carried:", step.explained_variance_ratio_.round(3))

# %% Step 3: what does LD1 lean on?
print(pd.Series(step.scalings_[:, 0], index=numeric).round(2).sort_values().to_string())

# %% Step 4: LD1 per category, in amount order
ld = pd.DataFrame(lda.transform(X_test)[:, :2], columns=["LD1", "LD2"], index=X_test.index)
ld["category"] = c_test
ld["amount"] = X_test["amount"]
print(ld.groupby("category")[["LD1", "amount"]].median().round(2).sort_values("LD1").to_string())

# %% Step 5: as a classifier, against the baseline
base = DummyClassifier(strategy="most_frequent").fit(X_train, c_train)
print(f"\nbaseline accuracy {base.score(X_test, c_test):.3f}   LDA accuracy {lda.score(X_test, c_test):.3f}")
pred = pd.Series(lda.predict(X_test), index=X_test.index)
print("recall by category:", (pred == c_test).groupby(c_test).mean().round(2).sort_values().to_dict())
