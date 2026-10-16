# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.1 A: bring the product category into the fraud model.

Every fraud model so far used nine numeric columns and left `category` (text) on the shelf. The
payments team wants it in, and wants proof that the model copes with next month's feed: a product
category the shop has never sold (Bikes) and a guest checkout with no account age.
Work through TODO 1 to 6; run the file after each one. It runs as it is.
"""
# %% Step 1: load and split (done for you: stratified 70/30, as in every fraud lab)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 120)
tx = pd.read_csv(DATA / "transactions.csv")
numeric = ["amount", "account_age_days", "hour", "items", "device_new",
           "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(
    tx[numeric + ["category"]], tx["is_fraud"], test_size=0.3, stratify=tx["is_fraud"], random_state=42)
print(X_train.shape, X_test.shape, "| text columns:", X_train.select_dtypes("str").columns.tolist())

# %% Step 2: look before you encode: is the category worth a column? (training rows only)
# TODO 1: the fraud rate per category on the TRAINING rows: y_train.groupby(X_train["category"]).mean(),
#         sorted from high to low, rounded to 4 decimals. Which category stands out?
print("TODO 1: fraud rate by category")

# %% Step 3: one ColumnTransformer for the whole table
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO 2: two branches, then .set_output(transform="pandas"):
#         "num": make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) on `numeric`
#               (the training data has no gaps, but next month's feed will)
#         "cat": OneHotEncoder(sparse_output=False, handle_unknown="ignore") on ["category"]
#         fit_transform on X_train, transform on X_test; print both shapes and columns 7 to 11.
preprocess = None
if preprocess is None:
    print("TODO 2: the ColumnTransformer, fitted on the training rows")

# %% Step 4: does the category help? Judge with the metrics that suit a rare class (Lesson 1.4)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# TODO 3: fit LogisticRegression() on all 19 prepared columns, and a second one on the 9 "num__" columns
#         only. Print ROC AUC and average precision for both (probabilities, not labels).
#         Predict first: will both metrics agree? ______
# TODO 4: the weight of "cat__category_Gift Cards" (pd.Series(model.coef_[0], index=...)), and the
#         name of the largest weight.
print("TODO 3-4: with and without the category: ROC AUC, average precision, the Gift Cards weight")

# %% Step 5: what did the preprocessor learn?
# TODO 5: preprocess.named_transformers_["num"][0] is the fitted imputer: print its fill value for
#         account_age_days (statistics_[1]). preprocess.named_transformers_["cat"].categories_ holds
#         the categories: how many?
print("TODO 5: the learned fill value and the number of categories")

# %% Step 6: next month's feed: a category never seen, a guest checkout with no account age
new = pd.DataFrame({
    "amount": [480.0, 129.0, 95.5], "account_age_days": [3.0, np.nan, 1200.0], "hour": [3, 14, 20],
    "items": [4, 1, 2], "device_new": [1, 0, 0], "billing_shipping_mismatch": [1, 0, 0],
    "country_mismatch": [0, 0, 0], "card_attempts": [3, 1, 1], "velocity_24h": [4, 0, 0],
    "category": ["Gift Cards", "Bikes", "Footwear"],
}, index=["P1", "P2", "P3"])
# TODO 6: transform `new` with the SAME fitted preprocess (never fit it again). Print the sum of P2's
#         "cat__" columns, P2's scaled account age, and the three fraud probabilities (3 decimals).
print("TODO 6: score P1, P2 and P3")

print("\nExpected when done: Gift Cards 0.0546, every other category 0.0057 to 0.0109; (35000, 19) and")
print("(15000, 19); 9 numeric columns ROC AUC 0.848 / AP 0.444, with the category ROC AUC 0.843 / AP 0.463;")
print("Gift Cards weight 0.95, the largest of all; fill value 447 days, 10 categories; P2's category")
print("columns sum to 0.0 and its scaled account age is -0.311; P1 1.000, P2 0.001, P3 0.000.")
