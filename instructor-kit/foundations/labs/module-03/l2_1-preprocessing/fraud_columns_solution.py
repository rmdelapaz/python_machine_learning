# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.1 A (SOLUTION): bring the product category into the fraud model.

Every fraud model so far used nine numeric columns and left `category` (text) on the shelf. The
payments team wants it in, and wants proof that the model copes with next month's feed: a product
category the shop has never sold (Bikes) and a guest checkout with no account age. One
ColumnTransformer: impute and scale the numbers, one-hot the category, all learned from training rows.
"""
# %% Step 1: load and split (stratified 70/30, as in every fraud lab)
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
rate = y_train.groupby(X_train["category"]).mean().sort_values(ascending=False)
print("\nfraud rate by category (training rows):")
print(rate.round(4).to_string())

# %% Step 3: one ColumnTransformer for the whole table
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"]),
]).set_output(transform="pandas")
Xt_train = preprocess.fit_transform(X_train)
Xt_test = preprocess.transform(X_test)
print()
print(Xt_train.shape, Xt_test.shape)
print(Xt_train.columns[7:12].tolist())

# %% Step 4: does the category help? Judge with the metrics that suit a rare class (Lesson 1.4)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

model = LogisticRegression().fit(Xt_train, y_train)
p_all = model.predict_proba(Xt_test)[:, 1]
num_cols = [c for c in Xt_train.columns if c.startswith("num__")]
model_num = LogisticRegression().fit(Xt_train[num_cols], y_train)
p_num = model_num.predict_proba(Xt_test[num_cols])[:, 1]
print()
for name, p in [("9 numeric columns", p_num), ("+ category (19 columns)", p_all)]:
    print(f"{name:24s} ROC AUC {roc_auc_score(y_test, p):.3f}   average precision {average_precision_score(y_test, p):.3f}")
weights = pd.Series(model.coef_[0], index=Xt_train.columns)
print(f"weight of cat__category_Gift Cards: {weights['cat__category_Gift Cards']:.2f} "
      f"(largest of all: {weights.idxmax()})")

# %% Step 5: what did the preprocessor learn?
imputer = preprocess.named_transformers_["num"][0]
encoder = preprocess.named_transformers_["cat"]
print(f"\nfill value for account_age_days: {imputer.statistics_[1]:.0f} days")
print(f"categories learned: {len(encoder.categories_[0])}")

# %% Step 6: next month's feed: a category never seen, a guest checkout with no account age
new = pd.DataFrame({
    "amount": [480.0, 129.0, 95.5], "account_age_days": [3.0, np.nan, 1200.0], "hour": [3, 14, 20],
    "items": [4, 1, 2], "device_new": [1, 0, 0], "billing_shipping_mismatch": [1, 0, 0],
    "country_mismatch": [0, 0, 0], "card_attempts": [3, 1, 1], "velocity_24h": [4, 0, 0],
    "category": ["Gift Cards", "Bikes", "Footwear"],
}, index=["P1", "P2", "P3"])
new_t = preprocess.transform(new)
print("\nP2's category columns (Bikes was never seen):", new_t.filter(like="cat__").loc["P2"].sum())
print("P2's account age after imputation (scaled):", round(new_t.loc["P2", "num__account_age_days"], 3))
print(pd.Series(model.predict_proba(new_t)[:, 1], index=new.index, name="fraud_probability").round(3).to_string())
