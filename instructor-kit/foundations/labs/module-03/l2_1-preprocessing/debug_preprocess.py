# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 2.1). This file is DELIBERATELY BROKEN.

It should prepare 12 churn columns with one ColumnTransformer and report a logistic regression's test
ROC AUC. Two bugs stop it with an error, one at a time: run it, read the LAST line of the traceback,
fix that one bug, run again. Two more bugs raise no error at all. Find them too: they change the
numbers without telling you.

When everything is fixed it prints:
    prepared: (4500, 31) (1500, 31)
    largest |mean| of the scaled TEST columns: 0.055
    loyalty_tier codes: Basic 0.0, Silver 1.0, Gold 2.0
    test ROC AUC: 0.824
(With the two silent bugs still in, it prints 0.0, "Silver 2.0, Gold 1.0" and 0.823: a mean of exactly
0.0 on the test set is a clue, not a success.)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
complete = ["tenure_months", "orders_12m", "days_since_last_order", "support_tickets_12m",
            "discount_share", "return_rate", "app_user"]
features = complete + ["email_opens_90d", "age_group", "loyalty_tier", "home_store", "preferred_category"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42)

preprocess = ColumnTransformer([
    ("scale", StandardScaler(), complete),
    ("gaps", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
     ["email_opens_90d", "age_group"]),
    ("tier", OrdinalEncoder(), ["loyalty_tier"]),
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["home_store", "preferred_category"]),
]).set_output(transform="pandas")

Xt_train = preprocess.fit_transform(X_train)
Xt_test = preprocess.fit_transform(X_test)
print("prepared:", Xt_train.shape, Xt_test.shape)
scaled = [c for c in Xt_test.columns if c.startswith("scale__")]
print("largest |mean| of the scaled TEST columns:", round(Xt_test[scaled].mean().abs().max(), 3))

codes = preprocess.named_transformers_["tier"].transform(pd.DataFrame({"loyalty_tier": ["Basic", "Silver", "Gold"]}))
print("loyalty_tier codes: " + ", ".join(f"{t} {c}" for t, c in zip(["Basic", "Silver", "Gold"], codes.to_numpy().ravel())))

model = LogisticRegression().fit(Xt_train, y_train)
print(f"test ROC AUC: {roc_auc_score(y_test, model.predict_proba(Xt_test)[:, 1]):.3f}")
