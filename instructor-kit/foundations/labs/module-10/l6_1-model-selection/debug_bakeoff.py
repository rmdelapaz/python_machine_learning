# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 6.1: a churn bake-off with FOUR bugs (deliberately broken handout).

Two bugs stop the script with an error: read the LAST line, fix, run again. Two more raise nothing
and quietly make the comparison unfair. Find them by reading the code against the lesson's
model-selection checklist (same folds, one metric, test set opened once, after the choice).

When everything is fixed the script prints:
  logistic  CV AUC 0.832 +/- 0.017
  tree_d5   CV AUC 0.842 +/- 0.018
  tree_d5 - logistic: mean +0.010, wins 8 of 10 folds
  best depth on the validation curve: 4
  chosen: tree_d5   test AUC (one look) 0.846
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

customers = pd.read_csv(DATA / "customers_churn.csv")
numeric = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
           "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
categorical = ["home_store", "region", "age_group", "loyalty_tier", "preferred_category"]
X_train, X_test, y_train, y_test = train_test_split(customers[numeric + categorical], customers["churned"],
                                                    test_size=0.25, stratify=customers["churned"], random_state=42)

def make_preprocess():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
    ])

candidates = {"logistic": make_pipeline(make_preprocess(), LogisticRegression(max_iter=1000)),
              "tree_d5": make_pipeline(make_preprocess(), DecisionTreeClassifier(max_depth=5, random_state=42))}

# 1. ten scores per candidate
folds = {}
for seed, (name, model) in enumerate(candidates.items()):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    folds[name] = cross_validate(model, X_train, y_train, cv=cv, scoring="auc")["test_score"]
    print(f"{name:9s} CV AUC {folds[name].mean():.3f} +/- {folds[name].std():.3f}")

# 2. is the gap real?
diff = folds["tree_d5"] - folds["logistic"]
print(f"tree_d5 - logistic: mean {diff.mean():+.3f}, wins {(diff > 0).sum()} of {len(diff)} folds")

# 3. the tree's depth
_, val = validation_curve(candidates["tree_d5"], X_train, y_train, param_name="max_depth",
                          param_range=[2, 3, 4, 5, 6], cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="roc_auc")
print("best depth on the validation curve:", [2, 3, 4, 5, 6][int(np.argmax(val.mean(axis=1)))])

# 4. pick the winner and report it
test_auc = {name: roc_auc_score(y_test, m.fit(X_train, y_train).predict_proba(X_test)[:, 1])
            for name, m in candidates.items()}
chosen = max(test_auc, key=test_auc.get)
print(f"chosen: {chosen}   test AUC (one look) {test_auc[chosen]:.3f}")
