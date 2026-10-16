# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.1 B: explaining the Foundations churn pipeline, categories and all.

Lesson 7.1 explained models on 11 numeric features. Real pipelines also one-hot encode categories.
Here you explain the full churn pipeline from Machine Learning Foundations (Lesson 2.2: 16 business
columns, a ColumnTransformer called preprocess, steps named "prep" and "model"): its coefficients as
odds ratios, the traps of overlapping columns, and permutation importance per BUSINESS column. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1: the Foundations pipeline on the canonical split (done for you)
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
complete = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
nominal = ["home_store", "region", "age_group", "preferred_category"]
features = complete + ["email_opens_90d", "loyalty_tier"] + nominal

train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
X_train, y_train = train[features], train["churned"]
X_test, y_test = test[features], test["churned"]

preprocess = ColumnTransformer([
    ("scale", StandardScaler(), complete),
    ("email", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
     ["email_opens_90d"]),
    ("tier", make_pipeline(OrdinalEncoder(categories=[["Basic", "Silver", "Gold"]]), StandardScaler()),
     ["loyalty_tier"]),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), nominal),
])
logreg = Pipeline([("prep", preprocess), ("model", LogisticRegression())]).fit(X_train, y_train)
print(f"logistic regression pipeline: test ROC AUC {roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]):.3f}")

# %% Step 2: coefficients as odds ratios, with the names the pipeline made
# TODO 1: names = logreg["prep"].get_feature_names_out(). Build a Series of np.exp(logreg["model"].coef_[0])
#         indexed by names, sort it, and print how many model columns there are, then the three lowest
#         and three highest odds ratios (.round(2).to_dict()).
names = None
if names is None:
    print("TODO 1: odds ratios with the pipeline's own column names")

# %% Step 3: two coefficients that mislead
# TODO 2: print the odds ratios of the five home_store columns, of onehot__region_Southeast and of
#         tier__loyalty_tier. Then print the REAL churn rate by loyalty_tier (a groupby on customers).
#         Two puzzles: why is Asheville "protective" while Southeast is "risky"? Why does tier get an odds
#         ratio above 1 when Gold customers churn least? (Hint: which columns already carry that fact?)
print("TODO 2: home_store, region and tier coefficients, and the real churn rate by tier")

# %% Step 4: permutation importance per business column (the boosting pipeline)
# TODO 3: boost = Pipeline([("prep", clone(preprocess)), ("model", HistGradientBoostingClassifier(random_state=42))]),
#         fit it, print its test ROC AUC. Then permutation_importance(boost, X_test, y_test,
#         scoring="roc_auc", n_repeats=5, random_state=42) on the RAW X_test: one score per business
#         column. Print the top 4 and the scores of the four nominal columns plus loyalty_tier.
print("TODO 3: permutation importance per business column")

# %% Step 5: shuffle the five descriptive columns together
# TODO 4: shuffle home_store, region, age_group, preferred_category and loyalty_tier with ONE shared
#         random order (rng = np.random.default_rng(42), 5 repeats) and print the mean and std of the AUC drop.
# TODO 5 (think): the model gains nothing from age_group. Does that prove it treats age groups alike?
print("TODO 4: grouped importance of the descriptive columns")

print("\nExpected when done: 40 model columns from 16; lowest spend_12m 0.65, home_store_Asheville 0.66; highest")
print("days_since_last_order 2.49; Asheville 0.66 but region Southeast 1.32; tier 1.48 although churn by tier is")
print("Basic 0.217 / Silver 0.129 / Gold 0.065; boosting AUC 0.863, top driver days_since_last_order 0.143;")
print("every descriptive column about 0, and all five shuffled together -0.002 +/- 0.002.")
