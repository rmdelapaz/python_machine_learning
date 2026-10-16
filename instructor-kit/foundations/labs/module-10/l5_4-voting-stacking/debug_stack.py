# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 5.4: a churn ensemble with FOUR bugs (deliberately broken handout).

Three bugs stop the script with an error. Fix them one at a time: run, read the LAST line of the
message, fix, run again. The fourth bug raises nothing: the script finishes and prints a respectable
number. Find it by reading the code, and say what it does to the meta-model's weights.

When everything is fixed the script prints:
  soft vote ROC AUC 0.858
  stack weights: [2.58 3.44]   stack ROC AUC 0.860
"""
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
numeric = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
           "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
categorical = ["home_store", "region", "age_group", "loyalty_tier", "preferred_category"]
X = customers[numeric + categorical].copy()
X[categorical] = X[categorical].astype("category")
y = customers["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

logreg = make_pipeline(
    make_column_transformer((make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                            (make_pipeline(SimpleImputer(strategy="most_frequent"),
                                           OneHotEncoder(handle_unknown="ignore")), categorical)),
    LogisticRegression(max_iter=1000))
boost = HistGradientBoostingClassifier(random_state=42)

# 1. a soft vote of the two members
vote = VotingClassifier([logreg, boost], voting="hard").fit(X_train, y_train)
print(f"soft vote ROC AUC {roc_auc_score(y_test, vote.predict_proba(X_test)[:, 1]):.3f}")

# 2. a stack of the same two members
stack = StackingClassifier(estimators=[("logreg", logreg), ("boost", boost)],
                           final_estimator=LogisticRegression(), cv=TimeSeriesSplit(n_splits=5))
stack.fit(X_train, y_train)

# 3. the same stack by hand, to check the weights
logreg.fit(X_train, y_train)
boost.fit(X_train, y_train)
meta_X = pd.DataFrame({"logreg": logreg.predict_proba(X_train)[:, 1],
                       "boost": boost.predict_proba(X_train)[:, 1]})
meta = LogisticRegression().fit(meta_X, y_train)
test_X = pd.DataFrame({"logreg": logreg.predict_proba(X_test)[:, 1], "boost": boost.predict_proba(X_test)[:, 1]})
print("stack weights:", meta.coef_[0].round(2),
      f"  stack ROC AUC {roc_auc_score(y_test, meta.predict_proba(test_X)[:, 1]):.3f}")
