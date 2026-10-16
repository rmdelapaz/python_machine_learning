# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.1 B (SOLUTION): explaining the Foundations churn pipeline, categories and all.

Lesson 7.1 explained models on 11 numeric features. Real pipelines also one-hot encode categories.
Here you explain the full churn pipeline from Machine Learning Foundations (Lesson 2.2: 16 business
columns, a ColumnTransformer called preprocess, steps named "prep" and "model"): its coefficients as
odds ratios, the traps of overlapping columns, and permutation importance per BUSINESS column.
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
names = logreg["prep"].get_feature_names_out()
odds = pd.Series(np.exp(logreg["model"].coef_[0]), index=names).sort_values()
print(len(names), "model columns from", len(features), "business columns")
print("lowest odds ratios :", odds.head(3).round(2).to_dict())
print("highest odds ratios:", odds.tail(3).round(2).to_dict())

# %% Step 3: two coefficients that mislead
store = odds[[n for n in names if n.startswith("onehot__home_store")]]
print("home_store:", store.round(2).to_dict())
print("region Southeast:", round(odds["onehot__region_Southeast"], 2))
print("loyalty_tier (per step Basic -> Silver -> Gold, scaled):", round(odds["tier__loyalty_tier"], 2))
print("real churn rate by tier:", customers.groupby("loyalty_tier")["churned"].mean().round(3).to_dict())
# Asheville is the only Southeast store, so home_store and region share one fact and split its credit.
# Tier is computed from spend and tenure, which are already in the model: "holding them fixed" asks
# about a Gold customer with a Basic customer's spend, who does not exist, so the sign flips.

# %% Step 4: permutation importance per business column (the boosting pipeline)
boost = Pipeline([("prep", clone(preprocess)), ("model", HistGradientBoostingClassifier(random_state=42))])
boost.fit(X_train, y_train)
print(f"boosting pipeline: test ROC AUC {roc_auc_score(y_test, boost.predict_proba(X_test)[:, 1]):.3f}")
result = permutation_importance(boost, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42)
importance = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
print("top 4:", importance.head(4).round(3).to_dict())
print("the five descriptive columns:", importance[nominal + ["loyalty_tier"]].round(3).to_dict())

# %% Step 5: shuffle the five descriptive columns together
rng = np.random.default_rng(42)
base = roc_auc_score(y_test, boost.predict_proba(X_test)[:, 1])
group = nominal + ["loyalty_tier"]
drops = []
for _ in range(5):
    shuffled = X_test.copy()
    shuffled[group] = X_test[group].to_numpy()[rng.permutation(len(X_test))]
    drops.append(base - roc_auc_score(y_test, boost.predict_proba(shuffled)[:, 1]))
print(f"descriptive columns shuffled together: AUC drop {np.mean(drops):.3f} +/- {np.std(drops):.3f}")
