# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Final project scaffold (instructor reference): the leak-safe split, the bar to beat and a first
cross-validated model for all three final-project problems.

Learners run the starter for ONE problem (PROBLEM = "churn", "demand" or "fraud"). This reference runs
all three, so you can check any learner's first milestone at a glance. Every split is the course's
canonical one; the test rows are used here only for the do-nothing baselines, which learn nothing.
"""
# %% Shared imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

# %% Problem A: churn (stratified 75/25, random_state=42; $15 per offer, $60 per churner reached)
customers = pd.read_csv(DATA / "customers_churn.csv")
numeric = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
           "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
categorical = ["home_store", "region", "age_group", "loyalty_tier", "preferred_category"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
assert "cancellation_reason" not in numeric + categorical     # the leak stays out
print(f"CHURN  train {len(train):,} / test {len(test):,}; first test id {test['customer_id'].iloc[0]}")
y_test = test["churned"]
print(f"  bar to beat: random-order AP {y_test.mean():.3f}; contact everyone {60 * y_test.sum() - 15 * len(y_test):+,} dollars;"
      f" contact nobody 0 dollars; break-even probability {15 / 60:.2f}")
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical)])
ap = cross_val_score(make_pipeline(prep, LogisticRegression(max_iter=1000)), train[numeric + categorical],
                     train["churned"], cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="average_precision")
print(f"  first model: logistic regression, 5-fold CV AP {ap.mean():.3f} +/- {ap.std():.3f}")

# %% Problem B: weekly demand (train 2024, test 2025; MAE in units)
demand = pd.read_csv(DATA / "weekly_demand.csv")
features = ["store", "product_id", "category", "week_of_year", "list_price",
            "discount_pct", "price", "promo", "holiday_week", "avg_high_temp_f"]
cats = ["store", "product_id", "category"]
is_train = demand["year"] == 2024
d_train, d_test = demand[is_train], demand[~is_train]
print(f"DEMAND train {len(d_train):,} (2024) / test {len(d_test):,} (2025)")
product_mean = d_train.groupby("product_id")["units_sold"].mean()
print(f"  bar to beat: overall mean MAE {mean_absolute_error(d_test['units_sold'], np.full(len(d_test), d_train['units_sold'].mean())):.2f};"
      f" per-product mean MAE {mean_absolute_error(d_test['units_sold'], d_test['product_id'].map(product_mean)):.2f}")
Xd = d_train[features].copy()
Xd[cats] = Xd[cats].astype("category")
mae = -cross_val_score(HistGradientBoostingRegressor(loss="poisson", random_state=42), Xd, d_train["units_sold"],
                       cv=KFold(5), scoring="neg_mean_absolute_error")
print(f"  first model: Poisson boosting, CV on blocks of 2024 weeks, MAE {mae.mean():.2f} +/- {mae.std():.2f}")

# %% Problem C: fraud (train January to May, test June; $10 per review, $375 per missed fraud)
tx = pd.read_csv(DATA / "transactions.csv")
num = ["account_age_days", "amount", "hour", "items", "device_new",
       "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
is_june = tx["timestamp"] >= "2025-06-01"
t_train, t_test = tx[~is_june], tx[is_june]
print(f"FRAUD  train {len(t_train):,} Jan-May ({t_train['is_fraud'].sum()} frauds) / test {len(t_test):,} June ({t_test['is_fraud'].sum()} frauds)")
def fraud_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(10 * (tp + fp) + 375 * fn)
yt = t_test["is_fraud"]
print(f"  bar to beat: review nothing ${fraud_cost(yt, np.zeros(len(yt), dtype=int)):,};"
      f" review everything ${fraud_cost(yt, np.ones(len(yt), dtype=int)):,}; break-even probability {10 / 375:.3f}")
prep_t = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"])])
ap_t = cross_val_score(make_pipeline(prep_t, LogisticRegression(max_iter=1000)), t_train[num + ["category"]],
                       t_train["is_fraud"], cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="average_precision")
print(f"  first model: logistic regression, 5-fold CV AP on Jan-May {ap_t.mean():.3f} +/- {ap_t.std():.3f}")
