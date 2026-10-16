# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.4 B (SOLUTION): a data vendor's 1,000 columns.

A data vendor offers Pinecrest 1,000 extra attributes per customer and sends a free sample for a pilot of
300 customers. The pitch: "pick the 10 attributes that predict churn best, and watch your score jump".
In this lab the vendor's columns are SIMULATED as pure random noise, so you know the truth: they cannot
help. The question is whether your evaluation would have told you so.
"""
# %% Step 1: the pilot: 300 training customers, the 10 real features and 1,000 vendor columns
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
real = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
        "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
pilot = train.sample(300, random_state=42)

rng = np.random.default_rng(42)


def with_vendor_columns(frame, n=1000):
    vendor = pd.DataFrame(rng.normal(size=(len(frame), n)), index=frame.index).add_prefix("vendor_")
    return pd.concat([frame[real], vendor], axis=1)


X_pilot, y_pilot = with_vendor_columns(pilot), pilot["churned"]
print(X_pilot.shape, f"pilot churn rate {y_pilot.mean():.3f}")

# %% Step 2: two reference scores
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plain = make_pipeline(StandardScaler(), LogisticRegression())
real_only = cross_val_score(plain, X_pilot[real], y_pilot, cv=cv, scoring="roc_auc").mean()
everything = cross_val_score(plain, X_pilot, y_pilot, cv=cv, scoring="roc_auc").mean()
print(f"10 real columns     CV AUC {real_only:.3f}")
print(f"all 1,010 columns   CV AUC {everything:.3f}")

# %% Step 3: the vendor's recipe: choose the 10 best columns on all 300 rows, then cross-validate
from sklearn.feature_selection import SelectKBest, f_classif

picked = SelectKBest(f_classif, k=10).fit(X_pilot, y_pilot).get_feature_names_out()
leaky = cross_val_score(plain, X_pilot[picked], y_pilot, cv=cv, scoring="roc_auc").mean()
print(f"\nselect first, then CV  CV AUC {leaky:.3f}")
print(f"real columns among the 10 picked: {sum(c in real for c in picked)}")

# %% Step 4: the honest recipe: selection is a pipeline step, redone inside every fold
honest_pipe = make_pipeline(SelectKBest(f_classif, k=10), StandardScaler(), LogisticRegression())
honest = cross_val_score(honest_pipe, X_pilot, y_pilot, cv=cv, scoring="roc_auc").mean()
print(f"select inside CV       CV AUC {honest:.3f}")

# %% Step 5: the truth: 1,500 customers the pilot never saw (with their own vendor columns)
from sklearn.metrics import roc_auc_score

X_test, y_test = with_vendor_columns(test), test["churned"]
print()
for name, cols in [("10 real columns", real), ("the vendor's 10 picks", list(picked))]:
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_pilot[cols], y_pilot)
    print(f"{name:22s} test AUC {roc_auc_score(y_test, model.predict_proba(X_test[cols])[:, 1]):.3f}")
