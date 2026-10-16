# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.4 B: a data vendor's 1,000 columns.

A data vendor offers Pinecrest 1,000 extra attributes per customer and sends a free sample for a pilot of
300 customers. The pitch: "pick the 10 attributes that predict churn best, and watch your score jump".
In this lab the vendor's columns are SIMULATED as pure random noise, so you know the truth: they cannot
help. The question is whether your evaluation would have told you so. Work through TODO 1 to 5;
the file runs as it is.
"""
# %% Step 1 (done for you): the pilot: 300 training customers, the 10 real features and 1,000 vendor columns
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
# TODO 1: cross-validate `plain` (scoring="roc_auc") on X_pilot[real] and on all of X_pilot. Print both means.
#         Before you run it, write your prediction: do 1,000 extra columns help or hurt? ______
print("TODO 1: CV AUC with the 10 real columns, and with all 1,010")

# %% Step 3: the vendor's recipe
from sklearn.feature_selection import SelectKBest, f_classif

# TODO 2: fit SelectKBest(f_classif, k=10) on ALL of X_pilot and y_pilot; get_feature_names_out() gives
#         the 10 picked columns. Cross-validate `plain` on X_pilot[picked] and print the mean.
# TODO 3: how many of the 10 picked columns are real ones? (sum(c in real for c in picked))
print("TODO 2-3: select first, then CV; how many real columns were picked")

# %% Step 4: the honest recipe
# TODO 4: put SelectKBest(f_classif, k=10) INSIDE the pipeline, in front of the scaler and the model,
#         and cross-validate the whole pipeline on X_pilot. Print the mean.
print("TODO 4: select inside CV")

# %% Step 5: the truth
from sklearn.metrics import roc_auc_score

# TODO 5: X_test, y_test = with_vendor_columns(test), test["churned"]. Fit a scaled logistic regression on
#         the pilot with the 10 real columns, and another with the vendor's 10 picks. Print each one's test
#         AUC on the 1,500 test customers. Which pilot score told the truth?
print("TODO 5: test AUC of the real 10 and of the vendor's 10 picks")

print("\nExpected when done: CV AUC 0.745 (10 real) and 0.603 (all 1,010); select first, then CV 0.825, with")
print("only 3 real columns among the 10 picked; select inside CV 0.660; test AUC 0.794 for the real 10 and")
print("0.609 for the vendor's picks.")
