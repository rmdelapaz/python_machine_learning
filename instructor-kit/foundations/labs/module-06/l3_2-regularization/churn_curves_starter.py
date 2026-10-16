# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.2 A: curves and penalties for a churn classifier.

Lesson 3.2 regularized regressions. The same ideas apply to logistic regression, whose dial is C (the
INVERSE of the penalty strength: small C, strong penalty). Give a churn model polynomial features, watch
it overfit a 1,000-customer pilot, let cross-validation choose C, and see what the full training set
changes. The test set is used once, at the end. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1 (done for you): the canonical split and a 1,000-customer pilot
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)   # C=np.inf on 285 columns: unpenalized fits may not converge
customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
pilot = train.sample(1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def poly_logit(degree, C):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))


print(len(pilot), "pilot customers")

# %% Step 2: flexibility without a penalty
# TODO 1: for degree 1, 2 and 3, cross_validate(poly_logit(degree, np.inf), pilot[features], pilot["churned"],
#         cv=cv, scoring="roc_auc", return_train_score=True) and print the mean train and CV AUC (and the number
#         of columns, PolynomialFeatures(degree, include_bias=False).fit(...).n_output_features_).
#         Which degree overfits, and how can you tell without a test set?
print("TODO 1: train and CV AUC for degrees 1 to 3")

# %% Step 3: the dial
from sklearn.model_selection import validation_curve

Cs = [0.001, 0.01, 0.1, 1, 10]
# TODO 2: validation_curve(poly_logit(3, 1.0), ..., param_name="logisticregression__C", param_range=Cs,
#         cv=cv, scoring="roc_auc") on the pilot. Print mean train and CV AUC per C, and the C with the best CV AUC.
#         Which way is "more penalty": small C or large C?
print("TODO 2: validation curve over C")

# %% Step 4: more data is also a regularizer
# TODO 3: on ALL training customers (train), CV AUC for degrees 1, 2, 3 with C=np.inf and C=0.1. Put the six
#         numbers in a table (rows: degree, columns: C). What changed compared with the pilot?
rows = []
print("TODO 3: the same comparison on 4,500 customers")

# %% Step 5: one test
from sklearn.metrics import roc_auc_score

# TODO 4: pick the row with the best CV AUC, fit that model on train, and print its test AUC next to a plain
#         degree-1 model (poly_logit(1, 1.0)).
# TODO 5: one sentence: why is it fair to look at the test set now, and not in Step 2?
print("TODO 4-5: the chosen model, tested once")

print("\nExpected when done: pilot degree 1/2/3: train AUC 0.823/0.899/1.000, CV AUC 0.807/0.807/0.706; degree 3")
print("by C: CV AUC 0.808, 0.810, 0.809, 0.776, 0.730 for C = 0.001 to 10 (best C=0.01); on 4,500 customers")
print("degree 2 reaches 0.851 with or without a penalty, degree 3 needs C=0.1 (0.851 against 0.827); test AUC of")
print("the chosen degree-2 model 0.853 against 0.823 for the plain model.")
