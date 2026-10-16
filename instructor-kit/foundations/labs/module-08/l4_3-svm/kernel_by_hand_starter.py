# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.3 B: the kernel trick, by hand and by kernel, on Pinecrest churn.

Lesson 4.3 lifted the ring data into 3-D with one hand-made feature (x² + y²). Here you do the same on
the real churn table: build every squared and pairwise-product feature yourself, feed them to a STRAIGHT
boundary (LinearSVC), and compare with a polynomial kernel that never builds them. Then count what
degree 3 would cost, and see where the RBF kernel lands.
Work through TODO 1 to 5; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the churn split from Lesson 1.1, and a report helper (done for you)
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)


def report(name, model):
    """Fit, then print test ROC AUC (from decision_function), churners in the top 50 and the fit time."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    scores = model.decision_function(X_test)
    top50 = int(y_test.to_numpy()[np.argsort(-scores, kind="stable")[:50]].sum())
    print(f"{name:30s} test AUC {roc_auc_score(y_test, scores):.3f}   top-50 churners {top50:2d}   fit {seconds:.2f} s")
    return model


# %% Step 2: a straight boundary on the ten features
# TODO 1: report("LinearSVC, 10 features", make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced")))
print("TODO 1: the straight boundary")

# %% Step 3: lift the data by hand: every square and every product of two features
# TODO 2: lift = PolynomialFeatures(degree=2).fit(X_train). Print lift.n_output_features_ and
#         lift.get_feature_names_out(features)[[1, 11, 15, 65]]. Then report a pipeline
#         StandardScaler -> PolynomialFeatures(degree=2) -> LinearSVC(class_weight="balanced") and keep it
#         as `lifted`. Predict first: will the AUC go up, down or stay put? ______
lifted = None
if lifted is None:
    print("TODO 2: lift by hand, then a straight boundary in the lifted space")

# %% Step 4: the kernel trick: the same degree-2 space, never built
# TODO 3: report SVC(kernel="poly", degree=2, coef0=1, class_weight="balanced") behind a StandardScaler,
#         and print how many support vectors it keeps (pipe[-1].n_support_.sum()).
print("TODO 3: the polynomial kernel")

# %% Step 5: what higher degrees would cost if you built the features yourself
# TODO 4: for degree 2 to 5, print PolynomialFeatures(degree=degree).fit(X_train).n_output_features_.
print("TODO 4: column counts for degrees 2 to 5")

# %% Step 6: the RBF kernel: an infinite lift, still no columns built
# TODO 5: report SVC(class_weight="balanced") (the default kernel is "rbf") behind a StandardScaler.
print("TODO 5: the RBF kernel")

# %% Stretch: which lifted columns does the straight boundary lean on most?
# Put lifted[-1].coef_[0] in a Series indexed by lifted[1].get_feature_names_out(features), drop "1",
# and print the five largest by absolute value. Which one is the tenure U-shape from Lesson 5.1?

print("\nExpected when done: LinearSVC AUC 0.826 (33 churners in the top 50); 66 columns after the lift")
print("(including 'tenure_months^2' and 'tenure_months days_since_last_order'); lift + LinearSVC AUC 0.852;")
print("poly kernel AUC 0.852 with 2,293 support vectors; 66 / 286 / 1,001 / 3,003 columns for degrees")
print("2 to 5; RBF AUC 0.850. Fit times are yours (they differ from machine to machine).")
