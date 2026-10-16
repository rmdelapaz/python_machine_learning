# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.3 B (SOLUTION): the kernel trick, by hand and by kernel, on Pinecrest churn.

Lesson 4.3 lifted the ring data into 3-D with one hand-made feature (x² + y²). Here you do the same on
the real churn table: build every squared and pairwise-product feature yourself, feed them to a STRAIGHT
boundary (LinearSVC), and compare with a polynomial kernel that never builds them. Then count what
degree 3 would cost, and see where the RBF kernel lands.
"""
# %% Step 1: the churn split from Lesson 1.1 (ten numeric features)
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
linear = report("LinearSVC, 10 features", make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced")))

# %% Step 3: lift the data by hand: every square and every product of two features
lift = PolynomialFeatures(degree=2).fit(X_train)
print("\ncolumns after the degree-2 lift:", lift.n_output_features_)
print("a few of them:", list(lift.get_feature_names_out(features)[[1, 11, 15, 65]]))
lifted = report("lift (deg 2) + LinearSVC",
                make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearSVC(class_weight="balanced")))

# %% Step 4: the kernel trick: the same degree-2 space, never built
poly_kernel = report("SVC(kernel='poly', degree=2)",
                     make_pipeline(StandardScaler(), SVC(kernel="poly", degree=2, coef0=1, class_weight="balanced")))
print("support vectors:", poly_kernel[-1].n_support_.sum(), "of", len(X_train))

# %% Step 5: what higher degrees would cost if you built the features yourself
for degree in [2, 3, 4, 5]:
    print(f"degree {degree}: {PolynomialFeatures(degree=degree).fit(X_train).n_output_features_:5d} columns")

# %% Step 6: the RBF kernel: an infinite lift, still no columns built
rbf = report("SVC(kernel='rbf')", make_pipeline(StandardScaler(), SVC(class_weight="balanced")))

# %% Stretch: which lifted columns does the straight boundary lean on most?
names = lifted[1].get_feature_names_out(features)
weights = pd.Series(lifted[-1].coef_[0], index=names).drop("1")
print("\nlargest lifted weights (absolute):")
print(weights.reindex(weights.abs().sort_values(ascending=False).index).head(5).round(2).to_string())
