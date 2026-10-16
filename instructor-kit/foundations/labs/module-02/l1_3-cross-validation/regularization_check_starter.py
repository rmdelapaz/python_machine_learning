# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.3 B: was C=0.01 really better?

In Session 1 (Lab 1.1 B) logistic regression with C=0.01 scored 0.853 accuracy on the test set against
0.848 for the default C=1.0. One split, one number each. Is the difference real, or is it the wobble
Lesson 1.2 measured? Settle it the Lesson 1.3 way: on the training customers only, with identical
repeated folds, a pipeline, and paired differences. Work through TODO 1 to 5; it runs as it is.
"""
# %% Step 1: the standard split; the test set stays locked (done for you)
import pandas as pd
from sklearn.model_selection import train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)
print(X_train.shape)

# %% Step 2: the leaky way and the pipeline way, side by side
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO 1: (a) scale all of X_train with StandardScaler().fit_transform, then cross_val_score a
#         LogisticRegression(C=0.01) on it; (b) cross_val_score make_pipeline(StandardScaler(),
#         LogisticRegression(C=0.01)) on the raw X_train. Print both means with 4 decimals.
#         Which one is honest, and how big is the difference here?
print("TODO 1: pre-scaled vs pipeline")

# %% Step 3: both settings on the same 15 folds, two metrics
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
# TODO 2: for C in [1.0, 0.01]: cross_validate the pipeline with cv=rcv and
#         scoring=["accuracy", "roc_auc"]; keep each result in a dict results[C]; print the mean and std
#         of test_accuracy and test_roc_auc.
results = {}
if not results:
    print("TODO 2: accuracy and AUC for C=1.0 and C=0.01 on 15 identical folds")

# %% Step 4: paired differences
# TODO 3: for each metric, diff = results[0.01][metric] - results[1.0][metric]. In how many folds is
#         C=0.01 ahead, and what is the mean difference? Compare it with the std from TODO 2.
print("TODO 3: paired differences, fold by fold")

# %% Step 5: a wider sweep, with training scores
# TODO 4: for C in [0.001, 0.01, 0.1, 1.0, 10.0]: cross_validate with cv=cv, scoring="roc_auc",
#         return_train_score=True; print mean train AUC and mean validation AUC. Does any C overfit?
print("TODO 4: train vs validation AUC across five values of C")

# %% Step 6: verdict
# TODO 5: write your verdict in two sentences, using the numbers (not a guess). Was Session 1's single
#         test split right about C=0.01?
print("TODO 5: the verdict")

print("\nExpected when done: pre-scaled 0.8493 vs pipeline 0.8491 (a tiny leak, here); C=1.0 accuracy 0.852,")
print("C=0.01 0.849, AUC 0.830 for both; C=0.01 ahead in only 2 of 15 folds on accuracy (mean -0.0026) and")
print("9 of 15 on AUC (+0.0006): a tie on ranking. C=0.001 is the only clear loser (validation AUC 0.825).")
