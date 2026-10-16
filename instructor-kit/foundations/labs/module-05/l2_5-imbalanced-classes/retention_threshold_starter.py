# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.5 A: who gets a retention offer?

Lesson 2.5 priced fraud reviews. Here the retention team prices a different decision. Each flagged
customer gets a call and a voucher, costing $15. Each churner nobody contacts costs Pinecrest about $60
of margin. (To keep the arithmetic simple, assume the offer keeps every churner it reaches.) Churn is
16.5%, rarer than "stays" but far less rare than fraud: does the 0.5 cut-off still go wrong?
Work through TODO 1 to 6; the file runs as it is.
"""
# %% Step 1 (done for you): the canonical split, 14 columns, a preprocessing function
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
num_cols = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
cat_cols = ["home_store", "loyalty_tier", "preferred_category"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
X_train, y_train = train[num_cols + cat_cols], train["churned"]
X_test, y_test = test[num_cols + cat_cols], test["churned"]


def make_preprocess():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


print(f"train {len(train)}, test {len(test)}, test churners {y_test.sum()}")

# %% Step 2: the cost of a set of decisions
from sklearn.metrics import confusion_matrix

OFFER_COST, CHURN_LOSS = 15, 60


def retention_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # TODO 1: return the total cost: OFFER_COST for every customer flagged (tp + fp) plus CHURN_LOSS for
    #         every churner missed (fn), as an int.
    return None


# TODO 2: print the cost of offering nobody (all zeros) and of offering everyone (all ones) on the test set.
print("TODO 1-2: the cost function and the two do-nothing policies")

# %% Step 3: a plain model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# TODO 3: fit make_pipeline(make_preprocess(), LogisticRegression()) on the training rows. Print its test
#         ROC AUC and AP (and y_test.mean(), the AP of a random ranking), then the number of offers and the
#         cost when predict() uses the default 0.5.
print("TODO 3: plain model, ranking scores and the cost at 0.5")

# %% Step 4: sweep the threshold, then the formula
# TODO 4: for thresholds 0.5, 0.4, 0.3, 0.25, 0.2, 0.15 and 0.1, count the offers and the cost of
#         (p >= threshold). Then compute the formula threshold, offer cost / churn loss. Does it land in the
#         cheap part of your sweep?
print("TODO 4: threshold sweep and formula")

# %% Step 5: class weights instead
# TODO 5: fit the same pipeline with LogisticRegression(class_weight="balanced"). Print its offers and cost
#         at 0.5, the mean predicted probability of both models next to the real churn rate, and the cost
#         when the BALANCED model's probabilities are cut at the formula threshold. What went wrong?
print("TODO 5: balanced weights at 0.5 and at the formula threshold")

# %% Step 6: let cross-validation choose
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO 6: wrap the plain pipeline in TunedThresholdClassifierCV with
#         scoring=make_scorer(retention_cost, greater_is_better=False) and cv=cv. Fit it on the training rows,
#         print best_threshold_, the number of offers, the churners reached and the test cost.
print("TODO 6: TunedThresholdClassifierCV")

print("\nExpected when done: nobody $14,880, everyone $22,500; ROC AUC 0.827, AP 0.496 (random 0.165);")
print("at 0.5: 112 offers, $12,540; the sweep bottoms out near 0.2 to 0.25 ($10,830 to $10,935); formula 0.25;")
print("balanced: 471 offers, $10,965 at 0.5, mean probability 0.377 against a real 0.165, and $14,070 at the")
print("formula threshold; tuned threshold 0.234: 343 offers, 153 of 248 churners reached, $10,845.")
