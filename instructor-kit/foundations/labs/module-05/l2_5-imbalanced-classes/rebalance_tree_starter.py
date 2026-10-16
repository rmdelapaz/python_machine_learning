# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.5 B : does rebalancing rescue a decision tree?

Lesson 2.5 found that for a logistic regression, class weights and every sampler mainly move the 0.5 cut.
Its Common Questions say SMOTE "can help models that struggle with very few minority examples, such as
k-nearest neighbors or a deep tree". Test that claim on fraud with a depth-6 decision tree (trees are
Lesson 5.1; here it is just one more estimator with fit and predict_proba). Judge the ranking by
cross-validated AP with every sampler inside an imblearn pipeline, then see what resampling outside
the folds would have reported. Work through TODO 1 to 4; the file runs as it is.
"""
# %% Step 1 (done for you): the lesson's stratified split, numeric columns only (trees need no scaling)
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
num_cols = ["account_age_days", "amount", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(tx[num_cols], tx["is_fraud"], test_size=0.3,
                                                    stratify=tx["is_fraud"], random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"train {len(y_train):,} rows, {y_train.sum()} frauds; random-ranking AP {y_train.mean():.4f}")

# %% Step 2: five candidates, compared by cross-validated AP
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier


def tree(**kwargs):
    return DecisionTreeClassifier(max_depth=6, random_state=42, **kwargs)


candidates = {
    "plain": make_pipeline(tree()),
    "balanced weights": make_pipeline(tree(class_weight="balanced")),
    # TODO 1: add "undersample", "oversample" and "SMOTE": an imblearn make_pipeline with the sampler
    #         (random_state=42) in front of tree().
}
# TODO 2: for every candidate, cross_val_score(..., cv=cv, scoring="average_precision") on the training
#         rows; print the mean and std. Before you run it, predict: which one wins? ______
print("TODO 1-2: five candidates by CV AP")

# %% Step 3: what each one does at its own 0.5 cut
from sklearn.metrics import average_precision_score, precision_score, recall_score

# TODO 3: fit every candidate on the training rows; print its test AP, how many test payments predict()
#         flags, its recall and its precision (zero_division=0). Which numbers moved, and which did not?
print("TODO 3: test AP, flags, recall and precision at 0.5")

# %% Step 4: the leak
# TODO 4: resample ALL training rows with SMOTE(random_state=42).fit_resample(X_train, y_train), then
#         cross-validate tree() on the result. Print the mean AP and the share of fraud in y_smote.
print("TODO 4: SMOTE first, then CV")

print("\nExpected when done: CV AP plain 0.335, balanced 0.269, undersample 0.044, oversample 0.271,")
print("SMOTE 0.182; at 0.5 plain flags 44 (recall 0.27, precision 0.80) while the others flag 1,517 to 2,242")
print("with recall near 0.65 and precision 0.04 to 0.06; SMOTE first, then CV: AP 0.873 on folds that are")
print("50% fraud.")
