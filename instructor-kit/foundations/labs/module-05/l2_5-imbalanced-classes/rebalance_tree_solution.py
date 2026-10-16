# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.5 B (SOLUTION): does rebalancing rescue a decision tree?

Lesson 2.5 found that for a logistic regression, class weights and every sampler mainly move the 0.5 cut.
Its Common Questions say SMOTE "can help models that struggle with very few minority examples, such as
k-nearest neighbors or a deep tree". Test that claim on fraud with a depth-6 decision tree (trees are
Lesson 5.1; here it is just one more estimator with fit and predict_proba). Judge the ranking by
cross-validated AP with every sampler inside an imblearn pipeline, then see what resampling outside
the folds would have reported.
"""
# %% Step 1: the lesson's stratified split, numeric columns only (trees need no scaling)
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
num_cols = ["account_age_days", "amount", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(tx[num_cols], tx["is_fraud"], test_size=0.3,
                                                    stratify=tx["is_fraud"], random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"train {len(y_train):,} rows, {y_train.sum()} frauds; random-ranking AP {y_train.mean():.4f}")

# %% Step 2: five candidates, each a pipeline, compared by cross-validated AP
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier


def tree(**kwargs):
    return DecisionTreeClassifier(max_depth=6, random_state=42, **kwargs)


candidates = {
    "plain": make_pipeline(tree()),
    "balanced weights": make_pipeline(tree(class_weight="balanced")),
    "undersample": make_pipeline(RandomUnderSampler(random_state=42), tree()),
    "oversample": make_pipeline(RandomOverSampler(random_state=42), tree()),
    "SMOTE": make_pipeline(SMOTE(random_state=42), tree()),
}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
    print(f"{name:17s} CV AP {scores.mean():.3f} +/- {scores.std():.3f}")

# %% Step 3: what each one does at its own 0.5 cut, on the test set
from sklearn.metrics import average_precision_score, precision_score, recall_score

print()
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    ap = average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"{name:17s} test AP {ap:.3f}   flagged {pred.sum():5d}   recall {recall_score(y_test, pred):.2f}"
          f"   precision {precision_score(y_test, pred, zero_division=0):.2f}")

# %% Step 4: the leak: SMOTE on all training rows first, then cross-validate
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
leaky = cross_val_score(tree(), X_smote, y_smote, cv=cv, scoring="average_precision")
print(f"\nSMOTE first, then CV: AP {leaky.mean():.3f}   (validation folds are {y_smote.mean():.0%} fraud)")
