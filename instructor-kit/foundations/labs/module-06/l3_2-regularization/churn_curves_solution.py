# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 3.2 A (SOLUTION): curves and penalties for a churn classifier.

Lesson 3.2 regularized regressions. The same ideas apply to logistic regression, whose dial is C (the
INVERSE of the penalty strength: small C, strong penalty). Give a churn model polynomial features, watch
it overfit a 1,000-customer pilot, let cross-validation choose C, and see what the full training set
changes. The test set is used once, at the end.
"""
# %% Step 1: the canonical split and a 1,000-customer pilot
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

# %% Step 2: flexibility without a penalty: training against validation AUC
for degree in [1, 2, 3]:
    scores = cross_validate(poly_logit(degree, np.inf), pilot[features], pilot["churned"], cv=cv,
                            scoring="roc_auc", return_train_score=True)
    n_cols = PolynomialFeatures(degree, include_bias=False).fit(pilot[features]).n_output_features_
    print(f"degree {degree} ({n_cols:3d} columns)  train AUC {scores['train_score'].mean():.3f}"
          f"   CV AUC {scores['test_score'].mean():.3f}")

# %% Step 3: the dial: C for degree 3 on the pilot
from sklearn.model_selection import validation_curve

Cs = [0.001, 0.01, 0.1, 1, 10]
tr, va = validation_curve(poly_logit(3, 1.0), pilot[features], pilot["churned"],
                          param_name="logisticregression__C", param_range=Cs, cv=cv, scoring="roc_auc")
for C, a, b in zip(Cs, tr.mean(axis=1), va.mean(axis=1)):
    print(f"C={C:<6} train AUC {a:.3f}   CV AUC {b:.3f}")
best_C = Cs[int(np.argmax(va.mean(axis=1)))]
print("chosen C for the pilot:", best_C)

# %% Step 4: more data is also a regularizer: the same comparison on all 4,500 training customers
rows = []
for degree in [1, 2, 3]:
    for C in [np.inf, 0.1]:
        s = cross_validate(poly_logit(degree, C), train[features], train["churned"], cv=cv, scoring="roc_auc")
        rows.append({"degree": degree, "C": C, "cv_auc": round(s["test_score"].mean(), 3)})
print()
print(pd.DataFrame(rows).pivot(index="degree", columns="C", values="cv_auc").to_string())

# %% Step 5: one test, for the model the training data chose
from sklearn.metrics import roc_auc_score

best = max(rows, key=lambda r: r["cv_auc"])
final = poly_logit(best["degree"], best["C"]).fit(train[features], train["churned"])
plain = poly_logit(1, 1.0).fit(train[features], train["churned"])
print(f"\nchosen: degree {best['degree']}, C={best['C']}")
print(f"test AUC chosen {roc_auc_score(test['churned'], final.predict_proba(test[features])[:, 1]):.3f}"
      f"   plain degree 1 {roc_auc_score(test['churned'], plain.predict_proba(test[features])[:, 1]):.3f}")
