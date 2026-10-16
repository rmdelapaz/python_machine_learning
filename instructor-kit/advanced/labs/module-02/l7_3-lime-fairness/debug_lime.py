# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 7.3: a LIME script with four bugs (deliberately broken handout).

Three bugs stop the script with an error (and the messages do not say what is wrong: read them twice).
One does not: the explanation it prints is built on far too few variants to report. When all four are
fixed, the explainer uses 5,000 variants and the script prints, for the riskiest test customer:
  C104051  model 0.972
    days_since_last_order > 110.00      +0.295
    discount_share > 0.29               +0.240
    39.00 < tenure_months <= 68.00      -0.071
"""
# %%
import warnings

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user",
            "email_opens_90d"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
X_train, X_test = train[features].astype(float), test[features].astype(float)
model = HistGradientBoostingClassifier(random_state=42).fit(X_train, train["churned"])
proba = model.predict_proba(X_test)[:, 1]
i = int(proba.argmax())
fill = X_train.median()


def predict_fn(rows):
    return model.predict(pd.DataFrame(rows, columns=features))


# %%
explainer = LimeTabularExplainer(X_train.to_numpy(), feature_names=features,
                                 class_names=["stayed", "churned"], random_state=42)
row = X_test.iloc[i].fillna(fill)
e = explainer.explain_instance(row, predict_fn, num_features=3, num_samples=100)
print(f"{test['customer_id'].iloc[i]}  model {proba[i]:.3f}")
for rule, w in e.as_list():
    print(f"    {rule:35s} {w:+.3f}")
