# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.1 B (SOLUTION): score three customers who are not in any file.

A model is only useful if it can score NEW rows. The care team describes three customers over the
phone; you turn them into rows, push them through the SAME fitted scaler and model, and rank them.
Along the way you look inside the fitted scaler and change a setting with set_params.
"""
# %% Step 1: the lesson's split (same features, same rows)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42)
print(X_train.shape, X_test.shape)

# %% Step 2: what did the scaler learn? Check it against pandas
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform="pandas").fit(X_train)
print("mean_ equals X_train.mean():        ", np.allclose(scaler.mean_, X_train.mean()))
print("scale_ equals X_train.std(ddof=0):  ", np.allclose(scaler.scale_, X_train.std(ddof=0)))
print("scale_ equals X_train.std() (ddof=1):", np.allclose(scaler.scale_, X_train.std()))

# %% Step 3: fit the model and read its settings
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(scaler.transform(X_train), y_train)
print(f"\nC = {model.get_params()['C']}, max_iter = {model.get_params()['max_iter']}, "
      f"test accuracy {model.score(scaler.transform(X_test), y_test):.3f}")

# %% Step 4: three customers described over the phone
new = pd.DataFrame({
    "tenure_months":         [3, 48, 20],
    "orders_12m":            [1, 9, 4],
    "spend_12m":             [60.0, 900.0, 400.0],
    "avg_order_value":       [60.0, 100.0, 100.0],
    "days_since_last_order": [200, 12, 160],
    "returns_12m":           [0, 1, 0],
    "return_rate":           [0.0, 0.05, 0.0],
    "support_tickets_12m":   [2, 0, 3],
    "discount_share":        [0.5, 0.1, 0.3],
    "app_user":              [0, 1, 0],
}, index=["Avery", "Blake", "Casey"])
risk = model.predict_proba(scaler.transform(new))[:, 1]
print()
print(pd.Series(risk, index=new.index, name="churn_probability").sort_values(ascending=False).round(3).to_string())

# %% Step 5: the same customers with the columns in another order
shuffled = new[features[::-1]]
try:
    scaler.transform(shuffled)
except ValueError as err:
    print("\nValueError:", str(err).splitlines()[0])
print("fixed by selecting the training columns again:", scaler.transform(shuffled[features]).shape)

# %% Step 6: change a setting with set_params, refit, compare
strong = LogisticRegression().set_params(C=0.01).fit(scaler.transform(X_train), y_train)
print(f"\nC=0.01: test accuracy {strong.score(scaler.transform(X_test), y_test):.3f}")
print("new customers:", pd.Series(strong.predict_proba(scaler.transform(new))[:, 1], index=new.index).round(3).to_dict())
