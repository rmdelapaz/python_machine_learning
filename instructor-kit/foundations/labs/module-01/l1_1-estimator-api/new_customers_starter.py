# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.1 B: score three customers who are not in any file.

A model is only useful if it can score NEW rows. The care team describes three customers over the
phone; you turn them into rows, push them through the SAME fitted scaler and model, and rank them.
Work in pairs: one types, one predicts each answer out loud before the run; swap at TODO 4.
The file runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the lesson's split (done for you: same features, same rows as Lesson 1.1)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42)
print(X_train.shape, X_test.shape)

# %% Step 2: what did the scaler learn?
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform="pandas").fit(X_train)
# TODO 1: predict first, then check with np.allclose: is scaler.mean_ equal to X_train.mean()?
#         Is scaler.scale_ equal to X_train.std(ddof=0)? And to X_train.std() (pandas' default, ddof=1)?
print("TODO 1: three np.allclose checks on mean_ and scale_")

# %% Step 3: fit the model and read its settings
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(scaler.transform(X_train), y_train)
# TODO 2: print C and max_iter from model.get_params(), and the test accuracy (scale X_test first!).
print("TODO 2: C, max_iter and the test accuracy")

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
# Before you run anything: rank Avery, Blake and Casey from most to least likely to churn: ______
# TODO 3: transform `new` with the fitted scaler (transform, NOT fit_transform), then print each
#         customer's churn probability (column 1 of predict_proba), highest first, rounded to 3 places.
print("TODO 3: churn probability for Avery, Blake and Casey")

# %% Step 5: the same customers with the columns in another order
shuffled = new[features[::-1]]
# TODO 4: call scaler.transform(shuffled) inside try/except ValueError and print the first line of the
#         message. Then show the fix: shuffled[features] puts the columns back in the training order.
print("TODO 4: what happens when the columns arrive in a different order?")

# %% Step 6: change a setting with set_params
# TODO 5: LogisticRegression().set_params(C=0.01), fit on the scaled training rows, print the test
#         accuracy and the three new customers' probabilities. (Smaller C = a more cautious model;
#         Lesson 3.2 explains why.) Which probabilities moved toward the middle?
print("TODO 5: C=0.01, refit, compare")

print("\nExpected when done: True, True, False; C = 1.0, max_iter = 100, accuracy 0.848; Avery 0.808,")
print("Casey 0.550, Blake 0.010; a ValueError that the feature names should match; with C=0.01 accuracy")
print("0.853 and Avery 0.719, Blake 0.018, Casey 0.481.")
