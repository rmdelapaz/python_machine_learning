# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 6.1 B (solution): curves for a demand forecaster, with folds that respect time.

Lesson 6.1 drew a validation curve and learning curves for churn. The demand forecaster from Lesson
5.3 (train 2024, test 2025) raises two new questions: are shuffled folds a fair way to score a
forecaster, and would it gain from more history?
Scores are mean absolute errors in units (scikit-learn reports them negated: bigger is better).
"""
# %% Step 1: the 2024 training rows (2025 stays locked in this lab)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, learning_curve

demand = pd.read_csv(DATA / "weekly_demand.csv")
features = ["store", "product_id", "category", "week_of_year", "list_price",
            "discount_pct", "price", "promo", "holiday_week", "avg_high_temp_f"]
cats = ["store", "product_id", "category"]
X = demand[features].copy()
X[cats] = X[cats].astype("category")
train = demand["year"] == 2024
X_train, y_train = X[train], demand.loc[train, "units_sold"]
model = HistGradientBoostingRegressor(loss="poisson", random_state=42)
print(len(X_train), "rows of 2024, sorted by week")

# %% Step 2 (TODO 1): shuffled folds against blocks of weeks
for name, cv in [("shuffled KFold", KFold(5, shuffle=True, random_state=42)), ("blocks of weeks", KFold(5))]:
    mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name:16s} CV MAE {mae.mean():.2f} +/- {mae.std():.2f}")
print("2025 test MAE of this model (Lesson 5.3): 8.05")

# %% Step 3 (TODO 2): a learning curve on blocks of weeks: would more history help?
blocks = KFold(5)
n, tr_l, va_l = learning_curve(model, X_train, y_train, train_sizes=[0.33, 0.67, 1.0], cv=blocks,
                               scoring="neg_mean_absolute_error")
print("\ntraining rows  train MAE  validation MAE")
for k, a, b in zip(n, -tr_l.mean(axis=1), -va_l.mean(axis=1)):
    print(f"{k:13d}  {a:9.2f}  {b:14.2f}")

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
ax.plot(n, -tr_l.mean(axis=1), marker="o", label="training MAE")
ax.plot(n, -va_l.mean(axis=1), marker="s", label="validation MAE (blocks of weeks)")
ax.set_xlabel("Training rows")
ax.set_ylabel("Mean absolute error (units)")
ax.set_title("The gap stays wide: more history would still help")
ax.legend()
fig.savefig(OUTPUT / "demand_learning_curve.png", dpi=100)
print("saved demand_learning_curve.png")

# %% Step 4 (TODO 3): the diagnosis, printed from the variables
gap = (-va_l.mean(axis=1) + tr_l.mean(axis=1))[-1]
drop = (-va_l.mean(axis=1))[-2] - (-va_l.mean(axis=1))[-1]
print(f"\nat full size: validation MAE {-va_l.mean(axis=1)[-1]:.2f}, gap to training {gap:.2f} units; "
      f"the last third of the rows cut validation MAE by {drop:.2f}")
print("Diagnosis:", "high variance and still improving: more history (a third year) should help"
      if drop > 0.2 else "flat: more rows of the same data will not help much")
