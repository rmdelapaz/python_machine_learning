# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 2.2). This file is DELIBERATELY BROKEN.

A demand pipeline: scale four numeric columns, one-hot encode product and store, Ridge with alpha=10,
trained on 2024 and scored on 2025. Three bugs stop it with an error, one at a time: run it, read the
error (the important part is near the START of a long pipeline message), fix that one bug, run again.
A fourth bug raises no error at all. Find it too: it makes the model look better than it is.

When everything is fixed it prints:
    5088 training weeks, 4896 test weeks
    2025 test MAE: 11.75 units
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

demand = pd.read_csv(DATA / "weekly_demand.csv")
numeric = ["price", "promo", "holiday_week", "avg_high_temp_f"]
cats = ["product_id", "store"]
train = demand[demand["year"] == 2024]
test = demand[demand["year"] != 2025]
print(len(train), "training weeks,", len(test), "test weeks")

prep = ColumnTransformer([
    ("scale", StandardScaler(), numeric),
    ("one__hot", OneHotEncoder(handle_unknown="ignore"), cats),
])
pipe = Pipeline([("prep", prep), ("model", Ridge())])
pipe.set_params(alpha=10.0)
pipe.fit(train[numeric + cats], train["units_sold"])
pred = pipe.predict(test[numeric])
print(f"2025 test MAE: {mean_absolute_error(test['units_sold'], pred):.2f} units")
