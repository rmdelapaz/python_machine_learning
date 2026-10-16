# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Diagnostic follow-up: six prerequisite skills, one small function each.

This course assumes Python and pandas at the level of the Python Data Science course. Each function
below matches one diagnostic question (D1 to D6 on the workbook's Start Here page). Replace each
`return None` with a real answer, run the file, and watch "not yet" turn into PASS. The last cell runs
your pandas functions on the real customer table once all six pass.
Review if stuck: D1-D3 Intermediate Python (lists, dicts, functions); D4-D6 Python Data Science
Lessons 2.1 (selecting and filtering), 2.3 (missing values) and 2.4 (groupby).
"""
# %% The six functions
import pandas as pd


def last_three(items):
    """D1 slicing: the last three items of a list, in order."""
    return None  # TODO 1: one slice with a negative start


def count_by(values):
    """D2 dictionaries: how many times each value appears, as a dict."""
    return None  # TODO 2: start with {} and use counts.get(v, 0) + 1


def discounted(price, rate=0.10):
    """D3 functions with a default: the price after a discount, rounded to cents."""
    return None  # TODO 3: price * (1 - rate), rounded to 2 places


def mean_where(df, flag, column):
    """D4 boolean filtering: the mean of `column` over the rows where `flag` is 1."""
    return None  # TODO 4: df.loc[<condition>, column].mean(), wrapped in float()


def rate_by(df, group, target):
    """D5 groupby: the mean of a 0/1 `target` for each value of `group`, highest first."""
    return None  # TODO 5: groupby, pick the target column, mean, sort_values(ascending=False)


def missing_share(df, column):
    """D6 missing values: the share of rows where `column` is empty."""
    return None  # TODO 6: the mean of isna(), wrapped in float()


# %% Check each one on a tiny example (do not edit below this line)
tiny = pd.DataFrame({"app_user": [1, 0, 1, 0], "spend": [100.0, 50.0, 300.0, None],
                     "tier": ["Gold", "Basic", "Basic", "Basic"], "churned": [0, 1, 0, 1]})


def _d5():
    r = rate_by(tiny, "tier", "churned")
    return None if r is None else r.round(3).to_dict()


checks = [
    ("D1 slicing", lambda: last_three([5, 6, 7, 8, 9]), [7, 8, 9]),
    ("D2 dictionaries", lambda: count_by(["Bend", "Boise", "Bend"]), {"Bend": 2, "Boise": 1}),
    ("D3 default argument", lambda: (discounted(80), discounted(80, rate=0.25)), (72.0, 60.0)),
    ("D4 boolean filter", lambda: mean_where(tiny, "app_user", "spend"), 200.0),
    ("D5 groupby", _d5, {"Basic": 0.667, "Gold": 0.0}),
    ("D6 missing values", lambda: missing_share(tiny, "spend"), 0.25),
]
passed = 0
for name, fn, want in checks:
    try:
        got = fn()
    except Exception as err:          # a half-written function: show the error, keep going
        got = f"{type(err).__name__}: {err}"
    ok = got == want
    passed += ok
    print(f"{'PASS' if ok else 'not yet':8s} {name:20s} got {got!r}")
print(f"{passed} of {len(checks)} checks pass.")

# %% The same functions on the real customers (runs once all six pass)
if passed == len(checks):
    customers = pd.read_csv(DATA / "customers_churn.csv")
    print(f"\nMean 12-month spend of app users: ${mean_where(customers, 'app_user', 'spend_12m'):,.2f}")
    print("Churn rate by loyalty tier:")
    print(rate_by(customers, "loyalty_tier", "churned").round(3).to_string())
    print(f"Share of customers with no age_group: {missing_share(customers, 'age_group'):.1%}")
else:
    print("\nExpected when done: six PASS lines, then app users' mean spend $663.21, churn by tier")
    print("Basic 0.217, Silver 0.129, Gold 0.065, and 8.1% of customers with no age_group.")
