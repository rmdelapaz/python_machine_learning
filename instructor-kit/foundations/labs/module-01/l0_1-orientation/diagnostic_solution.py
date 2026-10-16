# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Diagnostic follow-up (SOLUTION): six prerequisite skills, one small function each.

The course assumes Python and pandas at the level of the Python Data Science course. These six
functions match the six diagnostic questions D1 to D6 on the workbook's Start Here page. Each is
checked on a tiny example, then the pandas ones run on the real customer table.
"""
# %% The six functions
import pandas as pd


def last_three(items):
    """D1 slicing: the last three items of a list, in order."""
    return items[-3:]


def count_by(values):
    """D2 dictionaries: how many times each value appears, as a dict."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def discounted(price, rate=0.10):
    """D3 functions with a default: the price after a discount, rounded to cents."""
    return round(price * (1 - rate), 2)


def mean_where(df, flag, column):
    """D4 boolean filtering: the mean of `column` over the rows where `flag` is 1."""
    return float(df.loc[df[flag] == 1, column].mean())


def rate_by(df, group, target):
    """D5 groupby: the mean of a 0/1 `target` for each value of `group`, highest first."""
    return df.groupby(group)[target].mean().sort_values(ascending=False)


def missing_share(df, column):
    """D6 missing values: the share of rows where `column` is empty."""
    return float(df[column].isna().mean())


# %% Check each one on a tiny example
tiny = pd.DataFrame({"app_user": [1, 0, 1, 0], "spend": [100.0, 50.0, 300.0, None],
                     "tier": ["Gold", "Basic", "Basic", "Basic"], "churned": [0, 1, 0, 1]})
checks = [
    ("D1 slicing", last_three([5, 6, 7, 8, 9]), [7, 8, 9]),
    ("D2 dictionaries", count_by(["Bend", "Boise", "Bend"]), {"Bend": 2, "Boise": 1}),
    ("D3 default argument", (discounted(80), discounted(80, rate=0.25)), (72.0, 60.0)),
    ("D4 boolean filter", mean_where(tiny, "app_user", "spend"), 200.0),
    ("D5 groupby", rate_by(tiny, "tier", "churned").round(3).to_dict(), {"Basic": 0.667, "Gold": 0.0}),
    ("D6 missing values", missing_share(tiny, "spend"), 0.25),
]
passed = 0
for name, got, want in checks:
    ok = got == want
    passed += ok
    print(f"{'PASS' if ok else 'not yet':8s} {name:20s} got {got!r}")
print(f"{passed} of {len(checks)} checks pass.")

# %% The same functions on the real customers
customers = pd.read_csv(DATA / "customers_churn.csv")
print(f"\nMean 12-month spend of app users: ${mean_where(customers, 'app_user', 'spend_12m'):,.2f}")
print("Churn rate by loyalty tier:")
print(rate_by(customers, "loyalty_tier", "churned").round(3).to_string())
print(f"Share of customers with no age_group: {missing_share(customers, 'age_group'):.1%}")
