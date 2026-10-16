"""Generate the Pinecrest Outfitters machine-learning data.

Pinecrest Outfitters is a FICTIONAL outdoor-gear retailer (an online shop plus
stores in Boise, Bend, Flagstaff and Asheville). Every person, product, review
and number in these files is invented for the Python Machine Learning course.

    python data/make_ml_data.py            # write everything into data/
    python data/make_ml_data.py --out DIR  # write somewhere else

Needs only the standard library, numpy and pandas.

Reproducibility: every dataset draws from its OWN seeded random stream
(rng_for(n) below), so running the script twice gives byte-identical files, and
changing one dataset never shifts the numbers of another.

Fictional data written for this course. (c) 2026 Ray de la Paz.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261015
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
STORES = ["Boise", "Bend", "Flagstaff", "Asheville"]


def rng_for(n: int) -> np.random.Generator:
    """An independent, reproducible random stream per dataset."""
    return np.random.default_rng([SEED, n])


def to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


CATEGORIES = ["Tents", "Sleeping", "Packs", "Footwear", "Apparel", "Climbing", "Camp Kitchen", "Snow", "Accessories"]


# ==========================================================================
# 1. customers_churn.csv
# ==========================================================================
SNAPSHOT = pd.Timestamp("2025-06-30")      # features describe the 12 months up to this date
# Five behavioral segments (hidden; recoverable from orders_12m, avg_order_value, discount_share, return_rate).
#           share, orders_12m mean, AOV median, discount_share (a, b beta), return_rate mean, app prob, favorite categories
SEGMENTS = {
    "Trail Regulars":    (0.24, 9.0, 85.0, (2, 18), 0.04, 0.55, ["Footwear", "Accessories", "Apparel"]),
    "Big-Trip Planners": (0.18, 2.4, 330.0, (2, 16), 0.05, 0.30, ["Tents", "Sleeping", "Packs"]),
    "Deal Hunters":      (0.22, 5.5, 62.0, (9, 9), 0.08, 0.45, ["Apparel", "Camp Kitchen", "Accessories"]),
    "Gear Testers":      (0.14, 6.5, 165.0, (3, 14), 0.27, 0.50, ["Footwear", "Climbing", "Apparel"]),
    "Casual Campers":    (0.22, 1.8, 95.0, (2, 12), 0.06, 0.20, ["Camp Kitchen", "Accessories", "Sleeping"]),
}
HOME_STORE = {"Online": 0.44, "Boise": 0.17, "Bend": 0.13, "Flagstaff": 0.13, "Asheville": 0.13}
STORE_REGION = {"Boise": "Northwest", "Bend": "Northwest", "Flagstaff": "Southwest", "Asheville": "Southeast"}
ONLINE_REGIONS = (["Northwest", "Southwest", "Southeast", "Mountain", "Midwest", "Northeast"],
                  [0.24, 0.22, 0.18, 0.16, 0.11, 0.09])
AGE_GROUPS = (["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], [0.11, 0.22, 0.22, 0.19, 0.15, 0.11])
CANCEL_REASONS = (["moved away", "prices too high", "found another retailer", "poor service experience",
                   "no longer doing outdoor activities", "too many emails"], [0.14, 0.27, 0.24, 0.17, 0.10, 0.08])


def make_churn() -> tuple[pd.DataFrame, dict]:
    rng = rng_for(1)
    n = 6000
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s][0] for s in names])
    seg = rng.choice(len(names), n, p=shares / shares.sum())
    P = {k: np.array([SEGMENTS[s][i] for s in names], dtype=object) for i, k in
         enumerate(["share", "orders", "aov", "disc", "ret", "app", "cats"])}

    # tenure: 1..96 months; a fifth are new customers (under 6 months)
    tenure = np.where(rng.random(n) < 0.2, rng.integers(1, 6, n), rng.integers(6, 97, n))
    orders = np.round(np.array(P["orders"][seg], float) * rng.lognormal(0, 0.28, n))
    orders = np.maximum(1, np.minimum(orders, np.maximum(1, tenure * 2))).astype(int)   # new customers: few orders
    aov = np.clip(np.array(P["aov"][seg], float) * rng.lognormal(0, 0.22, n), 12, 1500)
    spend = np.round(orders * aov, 2)
    aov = np.round(spend / orders, 2)
    da = np.array([P["disc"][s][0] for s in seg], float)
    db = np.array([P["disc"][s][1] for s in seg], float)
    discount_share = np.round(rng.beta(da, db), 3)
    # return_rate: share of 12-month spend refunded; returns_12m: orders with at least one returned item
    ret_mean = np.array(P["ret"][seg], float)
    return_rate = np.round(rng.beta(ret_mean * 24, (1 - ret_mean) * 24), 3)
    returns = np.minimum(orders, rng.binomial(orders, np.clip(return_rate * 1.3, 0, 1)))
    app = rng.random(n) < np.array(P["app"][seg], float)
    tickets = rng.poisson(0.25 + 0.12 * orders ** 0.8 + 2.2 * return_rate + 0.6 * (discount_share > 0.4))
    email = rng.poisson(np.clip(rng.gamma(2.0, 3.2, n) + 2.0 * app, 0, None))
    email = np.minimum(email, 40)
    # recency: frequent buyers bought recently; drawn before the outcome, it is a cause, not a consequence
    gap = 365 / (orders + 0.5)
    recency = np.clip(np.round(rng.exponential(gap * 0.9) + rng.integers(0, 10, n)), 0, 364).astype(int)

    # ---- churn: did NOT buy in the 6 months after the snapshot
    rec_z = recency / 100
    logit = (-4.3
             + 0.15 * rec_z + 3.1 * sigmoid((recency - 150) / 15)              # recency, mostly a threshold
             + np.where(app, 0.04, 0.60) * tickets                             # interaction: app users tolerate tickets
             + 0.8 * (discount_share - 0.25) + 1.9 * (discount_share > 0.35)   # discount reliance
             - 0.07 * (email - 6)                                              # email engagement
             + 2.1 * (tenure < 6) + 1.4 * (tenure > 72)                        # tenure U-shape
             + 1.2 * return_rate
             - 0.08 * np.minimum(orders, 12)
             + rng.normal(0, 0.35, n))
    churned = (rng.random(n) < sigmoid(logit)).astype(int)

    # ---- descriptive columns
    stores = rng.choice(list(HOME_STORE), n, p=list(HOME_STORE.values()))
    region = np.array([STORE_REGION.get(s, "") for s in stores], dtype=object)
    online = stores == "Online"
    region[online] = rng.choice(ONLINE_REGIONS[0], online.sum(), p=ONLINE_REGIONS[1])
    age = rng.choice(AGE_GROUPS[0], n, p=AGE_GROUPS[1]).astype(object)
    age[rng.random(n) < np.where(online, 0.13, 0.04)] = np.nan            # optional profile field
    email_out = email.astype(float)
    email_out[rng.random(n) < np.where(churned == 1, 0.13, 0.08)] = np.nan  # tracking blocked / unsubscribed
    tier_score = spend * rng.lognormal(0, 0.15, n) + 8 * tenure
    tier = np.select([tier_score >= 1700, tier_score >= 850], ["Gold", "Silver"], "Basic")
    cats = np.array([rng.choice(P["cats"][s]) if rng.random() < 0.7 else rng.choice(CATEGORIES) for s in seg])
    signup = SNAPSHOT - pd.to_timedelta(np.round(tenure * 30.44 + rng.integers(0, 30, n)).astype(int), unit="D")
    # LEAK: recorded when an account is closed AFTER the snapshot, so it exists only for churners
    reason = np.full(n, np.nan, dtype=object)
    closed = (churned == 1) & (rng.random(n) < 0.62)
    reason[closed] = rng.choice(CANCEL_REASONS[0], closed.sum(), p=CANCEL_REASONS[1])

    order = rng.permutation(n)
    df = pd.DataFrame({
        "customer_id": [f"C{100001 + i}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d")[order],
        "home_store": stores[order], "region": region[order], "age_group": age[order],
        "loyalty_tier": tier[order], "app_user": app.astype(int)[order], "tenure_months": tenure[order],
        "orders_12m": orders[order], "spend_12m": spend[order], "avg_order_value": aov[order],
        "days_since_last_order": recency[order], "returns_12m": returns[order], "return_rate": return_rate[order],
        "support_tickets_12m": tickets[order], "discount_share": discount_share[order],
        "email_opens_90d": email_out[order], "preferred_category": cats[order],
        "cancellation_reason": reason[order], "churned": churned[order],
    })
    df["email_opens_90d"] = df["email_opens_90d"].astype("Int64")
    truth = {"segment": np.array(names)[seg][order]}
    return df, truth


# ==========================================================================
# 2. weekly_demand.csv
# ==========================================================================
CAT_SEASON = {  # monthly multipliers Jan..Dec (from the Data Science course data)
    "Tents":        [0.4, 0.5, 0.8, 1.1, 1.6, 1.9, 1.8, 1.4, 0.9, 0.6, 0.5, 0.6],
    "Sleeping":     [0.5, 0.6, 0.8, 1.1, 1.5, 1.7, 1.6, 1.3, 1.0, 0.7, 0.6, 0.7],
    "Packs":        [0.7, 0.7, 0.9, 1.1, 1.3, 1.4, 1.4, 1.3, 1.0, 0.8, 0.8, 1.0],
    "Footwear":     [0.8, 0.8, 1.0, 1.2, 1.3, 1.2, 1.1, 1.1, 1.1, 1.0, 0.9, 1.0],
    "Apparel":      [1.1, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 1.0, 1.2, 1.4, 1.7],
    "Climbing":     [0.6, 0.7, 1.0, 1.3, 1.4, 1.3, 1.1, 1.1, 1.2, 1.1, 0.7, 0.8],
    "Camp Kitchen": [0.6, 0.6, 0.8, 1.0, 1.4, 1.6, 1.6, 1.4, 1.0, 0.7, 0.7, 1.0],
    "Snow":         [2.4, 2.0, 0.9, 0.2, 0.1, 0.05, 0.05, 0.05, 0.2, 0.8, 2.0, 3.0],
    "Accessories":  [0.8, 0.8, 0.9, 1.0, 1.2, 1.3, 1.3, 1.2, 1.0, 0.9, 1.0, 1.3],
}
TEMP_EFFECT = {"Tents": 0.010, "Sleeping": 0.006, "Packs": 0.004, "Footwear": 0.0, "Apparel": -0.006,
               "Climbing": 0.006, "Camp Kitchen": 0.010, "Snow": -0.022, "Accessories": 0.003}  # log units per F above 55
DEMAND_PRODUCTS = [  # (product_id, name, category, list price, weekly units at a medium store at list price)
    ("TNT-2001", "Timberline 2-Person Tent", "Tents", 289.95, 9), ("TNT-2002", "Basecamp 4-Person Tent", "Tents", 419.95, 6),
    ("TNT-2003", "Drift Ultralight Tarp", "Tents", 89.95, 10),
    ("SLP-2001", "Cascade 20F Sleeping Bag", "Sleeping", 219.95, 9), ("SLP-2002", "Meadow Sleeping Pad", "Sleeping", 109.95, 13),
    ("PCK-2001", "Switchback 30L Daypack", "Packs", 129.95, 14), ("PCK-2002", "Ridgeline 65L Backpack", "Packs", 289.95, 7),
    ("FTW-2001", "Talus Hiking Boots", "Footwear", 189.95, 15), ("FTW-2002", "Foothill Trail Runners", "Footwear", 139.95, 18),
    ("FTW-2003", "Cairn Hiking Socks", "Footwear", 21.95, 55),
    ("APL-2001", "Alpenglow Rain Jacket", "Apparel", 179.95, 14), ("APL-2002", "Ember Fleece Pullover", "Apparel", 89.95, 20),
    ("APL-2003", "Summit Down Jacket", "Apparel", 279.95, 9), ("APL-2004", "Lookout Sun Hat", "Apparel", 34.95, 22),
    ("CLM-2001", "Sawtooth Climbing Harness", "Climbing", 79.95, 7), ("CLM-2002", "Mesa Chalk Bag", "Climbing", 24.95, 14),
    ("KIT-2001", "Headwater Water Filter", "Camp Kitchen", 59.95, 16), ("KIT-2002", "Canyon Backpacking Stove", "Camp Kitchen", 89.95, 11),
    ("KIT-2003", "Wayfinder Insulated Mug", "Camp Kitchen", 24.95, 30),
    ("SNW-2001", "Tundra Snowshoes", "Snow", 219.95, 6), ("SNW-2002", "Solstice Snow Goggles", "Snow", 119.95, 8),
    ("ACC-2001", "Skyline Headlamp", "Accessories", 49.95, 24), ("ACC-2002", "Trailhead Trekking Poles", "Accessories", 119.95, 11),
    ("ACC-2003", "Backcountry First-Aid Kit", "Accessories", 39.95, 17),
]
STORE_DEMAND = {"Boise": (18000, 1.30, 63, 27), "Bend": (12500, 1.00, 59, 23),
                "Flagstaff": (11000, 0.95, 61, 20), "Asheville": (9500, 0.80, 67, 19)}  # sqft, demand mult, mean high F, amplitude
ELASTICITY = -1.8          # log units per log price (within a product)
PROMO_LIFT = 0.25          # log units when featured in the weekly flyer
HOLIDAY_LIFT = 0.35        # log units in holiday weeks
DEMAND_TREND = 0.07        # log units per year


def holiday_weeks(weeks: pd.DatetimeIndex) -> np.ndarray:
    days = ["2024-05-27", "2024-07-04", "2024-09-02", "2024-11-29", "2024-12-23", "2025-05-26", "2025-07-04",
            "2025-09-01", "2025-11-28", "2025-12-22"]
    hol = set()
    for d in pd.to_datetime(days):
        hol.add(d - pd.Timedelta(days=d.weekday()))
    return np.array([w in hol for w in weeks])


def make_weekly_demand() -> pd.DataFrame:
    rng = rng_for(2)
    weeks = pd.date_range("2024-01-01", periods=104, freq="7D")      # Mondays
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    doy = weeks.dayofyear.to_numpy() + 3
    month_pos = (weeks.month.to_numpy() - 1) + (weeks.day.to_numpy() + 3) / 31
    hol = holiday_weeks(weeks)
    t_years = np.arange(104) / 52
    rows = []
    for store, (sqft, mult, tmean, tamp) in STORE_DEMAND.items():
        temp = tmean - tamp * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 4.5, 104)
        for pid, name, cat, list_price, base in DEMAND_PRODUCTS:
            season = np.interp(month_pos, np.arange(12) + 0.5, CAT_SEASON[cat], period=12)
            disc = rng.choice([0, 0, 0, 0, 0, 0, 10, 15, 20, 25, 30], 104)
            # clearance pricing in a category's off-season
            off = season < 0.6
            disc = np.where(off & (rng.random(104) < 0.5), np.maximum(disc, rng.choice([20, 25, 30, 40], 104)), disc)
            promo = ((disc >= 20) & (rng.random(104) < 0.7)) | (rng.random(104) < 0.05)
            price = np.round(list_price * (1 - disc / 100), 2)
            log_mu = (np.log(base * mult) + np.log(season) + ELASTICITY * np.log(price / list_price)
                      + PROMO_LIFT * promo + HOLIDAY_LIFT * hol + DEMAND_TREND * t_years
                      + TEMP_EFFECT[cat] * (temp - 55) + rng.normal(0, 0.06))
            mu = np.exp(log_mu)
            units = rng.poisson(mu * rng.gamma(12, 1 / 12, 104))          # negative binomial noise
            for k in range(104):
                rows.append((weeks[k].strftime("%Y-%m-%d"), int(weeks[k].year), int(woy[k]), store, sqft, pid, name, cat,
                             list_price, int(disc[k]), float(price[k]), int(promo[k]), int(hol[k]),
                             round(float(temp[k]), 1), int(units[k])))
    df = pd.DataFrame(rows, columns=["week_start", "year", "week_of_year", "store", "store_sqft", "product_id",
                                     "product_name", "category", "list_price", "discount_pct", "price", "promo",
                                     "holiday_week", "avg_high_temp_f", "units_sold"])
    # stockouts: the shelf was empty for a week, so nothing sold (not flagged)
    busy = df.index[df["units_sold"] >= 12].to_numpy()
    out = rng.choice(busy, 14, replace=False)
    df.loc[out, "units_sold"] = 0
    return df.sort_values(["week_start", "store", "product_id"]).reset_index(drop=True)


# ==========================================================================
# 3. transactions.csv
# ==========================================================================
TX_CATEGORIES = (["Tents", "Sleeping", "Packs", "Footwear", "Apparel", "Climbing", "Camp Kitchen", "Snow",
                  "Accessories", "Gift Cards"],
                 [0.08, 0.08, 0.09, 0.15, 0.22, 0.06, 0.10, 0.05, 0.14, 0.03],
                 [0.10, 0.06, 0.07, 0.12, 0.13, 0.06, 0.03, 0.06, 0.07, 0.30])   # normal mix, fraud mix
TX_AMOUNT = {"Tents": 250, "Sleeping": 160, "Packs": 120, "Footwear": 130, "Apparel": 95, "Climbing": 110,
             "Camp Kitchen": 60, "Snow": 170, "Accessories": 45, "Gift Cards": 100}
HOUR_P = np.array([1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2.2, 3.4, 4.4, 5.2, 5.8, 6.2, 6.0, 5.6, 5.4, 5.6, 6.2,
                   6.9, 7.4, 7.0, 5.6, 3.6, 2.0])


def make_transactions() -> pd.DataFrame:
    rng = rng_for(3)
    n = 50000
    fraud = rng.random(n) < 0.0092
    # a third of fraud is "camouflaged": it looks like an ordinary purchase on most features
    camo = fraud & (rng.random(n) < 0.33)
    loud = fraud & ~camo
    start = pd.Timestamp("2025-01-01")
    seconds = np.sort(rng.integers(0, 181 * 86400, n))                # 2025-01-01 .. 2025-06-30
    ts = start + pd.to_timedelta(seconds, unit="s")
    # hour comes from the timestamp; give loud fraud a night-time shift
    hours_normal = rng.choice(24, n, p=HOUR_P / HOUR_P.sum())
    device_new = rng.random(n) < np.where(loud, 0.50, 0.09)
    # loud fraud from a new device tends to happen at night (an interaction trees can find)
    night = loud & device_new & (rng.random(n) < 0.7)
    hour = np.where(night, rng.integers(0, 6, n), hours_normal)
    ts = ts.normalize() + pd.to_timedelta(hour * 3600 + rng.integers(0, 3600, n), unit="s")

    cat = np.where(loud, rng.choice(TX_CATEGORIES[0], n, p=TX_CATEGORIES[2]), rng.choice(TX_CATEGORIES[0], n, p=TX_CATEGORIES[1]))
    base = np.array([TX_AMOUNT[c] for c in cat], float)
    items = 1 + rng.poisson(np.where(loud, 1.6, 0.9))
    cheap = np.isin(cat, ["Accessories", "Camp Kitchen", "Apparel", "Gift Cards"])
    amount = base * items ** 0.7 * rng.lognormal(0, 0.45, n) * np.where(loud, np.where(cheap, 2.6, 1.15), 1.0)
    amount = np.where(cat == "Gift Cards", np.round(np.clip(amount, 25, 500) / 25) * 25, amount)
    amount = np.round(np.clip(amount, 5, 4000), 2)
    account_age = np.where(loud, rng.exponential(140, n), rng.exponential(700, n))
    account_age = np.where(rng.random(n) < 0.05, 0, np.round(account_age)).astype(int)   # guest-like new accounts
    bs_mismatch = rng.random(n) < np.where(loud, 0.40, 0.07)
    country_mismatch = rng.random(n) < np.where(loud, 0.20, 0.012)
    attempts = 1 + rng.poisson(np.where(loud, 0.7, 0.06))
    velocity = rng.poisson(np.where(loud, 2.2, 0.35))
    # camouflaged fraud: only a faint nudge on two features
    device_new |= camo & (rng.random(n) < 0.20)
    velocity = velocity + (camo & (rng.random(n) < 0.3))
    df = pd.DataFrame({
        "transaction_id": [f"T{1 + i:06d}" for i in range(n)],
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "account_age_days": account_age, "amount": amount, "hour": hour, "items": items,
        "category": cat, "device_new": device_new.astype(int), "billing_shipping_mismatch": bs_mismatch.astype(int),
        "country_mismatch": country_mismatch.astype(int), "card_attempts": attempts, "velocity_24h": velocity,
        "is_fraud": fraud.astype(int),
    })
    return df.sort_values("timestamp", kind="stable").assign(
        transaction_id=[f"T{1 + i:06d}" for i in range(n)]).reset_index(drop=True)


# ==========================================================================
# 5. products_ml.csv + interactions.csv  (made before reviews: reviews use the products)
# ==========================================================================
TASTES = ["hiking", "climbing", "paddling", "camping", "running"]
REC_CATALOG = {  # category: [(product type, price low, price high, activity weights h/cl/pa/ca/ru)]
    "Footwear": [("Hiking Boots", 130, 290, (1, 0.1, 0, 0.3, 0)), ("Trail Runners", 110, 180, (0.3, 0, 0, 0, 1)),
                 ("Approach Shoes", 110, 190, (0.2, 1, 0, 0, 0)), ("Water Shoes", 45, 110, (0, 0, 1, 0.2, 0)),
                 ("Hiking Socks", 14, 28, (0.8, 0.1, 0, 0.3, 0.5))],
    "Apparel": [("Rain Jacket", 90, 320, (0.8, 0.2, 0.5, 0.5, 0.2)), ("Running Tights", 45, 95, (0, 0, 0, 0, 1)),
                ("Fleece Pullover", 55, 150, (0.5, 0.2, 0.2, 0.8, 0.1)), ("Sun Hoodie", 40, 90, (0.4, 0.2, 1, 0.2, 0.3)),
                ("Climbing Pants", 60, 120, (0, 1, 0, 0, 0)), ("Running Shorts", 30, 60, (0, 0, 0, 0, 1))],
    "Packs": [("30L Daypack", 80, 180, (1, 0.2, 0, 0.2, 0)), ("65L Backpack", 200, 380, (1, 0, 0, 0.6, 0)),
              ("Running Vest", 70, 160, (0.1, 0, 0, 0, 1)), ("Crag Pack", 90, 170, (0, 1, 0, 0, 0)),
              ("Dry Bag", 15, 45, (0, 0, 1, 0.3, 0))],
    "Tents": [("2-Person Tent", 180, 480, (0.6, 0, 0, 1, 0)), ("6-Person Family Tent", 320, 700, (0, 0, 0, 1, 0)),
              ("Ultralight Shelter", 220, 520, (1, 0, 0, 0.3, 0)), ("Camp Canopy", 120, 260, (0, 0, 0.3, 1, 0))],
    "Sleeping": [("20F Sleeping Bag", 120, 380, (0.8, 0, 0, 0.8, 0)), ("Sleeping Pad", 45, 210, (0.6, 0, 0, 1, 0)),
                 ("Camp Pillow", 18, 55, (0.1, 0, 0, 1, 0))],
    "Climbing": [("Climbing Harness", 55, 130, (0, 1, 0, 0, 0)), ("Climbing Shoes", 90, 190, (0, 1, 0, 0, 0)),
                 ("Locking Carabiner", 10, 24, (0, 1, 0, 0, 0)), ("60m Rope", 170, 300, (0, 1, 0, 0, 0)),
                 ("Chalk Bag", 16, 36, (0, 1, 0, 0, 0))],
    "Paddling": [("Kayak Paddle", 90, 260, (0, 0, 1, 0, 0)), ("Life Vest", 70, 160, (0, 0, 1, 0, 0)),
                 ("Spray Skirt", 60, 140, (0, 0, 1, 0, 0)), ("Paddle Leash", 15, 30, (0, 0, 1, 0, 0)),
                 ("Deck Bag", 35, 80, (0, 0, 1, 0.2, 0))],
    "Camp Kitchen": [("Backpacking Stove", 45, 170, (0.6, 0, 0, 1, 0)), ("Cookset", 40, 130, (0.3, 0, 0, 1, 0)),
                     ("Cooler", 60, 320, (0, 0, 0.5, 1, 0)), ("Water Filter", 30, 120, (1, 0, 0.3, 0.5, 0.2))],
    "Running": [("Running Belt", 20, 45, (0, 0, 0, 0, 1)), ("Soft Flask", 18, 35, (0.2, 0, 0, 0, 1)),
                ("GPS Watch Band", 25, 60, (0.2, 0, 0, 0, 1))],
    "Accessories": [("Headlamp", 22, 110, (0.6, 0.4, 0.2, 0.8, 0.5)), ("Trekking Poles", 60, 190, (1, 0, 0, 0.2, 0)),
                    ("First-Aid Kit", 18, 65, (0.6, 0.4, 0.5, 0.6, 0.3)), ("Sunscreen SPF 50", 9, 18, (0.5, 0.4, 0.8, 0.5, 0.6)),
                    ("Multitool", 30, 120, (0.4, 0.2, 0.3, 0.8, 0))],
}
REC_N = {"Footwear": 26, "Apparel": 30, "Packs": 20, "Tents": 16, "Sleeping": 14, "Climbing": 22, "Paddling": 20,
         "Camp Kitchen": 18, "Running": 12, "Accessories": 22}       # 200 products
REC_PREFIX = {"Footwear": "FTW", "Apparel": "APL", "Packs": "PCK", "Tents": "TNT", "Sleeping": "SLP", "Climbing": "CLM",
              "Paddling": "PDL", "Camp Kitchen": "KIT", "Running": "RUN", "Accessories": "ACC"}
BRANDS = ["Pinecrest", "Larkspur Ridge", "Granite Fork", "Wrenfield", "Cedar Notch", "Tallgrass", "Juniper Vane", "Quillback"]
MODELS = ["Summit", "Switchback", "Timberline", "Ridgeline", "Cascade", "Basecamp", "Trailhead", "Alpenglow", "Skyline",
          "Backcountry", "Wayfinder", "Cairn", "Talus", "Meadow", "Ember", "Drift", "Solstice", "Tundra", "Canyon",
          "Foothill", "Lookout", "Headwater", "Sawtooth", "Mesa", "Eddy", "Riffle", "Bouldery", "Tempo"]


def make_rec_products() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = rng_for(5)
    rows, aff, used = [], [], set()
    for cat, types in REC_CATALOG.items():
        for i in range(REC_N[cat]):
            t, lo, hi, w = types[i % len(types)]
            while True:
                name = f"{rng.choice(BRANDS)} {rng.choice(MODELS)} {t}"
                if name not in used:
                    used.add(name)
                    break
            dollars = int(rng.uniform(lo, hi))
            w = np.array(w, float) * rng.uniform(0.8, 1.2, 5)
            rows.append({"product_id": f"{REC_PREFIX[cat]}-{3001 + i}", "name": name, "category": cat, "product_type": t,
                         "price": float(dollars) + 0.95,
                         "primary_activity": TASTES[int(np.argmax(w))] if w.max() - np.sort(w)[-2] > 0.15 else "general"})
            aff.append(w / w.max())
    df = pd.DataFrame(rows)
    df.insert(4, "brand", [next(b for b in BRANDS if nm.startswith(b + " ")) for nm in df["name"]])
    quality = rng.normal(0, 0.45, len(df))
    pop = rng.lognormal(0, 0.8, len(df)) * (80 / df["price"].to_numpy()) ** 0.35
    df["launch_year"] = rng.integers(2019, 2026, len(df))
    df["_quality"] = quality
    df["_pop"] = pop
    return df, np.array(aff), quality


def make_interactions(customers: pd.DataFrame, products: pd.DataFrame, aff: np.ndarray, quality: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rng = rng_for(6)
    ids = customers["customer_id"].to_numpy()
    users = np.sort(rng.choice(ids, 3000, replace=False))
    nu, ni = len(users), len(products)
    # each customer: one dominant taste (70%), a little of the rest
    dom = rng.choice(5, nu, p=[0.32, 0.16, 0.14, 0.24, 0.14])
    taste = rng.dirichlet(np.full(5, 0.35), nu) * 0.35
    taste[np.arange(nu), dom] += 0.65
    pop = products["_pop"].to_numpy()
    n_items = np.clip(np.round(rng.lognormal(np.log(9), 0.6, nu)), 3, 45).astype(int)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for u in range(nu):
        score = 5.0 * (aff @ taste[u]) + 0.45 * np.log(pop)
        p = np.exp(score - score.max())
        p /= p.sum()
        chosen = rng.choice(ni, min(n_items[u], ni), replace=False, p=p)
        for i in chosen:
            match = aff[i] @ taste[u]
            purchases = 1 + rng.poisson(0.25 + 0.6 * (products["price"].iat[i] < 40))
            rated = rng.random() < 0.38
            rating = int(np.clip(np.round(3.1 + 1.3 * match + quality[i] + rng.normal(0, 0.75)), 1, 5)) if rated else None
            day = start + pd.Timedelta(days=int(rng.integers(0, 731)))
            rows.append((users[u], products["product_id"].iat[i], int(purchases), rating, day.strftime("%Y-%m-%d")))
    df = pd.DataFrame(rows, columns=["customer_id", "product_id", "purchases", "rating", "last_purchase_date"])
    df["rating"] = df["rating"].astype("Int64")
    df = df.sort_values(["customer_id", "last_purchase_date", "product_id"]).reset_index(drop=True)
    return df, {"dominant_taste": dict(zip(users, np.array(TASTES)[dom]))}


# ==========================================================================
# 4. reviews.csv
# ==========================================================================
ASPECTS = {  # aspect: (positive phrases, negative phrases); {p} = product type in lower case
    "waterproofing": (["kept me completely dry in a downpour", "is truly waterproof", "shed rain all night",
                       "stayed dry inside after a full day of rain", "the seams never leaked"],
                      ["is not waterproof at all", "leaked at the seams the first time it rained",
                       "soaked through in light rain", "did not keep me dry", "let water in within an hour"]),
    "warmth": (["is warm without being bulky", "kept me toasty on a cold night", "is warmer than I expected",
                "blocks the wind really well", "was plenty warm at freezing temperatures"],
               ["is not warm enough for cold nights", "left me shivering", "is much colder than the rating claims",
                "does nothing against the wind", "was never warm, even with layers"]),
    "fit": (["fits true to size", "fits perfectly", "is comfortable right out of the box", "has a great fit",
             "sizing is spot on"],
            ["runs small", "fits badly", "is way too tight in the shoulders", "runs a full size large",
             "rubbed and gave me blisters"]),
    "durability": (["still looks new after a season of use", "is built to last", "has held up to rough trails",
                    "feels sturdy and well made", "survived two years of hard use"],
                   ["fell apart after three trips", "ripped on the second outing", "broke within a month",
                    "feels flimsy and cheap", "the zipper failed after a week"]),
    "price": (["is a great value", "is worth every penny", "is fairly priced for the quality",
               "cost less than similar gear and works better"],
              ["is overpriced for what you get", "is not worth the money", "is far too expensive for the quality",
               "costs twice what it should"]),
    "shipping": (["arrived quickly", "shipped fast and well packed", "showed up two days early"],
                 ["arrived late", "came with a damaged box", "took three weeks to arrive", "shipped the wrong color"]),
    "weight": (["is impressively light", "barely adds any weight to my pack", "is lighter than my old one"],
               ["is way too heavy", "is heavier than advertised", "weighs a ton for what it is"]),
    "setup": (["is easy to set up", "goes together in minutes", "is simple to use even in the dark"],
              ["is a pain to set up", "has confusing instructions", "was frustrating to use"]),
}
CAT_ASPECTS = {"Tents": ["waterproofing", "setup", "weight", "durability", "price", "shipping"],
               "Sleeping": ["warmth", "weight", "durability", "price", "shipping"],
               "Packs": ["fit", "durability", "weight", "waterproofing", "price", "shipping"],
               "Footwear": ["fit", "durability", "waterproofing", "price", "shipping"],
               "Apparel": ["fit", "warmth", "waterproofing", "durability", "price", "shipping"],
               "Climbing": ["fit", "durability", "weight", "price", "shipping"],
               "Paddling": ["fit", "durability", "waterproofing", "price", "shipping"],
               "Camp Kitchen": ["setup", "durability", "weight", "price", "shipping"],
               "Running": ["fit", "weight", "durability", "price", "shipping"],
               "Accessories": ["durability", "weight", "setup", "price", "shipping"]}
NEUTRAL = ["I bought this for a trip to the Sawtooths.", "Picked it up at the Bend store.", "Ordered it online in the fall.",
           "I used it on a week-long trip.", "This is my second one.", "Bought it as a gift for my brother.",
           "I have used it on about ten trips now.", "My partner has the same one.", "Got it during the spring sale.",
           "I compared a few before choosing this one.", "Used it mostly on weekend trips.", "I am fairly new to backpacking."]
POS_OPEN = ["Love it.", "Great purchase.", "Really happy with this.", "Highly recommend.", "Exactly what I needed.",
            "Five stars from me.", "Would buy again.", "No regrets at all.", "Not bad at all.", "Never had a problem with it."]
NEG_OPEN = ["Very disappointed.", "Would not buy again.", "Not happy with this.", "Save your money.", "Returned it.",
            "Not what I expected.", "Do not recommend.", "Regret this purchase.", "Pretty bad overall.", "Never again."]
MIX_JOIN = ["but", "although", "however"]
POS_TITLES = ["Great {p}", "Love this {p}", "Best {p} I have owned", "Solid choice", "Worth it", "Excellent",
              "Does the job well", "Happy camper", "Highly recommended", "Exceeded expectations"]
NEG_TITLES = ["Disappointing {p}", "Not worth it", "Returned it", "Poor quality", "Would not buy again",
              "Do not bother", "Not as advertised", "Fell short", "Frustrating", "Not impressed"]
MID_TITLES = ["It is okay", "Mixed feelings", "Good and bad", "Decent {p}", "Fine for the price", "Average"]
TYPOS = {"waterproof": "waterprof", "comfortable": "comfortible", "definitely": "definately", "really": "realy",
         "expected": "expectd", "recommend": "reccomend", "quality": "qaulity", "perfectly": "perfectely",
         "disappointed": "dissapointed", "weight": "wieght"}


def _sentence(subject: str, phrase: str) -> str:
    return f"{subject} {phrase}."


def make_reviews(products: pd.DataFrame) -> pd.DataFrame:
    rng = rng_for(4)
    n = 4000
    pidx = rng.choice(len(products), n, p=(products["_pop"] / products["_pop"].sum()).to_numpy())
    q = products["_quality"].to_numpy()[pidx]
    stars = np.clip(np.round(3.7 + 1.1 * q + rng.normal(0, 1.35, n)), 1, 5).astype(int)
    rows = []
    for k in range(n):
        prod = products.iloc[pidx[k]]
        ptype = prod["product_type"].lower()
        subj = rng.choice([f"The {ptype}", "It", "This one", f"This {ptype}"])
        aspects = list(rng.choice(CAT_ASPECTS[prod["category"]], int(rng.integers(1, 4)), replace=False))
        r = int(stars[k])
        sent_score = {5: 1.0, 4: 0.75, 3: 0.5, 2: 0.22, 1: 0.05}[r]
        mismatch = rng.random() < 0.025          # rating clicked wrong: text says the opposite
        if mismatch and r != 3:
            sent_score = 1 - sent_score
        parts = []
        if rng.random() < 0.55:
            parts.append(str(rng.choice(NEUTRAL)))
        for a in aspects:
            pos, neg = ASPECTS[a]
            good = rng.random() < sent_score
            parts.append(_sentence(subj if not parts or rng.random() < 0.5 else "It", str(rng.choice(pos if good else neg))))
        if r == 3 or (rng.random() < 0.25):          # mixed review: one counterpoint
            a = str(rng.choice(CAT_ASPECTS[prod["category"]]))
            pos, neg = ASPECTS[a]
            other = str(rng.choice(neg if sent_score >= 0.5 else pos))
            parts.append(f"{str(rng.choice(MIX_JOIN)).capitalize()} it {other}.")
        opener = rng.random()
        if opener < 0.6:
            parts.insert(0, str(rng.choice(POS_OPEN if sent_score >= 0.5 else NEG_OPEN)))
        text = " ".join(parts)
        if rng.random() < 0.06:                      # a typo or two
            for good_w, bad_w in TYPOS.items():
                if good_w in text and rng.random() < 0.7:
                    text = text.replace(good_w, bad_w, 1)
        if rng.random() < 0.03:
            text = text.lower()
        titles = POS_TITLES if r >= 4 else NEG_TITLES if r <= 2 else MID_TITLES
        if mismatch and r != 3:
            titles = NEG_TITLES if r >= 4 else POS_TITLES
        title = str(rng.choice(titles)).format(p=ptype)
        recommended = int(rng.random() < {5: 0.97, 4: 0.88, 3: 0.45, 2: 0.10, 1: 0.03}[r])
        rows.append((prod["product_id"], prod["category"], r, title, text, recommended))
    df = pd.DataFrame(rows, columns=["product_id", "category", "rating", "title", "text", "recommended"])
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 731, n)), unit="D")
    df.insert(0, "review_date", dates.strftime("%Y-%m-%d"))
    df.insert(0, "review_id", [f"R{1 + i:05d}" for i in range(n)])
    return df


# ==========================================================================
# 6. daily_store_sales.csv
# ==========================================================================
DAILY_STORES = {"Boise": (6200, 0.08), "Bend": (4700, 0.03), "Flagstaff": (4300, 0.15), "Asheville": (4500, -0.04)}
DOW = np.array([0.84, 0.80, 0.85, 0.90, 1.08, 1.47, 1.26])            # Mon..Sun
SALE_EVENTS = {  # name: list of (start, days, lift)
    "Memorial Day sale": [("2023-05-26", 4), ("2024-05-24", 4), ("2025-05-23", 4)],
    "Labor Day sale": [("2023-09-01", 4), ("2024-08-30", 4), ("2025-08-29", 4)],
    "Black Friday": [("2023-11-24", 4), ("2024-11-29", 4), ("2025-11-28", 4)],
    "Holiday rush": [("2023-12-18", 6), ("2024-12-18", 6), ("2025-12-18", 6)],
}
EVENT_LIFT = {"Memorial Day sale": 0.45, "Labor Day sale": 0.40, "Black Friday": 0.85, "Holiday rush": 0.35}
CLOSED = ["2023-11-23", "2023-12-25", "2024-11-28", "2024-12-25", "2025-11-27", "2025-12-25"]


def make_daily_sales() -> pd.DataFrame:
    rng = rng_for(7)
    days = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    nd = len(days)
    doy = days.dayofyear.to_numpy()
    yearly = 1 + 0.22 * np.sin(2 * np.pi * (doy - 100) / 365.25) + 0.12 * np.exp(-((doy - 350) / 12) ** 2)
    event = np.full(nd, "", dtype=object)
    lift = np.zeros(nd)
    for name, spans in SALE_EVENTS.items():
        for s, length in spans:
            m = (days >= pd.Timestamp(s)) & (days < pd.Timestamp(s) + pd.Timedelta(days=length))
            event[m] = name
            lift[m] = EVENT_LIFT[name]
    closed = days.isin(pd.to_datetime(CLOSED))
    rows = []
    for store, (base, growth) in DAILY_STORES.items():
        t = np.arange(nd) / 365.25
        ar = np.zeros(nd)
        eps = rng.normal(0, 0.075, nd)
        for i in range(1, nd):                                         # slow-moving local conditions
            ar[i] = 0.82 * ar[i - 1] + eps[i]
        log_sales = (np.log(base) + growth * t + np.log(yearly) + np.log(DOW[days.dayofweek]) + np.log1p(lift)
                     + ar + rng.normal(0, 0.05, nd))
        sales = np.round(np.exp(log_sales), 2)
        sales[closed] = 0.0
        tx = np.round(sales / rng.normal(72, 3, nd)).astype(int)
        for i in range(nd):
            rows.append((days[i].strftime("%Y-%m-%d"), store, float(sales[i]), int(tx[i]), int(lift[i] > 0),
                         event[i] if event[i] else ("Closed" if closed[i] else "")))
    df = pd.DataFrame(rows, columns=["date", "store", "sales", "transactions", "promo", "event"])
    df["event"] = df["event"].replace("", np.nan)
    return df.sort_values(["date", "store"]).reset_index(drop=True)


# ==========================================================================
ZIP_MEMBERS = ["README.md", "customers_churn.csv", "weekly_demand.csv", "transactions.csv", "reviews.csv",
               "products_ml.csv", "interactions.csv", "daily_store_sales.csv", "make_ml_data.py"]


def write_zip(out: Path) -> list[str]:
    """data/pinecrest-ml-data.zip unpacks to a data/ folder next to the learner's notebooks."""
    missing = []
    with zipfile.ZipFile(out / "pinecrest-ml-data.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in ZIP_MEMBERS:
            src = out / name
            if not src.exists():
                src = Path(__file__).parent / name
            if not src.exists():
                missing.append(name)
                continue
            zi = zipfile.ZipInfo(f"data/{name}", FIXED_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, src.read_bytes())
    return missing


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    churn, _ = make_churn()
    demand = make_weekly_demand()
    tx = make_transactions()
    rec_products, aff, quality = make_rec_products()
    inter, _ = make_interactions(churn, rec_products, aff, quality)
    reviews = make_reviews(rec_products)
    daily = make_daily_sales()

    to_csv(churn, out / "customers_churn.csv")
    to_csv(demand, out / "weekly_demand.csv")
    to_csv(tx, out / "transactions.csv")
    to_csv(reviews, out / "reviews.csv")
    to_csv(rec_products.drop(columns=["_quality", "_pop"]), out / "products_ml.csv")
    to_csv(inter, out / "interactions.csv")
    to_csv(daily, out / "daily_store_sales.csv")
    missing = write_zip(out)
    counts = {"customers_churn": len(churn), "weekly_demand": len(demand), "transactions": len(tx),
              "reviews": len(reviews), "products_ml": len(rec_products), "interactions": len(inter),
              "daily_store_sales": len(daily)}
    print("Rows written:")
    for k, v in counts.items():
        print(f"  {k:<18}{v:>8,}")
    if missing:
        print("Not in the zip (file not found):", ", ".join(missing))
    print(f"pinecrest-ml-data.zip: {(out / 'pinecrest-ml-data.zip').stat().st_size / 1e6:.2f} MB")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Pinecrest Outfitters machine-learning data.")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent, help="output folder (default: this script's folder)")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
