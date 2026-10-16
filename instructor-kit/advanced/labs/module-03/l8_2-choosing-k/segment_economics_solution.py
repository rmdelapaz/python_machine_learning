# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.2 B (SOLUTION): should marketing target segments, or customers?

Lesson 8.2's segment brief suggests an action per segment. The retention team asks a sharper question
with the Foundations economics ($15 per offer, $60 per real churner reached): is it worth sending the
retention offer to a whole segment? And how does that compare with targeting customers one by one with
the churn model? Segments are built on all customers (description); the model comparison uses only the
1,500 test customers of the canonical split.
"""
# %% Step 1: Lesson 8.2's segmenter, with names read from the profile (done for you)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

pd.set_option("display.width", 120)
customers = pd.read_csv(DATA / "customers_churn.csv")
behavior = ["orders_12m", "avg_order_value", "discount_share", "return_rate"]
log_some = ColumnTransformer(
    [("log", FunctionTransformer(np.log, feature_names_out="one-to-one"), ["orders_12m", "avg_order_value"])],
    remainder="passthrough", verbose_feature_names_out=False,
)
segmenter = make_pipeline(log_some, StandardScaler(), KMeans(n_clusters=5, n_init=10, random_state=42))
customers["cluster"] = segmenter.fit_predict(customers[behavior])
profile = customers.groupby("cluster")[behavior].mean()
names = {}
for cl, row in profile.iterrows():                    # named from the profile, never from the number
    if row["discount_share"] > 0.4:
        names[cl] = "Deal Hunters"
    elif row["return_rate"] > 0.2:
        names[cl] = "Gear Testers"
    elif row["avg_order_value"] > 250:
        names[cl] = "Big-Trip Planners"
    elif row["orders_12m"] > 7:
        names[cl] = "Trail Regulars"
    else:
        names[cl] = "Casual Campers"
customers["segment"] = customers["cluster"].map(names)
print(customers["segment"].value_counts().to_dict())

# %% Step 2: a blanket offer to a whole segment
OFFER, SAVE = 15, 60
blanket = customers.groupby("segment").agg(customers=("churned", "size"), churn=("churned", "mean"))
blanket["profit_per_customer"] = SAVE * blanket["churn"] - OFFER
print(blanket.sort_values("churn", ascending=False).round(2))
print(f"break-even churn rate for a blanket offer: {OFFER / SAVE:.2f}")

# %% Step 3: the churn model on the canonical split, offers at the 0.25 break-even
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user",
            "email_opens_90d"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
model = HistGradientBoostingClassifier(random_state=42).fit(train[features].astype(float), train["churned"])
test = test.assign(offer=model.predict_proba(test[features].astype(float))[:, 1] >= OFFER / SAVE)

# %% Step 4: blanket vs model, segment by segment, on the same 1,500 test customers
hit = test["offer"] & (test["churned"] == 1)
compare = pd.DataFrame({
    "customers": test.groupby("segment").size(),
    "blanket_profit": SAVE * test.groupby("segment")["churned"].sum() - OFFER * test.groupby("segment").size(),
    "model_offers": test.groupby("segment")["offer"].sum(),
    "model_profit": SAVE * hit.groupby(test["segment"]).sum() - OFFER * test.groupby("segment")["offer"].sum(),
})
print(compare)
print("totals:", compare.sum().to_dict())
