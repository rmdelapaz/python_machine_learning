# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.2 B: should marketing target segments, or customers?

Lesson 8.2's segment brief suggests an action per segment. The retention team asks a sharper question
with the Foundations economics ($15 per offer, $60 per real churner reached): is it worth sending the
retention offer to a whole segment? And how does that compare with targeting customers one by one with
the churn model? Segments are built on all customers (description); the model comparison uses only the
1,500 test customers of the canonical split. Work through TODO 1 to 4; the file runs as it is.
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
# TODO 1: per segment: customers, churn rate, and the profit per customer of offering EVERYONE in it:
#         SAVE * churn - OFFER. Sort by churn and print. At what churn rate does a blanket offer break even?
print("TODO 1: blanket-offer profit per customer, by segment")

# %% Step 3: the churn model on the canonical split, offers at the 0.25 break-even
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user",
            "email_opens_90d"]
# TODO 2: train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42);
#         HistGradientBoostingClassifier(random_state=42) on train[features].astype(float); add a column
#         offer = predicted probability >= OFFER / SAVE to test.
print("TODO 2: the churn model and its offers on the test customers")

# %% Step 4: blanket vs model, segment by segment, on the same 1,500 test customers
# TODO 3: per segment on the TEST customers: the blanket profit (60 x churners - 15 x customers), the model's
#         number of offers and its profit (60 x offered churners - 15 x offers). Print the table and totals.
# TODO 4 (think): what are segments good for, then, if not for choosing whom to call?
print("TODO 3: blanket vs model profit by segment")

print("\nExpected when done: only Deal Hunters (churn 0.31) beat the 0.25 break-even: +$3.72 per customer;")
print("Trail Regulars lose $12.56 each. On the 1,500 test customers: blanket offers to everyone lose $7,620,")
print("the model's 300 offers earn $4,920, and the model wins inside every segment (Deal Hunters $2,310 vs $1,425).")
