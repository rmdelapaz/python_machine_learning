# Pinecrest Outfitters machine-learning data

Pinecrest Outfitters is a **fictional** outdoor-gear retailer with an online shop and four stores
(Boise, Bend, Flagstaff, Asheville). You may know it from the
[Python Data Science](https://rays-python-datascience.netlify.app/) course. These files are new
tables built for the [Python Machine Learning](https://rays-python-machine-learning.netlify.app/)
course: each one is shaped for a family of machine-learning problems. Every customer, product,
review and number is invented.

**License:** Fictional data written for this course. © 2026 Ray de la Paz.

## Using it

Unzip `pinecrest-ml-data.zip` next to your notebooks so you have a `data/` folder, then load files with
a relative path:

```python
import pandas as pd
customers = pd.read_csv("data/customers_churn.csv")
```

Rebuild everything (seeded, so you get the same files): `python data/make_ml_data.py`. Every file is
byte-identical from run to run, and each table has its own random stream, so the tables do not depend
on one another. `make_ml_data.py` needs only numpy and pandas.

The image lessons (CNNs and transfer learning) use public datasets that PyTorch downloads for you:
Fashion-MNIST and EuroSAT. Those lessons show how.

## Files at a glance

| File | Rows | One row is | Used for |
|---|---:|---|---|
| `customers_churn.csv` | 6,000 | a customer at the 2025-06-30 snapshot | classification (churn), feature work, clustering |
| `weekly_demand.csv` | 9,984 | one product in one store in one week | regression (units sold) |
| `transactions.csv` | 50,000 | one online card payment | imbalanced classification, anomaly detection |
| `reviews.csv` | 4,000 | one product review | text classification (sentiment) |
| `products_ml.csv` | 200 | a product in the recommendation catalog | recommendation systems |
| `interactions.csv` | 32,178 | one customer's history with one product | recommendation systems |
| `daily_store_sales.csv` | 4,384 | one store on one day (2023 to 2025) | sequence models and forecasting |
| `make_ml_data.py` | | the generator | |

Missing values are empty cells. Dates are `YYYY-MM-DD` text (pass `parse_dates=[...]` to get dates).

## customers_churn.csv

One row per customer who bought at least once in the 12 months up to **2025-06-30**. The features
describe those 12 months. The target, `churned`, says whether the customer then made **no purchase in
the following six months** (2025-07-01 to 2025-12-31).

| Column | Type | Notes |
|---|---|---|
| `customer_id` | text | key, `C100001` to `C106000` |
| `signup_date` | text (date) | when the account was opened |
| `home_store` | text | where they shop most: `Online`, `Boise`, `Bend`, `Flagstaff`, `Asheville` |
| `region` | text | `Northwest`, `Southwest`, `Southeast`, `Mountain`, `Midwest`, `Northeast` (store shoppers get their store's region) |
| `age_group` | text | `18-24` to `65+`; 487 missing (an optional profile field, skipped more often online) |
| `loyalty_tier` | text | `Basic`, `Silver`, `Gold` (from spend and tenure) |
| `app_user` | int (0/1) | uses the Pinecrest mobile app |
| `tenure_months` | int | months since signup, 1 to 96 |
| `orders_12m` | int | orders in the 12 months, at least 1 |
| `spend_12m` | float | dollars spent in the 12 months |
| `avg_order_value` | float | `spend_12m / orders_12m` (so these three columns are redundant) |
| `days_since_last_order` | int | days from the last order to the snapshot, 0 to 364 |
| `returns_12m` | int | orders with at least one returned item |
| `return_rate` | float | share of the 12-month spend that was refunded, 0 to 1 |
| `support_tickets_12m` | int | customer-service tickets opened |
| `discount_share` | float | share of spend that used a discount or promo code |
| `email_opens_90d` | int | marketing emails opened in the last 90 days; 537 missing (tracking blocked or unsubscribed) |
| `preferred_category` | text | the category they buy most (9 categories) |
| `cancellation_reason` | text | reason given when the account was closed; filled for 625 customers, empty for the rest. **Read the warning below before you use it.** |
| `churned` | int (0/1) | the target: 1 = no purchase in the next six months (16.5% of customers) |

> **Warning: `cancellation_reason` is recorded AFTER the snapshot.** Only customers who later closed
> their account have one, and every one of them churned. A model that uses it looks brilliant in
> testing and is useless in real life, because on the day you need a prediction the column is
> always empty. This is called *target leakage*, and the course uses this column to show it.

## weekly_demand.csv

Units sold of 24 products in the four stores, every week of 2024 and 2025 (24 × 4 × 104 rows), sorted by
week, store and product.

| Column | Type | Notes |
|---|---|---|
| `week_start` | text (date) | the Monday that starts the week, 2024-01-01 to 2025-12-22 |
| `year`, `week_of_year` | int | from `week_start` (ISO week, 1 to 52) |
| `store` | text | Boise, Bend, Flagstaff, Asheville |
| `store_sqft` | int | store size: Boise 18,000, Bend 12,500, Flagstaff 11,000, Asheville 9,500 |
| `product_id`, `product_name` | text | 24 products, 2 or 3 per category, ids like `TNT-2001` |
| `category` | text | the 9 Pinecrest categories |
| `list_price` | float | regular price (does not change over the period) |
| `discount_pct` | int | 0, 10, 15, 20, 25, 30 or 40 (40 only as off-season clearance) |
| `price` | float | shelf price that week: `list_price * (1 - discount_pct / 100)` |
| `promo` | int (0/1) | featured in the weekly flyer |
| `holiday_week` | int (0/1) | the weeks of Memorial Day, July 4, Labor Day, Black Friday and the week before Christmas |
| `avg_high_temp_f` | float | average daily high in the store's city that week (made-up values) |
| `units_sold` | int | the target; skewed to the right (mean 25, median 18, max 420) |

Zeros: winter gear sells nothing in high summer, so some zeros are real. A few zeros in busy weeks are
not: the shelf was empty (a stockout) and nothing could sell. They are not flagged.

## transactions.csv

Online card payments from 2025-01-01 to 2025-06-30, in time order. About 0.9% are fraudulent.

| Column | Type | Notes |
|---|---|---|
| `transaction_id` | text | key, `T000001` to `T050000`, in time order |
| `timestamp` | text | `YYYY-MM-DD HH:MM:SS` |
| `account_age_days` | int | age of the shopper's account; 0 for brand-new accounts |
| `amount` | float | dollars |
| `hour` | int | hour of day, 0 to 23 (the same as in `timestamp`) |
| `items` | int | items in the cart |
| `category` | text | main category of the cart: the 9 Pinecrest categories plus `Gift Cards` |
| `device_new` | int (0/1) | first purchase from this device |
| `billing_shipping_mismatch` | int (0/1) | billing and shipping addresses differ |
| `country_mismatch` | int (0/1) | the card's country differs from the shipping country |
| `card_attempts` | int | card entries tried before this payment went through |
| `velocity_24h` | int | other payments from the same account in the previous 24 hours |
| `is_fraud` | int (0/1) | the target, confirmed by a later chargeback |

## reviews.csv

Product reviews from the website, 2024 and 2025. Written from templates, so they are short and a little
repetitive, but they mix praise and complaints, use negation ("not waterproof at all"), include a few
typos, and a few have a star rating that does not match the text (the reviewer clicked the wrong star).

| Column | Type | Notes |
|---|---|---|
| `review_id` | text | key, `R00001` to `R04000`, in date order |
| `review_date` | text (date) | |
| `product_id` | text | joins to `products_ml.csv` |
| `category` | text | the product's category |
| `rating` | int | 1 to 5 stars |
| `title` | text | short headline |
| `text` | text | the review, 3 to 50 words |
| `recommended` | int (0/1) | answered "Would you recommend this product?" with yes |

For a positive/negative sentiment task, a common choice is to drop the 3-star reviews and call 4 and 5
stars positive.

## products_ml.csv

The catalog for the recommendation lessons: 200 products in 10 categories (the usual 9, minus Snow, plus
`Paddling` and `Running`). Ids look like `PDL-3004`.

| Column | Type | Notes |
|---|---|---|
| `product_id` | text | key |
| `name` | text | brand + model + product type, unique |
| `category` | text | 10 categories |
| `product_type` | text | for example `Kayak Paddle` |
| `brand` | text | 8 invented brands |
| `price` | float | list price in USD |
| `primary_activity` | text | `hiking`, `climbing`, `paddling`, `camping`, `running`, or `general` for all-round gear |
| `launch_year` | int | 2019 to 2025 |

## interactions.csv

What each of 3,000 customers bought from the `products_ml.csv` catalog in 2024 and 2025. The
`customer_id` values are a subset of those in `customers_churn.csv`.

| Column | Type | Notes |
|---|---|---|
| `customer_id` | text | 3,000 customers, 3 to 45 products each (median 9) |
| `product_id` | text | joins to `products_ml.csv` |
| `purchases` | int | how many times they bought it (implicit feedback), 1 to 7 |
| `rating` | int | their star rating, 1 to 5, when they left one (12,208 rows); empty otherwise |
| `last_purchase_date` | text (date) | the most recent purchase of that product |

Each (`customer_id`, `product_id`) pair appears once.

## daily_store_sales.csv

Register sales of the four stores, every day from 2023-01-01 to 2025-12-31, sorted by date and store.

| Column | Type | Notes |
|---|---|---|
| `date` | text (date) | |
| `store` | text | Asheville, Bend, Boise, Flagstaff (alphabetical within a date) |
| `sales` | float | dollars; 0.00 on the six days all stores were closed |
| `transactions` | int | register transactions |
| `promo` | int (0/1) | a store-wide sale was running |
| `event` | text | `Memorial Day sale`, `Labor Day sale`, `Black Friday`, `Holiday rush`, `Closed` (Thanksgiving and Christmas Day), or empty |
