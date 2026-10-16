# Session 7 labs: Nearest Neighbors and Naive Bayes

Lessons **4.1 k-Nearest Neighbors** and **4.2 Naive Bayes**. Every lab is new practice on the lesson's
skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and `debug_knn.py`. Keep every `*_solution.py` for yourself until after
the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l4_1-knn/fraud_neighbors_starter.py` / `_solution.py` | Who decides an unscaled distance, scaled against unscaled kNN, choosing k by CV average precision, ties at a small k, on fraud | 12 min | `account_age_days` 99.7% of the distance; AP 0.056 unscaled, 0.397 scaled (LR 0.444); CV AP 0.362 / 0.448 / 0.468 for k = 5 / 51 / 201; test AP 0.329 (k=5) and 0.419 (k=201) |
| `l4_1-knn/lookalike_days_starter.py` / `_solution.py` | `KNeighborsRegressor` for Boise's daily sales: raw against cyclical calendar features, k, and why averaging the past misses growth | 8 min (pairs) | weekday mean $1,669; raw unscaled $1,889, scaled $1,327; cycles + events k=5 $1,244 (+$921 bias); 2024 only $1,202 (+$458) |
| `l4_1-knn/debug_knn.py` | Two loud bugs (k larger than the rows; NaN reaches kNN) and two silent ones (no scaler; `kneighbors` on an unscaled row) (deliberately broken) | fast finishers | Fixed: `CV ROC AUC 0.854`, `3 of 51 lookalikes churned; ... median ... 46 (the customer's: 53)` |
| `l4_2-naive-bayes/fraud_flags_bayes_starter.py` / `_solution.py` | Bayes' rule on a fraud flag, likelihood ratios, the naive product against direct counts, `BernoulliNB` on nine flags | 12 min | country mismatch posterior 0.091 both ways; night + new device naive 0.280 vs counted 0.425; BernoulliNB AUC 0.810, AP 0.439 (LR 0.810, 0.435) |
| `l4_2-naive-bayes/review_category_starter.py` / `_solution.py` | Ten-class text classification of review categories: baseline, Multinomial / Complement NB, alpha by CV, telling words, mix-ups | 8 min (pairs) | baseline 0.237; MultinomialNB 0.586, ComplementNB 0.592, LR 0.605; top mix-up Camp Kitchen → Accessories |

## How to run

```bash
python labs/module-07/l4_1-knn/fraud_neighbors_starter.py
python labs/module-07/l4_2-naive-bayes/fraud_flags_bayes_starter.py
```

`fraud_neighbors` Step 4 cross-validates kNN three times on 35,000 payments (about 10 seconds).

Check everything in this session at once (8 files; about a minute):

```bash
python3 labs/check_labs.py module-07
```

## Timing in Session 7

| Clock | Lab |
|-------|-----|
| 0:50 | `fraud_neighbors_starter.py` (12 min), then `lookalike_days_starter.py` in pairs (8 min); fast finishers take `debug_knn.py` |
| 2:15 | `fraud_flags_bayes_starter.py` (12 min), then `review_category_starter.py` in pairs (8 min) |
