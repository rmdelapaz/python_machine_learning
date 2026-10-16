# Session 2 labs: Local Explanations, Fairness and K-Means

Lessons **7.3 LIME and Fairness Checks** and **8.1 K-Means Clustering**. Every lab is new practice on the
lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` and `debug_*.py` files. Keep every `*_solution.py` until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l7_3-lime-fairness/lime_fraud_starter.py` / `_solution.py` | LIME on the fraud model: rules, fidelity, the surrogate's prediction, stability at 500 and 5,000 variants, SHAP as a second opinion | 12 min | T031633 (0.990): fidelity 0.12, surrogate 0.077; top rule agrees in 3/2/1 of 3 seeds at 500 variants, 2/3/3 at 5,000; SHAP: velocity, velocity, new device |
| `l7_3-lime-fairness/fairness_calibration_starter.py` / `_solution.py` | Calibration (predicted vs real churn) and campaign profit ($15 / $60) by age group; a permutation test on the largest calibration gap | 8 min (pairs, TODO 1-2) + homework | 300 offers (52.3% churners); gaps -0.042 to +0.025; profit per customer $2.00 (45-54) to $4.80 (unknown); largest gap 0.042 vs 0.046 by chance, p = 0.642 |
| `l7_3-lime-fairness/debug_lime.py` | A pandas row (`KeyError: 0`), gaps in the explainer's data, labels instead of probabilities, too few variants (deliberately broken) | fast finishers | Fixed: C104051, recency +0.295, discounts +0.240, tenure -0.071 |
| `l8_1-kmeans/scaling_choices_starter.py` / `_solution.py` | K-Means on recency, spend and tickets: raw, standardized, recency weighted twice; centers in real units; churn by cluster; crosstab | 12 min | Raw = spend bands; standardized finds 780 silent customers (271 days, churn 51%) and 1,201 who complain (2.5 tickets, 22%); recency x2: 539 at 307 days, churn 53% |
| `l8_1-kmeans/payment_clusters_starter.py` / `_solution.py` | K-Means on 50,000 payments: raw vs log + scale, seeds, naming from profiles, fraud rate by group | 13 min | Raw = account-age bands; scaled: small everyday orders 0.27% fraud, big baskets 0.76%, brand-new accounts 0.78%, busy cards 2.13% (243 of 438 frauds); inertia 97,067 for every seed |
| `l8_1-kmeans/debug_kmeans.py` | A text column, gaps, scaled centers shown as real, a hard-coded cluster number (deliberately broken) | fast finishers | Fixed: centers in real units; `big-ticket segment: 1073 customers, average order $336.92, churn 0.159` |

## How to run

```bash
python labs/module-02/l7_3-lime-fairness/lime_fraud_starter.py
python labs/module-02/l8_1-kmeans/scaling_choices_starter.py
python3 labs/check_labs.py module-02          # all 8 files, about a minute
```

## Timing in Session 2

| Clock | Lab |
|-------|-----|
| 0:45 | `lime_fraud_starter.py` (12 min), then `fairness_calibration_starter.py` TODO 1-2 in pairs (8 min) |
| 2:10 | `scaling_choices_starter.py` (12 min), then `payment_clusters_starter.py` (13 min) |
