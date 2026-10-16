# Session 5 labs: Anomaly Detection and PCA

Lessons **8.6 Anomaly Detection** and **9.1 Principal Component Analysis**. Every lab is new practice on
the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and `debug_anomaly.py`. Keep every `*_solution.py` for yourself until
after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l8_6-anomaly-detection/monthly_queue_starter.py` / `_solution.py` | An Isolation Forest pipeline fitted once on January to March (no labels), the sign convention, a 2% review queue for April, May and June, and a fixed `contamination` threshold that drifts | 25 min | AUC of `score_samples` 0.153 vs 0.847 negated; caught 33 of 72, 35 of 65, 33 of 64 (random queue about 1.3); fixed threshold flags 1.72% to 2.16% |
| `l8_6-anomaly-detection/unusual_customers_starter.py` / `_solution.py` | Isolation Forest and LOF on customers: two anomaly scores, rank agreement, top-60 overlap, profiles; "unusual" is not "bad" | pairs / fast finishers | Rank correlation 0.515; 9 in both lists; IF's top 60 heavy buyers churn 0.067; LOF's top 60 churn 0.333 (overall 0.165) |
| `l8_6-anomaly-detection/debug_anomaly.py` | One loud bug (default LOF has no `score_samples`) and three silent ones (a random split for time-ordered data, `pred == 1` counted as outliers, the score's sign) (deliberately broken) | fast finishers | Fixed: 1.65% flagged in June, AUC 0.848, 32 frauds in the top 165, LOF novelty AUC 0.757 |
| `l9_1-pca/payment_pca_starter.py` / `_solution.py` | PCA on nine payment features: unscaled vs scaled, scree and the 90% count, naming PC1 and PC2 from loadings, fraud on the components | 25 min | Unscaled PC1 = account age (0.95); scaled: eight components for 90%; PC1 basket size, PC2 risk signals; fraud 0.206 in the top 2% of PC2 vs base 0.009 |
| `l9_1-pca/daily_sales_pca_starter.py` / `_solution.py` | PCA where it shines: days × four stores in one unit; PC1 the chain's day (events), PC2 a growth contrast between stores; reconstruction from one component | pairs / homework | Shares 0.841, 0.067, ...; PC1 1.16 to 1.23 on sale events; PC2 0.117 → -0.130 from 2023 to 2025; gap 0.105 log units |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" line. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder (the one with `data/` in it):

```bash
python labs/module-05/l8_6-anomaly-detection/monthly_queue_starter.py
python labs/module-05/l9_1-pca/payment_pca_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. Check everything in this
session at once (8 files; about 30 seconds):

```bash
python3 labs/check_labs.py module-05
```

## Timing in Session 5

| Clock | Lab |
|-------|-----|
| 0:45 | `monthly_queue_starter.py` (15 min, everyone), then `unusual_customers_starter.py` in pairs (10 min); fast finishers take `debug_anomaly.py` |
| 2:10 | `payment_pca_starter.py` (15 min, everyone), then `daily_sales_pca_starter.py` in pairs; it goes home for anyone who does not reach it |
