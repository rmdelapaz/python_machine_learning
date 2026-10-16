# Session 3 labs: Metrics and Preprocessing

Lessons **1.4 Metrics for Classification and Regression** and **2.1 Impute, Scale and Encode**. Every lab is
new practice on the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and the two `debug_*.py` files.
Keep every `*_solution.py` for yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l1_4-metrics/fraud_review_starter.py` / `_solution.py` | A fraud review queue framed as a cost ($10 per review, $375 per missed fraud): confusion matrix, precision/recall/F1, ROC AUC vs average precision, out-of-fold threshold choice, one test-set comparison, `FixedThresholdClassifier`; stretch `TunedThresholdClassifierCV` with a cost scorer | 20 min | TN 14867 FP 2 FN 93 TP 38; precision 0.950, recall 0.290; AUC 0.848, AP 0.444 vs fraud rate 0.0087; cheapest training threshold 0.04; test cost 24,525 vs 35,275 at 0.5, 49,125 reviewing nothing, 150,000 reviewing all; stretch threshold 0.040, cost 24,485 |
| `l1_4-metrics/store_forecast_starter.py` / `_solution.py` | Regression metrics on daily store sales (train 2023-2024, test 2025): four no-model forecasts, MAE/RMSE/R²/median error, the MAPE trap on closed days, error by store | homework / fast finishers | Same day last week wins: MAE 1,072, RMSE 1,628, R² 0.620; overall mean R² -0.048; MAPE 1.33e+17 on all days, 0.168 on open days; MAE / mean 0.173 |
| `l1_4-metrics/debug_metrics.py` | Two loud metric bugs (both probability columns; text labels without `pos_label`) and two silent ones (swapped arguments; AUC from labels) (deliberately broken) | fast finishers | Fixed: 0.823, 0.705, 0.395, 0.865 |
| `l2_1-preprocessing/fraud_columns_starter.py` / `_solution.py` | A ColumnTransformer on the fraud table (impute + scale the numbers, one-hot `category`), judged with AUC and AP; scoring an unseen category and a missing account age | 20 min | Gift Cards fraud rate 0.0546; (35000, 19); AUC 0.848 → 0.843 but AP 0.444 → 0.463; Gift Cards weight 0.95; fill value 447 days; P1 1.000, P2 0.001, P3 0.000 |
| `l2_1-preprocessing/demand_categories_starter.py` / `_solution.py` | One-hot store / category / product for a linear demand forecast (train 2024, test 2025, MAE); what "all zeros" means for a product never seen | homework / fast finishers | Baseline 17.13; numbers 14.71, + store 14.26, + category 13.26, + product 11.31; new tent: 52.6 (category model), 87.4 (product model); stretch 11.31 / 59.6 |
| `l2_1-preprocessing/debug_preprocess.py` | Two loud bugs (median on a text column; sparse output with pandas) and two silent ones (alphabetical tiers; `fit_transform` on the test rows) (deliberately broken) | fast finishers | Fixed: (4500, 31) (1500, 31), test mean 0.055, Basic 0 / Silver 1 / Gold 2, ROC AUC 0.824 |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" description. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder:

```bash
python labs/module-03/l1_4-metrics/fraud_review_starter.py
python labs/module-03/l2_1-preprocessing/fraud_columns_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. No lab writes any file.

Check everything in this session at once (8 files; about 30 seconds):

```bash
python3 labs/check_labs.py module-03
```

## Timing in Session 3

| Clock | Lab |
|-------|-----|
| 0:55 | `fraud_review_starter.py` (20 min); fast finishers take the Stretch line, then `debug_metrics.py` |
| 2:15 | `fraud_columns_starter.py` (20 min); fast finishers take `debug_preprocess.py` |

`store_forecast_starter.py` and `demand_categories_starter.py` are optional homework.
