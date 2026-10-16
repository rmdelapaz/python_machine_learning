# Session 4 labs: Pipelines and Creating Features

Lessons **2.2 Pipelines** and **2.3 Creating Features**. Every lab is new practice on the lesson's skills with
the Pinecrest data (fraud and daily store sales, where the lessons used churn and weekly demand); none of
them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and `debug_pipeline.py`. Keep every `*_solution.py` for yourself until
after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l2_2-pipelines/fraud_pipeline_starter.py` / `_solution.py` | A `Pipeline` with a `ColumnTransformer` (`prep`) and a model (`model`) on card fraud: CV against a baseline pipeline (ROC AUC and average precision), `named_transformers_`, feature names, `model__C`, one test look, `joblib` save/reload to `OUTPUT`, raw payments with an unknown category | 12 min | Baseline CV 0.500 / AP 0.009; logistic regression 0.867 / 0.535; 10 categories, largest weight `onehot__category_Gift Cards` 0.95; C=0.01 0.518; test ROC AUC 0.843, AP 0.463, 56 frauds in the top 100; probabilities `[0.999, 0.0, 0.002]` |
| `l2_2-pipelines/leak_size_starter.py` / `_solution.py` | How big is a leak? A target rate per dollar amount, a scaler and `np.log1p`, each fitted before CV and inside a pipeline | 8 min (pairs) | 1,016 amounts (434 seen 5 times or fewer); target rate 0.913 before CV, 0.583 inside, 0.568 test; scaler 0.8613 both ways (-0.00001); log1p 0.8617 both ways (+0.00000) |
| `l2_2-pipelines/debug_pipeline.py` | Three loud pipeline bugs (`alpha` without a step name, a step named `one__hot`, a missing column at prediction) and a silent one (the "test" rows are the training year) (deliberately broken) | fast finishers | Fixed: `5088 training weeks, 4896 test weeks`, `2025 test MAE: 11.75 units` (11.02 while the silent bug remains) |
| `l2_3-creating-features/night_features_starter.py` / `_solution.py` | Looking at training rows (hour bands, night × new device), a pure feature function in a pipeline via `FunctionTransformer`, CV lift, weights by name, one test look | 12 min | Night 6.71% fraud vs about 0.65%; night + new device 0.425; CV ROC AUC 0.867 → 0.864, CV AP 0.535 → 0.564; `hour` weight -0.28 → 0.01; test AP 0.463 → 0.504, top 100: 56 → 61 |
| `l2_3-creating-features/daily_sales_features_starter.py` / `_solution.py` | Calendar parts, per-store lags and a shifted rolling mean on daily sales; history features as forecasts; three Ridge pipelines; a leaky window | 8 min (pairs; TODO 5 can go home) | Mon..Sun mean 4639 … 8209 (Sat) …; forecasts lag_1 1,350, lag_7 1,072, rolling_7 1,451; Ridge 996 → 708 → 586; leaky window 551 |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block. Only `fraud_pipeline_solution.py` writes a file: `fraud_pipeline.joblib` in `OUTPUT`
(`output/` next to the lab; a temporary folder during the kit check).

## How to run

With the course environment active, from the `pinecrest-ml` project folder:

```bash
python labs/module-04/l2_2-pipelines/fraud_pipeline_starter.py
python labs/module-04/l2_3-creating-features/night_features_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. Check everything in this
session at once (8 files; about 35 seconds):

```bash
python3 labs/check_labs.py module-04
```

## Timing in Session 4

| Clock | Lab |
|-------|-----|
| 0:50 | `fraud_pipeline_starter.py` (12 min), then `leak_size_starter.py` in pairs (8 min); fast finishers take `debug_pipeline.py` |
| 2:15 | `night_features_starter.py` (12 min), then `daily_sales_features_starter.py` in pairs (8 min) |
