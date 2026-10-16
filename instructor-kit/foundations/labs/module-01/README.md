# Session 1 labs: Your ML Toolkit and the Estimator API

Lessons **0.1 Your ML Toolkit** and **1.1 The scikit-learn Estimator API**. Every lab is new practice on
the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files, `check_environment.py` and `debug_estimator.py`.
Keep every `*_solution.py` for yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l0_1-orientation/check_environment.py` | A tool, not an exercise: Python 3.12+, an active environment, every course package (with versions), the seven data files and their row counts | 12 min (opening) | Last line `READY for Python Machine Learning`, or `NOT READY` with a numbered fix list. Takes 15 to 60 s (PyTorch and MLflow import slowly) |
| `l0_1-orientation/diagnostic_starter.py` / `_solution.py` | Six prerequisite skills, one function each: slicing, dictionaries, default arguments, boolean filtering, `groupby`, missing values | homework (optional) | Six `PASS` lines, `6 of 6 checks pass.`, app users' mean spend $663.21, churn by tier Basic 0.217 / Silver 0.129 / Gold 0.065, 8.1% with no `age_group` |
| `l0_1-orientation/missing_values_starter.py` / `_solution.py` | Asking WHY a value is missing: `isna()` shares by group, churn by "is missing", a crosstab that exposes the `cancellation_reason` leak | 15 min | `age_group` missing 12.7% online vs 4.4% in stores; churn 26.1% when `email_opens_90d` is missing vs 15.6%; 625 of 625 cancellation reasons belong to churners; 19,970 of 32,178 interactions (62.1%) unrated |
| `l0_1-orientation/frame_targets_starter.py` / `_solution.py` | Framing three new requests: table, target, kind of problem, and what "knowing nothing" scores | fast finishers / homework | `recommended`: 2 values, 65.5% are 1; Boise: 1,096 days, 1,091 distinct sales values, mean $7,664.57, 6 closed days; fraud 1.00% (January) to 0.77% (June), so "always legitimate" scores 0.9900 then 0.9923 |
| `l1_1-estimator-api/fraud_baseline_starter.py` / `_solution.py` | The whole estimator API on card fraud: `X`/`y`, stratified 70/30 split, `DummyClassifier`, `StandardScaler`, `LogisticRegression`, `predict` vs `predict_proba`, ranking, `coef_` | 15 min | 35,000 / 15,000 rows (307 / 131 frauds); baseline 0.9913; logistic regression 0.9937, flags 40 (38 fraud, 93 missed); 53 frauds in the 100 riskiest vs 0.9 at random; `hour` weight -0.31; stretch: forest 0.9935, 58 in its top 100 |
| `l1_1-estimator-api/new_customers_starter.py` / `_solution.py` | Scoring rows that are in no file: learned attributes (`mean_`, `scale_`), `get_params` / `set_params`, `transform` on new data, column order | 10 min (pairs) | `True True False`; C 1.0, accuracy 0.848; Avery 0.808, Casey 0.550, Blake 0.010; `ValueError: The feature names should match those that were passed during fit.`; C=0.01: accuracy 0.853, Avery 0.719 / Blake 0.018 / Casey 0.481 |
| `l1_1-estimator-api/debug_estimator.py` | Three bugs that stop the script (`KeyError: 'churn'`, a Series as `X`, `NotFittedError`) and one that does not (scaler fitted on the test set) (deliberately broken) | fast finishers | Fixed: `one feature (days_since_last_order): accuracy 0.841`, `ten features: accuracy 0.848` (0.847 while the silent bug remains) |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" line. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder (the one with `data/` in it):

```bash
python labs/module-01/l0_1-orientation/check_environment.py
python labs/module-01/l0_1-orientation/missing_values_starter.py
python labs/module-01/l1_1-estimator-api/fraud_baseline_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells, so learners can run one step
at a time. The labs find the data through `labs_paths.py` (see `labs/README.md`): keep the `labs/` folder
inside the project folder, next to `data/`, or set `PINECREST_ML_DATA`. `check_environment.py` does not
use `labs_paths.py`; run it from the project folder, or by its path as above.

Check everything in this session at once (10 files; about 20 seconds):

```bash
python3 labs/check_labs.py module-01
```

## Timing in Session 1

| Clock | Lab |
|-------|-----|
| 0:10 | `check_environment.py` on every machine (12 min, inside the 30-minute opening) |
| 1:00 | `missing_values_starter.py` (15 min); fast finishers take `frame_targets_starter.py` |
| 2:15 | `fraud_baseline_starter.py` (12 min), then `new_customers_starter.py` in pairs (8 min); fast finishers take `debug_estimator.py` |

`diagnostic_starter.py` is optional homework for anyone who missed two or more of the six diagnostic
questions on the workbook's Start Here page. `frame_targets_starter.py` is homework for anyone who did
not reach it in class.
