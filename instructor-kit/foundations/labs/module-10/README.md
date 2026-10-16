# Session 10 labs: Ensembles and Model Selection

Lessons **5.4 Voting and Stacking** and **6.1 Model Selection and Learning Curves**. Every lab is new
practice on the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.
The lessons work on churn; these labs move the same questions to card fraud (train January to May,
judge June) and to the weekly demand forecaster (train 2024, judge 2025), where time matters.

Hand out the `*_starter.py` files and the two `debug_*.py` files. Keep every `*_solution.py` for yourself
until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l5_4-voting-stacking/fraud_ensemble_starter.py` / `_solution.py` | Out-of-fold predictions, member diversity, choosing a soft vote on out-of-fold AP, `StackingClassifier`, opening the test month once, a paired bootstrap of the gain | 20 min (pairs) | OOF AP logreg 0.506, forest 0.513, boost 0.524; correlations 0.89 to 0.92; votes 0.541 / 0.531 / 0.539 (all three chosen); meta weights [5.04 4.45 5.5]; June AP 0.539 / 0.509 / 0.556 / vote 0.570 / stack 0.573, all 35 or 36 frauds in the top 100; stack minus boost +0.016, 95% interval [-0.016, +0.057]: keep one model. About 40 s |
| `l5_4-voting-stacking/demand_blend_starter.py` / `_solution.py` | Blending on a LATER period (time-aware), weights chosen on a holdout, refit, test once | fast finishers / homework | Oct-Dec 2024 holdout MAE boost 9.78, forest 10.92, correlation 0.95; best weight 1.0 (all booster); 2025 MAE boost 8.05, forest 8.47, blend 8.05, 50/50 average 8.11: keep the booster. About 10 s |
| `l5_4-voting-stacking/debug_stack.py` | Members without names (`ValueError`), `predict_proba` on a hard vote (`AttributeError`), `TimeSeriesSplit` as stacking cv (`ValueError: cross_val_predict only works for partitions`), and a silent in-sample meta-model (deliberately broken) | fast finishers | Fixed: `soft vote ROC AUC 0.858`, `stack weights: [2.58 3.44]   stack ROC AUC 0.860`; with the silent bug left in: weights [-2.44 13.59], AUC 0.853 |
| `l6_1-model-selection/fraud_bakeoff_starter.py` / `_solution.py` | One splitter for all candidates, a metric chosen in advance (AP), paired differences, test month opened once | 20 min | CV AP boost_gentle 0.521 +/- 0.057, logistic 0.504, boost_default 0.446, baseline 0.009 (AUC ranks logistic first, 0.855); gentle - default +0.075 (10 of 10), gentle - logistic +0.017 (std 0.023, 7 of 10); June AP 0.556 vs 0.539, both 36 frauds in the top 100. About 30 s |
| `l6_1-model-selection/demand_curves_starter.py` / `_solution.py` | Shuffled folds vs blocks of weeks for a forecaster; a learning curve on time-respecting folds | fast finishers / homework | Shuffled 7.60 +/- 0.19 vs blocks 8.49 +/- 1.34 (2025 test 8.05); learning curve 1,343 / 2,726 / 4,070 rows: validation MAE 10.66 / 9.96 / 8.41; more history should help. About 70 s |
| `l6_1-model-selection/debug_bakeoff.py` | `scoring="auc"` (`InvalidParameterError`), a pipeline parameter without its step name (`ValueError`), a different shuffle per candidate (silent), the winner picked on the test set (silent) (deliberately broken) | fast finishers | Fixed: logistic 0.832 +/- 0.017, tree_d5 0.842 +/- 0.018, +0.010 winning 8 of 10 folds, best depth 4, chosen tree_d5, test AUC 0.846 |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder:

```bash
python labs/module-10/l5_4-voting-stacking/fraud_ensemble_starter.py
python labs/module-10/l6_1-model-selection/fraud_bakeoff_starter.py
```

Check everything in this session at once (8 files, about 4 minutes on a busy machine):

```bash
python3 labs/check_labs.py module-10
```

## Timing in Session 10

| Clock | Lab |
|-------|-----|
| 0:45 | `fraud_ensemble_starter.py` (20 min, pairs); fast finishers take `debug_stack.py`, then `demand_blend_starter.py` |
| 2:00 | `fraud_bakeoff_starter.py` (20 min); fast finishers take `debug_bakeoff.py`, then `demand_curves_starter.py` |

The fraud labs fit several models on 33,000 to 42,000 rows; on an older laptop allow a minute per run
and start the run before discussing the previous step.
