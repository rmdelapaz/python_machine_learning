# Session 9 labs: Random Forests and Gradient Boosting

Lessons **5.2 Bagging and Random Forests** and **5.3 Gradient Boosting**. Every lab is new practice on
the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise. The two
fraud labs continue a thread that runs through the kit: logistic regression put 53 frauds in its 100
riskiest payments (Session 1), the eight-rule tree scored AP 0.253 (Session 8), and now a forest and
three boosting libraries try to do better.

Hand out the `*_starter.py` files and the two `debug_*.py` files. Keep every `*_solution.py` for
yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l5_2-random-forests/fraud_forest_starter.py` / `_solution.py` | A forest for the fraud review queue: leaf size tuned out-of-bag (`oob_decision_function_`), one test against logistic regression and a single tree, distinct scores, MDI against permutation importance (average precision) | 13 min | OOB AP 0.489 / 0.511 / 0.513 / 0.503 for leaf 1 / 3 / 5 / 10; test: logistic regression AUC 0.848, AP 0.444, 53 in the top 100; one tree 0.684 / 0.129 / 41; forest 0.842 / 0.479 / 57; distinct scores 2 vs 10,197; `device_new` MDI 0.081 but permutation 0.122, `amount` 0.176 vs 0.058. About 20 s |
| `l5_2-random-forests/decorrelate_starter.py` / `_solution.py` | The variance formula measured: per-tree predictions, single-tree AUC and tree correlation as `max_features` goes 1 → 10, the first-k-trees plateau, the panel's disagreement per customer | 12 min (pairs) | Forest AUC 0.865 = the average of its trees; single-tree AUC 0.709 → 0.747 and correlation 0.357 → 0.535; forest test AUC 0.854 / 0.864 / 0.865 / 0.861 / 0.856 for 1 / 2 / 3 / 5 / 10; first k trees 0.755 (1) … 0.865 (200); top 50 holds 41 churners, spread 0.284 vs 0.178 |
| `l5_2-random-forests/debug_forest.py` | Two bugs that stop it (`base_estimator=`, OOB with `bootstrap=False`) and two that do not (training AUC printed as "OOB", no `random_state`) (deliberately broken) | fast finishers | Fixed, every run: `bagged trees test ROC AUC 0.849`, `forest OOB ROC AUC 0.866 test ROC AUC 0.865` (the silent bug prints "OOB" 0.979) |
| `l5_3-gradient-boosting/boosting_by_hand_starter.py` / `_solution.py` | Classification boosting by hand on churn: start at the log-odds of the churn rate, fit depth-3 regression trees to y − p, step 0.3, choose the number of trees on a validation set, test once, compare with `HistGradientBoostingClassifier` | 13 min | Start −1.619; validation loss 0.4375 (1 tree) → 0.2898 (300), best 275 trees (0.2892); hand-made test AUC 0.870, log loss 0.3082; HGB 165 trees, 0.866 / 0.3091; stretch: rate 1.0 best at 72 trees, 0.3072 after 300 |
| `l5_3-gradient-boosting/fraud_boosters_starter.py` / `_solution.py` | HistGradientBoosting (default and gentle), XGBoost and LightGBM on fraud with a native category column, early stopping on a validation set (`eval_metric="aucpr"` / `"average_precision"`), and the peeking mistake | 12 min (pairs) | HGB defaults AUC 0.841, AP 0.415, 55 in the top 100 (46 trees: early stopping is on above 10,000 rows); gentle 0.857 / 0.475 / 57; XGBoost 0.862 / 0.503 / 61 (best iteration 268); LightGBM 0.861 / 0.492 / 59 (99); peeking 0.867 / 0.510 / 60. XGBoost and LightGBM may vary in the last digit |
| `l5_3-gradient-boosting/debug_boosting.py` | Two bugs that stop it (text columns without the category dtype, `early_stopping_rounds` in `fit()`) and one that does not (early stopping on the test set) (deliberately broken) | fast finishers | Fixed: `HistGradientBoosting test ROC AUC 0.862`, `XGBoost test ROC AUC 0.864 best iteration 218` (peeking: 0.866, 167) |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder:

```bash
python labs/module-09/l5_2-random-forests/fraud_forest_starter.py
python labs/module-09/l5_3-gradient-boosting/boosting_by_hand_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. Check everything in this
session at once (8 files; about 50 seconds):

```bash
python3 labs/check_labs.py module-09
```

## Timing in Session 9

| Clock | Lab |
|-------|-----|
| 0:50 | `fraud_forest_starter.py` (13 min, everyone), then `decorrelate_starter.py` in pairs (12 min); fast finishers take `debug_forest.py` |
| 2:15 | `boosting_by_hand_starter.py` (13 min, everyone), then `fraud_boosters_starter.py` in pairs (12 min); fast finishers take `debug_boosting.py` |

Whatever is not finished in class is homework.
