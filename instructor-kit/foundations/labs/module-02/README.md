# Session 2 labs: Splits, Leakage and Cross-Validation

Lessons **1.2 Splits, Baselines and Leakage** and **1.3 Cross-Validation**. Every lab is new practice on
the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and the two `debug_*.py` files.
Keep every `*_solution.py` for yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l1_2-splits-and-leakage/demand_forecast_starter.py` / `_solution.py` | A time split (2024 train, 2025 test) of `weekly_demand.csv`, a random split of the same size for contrast, the same forest graded both ways, and three free forecast baselines | 20 min | 5,088 / 4,896 rows (53 and 51 weeks); forest MAE 10.45 (random split) vs 10.27 (time split): no flattery on this data; baselines: each product's 2024 average 12.95, same week last year 13.81, training mean 17.13 |
| `l1_2-splits-and-leakage/fraud_validation_starter.py` / `_solution.py` | Train / validation / test (60/20/20, stratified twice) for a fraud review queue: choose among four models by frauds in the 100 riskiest validation payments, then open the test set once | fast finishers / homework | 263 / 87 / 88 frauds; validation top 100: logistic regression 45, tree depth 3 30, depth 8 34, forest 47 (chosen); test top 100: 41 frauds, 47x a random 100 |
| `l1_2-splits-and-leakage/debug_leaky_split.py` | Two bugs that stop it (stratify on a number, the split unpacked in the wrong order) and two that do not (`test_size=25`, the `has_reason` leak) (deliberately broken) | fast finishers | Fixed: `test customers: 1500 (churn rate 0.165)`, `baseline accuracy 0.835`, `logistic regression accuracy 0.848` |
| `l1_3-cross-validation/fraud_folds_starter.py` / `_solution.py` | Cross-validation for a model used on next month's payments: what unshuffled vs stratified folds hold, `cross_validate` with training scores, paired folds, next-month folds built by hand, June opened once | 20 min | frauds per fold `[83, 84, 71, 75, 61]` vs `[75, 75, 75, 74, 75]`; AUC baseline 0.500, logistic regression 0.854 +/- 0.023, tree 0.776 (ahead in 0 of 5 folds); next-month mean 0.862; June 0.874 |
| `l1_3-cross-validation/regularization_check_starter.py` / `_solution.py` | Was Session 1's C=0.01 really better? Pre-scaled vs pipeline, 15 paired folds, a C sweep with training scores | fast finishers / homework | pre-scaled 0.8493 vs pipeline 0.8491; C=0.01 behind C=1.0 on accuracy in 11 of 15 folds (mean -0.0026), tied on AUC (+0.0006); C=0.001 the only clear loser (0.825) |
| `l1_3-cross-validation/debug_cross_validation.py` | Two loud bugs (a seed without a shuffle, a misspelled scorer) and three silent ones (pre-scaling, the test rows inside CV, reporting the best fold) (deliberately broken) | fast finishers | Fixed with `StratifiedKFold(5, shuffle=True, random_state=42)`: `cross-validated on 4500 training customers, 5 folds`, `ROC AUC 0.829 +/- 0.025` |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" description. "Done" for each lab is the output in the last column. No lab writes files.

## How to run

With the course environment active, from the `pinecrest-ml` project folder (the one with `data/` in it):

```bash
python labs/module-02/l1_2-splits-and-leakage/demand_forecast_starter.py
python labs/module-02/l1_3-cross-validation/fraud_folds_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. The labs find the data
through `labs_paths.py` (see `labs/README.md`).

Check everything in this session at once (8 files; about 30 seconds; each solution runs in under 10 seconds):

```bash
python3 labs/check_labs.py module-02
```

## Timing in Session 2

| Clock | Lab |
|-------|-----|
| 0:50 | `demand_forecast_starter.py` (20 min); fast finishers take `fraud_validation_starter.py` or `debug_leaky_split.py` |
| 2:15 | `fraud_folds_starter.py` (20 min); fast finishers take `regularization_check_starter.py` or `debug_cross_validation.py` |

The second lab of each lesson is optional homework for anyone who did not reach it in class.
