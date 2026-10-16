# Session 5 labs: Feature Selection and Imbalanced Classes

Lessons **2.4 Feature Selection** and **2.5 Imbalanced Classes**. Every lab is new practice on the lesson's
skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and `debug_imbalance.py`. Keep every `*_solution.py` for yourself until
after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l2_4-feature-selection/lean_fraud_starter.py` / `_solution.py` | Redundancy check, the F-test, forward selection judged by cross-validated average precision, the tolerance rule, one test, on 9 raw + 5 engineered fraud columns | 12 min | `amount`/`log_amount` 0.88; F-test top `night_new_device`; CV AP 0.5512 ± 0.0618 (14 columns), 0.5202 (raw 9); 8 columns within 0.01; test AP raw 0.444, all 14 0.505, lean 8 0.505 |
| `l2_4-feature-selection/vendor_columns_starter.py` / `_solution.py` | Selection leakage with real and useless columns mixed: 1,000 simulated vendor columns on a 300-customer pilot | 8 min (pairs) | CV AUC 0.745 (real 10), 0.603 (all 1,010); select first 0.825 (3 real of 10 picked); select inside CV 0.660; test AUC 0.794 vs 0.609 |
| `l2_5-imbalanced-classes/retention_threshold_starter.py` / `_solution.py` | A cost function, threshold sweep, the formula, balanced weights and their inflated probabilities, `TunedThresholdClassifierCV`, on churn with a $15 offer and $60 per lost churner | 12 min | nobody $14,880, everyone $22,500; AUC 0.827, AP 0.496; 0.5: 112 offers $12,540; formula 0.25 ($10,935); balanced $10,965 at 0.5, $14,070 at the formula; tuned 0.234: 343 offers, $10,845 |
| `l2_5-imbalanced-classes/rebalance_tree_starter.py` / `_solution.py` | Class weights and three samplers for a depth-6 decision tree inside imblearn pipelines; the resample-before-CV leak | 8 min (pairs) | CV AP plain 0.335, balanced 0.269, under 0.044, over 0.271, SMOTE 0.182; leak 0.873 |
| `l2_5-imbalanced-classes/debug_imbalance.py` | Two loud bugs (scikit-learn pipeline with a sampler; SMOTE before the encoder) and two silent ones (default scoring; `ravel()` order) (deliberately broken) | fast finishers | Fixed: `plain  CV AP 0.535`, `SMOTE  CV AP 0.495`, `611 flagged, cost $25,985` |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder:

```bash
python labs/module-05/l2_4-feature-selection/lean_fraud_starter.py
python labs/module-05/l2_5-imbalanced-classes/retention_threshold_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. `lean_fraud` Step 4 fits about
500 small models (about 15 s).

Check everything in this session at once (8 files; under a minute):

```bash
python3 labs/check_labs.py module-05
```

## Timing in Session 5

| Clock | Lab |
|-------|-----|
| 0:50 | `lean_fraud_starter.py` (12 min, everyone), then `vendor_columns_starter.py` in pairs (8 min) |
| 2:15 | `retention_threshold_starter.py` (12 min, everyone), then `rebalance_tree_starter.py` in pairs (8 min); fast finishers take `debug_imbalance.py` |
