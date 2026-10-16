# Session 8 labs: Support Vector Machines and Decision Trees

Lessons **4.3 Support Vector Machines** and **5.1 Decision Trees**. Every lab is new practice on the
lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and the two `debug_*.py` files. Keep every `*_solution.py` for
yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l4_3-svm/review_svm_starter.py` / `_solution.py` | A linear SVM on review text (Lesson 4.2's labels and split): `decision_function` for ranking, "scaling" word counts with TF-IDF, `linearsvc__C` chosen by cross-validation, reading the weights, the street (\|score\| < 1), `CalibratedClassifierCV` | 13 min | Raw counts accuracy 0.859, AUC 0.921 (Naive Bayes 0.868 / 0.934); TF-IDF 0.884 / 0.936, 1,856 features; CV AUC 0.9077 / 0.9253 / 0.9269 / 0.9066 for C = 0.01 / 0.1 / 1 / 10; `not` -2.51, `would buy` 1.93; 445 of 774 inside the street, 90 on the wrong side; calibrated mean 0.734 vs 0.745; stretch RBF 0.876 / 0.928 |
| `l4_3-svm/kernel_by_hand_starter.py` / `_solution.py` | The kernel trick on churn: `PolynomialFeatures(2)` + `LinearSVC` (an explicit lift) against `SVC(kernel="poly", degree=2)` (the same space, never built), column counts by degree, the RBF kernel | 12 min (pairs) | LinearSVC AUC 0.826 (33 in the top 50); 66 lifted columns; lift + LinearSVC 0.852; poly kernel 0.852 with 2,293 support vectors; 66 / 286 / 1,001 / 3,003 columns for degrees 2 to 5; RBF 0.850; stretch: `tenure_months^2` weight +0.31 (the U-shape) |
| `l4_3-svm/debug_svm.py` | Two bugs that stop the script (grid key `C` without the step name, `predict_proba` on an SVM) and two that do not (no scaler; the top 50 taken from the wrong end) (deliberately broken) | fast finishers | Fixed: `best C: 1   CV ROC AUC 0.848`, `test ROC AUC 0.850`, 32 real churners among the 50 riskiest (with both silent bugs: 0.779 / 0.781 and 2) |
| `l5_1-decision-trees/fraud_rules_starter.py` / `_solution.py` | A fraud rule book for payment reviewers: `export_text` with weights, the root's Gini decrease by hand against `tree_.impurity`, an unlimited tree judged honestly, `max_leaf_nodes` chosen by CV average precision, a segment table with `apply`, a zero importance | 13 min | Leaf "velocity > 3.5 and account age <= 366 days": 40 frauds of 42; Gini 0.0174 / 0.0151 / 0.3457, decrease 0.0018; unlimited tree 433 leaves, train AUC 1.000, test 0.684, AP 0.129; CV AP 0.178 / 0.259 / 0.287 / 0.295 for 4 / 8 / 12 / 16 leaves; 8-rule book test AUC 0.756, AP 0.253, riskiest segment 15 of 19 fraud; `items` importance 0.000 although frauds average 2.39 items vs 1.89 |
| `l5_1-decision-trees/sales_tree_starter.py` / `_solution.py` | A regression tree for daily store sales: calendar features, a time split (2023–2024 train, 2025 test), `TimeSeriesSplit` to choose the depth, leaves as averages, why a tree cannot follow a trend | 12 min (pairs) | Baselines MAE $1,967 (mean) / $1,670 (per store); depth-3 root `day_of_week <= 4`, biggest leaf Boise weekends from May ($10,059); CV MAE $1,374 (depth 2) to $901 (depth 10), $913 unlimited; depth 10 test MAE $976, 298 leaves; Boise 2025 $8,342 actual vs $7,339 predicted |
| `l5_1-decision-trees/debug_tree.py` | Two bugs that stop it (`max_depth=0`, a text column) and two that do not (class names in the wrong order, training accuracy reported as accuracy on new customers) (deliberately broken) | fast finishers | Fixed: first leaf `class: stayed`, lapsed newcomers `class: churned`, `accuracy on new customers 0.851 (baseline 0.835)` (0.856 while the silent bug remains) |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block. "Done" for each lab is the output in the last column. Fit times printed by
`kernel_by_hand` are the only numbers that differ from machine to machine.

## How to run

With the course environment active, from the `pinecrest-ml` project folder (the one with `data/` in it):

```bash
python labs/module-08/l4_3-svm/review_svm_starter.py
python labs/module-08/l5_1-decision-trees/fraud_rules_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. Every lab finishes in
under 10 seconds. Check everything in this session at once (8 files; about 30 seconds):

```bash
python3 labs/check_labs.py module-08
```

## Timing in Session 8

| Clock | Lab |
|-------|-----|
| 0:50 | `review_svm_starter.py` (13 min, everyone), then `kernel_by_hand_starter.py` in pairs (12 min); fast finishers take `debug_svm.py` |
| 2:15 | `fraud_rules_starter.py` (13 min, everyone), then `sales_tree_starter.py` in pairs (12 min); fast finishers take `debug_tree.py` |

Whatever is not finished in class is homework: each starter's "Expected when done" block lets learners
check themselves.
