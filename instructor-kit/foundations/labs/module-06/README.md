# Session 6 labs: Linear Models and Regularization (and Checkpoint 1)

Lessons **3.1 Linear and Logistic Regression** and **3.2 Regularization and the Bias-Variance Trade-off**,
plus **Checkpoint 1** on Modules 1 and 2. Every lab is new practice on the lesson's skills with the Pinecrest
data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files, including `checkpoint-1/checkpoint1_starter.py`. Keep every
`*_solution.py` (and the checkpoint model answer) for yourself.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `checkpoint-1/checkpoint1_starter.py` / `_solution.py` | Checkpoint 1: audit a leaky churn notebook, rebuild it in a pipeline, CV ROC AUC and AP against a dummy, a threshold from a $5 email / $50 churner cost, one test, three sentences | 30 min (0:10) | Colleague 0.947; CV AUC 0.832 vs 0.500, CV AP 0.528 vs 0.165; threshold 0.107; test AUC 0.826, 654 emails, 214 of 248 churners, $4,970 (everyone $7,500, 0.5 $9,630, nobody $12,400) |
| `l3_1-linear-logistic/daily_sales_line_starter.py` / `_solution.py` | Baselines, a slope by hand, coefficients in dollars against a reference day, a twin column (`promo` vs `event`), a log target in percent, on daily store sales | 12 min | baselines $1,670 / $1,292; Boise +$1,333 a year; MAE $949; promo twin (same predictions: True); log MAE $853, Saturday +74.4%, trend +8.1% a year |
| `l3_1-linear-logistic/fraud_odds_starter.py` / `_solution.py` | The sigmoid by hand, odds ratios in the data's own units, hour as a line against a night flag, `decision_function` | 8 min (pairs) | z = -5.984 + 1.515 × velocity; odds ratios device_new 7.11, per $100 1.31; hour 0.90 vs night 10.53; 21 flagged, 18 fraud; AP 0.332 |
| `l3_2-regularization/churn_curves_starter.py` / `_solution.py` | Polynomial logistic regression, train vs CV AUC, a validation curve over `C`, more data as a regularizer, one test | 12 min | pilot CV AUC 0.807 / 0.807 / 0.706 (degrees 1 to 3); best C 0.01; full data degree 2 0.851; test 0.853 vs 0.823 |
| `l3_2-regularization/variance_without_truth_starter.py` / `_solution.py` | Measuring variance on real data by refitting on 30 pilot samples; Ridge alpha against spread and error | 8 min (pairs) | spread $1,217 / $526 / $275 / $167 (least squares, alpha 1, 10, 100); mean MAE $1,351 / $1,089 / $1,122 / $1,527 |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block.

## How to run

```bash
python labs/module-06/checkpoint-1/checkpoint1_starter.py
python labs/module-06/l3_1-linear-logistic/daily_sales_line_starter.py
```

Check everything in this session at once (10 files; about a minute):

```bash
python3 labs/check_labs.py module-06
```

## Timing in Session 6

| Clock | Lab |
|-------|-----|
| 0:10 | Checkpoint 1 (30 min, alone, open notebook) |
| 1:10 | `daily_sales_line_starter.py` (12 min), then `fraud_odds_starter.py` in pairs (8 min) |
| 2:20 | `churn_curves_starter.py` (12 min), then `variance_without_truth_starter.py` in pairs (8 min) |
