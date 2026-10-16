# Session 11 labs: Hyperparameter Tuning and Final Project Studio

Lesson **6.2 Hyperparameter Tuning**, then the final-project studio. The two tuning labs are new practice
on the lesson's skills (the lesson tunes churn; these tune fraud and make the winner's curse visible);
the scaffold pair is milestone 1 of the final project (see `final-assessment.html` in the kit root).

Hand out the `*_starter.py` files. Keep every `*_solution.py` for yourself.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l6_2-hyperparameter-tuning/tune_fraud_starter.py` / `_solution.py` | A written search space (log scales), `RandomizedSearchCV` on the training months, reading `cv_results_`, the test month opened once, saving the settings as JSON | 20 min | Default 3-fold AP 0.443 +/- 0.032; 24 fits; best CV AP 0.515 (learning_rate 0.0186, max_leaf_nodes 24, min_samples_leaf 170); 5 of 8 settings within one std; June AP 0.477 default vs 0.544 tuned, 32 vs 38 frauds in the top 100; `fraud_best_params.json` written to `output/`. About 40 s |
| `l6_2-hyperparameter-tuning/winners_curse_starter.py` / `_solution.py` | `best_score_` optimism made visible on shuffled labels; nested cross-validation; the real labels for comparison | fast finishers / homework | Noise: best_score_ 0.504 / 0.514 / 0.537 after 1 / 10 / 40 settings; nested 0.501, test 0.524. Real labels: best_score_ 0.848, nested 0.840, test 0.859. About 25 s |
| `l6_2-hyperparameter-tuning/final_project_starter.py` / `_solution.py` | Final-project milestone 1: the canonical leak-safe split for the chosen problem, the bar to beat, a first cross-validated model (the solution runs all three problems) | studio, first 15 min | churn 4,500 / 1,500, random-order AP 0.165, contact everyone -7,620 dollars, logistic CV AP 0.528 +/- 0.042; demand 5,088 / 4,896, MAE 17.13 / 13.42, boosting CV MAE 8.49 +/- 1.34; fraud 41,738 / 8,262 (374 / 64), $24,000 / $82,620, logistic CV AP 0.506 +/- 0.054. About 25 s |

## How to run

```bash
python labs/module-11/l6_2-hyperparameter-tuning/tune_fraud_starter.py
python labs/module-11/l6_2-hyperparameter-tuning/final_project_starter.py     # set PROBLEM first
python3 labs/check_labs.py module-11                                            # 6 files, about 2 minutes
```

## Timing in Session 11

| Clock | Lab |
|-------|-----|
| 0:50 | `tune_fraud_starter.py` (20 min); fast finishers take `winners_curse_starter.py` |
| 1:20 | `final_project_starter.py` opens the studio (milestone 1 check for anyone who has not done it) |
