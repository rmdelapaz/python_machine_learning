# Session 10 labs: Recommendations and Reproducibility

Lessons **11.2 Recommendation Systems** and **12.1 Reproducibility and Experiment Tracking**. Every lab is
new practice on the lesson's skills (new baselines and a new recommender for 11.2; the demand and fraud
models instead of the lesson's churn model for 12.1); none repeats a lesson project or exercise.

Hand out the `*_starter.py` files and the two `debug_*.py` files. Keep every `*_solution.py` for yourself
until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l11_2-recommendations/segment_baselines_starter.py` / `_solution.py` | Leave-last-out evaluation of two segment baselines (home-store and favorite-activity best sellers) against global best sellers and item-kNN, with coverage | 12 min | HR@10: global 0.209, home store 0.209 (36 products, 98% top-20 slots), favorite activity 0.265 (183, 31%), item-kNN 0.308 (161, 51%); stretch "general" fans 0.229 vs 0.280 |
| `l11_2-recommendations/user_based_starter.py` / `_solution.py` | User-based filtering with n neighbors chosen on a validation item, one test score, a last-purchase-only list | 8 min (pairs) | validation 0.202 / 0.282 / 0.292 / 0.266 (10 / 50 / 200 / 1,000 neighbors), item-kNN 0.286; test user-kNN (200) 0.321, item-kNN 0.308; last purchase only 0.212; stretch 0.305 / 0.298 |
| `l11_2-recommendations/debug_recommender.py` | A broadcasting error and the wrong similarity axis (loud); first purchase as the test item and an ascending sort (silent) (deliberately broken) | fast finishers | Fixed: random 0.047, best sellers 0.209, item-kNN 0.308 |
| `l12_1-reproducibility/track_demand_starter.py` / `_solution.py` | CONFIG and data fingerprint, one MLflow run per candidate (SQLite store and artifacts in `OUTPUT`), choosing by `search_runs` (ASC for an error), one test score, a filtered query, seed checks | 12 min | fingerprint `c48cd2582060519c`; validation MAE ridge 12.63, forest 11.31, hgb 10.74; test MAE 8.21; below 12: hgb, forest; 4 runs; hgb seed-independent (10.7445), forest 11.3146 / 11.2821 |
| `l12_1-reproducibility/safe_handover_starter.py` / `_solution.py` | skops save, a refused load, trust lists for three model kinds, a JSON model card, verifying a score from the card alone, fingerprints of a copy / re-save / one-cent edit | 8 min (pairs) | AP 0.4743; HGB `TreePredictor`, logistic regression `[]`, forest `Tree`; verify (True, True, 0.4743) ×3, then (False, True, 0.4743); skops 462 kB vs joblib 142 kB |
| `l12_1-reproducibility/debug_reproducibility.py` | `./mlruns` and a parameter logged twice (loud); an unseeded forest and an automatic trust list (silent) (deliberately broken) | fast finishers | Fixed, identical on every run: `reloaded forest, test AUC 0.8637` |

A starter runs without an error as handed out: it prints its `TODO` reminders and an "Expected when
done" line. The 12.1 labs print a few MLflow `INFO` lines on the error stream the first time they create
a database: that is normal.

## Where the 12.1 labs write

Everything goes to `OUTPUT`: an `output/` folder next to the lab file on a learner's machine (a temporary
folder under the checker). To browse a learner's runs:

```bash
mlflow ui --backend-store-uri sqlite:///labs/module-10/l12_1-reproducibility/output/mlflow.db
```

Rerunning `track_demand_*` adds runs to the same store (the "runs in the store" count grows); delete
`output/` to start fresh.

## How to run

```bash
python labs/module-10/l11_2-recommendations/segment_baselines_starter.py
python labs/module-10/l12_1-reproducibility/track_demand_starter.py
python3 labs/check_labs.py module-10        # all 8 files, about 60 seconds
```

## Timing in Session 10

| Clock | Lab |
|-------|-----|
| 0:50 | `segment_baselines_starter.py` (12 min), fast pairs take `user_based_starter.py` (8 min) |
| 2:15 | `track_demand_starter.py` (12 min), fast pairs take `safe_handover_starter.py` (8 min) |
