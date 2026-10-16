# Session 1 labs: Interpreting Models and SHAP

Lessons **7.1 Interpreting Models** and **7.2 SHAP Values**, plus the Kit B orientation. Every lab is new
practice on the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files, `check_environment.py` and the `debug_*.py` files.
Keep every `*_solution.py` for yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l7_0-orientation/check_environment.py` | A tool, not an exercise: Python 3.12+, an active environment, every package (shap, lime, PyTorch, torchvision ...), the seven data files, and the image data (Fashion-MNIST, EuroSAT) and two pretrained networks. `--download` fetches the images and weights once (about 175 MB) | 10 min (opening) | `READY for Advanced Machine Learning`; `READY for Sessions 1 to 4` when only the image data is missing (download before Session 5); or `NOT READY` with a numbered fix list. 15 to 60 s |
| `l7_0-orientation/diagnostic_starter.py` / `_solution.py` | Foundations refresher, one function each: the canonical split, a `Pipeline` named prep/model, 5-fold stratified CV, recall and precision, the $15 / $60 campaign profit, a threshold chosen by profit | homework (optional) | Six `PASS` lines; split 4500 / 1500 (C102112); CV ROC AUC 0.833 +/- 0.026; threshold 0.21 on out-of-fold probabilities; test: 385 offers, recall 0.641, precision 0.413, profit $3,765 |
| `l7_1-interpreting-models/fraud_drivers_starter.py` / `_solution.py` | Permutation importance under two metrics, partial dependence of the hour, an interaction (hour x new device), checking a curve against the data, on the fraud model | 15 min | AUC 0.863, AP 0.513; AP drops velocity 0.242, account age 0.120, new device 0.116; hour 0.074 by AP but -0.005 by AUC; PD night 0.0174 vs day 0.0072; new devices 0.122 at 2 a.m.; real night fraud 6.40% |
| `l7_1-interpreting-models/pipeline_importance_starter.py` / `_solution.py` | Odds ratios with `get_feature_names_out()` in the Foundations pipeline, two misleading coefficients, permutation importance per business column, grouped importance | fast finishers / homework | 40 model columns; Asheville 0.66 vs region Southeast 1.32; tier 1.48 although Gold churns 6.5%; boosting AUC 0.863; descriptive columns about 0, together -0.002 |
| `l7_1-interpreting-models/debug_interpret.py` | Two loud bugs (`coef_` on a pipeline, integer columns in a PDP) and two silent ones (log-odds printed as an odds ratio, importance on training rows) (deliberately broken) | fast finishers | Fixed: 1.36; [0.077, 0.214, 0.443]; days_since_last_order (0.140) |
| `l7_2-shap-values/shapley_by_hand_starter.py` / `_solution.py` | Shapley values by trying all 24 orders for a four-feature fraud rule; shap's Exact explainer; a null feature; how the reference changes the shares | 8 min (pairs) | velocity 0.2, new_device 0.15, night 0.15, gift_card 0.0 (sum 0.5); against a night payment 0.2 / 0.3 / 0.0 / 0.0 |
| `l7_2-shap-values/fraud_reasons_starter.py` / `_solution.py` | TreeExplainer on the fraud model: additivity, base value in two units, mean \|SHAP\|, a review queue with three reasons per payment, the night x new-device interaction in SHAP values | 12 min | base -5.840 = 0.0029; device_new 0.247, velocity 0.235; 25 of 25 queued payments are frauds; hour +1.24 log-odds at night on a new device |
| `l7_2-shap-values/debug_shap.py` | TreeExplainer on a pipeline, a forest's class axis, log-odds added to a probability (deliberately broken) | fast finishers | Fixed: rebuilt probability 0.972; forest's top reason days_since_last_order |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" line. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder (the one with `data/` in it):

```bash
python labs/module-01/l7_0-orientation/check_environment.py
python labs/module-01/l7_0-orientation/check_environment.py --download     # once, at home: images + weights
python labs/module-01/l7_1-interpreting-models/fraud_drivers_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. `check_environment.py` does
not use `labs_paths.py`; run it from the project folder, or by its path as above.

Check everything in this session at once (10 files; about a minute):

```bash
python3 labs/check_labs.py module-01
```

## Timing in Session 1

| Clock | Lab |
|-------|-----|
| 0:12 | `check_environment.py` on every machine (10 min, inside the 30-minute opening) |
| 1:05 | `fraud_drivers_starter.py` (15 min); fast finishers take `pipeline_importance_starter.py`, then `debug_interpret.py` |
| 2:15 | `shapley_by_hand_starter.py` in pairs (8 min), then `fraud_reasons_starter.py` (12 min); fast finishers take `debug_shap.py` |

`diagnostic_starter.py` is optional homework for anyone who missed two or more of the six refresher
questions on the workbook's Start Here page.
