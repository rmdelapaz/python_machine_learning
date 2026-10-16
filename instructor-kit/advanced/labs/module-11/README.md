# Session 11 labs: Serving, Monitoring and the Churn Case Study

Lessons **12.2 Serving and Monitoring Models** and **13.1 Case Study: Predicting Churn**. Every lab is new
practice on the lesson's skills with the Pinecrest data: the lesson served and monitored the CHURN model;
these labs serve and monitor a FRAUD model, and run a leak audit and cost scenarios around the case study.

Hand out the `*_starter.py` files and `debug_serving.py`. Keep every `*_solution.py` until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l12_2-serving/fraud_api_starter.py` / `_solution.py` | Packaging with skops (reviewed trusted types), serving metadata, a `Payment` pydantic schema, FastAPI `/score` and `/score_batch`, TestClient, the match test | 20 min | threshold 0.0775 (reference hold rate 0.0101); 4 problems in the bad payment, 422; first June payment 0.0021; `API matches the notebook: True`; June 83 of 8,262 held (1.00%), 35 of 64 frauds |
| `l12_2-serving/fraud_monitor_starter.py` / `_solution.py` | Monitoring on REAL months: numeric and category PSI, mean score and hold rate against a reference, the label check, a broken feed | fast finishers / homework | every PSI ≤ 0.003; hold rates 0.0089 / 0.0089 / 0.0080 vs 0.0100; average precision 0.519 / 0.521 / 0.531; broken `velocity_24h`: PSI 2.523, hold rate 0.0025, 10 of 35 frauds held (19 with the feed intact) |
| `l12_2-serving/debug_serving.py` | Four bugs between JSON and the model: a wrong field type, a threshold stored as text, scores rounded to 2 decimals, holds from `predict` (deliberately broken) | fast finishers | Fixed: `API matches the notebook: True`, `June: 83 held, 35 of them fraud` |
| `l13_1-case-study/leak_audit_starter.py` / `_solution.py` | A reusable leak screen for every column: single-column CV AUC, "filled for one class", ID-like text; on churn and on fraud | 10 min | churn: `cancellation_reason` FILLED FOR ONE CLASS (AUC 0.815; 63% of churners, 0% of stayers), `customer_id` ID-LIKE; best honest column `days_since_last_order` 0.715; fraud: only `transaction_id` flagged |
| `l13_1-case-study/cost_scenarios_starter.py` / `_solution.py` | Thresholds from costs on out-of-fold probabilities, the profit of each block of 100 offers, one look at the test set | 10 min (pairs) | best thresholds 0.17 / 0.24 / 0.31 / 0.43 for $10 / $15 / $20 / $25; first losing block ends at offer 1,200 (p 0.213); $25 offer on test: 0.25 → 360 offers, $1,620; 0.43 → 183 offers, $2,745 |

Starters run as handed out: they print their `TODO` reminders and an "Expected when done" line.

## How to run

With the course environment active, from the `pinecrest-ml` folder:

```bash
python labs/module-11/l12_2-serving/fraud_api_starter.py
python labs/module-11/l13_1-case-study/leak_audit_starter.py
```

The serving labs build their app inside the file and test it with TestClient: no server is started, no port
is opened. Files they save (the skops model, the serving JSON) go to `output/` next to the lab. Some machines
print a `StarletteDeprecationWarning` when FastAPI's test client is imported: harmless, the labs filter it.

Check the whole session (8 files, about 40 seconds): `python3 labs/check_labs.py module-11`.

## Timing in Session 11

| Clock | Lab |
|-------|-----|
| 0:50 | `fraud_api_starter.py` (20 min); fast finishers `fraud_monitor_starter.py` or `debug_serving.py` |
| 2:10 | `leak_audit_starter.py` (10 min), then `cost_scenarios_starter.py` in pairs (10 min) |
