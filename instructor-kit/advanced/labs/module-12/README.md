# Session 12 labs: Capstone Kickoff and Studio I

Lesson **13.2 Capstone Project**, Milestones 1 to 4. These are capstone tools, not exercises: each learner
sets `PROBLEM` to the letter they chose and does the part for that problem. The solutions run all five
problems, so they double as the instructor's reference numbers (the full notes are in `final-assessment.html`).

`capstone_start.py` is the lesson's starter template (the five `start_*` functions) as a module: unchanged
except that it reads from `labs_paths.DATA`. Both labs import it; it is a support file, not run by the checker.

| File | Practices | In class | Expected output (solution, all five problems) |
|------|-----------|---------:|----------------------------|
| `l13_2-capstone/baselines_starter.py` / `_solution.py` | Milestones 1 to 3: a second honest "no machine learning" baseline on the template's split and metric, and the "do not deploy" sentence | 12 min | demand per product × store 12.953, same week last year 13.810 (template 13.419); fraud velocity rule AP 0.250, 19 of 64 June frauds in its top 1%; reviews keyword rule accuracy 0.788, negative recall 0.477; recs random 0.053; segments five random groups -0.023 |
| `l13_2-capstone/harness_starter.py` / `_solution.py` | Milestones 3 and 4: ONE evaluation function on the training data (rolling folds, May validation, stratified folds with negative recall, a second hidden purchase, silhouette + seed stability) and a first pipeline | 23 + 30 min | demand ridge 13.128 vs baseline 13.924 (rolling folds); fraud LR AP 0.539 on May; reviews TF-IDF + LR 0.857 (negative recall 0.577); recs item-kNN 0.286, SVD k=5 0.285, popularity 0.209; segments k=5 silhouette 0.470, seed ARI 1.000 |

No lab here touches the test set. Starters run as handed out and print their TODOs and "Expected when done".

```bash
python labs/module-12/l13_2-capstone/baselines_starter.py
python3 labs/check_labs.py module-12        # 4 files, about 15 seconds
```
