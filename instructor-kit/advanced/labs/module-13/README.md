# Session 13 labs: Capstone Studio II and Showcase

Lesson **13.2 Capstone Project**, Milestones 6 to 9. Capstone tools, not exercises: the solutions work on
the fraud, demand and review-triage problems and hold reference numbers (full notes in `final-assessment.html`).
`capstone_start.py` is the lesson's starter template as a module (the same file as in `module-12`).

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l13_2-capstone/bootstrap_ci_starter.py` / `_solution.py` | Milestone 6: a paired bootstrap of average precision (two final candidates on June), recall at the team's 1% capacity with an interval, and row vs WEEK resampling on demand | 12 min | AP logistic regression 0.539 (0.408 to 0.658), boosting 0.556 (0.423 to 0.668), difference +0.017 (-0.046 to +0.077); top 1%: 35 of 64 (recall 0.55, 0.43 to 0.68); demand MAE 8.19 vs 13.42, saved 5.23, rows 4.90 to 5.53, weeks 4.68 to 5.81 |
| `l13_2-capstone/model_card_starter.py` / `_solution.py` | Milestones 8 and 9: choose on CV, test once with an interval, `results.json`, a skops round trip with a reviewed trusted list, a card written FROM the results, a number check | 10 min / home | balanced model chosen (CV negative recall 0.849); test accuracy 0.873 (0.85 to 0.895) vs 0.745, negative recall 0.873; no types to review; reload identical; a hand-typed 0.877 is caught |
| `l13_2-capstone/debug_capstone.py` | Review a classmate's capstone that breaks four rules (random split on weekly data, the actual temperature, the learning rate chosen on the test set, a weak baseline) | if short / home | As shipped: "beats the baseline by 55%". Fixed: learning rate 0.1 chosen by CV on 2024, 2025 MAE 8.29 vs 13.42 for the template baseline (38%) |

Everything a lab saves goes to `output/` next to it. Check: `python3 labs/check_labs.py module-13` (4 files, about 25 seconds).
