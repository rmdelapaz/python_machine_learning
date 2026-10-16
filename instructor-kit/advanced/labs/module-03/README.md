# Session 3 labs: Choosing k and Hierarchical Clustering

Lessons **8.2 Choosing k and Customer Segments** and **8.3 Hierarchical Clustering**. Every lab is new
practice on the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` and `debug_*.py` files. Keep every `*_solution.py` until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l8_2-choosing-k/choose_k_tastes_starter.py` / `_solution.py` | Choosing k from four kinds of evidence (elbow, silhouette, bootstrap ARI, a profile) for customers described by WHAT they buy (activity shares from `interactions.csv`) | 20 min | Silhouette peaks at k = 5 (0.453); inertia drop 115 (4→5) vs 51 (5→6); worst bootstrap ARI 0.641 / 0.925 / 0.950 / 0.658 for k = 4..7; five tastes (hiking 1,009 ... running 406); at k = 6 a "general gear" group of 484 |
| `l8_2-choosing-k/segment_economics_starter.py` / `_solution.py` | Usefulness in money: a blanket offer per segment vs the churn model's offers, with the $15 / $60 economics | fast finishers / homework | Only Deal Hunters (churn 0.31) beat the 0.25 break-even (+$3.72 per customer); on the 1,500 test customers blanket offers lose $7,620, the model's 300 offers earn $4,920 |
| `l8_2-choosing-k/debug_choose_k.py` | A silhouette for k = 1, k chosen by the lowest inertia, a scaler refitted on newcomers (deliberately broken) | fast finishers | Fixed: `k chosen by silhouette: 5 (0.479)`, `newcomers: [0, 0, 0]` |
| `l8_3-hierarchical/seasonal_families_starter.py` / `_solution.py` | A tree of the 24 products in `weekly_demand.csv` by their monthly sales profile: average linkage, correlation distance, cuts by height and number, Euclidean for contrast, cophenetic correlation | 13 min | Highest merges [1.54, 0.292, ...]: a winter family (4 Apparel + 2 Snow) and the rest; 3 families add the two spring footwear products; Euclidean isolates the two best sellers; cophenetic 0.979 |
| `l8_3-hierarchical/sample_then_assign_starter.py` / `_solution.py` | Lesson 8.3's big-data recipe: Ward on a sample, everyone to the nearest mean, K-Means polished from those means | 12 min | 17,997,000 distances for all vs 124,750 for 500; ARI vs K-Means 0.971 (500) and 0.957 (1,500); polished K-Means inertia 5,525, ARI 1.000 |
| `l8_3-hierarchical/debug_hierarchical.py` | `affinity=`, Ward with a precomputed distance, `n_clusters` with `distance_threshold`, a single-linkage cross-check (deliberately broken) | fast finishers | Fixed: `family sizes: [31, 36, 36, 38, 59]`, `agreement with SciPy's tree (ARI): 1.000` |

## How to run

```bash
python labs/module-03/l8_2-choosing-k/choose_k_tastes_starter.py
python labs/module-03/l8_3-hierarchical/seasonal_families_starter.py
python3 labs/check_labs.py module-03          # all 8 files, under a minute
```

## Timing in Session 3

| Clock | Lab |
|-------|-----|
| 0:50 | `choose_k_tastes_starter.py` (20 min); fast finishers take `segment_economics_starter.py` |
| 2:10 | `seasonal_families_starter.py` (13 min), then `sample_then_assign_starter.py` (12 min) |
