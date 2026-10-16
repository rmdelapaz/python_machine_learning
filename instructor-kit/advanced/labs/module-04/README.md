# Session 4 labs: Density-Based Clustering and Gaussian Mixtures

Lessons **8.4 Density-Based Clustering** and **8.5 Gaussian Mixture Models**. Every lab is new practice on
the lesson's skills with the Pinecrest data; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and `debug_density.py`. Keep every `*_solution.py` for yourself until
after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l8_4-density-clustering/payment_density_starter.py` / `_solution.py` | DBSCAN k-distances and an eps sweep, HDBSCAN, counting clusters without `-1`, profiling, the whole-number artifact, stability across `min_cluster_size`, on 5,000 card payments (fraud used only to profile) | 25 min | No eps works (69.7% noise at 0.3, 5.6% at 1.0); HDBSCAN 8 clusters, 8.8% noise; clusters sit on exact items/velocity values; noise holds 24 of the 38 frauds (5.4% fraud vs 0.31% in clusters) |
| `l8_4-density-clustering/rings_starter.py` / `_solution.py` | K-Means vs DBSCAN on a ring inside a ring, eps from k-distances, ARI, a silhouette that prefers the wrong answer | fast finishers / homework | K-Means ARI -0.002; eps 0.3 and 0.5 ARI 1.000; silhouette 0.328 (wrong K-Means) vs 0.158 (right DBSCAN) |
| `l8_4-density-clustering/debug_density.py` | Two loud bugs (`KeyError`, DBSCAN has no `predict`) and three silent ones (no scaling, `-1` counted as a cluster, silhouette with the noise included) (deliberately broken) | fast finishers | Fixed: 4 clusters, 7.4% noise, noise churn 0.083, silhouette of the clustered customers 0.268 |
| `l8_5-gaussian-mixtures/discount_mixture_starter.py` / `_solution.py` | A one-feature mixture on a real column (`discount_share`): BIC, naming components by their means, where the vote switches, soft membership against churn, borderline customers | 25 min | BIC drop 3,596 then 330; means 0.11 / 0.44; P(deal-driven) 0.471 at 0.25, 0.911 at 0.30; churn 0.119 vs 0.305; 430 borderline |
| `l8_5-gaussian-mixtures/lifecycle_mixture_starter.py` / `_solution.py` | Lifecycle stages (tenure, recency, spend): BIC without a minimum, profiles and names, label switching between seeds (ARI), confidence, scoring a new customer through the same scaler | pairs / fast finishers | Four stages: regulars 0.109, established 0.071, fading 0.205, newcomers 0.317 churn; ARI 0.981 between seeds; 1,340 below 0.8 confidence; newcomer 0.977 |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" line. "Done" for each lab is the output in the last column.

## How to run

With the course environment active, from the `pinecrest-ml` project folder (the one with `data/` in it):

```bash
python labs/module-04/l8_4-density-clustering/payment_density_starter.py
python labs/module-04/l8_5-gaussian-mixtures/discount_mixture_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. Check everything in this
session at once (8 files; about 30 seconds):

```bash
python3 labs/check_labs.py module-04
```

## Timing in Session 4

| Clock | Lab |
|-------|-----|
| 0:45 | `payment_density_starter.py` (25 min); fast finishers take `rings_starter.py`, then `debug_density.py` |
| 2:10 | `discount_mixture_starter.py` (20 min, everyone), then `lifecycle_mixture_starter.py` in pairs; it goes home for anyone who does not reach it |
