# Session 6 labs: LDA, t-SNE and UMAP (and Checkpoint 1)

Lessons **9.2 Linear Discriminant Analysis** and **9.3 t-SNE and UMAP**, plus **Checkpoint 1** on Modules
7 to 9. Every lab is new practice on the lesson's skills; none repeats a lesson project or exercise.

Hand out the `*_starter.py` files. Keep every `*_solution.py` (and `checkpoint_solution.py`, the model
answer) for yourself. **Fashion-MNIST must be on the machine** (26 MB): learners download it before the
session (bring list of Session 5); the labs find it with `image_root("FashionMNIST")` and never download
during `check_labs.py`.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l9_2-lda/payment_category_lda_starter.py` / `_solution.py` | LDA with ten classes: number of axes, separation per axis, reading `scalings_`, LDA as a classifier against a baseline, recall by class | 15 min | 9 axes, LD1 carries 0.995: price per item (amount 1.42, items -0.59); accuracy 0.290 vs 0.219; six categories never predicted |
| `l9_2-lda/noise_leak_starter.py` / `_solution.py` | A supervised reducer fitted outside cross-validation, on real customers plus 290 noise columns, with shuffled and true labels | pairs / homework | Shuffled: leaky AUC 0.998, honest 0.481; true: leaky 1.000, honest 0.541; ten real columns alone 0.776 |
| `l9_3-tsne-umap/fashion_map_starter.py` / `_solution.py` | t-SNE and UMAP on 1,500 Fashion-MNIST training images: neighbor checks, which garments mix, placing 500 new images with `transform`, saving a figure to `OUTPUT` | 20 min | Neighbors kept 0.195 / 0.573 / 0.495 (PCA view / t-SNE / UMAP); Shirt 0.32 hardest; 0.730 of new images among their own class. About 30 s |
| `l9_3-tsne-umap/fraud_map_starter.py` / `_solution.py` | Checking a "fraud is its own island" claim with numbers: fraud neighbors in the original space and on maps at two perplexities, then a clustering in the original space | pairs / homework | 0.614 original; 0.630 / 0.631 on the maps; K-Means: one cluster of 239 frauds + 20 honest, 199 frauds elsewhere |
| `checkpoint-1/checkpoint_starter.py` / `_solution.py` | **Checkpoint 1** (30 min, open notebook): permutation importance (Module 7), soft segments (Module 8), PCA vs LDA (Module 9), three written sentences | 30 min | Top columns days_since_last_order 0.154, discount_share 0.099; five components, Deal Hunters churn 0.298, 305 below 0.8; 7 components for 90%; LD1 led by days_since_last_order (0.92); AUC PC1 0.631 vs LD1 0.824 |

Rubric and model answer: the Session 6 answer key.

## How to run

```bash
python labs/module-06/l9_2-lda/payment_category_lda_starter.py
python labs/module-06/l9_3-tsne-umap/fashion_map_starter.py
python3 labs/check_labs.py module-06          # 10 files; about 90 seconds
```

## Timing in Session 6

| Clock | Lab |
|-------|-----|
| 0:40 | `payment_category_lda_starter.py` (everyone), then `noise_leak_starter.py` in pairs |
| 1:50 | `fashion_map_starter.py` (everyone; start TODO 1 and talk while it runs), then `fraud_map_starter.py` in pairs |
| 2:20 | `checkpoint_starter.py` (Checkpoint 1, 30 minutes) |
