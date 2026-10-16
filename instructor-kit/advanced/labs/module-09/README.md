# Session 9 labs: Transfer Learning and Text Classification

Lessons **10.5 Transfer Learning** and **11.1 Text Classification**. Every lab is new practice on the
lesson's skills; none repeats a lesson project or exercise (the EuroSAT tiles in Lab 10.5 A are not the
lesson's tiles, and the 11.1 labs use new targets on the review data).

Hand out the `*_starter.py` files, the `_fixtures/` folder and the two `debug_*.py` files.
Keep every `*_solution.py` for yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l10_5-transfer-learning/prepare_eurosat_features.py` | One-off: frozen ResNet-18 features (128 px, ImageNet-normalized, `eval()`) for EuroSAT positions 3,500 to 6,499 of the seed-42 permutation, saved to `_fixtures/eurosat_resnet18_feats.npz` (float16, 2.4 MB) | before the session, only if the fixture is missing | `features for 3,000 tiles in ... s`, `saved eurosat_resnet18_feats.npz: 2.45 MB` (about 30 s) |
| `l10_5-transfer-learning/feature_table_starter.py` / `_solution.py` | Pretrained features as an ordinary table: baseline, cross-validated head, cosine kNN, per-class accuracy and mix-ups, a confidence-based human-review queue | 12 min | baseline (SeaLake) 0.126; CV 0.899 ± 0.011 (scaled 0.896); test 0.903; kNN 0.850 / 0.866 / 0.861; weakest River 0.759; River -> Highway 11; < 0.8 confidence: 10.5% of tiles, 0.571 inside vs 0.942 outside; stretch PCA 10/50/200: 0.808/0.870/0.896 |
| `l10_5-transfer-learning/freeze_audit_starter.py` / `_solution.py` | Auditing a fine-tuning setup on one batch: freezing order, trainable counts per strategy, parameter groups that forget the head, what one step changes in eval() and train() mode | 8 min (pairs) | 5,130 and 0 (wrong order); 8,398,858 (75.11%) and 10,498,570 (93.89%); missing `['fc.weight', 'fc.bias']`; train() step also changes `bn1.running_mean` |
| `l10_5-transfer-learning/debug_transfer.py` | Two loud bugs (`classifier` instead of `fc`, head frozen too) and two silent ones (no Normalize, no `eval()`) (deliberately broken) | fast finishers | Fixed: `test accuracy of the linear head: 0.895` |
| `l11_1-text-classification/recommend_flag_starter.py` / `_solution.py` | A new target (the "would recommend" checkbox) on all 4,000 reviews: baseline, three vectorizer settings cross-validated, classification report, coefficients, accuracy by star rating, probes | 12 min | baseline 0.655; CV 0.726 / 0.719 / 0.732; test 0.755, not-recommended recall 0.481; by stars 0.794 / 0.543 / 0.464 / 0.845 / 0.963; "It leaked on the first night." P(recommended) 0.66 |
| `l11_1-text-classification/category_router_starter.py` / `_solution.py` | Ten-class routing of messages to product teams: macro F1, per-class recall, wrong routes, words of one class, confidence-based auto-routing | 8 min (pairs) | baseline 0.237; accuracy 0.615, macro F1 0.585; Tents recall 0.333; Paddling -> Apparel 63; 42.7% routed at 0.932; stretch balanced 0.580 / Tents 0.452 |
| `l11_1-text-classification/debug_text_pipeline.py` | A DataFrame and missing text (loud), stop words and a threshold chosen on the test set (silent) (deliberately broken) | fast finishers | Fixed: 0.877, threshold 0.2, P(negative) `[0.05 0.3 0.13]` |

A starter runs without an error as handed out: it prints its `TODO` reminders and an "Expected when
done" line. The 10.5 labs need EuroSAT and the ResNet-18 weights in a cache (the lessons' downloads in
`data/` and PyTorch's cache are found automatically by `labs_paths`); `feature_table_*` reads only the
fixture.

## Rebuilding the fixture

```bash
python labs/module-09/l10_5-transfer-learning/prepare_eurosat_features.py
```

It needs EuroSAT and the ResNet-18 weights (downloads them when online and not cached) and takes about
30 seconds on a 4-thread laptop CPU. Features are stored as float16, so a rebuilt file gives the same
numbers to the third decimal.

## How to run

```bash
python labs/module-09/l10_5-transfer-learning/feature_table_starter.py
python labs/module-09/l11_1-text-classification/recommend_flag_starter.py
python3 labs/check_labs.py module-09        # all 8 files, about 35 seconds
```

## Timing in Session 9

| Clock | Lab |
|-------|-----|
| 0:50 | `feature_table_starter.py` (12 min), pairs who finish take `freeze_audit_starter.py` (8 min) |
| 2:15 | `recommend_flag_starter.py` (12 min), fast pairs take `category_router_starter.py` (8 min) |
