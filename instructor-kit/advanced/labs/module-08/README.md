# Session 8 labs: CNNs and Sequence Models

Lessons **10.3 Convolutional Neural Networks** and **10.4 Sequence Models**. Every lab is new practice on
the lesson's skills; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files, the two `debug_*.py` files and the `_fixtures/` folder (learners need it:
`shift_test_*.py` loads the two saved models from it). Keep every `*_solution.py` for yourself until after
the lab. `prepare_fashion_checkpoints.py` is yours too (it rebuilds the fixtures).

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l10_3-cnn/edge_features_starter.py` / `_solution.py` | Hand-made filters with `F.conv2d`, pooling into features, and a race against the lesson's SmallCNN at 1,000 / 5,000 / 20,000 training images | 12 min | 48 features per image; hand-made + logistic regression 0.795 / 0.819 / 0.827 against SmallCNN (3 epochs) 0.676 / 0.782 / **0.863**: a person's filters win with little data, learned filters overtake at 20,000 |
| `l10_3-cnn/shift_test_starter.py` / `_solution.py` | Loading saved `state_dict`s with `weights_only=True`, `eval()`, and testing the "a CNN finds a pattern anywhere" claim by shifting the test images | 8 min (pairs), finish at home | Unshifted 0.840 / 0.864; shifted right 1 / 2 / 3 / 4 px: dense 0.756 / 0.552 / 0.355 / 0.252, CNN 0.838 / 0.743 / 0.471 / 0.301 (down: dense 0.778 / 0.708 / 0.661 / 0.553, CNN 0.821 / 0.770 / 0.726 / 0.592); worst CNN classes at 2 px: Dress -0.334, T-shirt/top -0.229, Coat -0.217 |
| `l10_3-cnn/debug_cnn.py` | Four bugs: no channel axis and a wrong first `Linear` size (both stop with a `RuntimeError`), a softmax before `CrossEntropyLoss` and evaluation without `model.eval()` (both silent) (deliberately broken) | fast finishers | Fixed: test accuracy 0.747 on both lines. Silent trail: 0.655 / 0.653 with both silent bugs, 0.688 with only the missing `eval()` |
| `l10_3-cnn/prepare_fashion_checkpoints.py` | Not a lab: trains the lesson's dense network and SmallCNN exactly as on the page and saves them to `_fixtures/` (`fashion_dense.pt`, `fashion_small_cnn.pt`, about 0.4 MB each) | before the session, once | `dense test accuracy 0.840, CNN test accuracy 0.864` (the page's numbers); 30 to 90 s |
| `l10_4-sequences/transactions_gru_starter.py` / `_solution.py` | The 10.4 workflow on a new target, tomorrow's NUMBER OF TRANSACTIONS: naive yardsticks, log + standardize on training years, windows, a GRU, with and without the calendar, MAE in transactions | 15 min | Naive 19.09, seasonal naive 15.50, GRU weekday only 9.15, GRU with promotion and closed flags 7.56 transactions per store-day; 2,812 / 1,460 windows; 4,065 / 4,257 parameters |
| `l10_4-sequences/attention_lookup_starter.py` / `_solution.py` | Hand-built attention: what the keys describe (weekday vs weekday + promotion + closed) and how far back the memory reaches (28 vs 364 days), scored separately on ordinary and special days; routing queries; reading the weights | 10 min, finish at home | MAE all / ordinary / special: seasonal naive $1,072 / $949 / $3,194; lesson attention $979 / $856 / $3,097; calendar keys 364 days $1,039 / $1,030 / $1,198; routed $875 / $856 / $1,198; Black Friday Boise $12,014 vs $12,702 actual, weight 0.226 on four past sale Fridays |
| `l10_4-sequences/debug_sequence.py` | Four bugs in a Boise forecaster: float64 windows (a `ValueError`), no `batch_first=True`, a random split of overlapping windows, the MAE in scaled units (all three silent) (deliberately broken) | fast finishers | Fixed: 365 test windows, LSTM MAE 1,039.2 against seasonal naive 1,355.6. With the silent bugs it prints a meaningless 0.2; in dollars the random split flatters it to 825.8 (863.2 without `batch_first`) |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block. PyTorch numbers are bit-identical on one machine with 4 threads; on another laptop the
last digit can differ ("yours may vary slightly"); the rankings hold.

## CPU time and the fixtures

Measured with 4 threads on the build machine (other jobs running; a quiet laptop is similar, an older one
up to twice as slow): edge-features race 30 s (most of it the CNN on 20,000 images), shift test 20 s (no
training: it loads the fixtures), transactions GRU 30 s (two 15-epoch trainings), attention 3 s. Fashion-MNIST
comes from the image cache (`image_root("FashionMNIST")`): nothing downloads under the checker, and on a
learner's laptop the labs reuse the copy the lessons downloaded into `data/`.

The fixtures are the lesson's own models, so the shift test starts from the page's numbers. Rebuild them
after a PyTorch upgrade, from the kit folder:

```bash
OMP_NUM_THREADS=4 python labs/module-08/l10_3-cnn/prepare_fashion_checkpoints.py
```

## How to run

With the course environment active, from the `pinecrest-ml` project folder:

```bash
python labs/module-08/l10_3-cnn/edge_features_starter.py
python labs/module-08/l10_4-sequences/transactions_gru_starter.py
```

Check everything in this session at once (8 files; about 2 minutes):

```bash
python3 labs/check_labs.py module-08
```

## Timing in Session 8

| Clock | Lab |
|-------|-----|
| 0:50 | `edge_features_starter.py` (12 min, everyone), then `shift_test_starter.py` in pairs (8 min); fast finishers take `debug_cnn.py` |
| 2:10 | `transactions_gru_starter.py` (15 min), then `attention_lookup_starter.py` (10 min); fast finishers take `debug_sequence.py` |
| homework | finish `shift_test_starter.py` and `attention_lookup_starter.py` |
