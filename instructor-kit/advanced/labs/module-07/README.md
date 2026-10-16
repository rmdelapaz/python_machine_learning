# Session 7 labs: Neural Networks and PyTorch

Lessons **10.1 Neural Networks from Scratch** and **10.2 PyTorch Fundamentals**. Every lab is new practice
on the lesson's skills; none of them repeats a lesson project or exercise.

Hand out the `*_starter.py` files and the two `debug_*.py` files. Keep every `*_solution.py` for
yourself until after the lab.

## What is here

| File | Practices | In class | Expected output (solution) |
|------|-----------|---------:|----------------------------|
| `l10_1-neural-networks/circles_network_starter.py` / `_solution.py` | Writing the forward pass and the loss by hand, a numeric gradient check on an output weight, the four-line training loop, width, reading a learning-rate problem from the loss | 15 min | Logistic regression 0.488 (a coin flip on a ring); untrained loss 0.6927; `W2[5, 0]` backprop and numeric both 0.03266046; 8 units: loss 0.6927 → 0.0156, test accuracy 0.984; width sweep 1/2/3/4/8 units: 0.648 / 0.752 / **0.992** / 0.984 / 0.984; learning rate 0.05 crawls (0.487 after 500 epochs), 0.5 reaches 0.045, 50 bounces near 4 |
| `l10_1-neural-networks/churn_neuron_starter.py` / `_solution.py` | One sigmoid neuron trained by hand on churn (its gradient is `(p - y) / n`), compared with `LogisticRegression`; why zero initialization is fine for one neuron and fatal for a hidden layer | 10 min (pairs) | Loss 0.6931 → 0.3689 (epoch 10) → 0.3416; test ROC AUC 0.8235 for the neuron AND for `LogisticRegression(C=1e6)`; largest weight difference 0.0022; biggest weights `days_since_last_order` 0.88, `discount_share` 0.67 |
| `l10_1-neural-networks/debug_backprop.py` | Three bugs in a NumPy network: a transposed `W2` (stops with a `ValueError`), a backward pass without the ReLU gate (the gradient check disagrees: 0.03319750 vs 0.00780703), an update that climbs uphill (loss 0.6714 → 13.8155, accuracy 0.500) (deliberately broken) | fast finishers | Fixed: `agree: True`, loss 0.6714 → 0.1432, test accuracy 0.972 (the lesson's numbers) |
| `l10_2-pytorch/fraud_mlp_starter.py` / `_solution.py` | The whole PyTorch workflow on card fraud: three-way split, `ColumnTransformer`, float32 tensors with `(n, 1)` targets, an `nn.Module`, mini-batches, early stopping, one look at the test set, an honest comparison | 25 min | 30,000 / 7,500 / 12,500 rows (263 / 66 / 109 frauds); 673 parameters; stopped after epoch 24, best epoch 19, validation loss 0.0245; test ROC AUC network 0.855 (AP 0.483), logistic regression 0.844 (0.456), default boosting 0.825 (0.399); 50 frauds in the network's top 100 vs 0.9 at random |
| `l10_2-pytorch/batch_size_starter.py` / `_solution.py` | The training step from memory; batch size vs number of steps (16, 64, 512, 3,600); a checkpoint that stores its own settings and the scaler's numbers | 10 min (pairs) or homework | After 10 epochs val AUC 0.871 (batch 16, 2,250 steps), 0.866 (64, 570), 0.819 (512, 80), 0.518 (3,600, 10); the full batch given 570 steps reaches 0.873; rebuilt checkpoint 0.866 = original |
| `l10_2-pytorch/debug_training_loop.py` | Five bugs: float64 tensors and flat targets (both stop with an error), a sigmoid before `BCEWithLogitsLoss`, a missing `zero_grad()`, evaluation without `model.eval()` (all three silent) (deliberately broken) | fast finishers | Fixed: validation AUC 0.839, the same on both repeat lines. Silent-bug trail: 0.503 / 0.502 / 0.499 with all three; 0.766 / 0.764 / 0.780 once the sigmoid is gone; 0.824 / 0.825 / 0.829 once `zero_grad()` is back |

A starter runs without an error exactly as handed out: it prints its `TODO` reminders and an "Expected
when done" block. Training numbers are bit-identical on one machine with 4 threads; on another laptop
the last digit can differ ("yours may vary slightly"), while the comparisons and rankings hold.

## CPU time

Everything here runs on a laptop CPU. Measured with 4 threads: churn neuron 3 s, circles network 5 s,
fraud network 20 s, batch-size sweep 22 s (most of it the 570-epoch full-batch run in Step 4). No
downloads, no image data, no GPU: Session 7 uses only the Pinecrest CSV files.

## How to run

With the course environment active, from the `pinecrest-ml` project folder (the one with `data/` in it):

```bash
python labs/module-07/l10_1-neural-networks/circles_network_starter.py
python labs/module-07/l10_2-pytorch/fraud_mlp_starter.py
```

Or open a file in VS Code or JupyterLab: the `# %%` lines are notebook cells. Labs find the data through
`labs_paths.py` (see `labs/README.md`); the batch-size lab writes its checkpoint to `OUTPUT`
(`output/` next to the lab, a temporary folder under the checker).

Check everything in this session at once (8 files; about 70 seconds):

```bash
python3 labs/check_labs.py module-07
```

## Timing in Session 7

| Clock | Lab |
|-------|-----|
| 0:50 | `circles_network_starter.py` (15 min, everyone); fast finishers take `debug_backprop.py` |
| 1:05 | `churn_neuron_starter.py` in pairs (5 min in class, finish at home) |
| 2:10 | `fraud_mlp_starter.py` (25 min); fast finishers take `debug_training_loop.py` |
| homework | `batch_size_starter.py` (or in class when the session runs short) |
