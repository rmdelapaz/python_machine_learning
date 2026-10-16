# Labs

All hands-on lab programs for **Advanced Machine Learning** (Python Machine Learning, Modules 7 to 13),
grouped by session (one folder per 3-hour session; `module-07` = Session 7). Each session folder has its
own `README.md` with the exact files, what each lab practices, how long it takes and what "done" looks
like. This file is the map and the rules.

```
labs/
├─ labs_paths.py    finds the course data (DATA), a safe place to write (OUTPUT), the OFFLINE switch,
│                   and the cached image datasets and pretrained weights (image_root, use_cached_weights)
├─ check_labs.py    runs every starter and solution and reports PASS/FAIL
├─ module-01/       Session 1 · lessons 7.1 + 7.2
│   ├─ README.md
│   ├─ l7_0-orientation/        (Session 1 only: environment check incl. PyTorch, refresher diagnostic)
│   ├─ l7_1-<slug>/             one folder per lesson: l<module>_<lesson>-<slug>
│   │   ├─ <name>_starter.py    the handout
│   │   ├─ <name>_solution.py   the instructor's reference
│   │   └─ debug_<name>.py      (optional) a deliberately broken "find the bug" handout
│   └─ _fixtures/               (optional) small saved inputs a lab reads (saved features, a model)
├─ module-02/ ... module-13/
```

## The data

Labs read the same Pinecrest Outfitters files the lessons use (`customers_churn.csv`,
`weekly_demand.csv`, `transactions.csv`, ...). They never contain a path to the data. They ask
`labs_paths` for it:

```python
from labs_paths import DATA
customers = pd.read_csv(DATA / "customers_churn.csv")
```

`DATA` is found in this order:

1. the environment variable **`PINECREST_ML_DATA`**, if it is set (the folder that holds `customers_churn.csv`);
2. a **`data/customers_churn.csv`** in the lab's own folder or any folder above it. Inside the course
   repository that is the course's `data/` folder. If you received the kit on its own, unzip
   `pinecrest-ml-data.zip` next to the `labs/` folder so that `data/customers_churn.csv` sits beside it;
   on a learner's laptop, put `labs/` inside the `pinecrest-ml` project folder from Lesson 0.1, next to `data/`;
3. otherwise the lab stops with a message that says exactly those two fixes.

**Labs only read from `DATA`.** Anything a lab writes (charts, CSV exports, a saved model, an MLflow
database) goes to `OUTPUT`:

```python
from labs_paths import DATA, OUTPUT
fig.savefig(OUTPUT / "risk_deciles.png")          # ./output/ next to the lab file
```

`OUTPUT` is an `output/` folder next to the lab file, created the first time a lab uses it (it is
git-ignored). When `LABS_OFFLINE=1` it is a temporary folder that disappears when the program ends.

## The bootstrap: the first lines of every lab

Every `*_starter.py`, `*_solution.py` and `debug_*.py` begins with exactly these lines
(change only the names imported on the last one):

```python
# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA
```

The middle line walks up from the lab file until it reaches the folder that holds `labs_paths.py`
(this `labs/` folder) and puts it on Python's import path. That is why a lab runs the same way from its
own folder, from `labs/`, from the kit root or from anywhere else, and as cells in VS Code (`# %%`). In a
plain Jupyter notebook there is no `__file__`, so the walk starts at the notebook's working folder
instead; keep the notebook somewhere under `labs/`. The last line may import any of `DATA`, `OUTPUT`,
`OFFLINE`, `image_root`, `use_cached_weights`, `fixtures`.

## Image datasets and pretrained weights (deep-learning labs)

Fashion-MNIST, EuroSAT and the pretrained torchvision weights are large downloads. Labs never hard-code
where they are and never download while the checker runs:

```python
from labs_paths import image_root, use_cached_weights, OFFLINE
train = datasets.FashionMNIST(image_root("FashionMNIST"), train=True, download=not OFFLINE,
                              transform=transforms.ToTensor())
use_cached_weights()                      # before models.resnet18(weights=...)
```

`image_root(name)` returns the first folder that already holds `<name>/`: `$ML_DATA_DIR`, the
instructor's shared cache `~/.cache/ml-course-data`, then the `data/` folder (where the lessons'
`root="data"` downloads on a learner's laptop). On a learner's machine that has run the lessons, the
labs reuse the lesson downloads. Slow steps (a pretrained network's features for a few thousand images)
belong in a `prepare_<name>.py` that writes a small file into the session's `_fixtures/` folder once;
the lab then loads that file in a second. `prepare_*.py` files are not run by the checker.

## Naming rules

| File | Meaning | Run by the checker |
|------|---------|--------------------|
| `<name>_starter.py` | The handout. **Runs without error exactly as shipped**: numbered `# TODO` comments, placeholder values, and a printed line that says what is expected once the TODOs are done. | yes |
| `<name>_solution.py` | The reference. Complete; prints its answers. Do not hand out before the lab. | yes |
| `debug_<name>.py` | A deliberately broken or wrong-answer "find the bug" handout. | **no** |
| `prepare_<name>.py` | A one-off step that computes a cached input (features, a trained model) into `_fixtures/`. | **no** |
| `_fixtures/...` | Small saved inputs. Not programs. | **no** |
| anything else (`check_environment.py`, helper modules, `README.md`) | Support files. | no |

- Starters and solutions come in **pairs** with the same `<name>`; a starter without a solution (or the
  reverse) fails the check.
- Lesson folders are `l<module>_<lesson>-<slug>`: `l7_2-shap-values`, `l10_3-cnn`.
- Use `# %%` cell markers so a lab opens as notebook cells in VS Code and Jupyter.
- Labs take no keyboard input (`input()` gets an empty stdin in the checker), seed every random step
  (`random_state=42`, `np.random.default_rng(42)`, `torch.manual_seed(42)`), and print rounded numbers.
- **CPU budget:** a solution finishes in 90 seconds or less on a 4-thread laptop CPU. Subset the data,
  train for one or two epochs, or load features made by a `prepare_*.py`.
- No `plt.show()` that blocks, no `webbrowser.open()`, no server left running. Save figures to `OUTPUT`.

## Checking the labs

From this kit's folder (any folder works):

```bash
python3 labs/check_labs.py                 # all sessions
python3 labs/check_labs.py module-07       # one session (also 07 or 7; several allowed)
python3 labs/check_labs.py module-07 -v    # also show the tail of each program's output
python3 labs/check_labs.py --timeout 300 module-08     # allow slower labs (default 150 s per file)
python3 labs/check_labs.py --python /path/to/python    # choose the interpreter that runs the labs
python3 labs/check_labs.py --strict        # all sessions; a session with no labs counts as a failure
```

What it does for every `*_starter.py` and `*_solution.py`:

- copies the whole session folder to a temporary folder and runs the file from its own lesson folder
  inside that copy, so nothing a lab writes can land in the kit;
- uses the lessons' settings: `LABS_OFFLINE=1`, `PINECREST_ML_DATA=<course>/data`, 4 threads
  (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `LOKY_MAX_CPU_COUNT`),
  `OMP_WAIT_POLICY=PASSIVE`, `PYTHONHASHSEED=0`, `TORCH_HOME` and `ML_DATA_DIR` pointing at
  `~/.cache/ml-course-data`, `MLFLOW_DISABLE_AGENT_HINT=1`, `MPLBACKEND=Agg`, `PYTHONPATH=<kit>/labs`,
  `PYTHONDONTWRITEBYTECODE=1`, and a private `TMPDIR`;
- fails the file if it exits non-zero, prints a `Traceback` on stderr, or runs past the timeout; prints
  the run time, and a `SLOW` note for a solution over the 90-second budget;
- prints one `PASS` / `FAIL` line per file, then `N passed, M failed`; the exit code is 0 only when
  nothing failed. Asking for one session that has no lab files is a failure.

After the runs it compares a listing of the checked session folders, `labs/`, the data folder and the
image cache with the listing it took before. A new, changed or missing file is a failure: some lab
wrote outside its temporary copy (usually into `DATA`) or downloaded something.

**Which Python runs the labs:** `--python`, else the environment variable `LABS_PYTHON`, else
`~/.cache/ml-course-venv/bin/python` when it exists (the author's course environment), else the
interpreter you started the checker with. On your own machine, activate the course virtual environment
from the setup guide and run `python labs/check_labs.py`.

**How the documents refer to labs:** the facilitator guide, workbook and answer key use paths like
`labs/module-07/l10_1-neural-networks/`, relative to this kit's folder (where the combined books live). Keep the
whole kit folder together so those pointers hold.
