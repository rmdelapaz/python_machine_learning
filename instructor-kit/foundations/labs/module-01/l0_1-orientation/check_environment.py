"""check_environment.py: is this computer ready for Python Machine Learning?

Run it from a terminal with the course environment active, from your pinecrest-ml project folder:

    python check_environment.py
    (or by its path:  python labs/module-01/l0_1-orientation/check_environment.py)

It checks four things and tells you exactly what to fix:
  1. Python is 3.12 or newer
  2. a virtual environment is active
  3. every course package imports (and prints its version)
  4. the Pinecrest machine-learning data can be found and has every file

The last line is either "READY for Python Machine Learning" or "NOT READY" with a numbered fix list.
This file is a tool, not an exercise: there is nothing to edit. The first run takes up to a minute
(some packages, such as PyTorch and MLflow, are large and slow to import the first time).
"""
import importlib
import importlib.metadata
import os
import pathlib
import platform
import sys
import warnings

warnings.filterwarnings("ignore")      # a version check, not a lesson: keep the output to the point
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # MLflow otherwise prints an unrelated hint

# (name you import, name on the pip line, where it is first needed, must it be there today?)
PACKAGES = [
    ("numpy", "numpy", "Lesson 0.1", True),
    ("pandas", "pandas", "Lesson 0.1", True),
    ("matplotlib", "matplotlib", "Lesson 0.1", True),
    ("sklearn", "scikit-learn", "Lesson 1.1", True),
    ("scipy", "scipy", "Lesson 2.5", True),
    ("imblearn", "imbalanced-learn", "Lesson 2.2", True),
    ("joblib", "joblib", "Lesson 2.2", True),
    ("skops", "skops", "Lesson 2.2", True),
    ("xgboost", "xgboost", "Lesson 5.3", True),
    ("lightgbm", "lightgbm", "Lesson 5.3", True),
    ("optuna", "optuna", "Lesson 6.2", True),
    ("shap", "shap", "Lesson 7.2", True),
    ("lime", "lime", "Lesson 7.3", True),
    ("umap", "umap-learn", "Lesson 9.3", True),
    ("torch", "torch", "Lesson 0.1 check, Module 10", True),
    ("torchvision", "torchvision", "Module 10", True),
    ("mlflow", "mlflow", "Lesson 12.1", True),
    ("fastapi", "fastapi", "Lesson 12.2", True),
    ("uvicorn", "uvicorn", "Lesson 12.2", True),
    ("httpx", "httpx", "Lesson 12.2", True),
    ("seaborn", "seaborn", "optional charts", False),
    ("pyarrow", "pyarrow", "faster file reading (optional)", False),
    ("jupyterlab", "jupyterlab", "notebooks in JupyterLab", False),
]

DATA_FILES = {                      # file -> expected number of data rows
    "customers_churn.csv": 6000, "weekly_demand.csv": 9984, "transactions.csv": 50000,
    "reviews.csv": 4000, "products_ml.csv": 200, "interactions.csv": 32178, "daily_store_sales.csv": 4384,
}

fixes = []      # things that must be fixed before Session 2
notes = []      # things that can wait


def line(mark, label, detail=""):
    print(f"  {'[' + mark + ']':<7}{label:<22} {detail}")


def find_data():
    """The folder that holds customers_churn.csv: PINECREST_ML_DATA, or a data/ folder here or above."""
    env = os.environ.get("PINECREST_ML_DATA", "").strip()
    if env:
        return pathlib.Path(env).expanduser()
    for start in (pathlib.Path.cwd(), pathlib.Path(__file__).resolve().parent):
        for folder in [start, *start.parents]:
            if (folder / "data" / "customers_churn.csv").is_file():
                return folder / "data"
    return None


print("Python Machine Learning: environment check")
print("=" * 64)

# 1. Python version
print("1. Python")
version = sys.version.split()[0]
if sys.version_info >= (3, 12):
    line("ok", "Python " + version, platform.system() + " " + platform.machine())
else:
    line("FIX", "Python " + version, "the course needs 3.12 or newer")
    fixes.append("Install Python 3.12 or newer from python.org, then rebuild the environment "
                 "(delete .venv, create it again, install from requirements.txt).")
print("       running from:", sys.executable)

# 2. Virtual environment
print("2. Virtual environment")
in_venv = sys.prefix != sys.base_prefix or bool(os.environ.get("CONDA_PREFIX"))
if in_venv:
    line("ok", "environment active", pathlib.Path(sys.prefix).name)
else:
    line("FIX", "no environment active", "this is the computer-wide Python")
    fixes.append("Activate the course environment in THIS terminal, then run the check again:\n"
                 "       Windows PowerShell:  .venv\\Scripts\\Activate.ps1\n"
                 "       macOS / Linux:       source .venv/bin/activate")

# 3. Packages
print("3. Course packages")
missing = []
for module, pip_name, first_used, core in PACKAGES:
    try:
        mod = importlib.import_module(module)
    except Exception as err:            # ImportError, or a broken install raising something else
        if core:
            line("FIX", pip_name, f"not importable ({type(err).__name__}); first needed in {first_used}")
            missing.append(pip_name)
        else:
            line("note", pip_name, f"not installed; used for {first_used}")
            notes.append(f"{pip_name} is missing (used for {first_used}); "
                         "python -m pip install -r requirements.txt installs it.")
        continue
    try:
        ver = getattr(mod, "__version__", None) or importlib.metadata.version(pip_name)
    except importlib.metadata.PackageNotFoundError:
        ver = "installed"
    line("ok", pip_name, ver)
if missing:
    fixes.append("Install the missing packages with the environment active:\n"
                 "       python -m pip install -r requirements.txt\n"
                 "     (missing: " + ", ".join(missing) + ")\n"
                 "     On Linux, install the CPU build of PyTorch first (Lesson 0.1 shows the line).")
try:
    import sklearn
    if not sklearn.__version__.startswith("1.9"):
        notes.append(f"scikit-learn is {sklearn.__version__}; the lessons print output from 1.9. "
                     "Some numbers will differ: reinstall from requirements.txt.")
except ImportError:
    pass
try:
    import torch
    gpu = torch.cuda.is_available()
    line("ok", "PyTorch device", "GPU found (fine; the course uses the CPU)" if gpu else "CPU only (expected)")
except ImportError:
    pass

# 4. Course data
print("4. Course data")
data = find_data()
if data is None or not (data / "customers_churn.csv").is_file():
    line("FIX", "data folder", "not found")
    fixes.append("Unzip the course data so that a data folder sits in your project folder:\n"
                 "       python -m zipfile -e pinecrest-ml-data.zip .\n"
                 "     then run this check from the project folder (the one that contains data/).")
else:
    line("ok", "data folder", str(data))
    absent = [name for name in DATA_FILES if not (data / name).is_file()]
    if absent:
        line("FIX", "files missing", ", ".join(absent))
        fixes.append("Unzip pinecrest-ml-data.zip again: these files are missing from the data folder: "
                     + ", ".join(absent))
    else:
        line("ok", f"{len(DATA_FILES)} data files", "all present")
        import csv
        wrong = []
        for name, rows in DATA_FILES.items():
            with open(data / name, newline="", encoding="utf-8") as f:
                n = sum(1 for _ in csv.reader(f)) - 1      # data rows, without the header
            if rows is not None and n != rows:
                wrong.append(f"{name} has {n:,} rows, expected {rows:,}")
        if wrong:
            line("FIX", "row counts", "; ".join(wrong))
            fixes.append("A data file has the wrong size. Download pinecrest-ml-data.zip again and unzip it.")
        else:
            line("ok", "row counts", "customers_churn.csv 6,000 customers, and the rest as expected")

print("=" * 64)
for text in notes:
    print("Note:", text)
print("Note: the deep-learning lessons (Module 10) download Fashion-MNIST, EuroSAT and pretrained weights")
print("      the first time they need them (about 175 MB). Nothing to do now.")
if fixes:
    print("NOT READY. Fix before Session 2:")
    for number, text in enumerate(fixes, start=1):
        print(f"  {number}. {text}")
    print("Then run this check again. Stuck? Show this output to your instructor.")
else:
    print("READY for Python Machine Learning")
