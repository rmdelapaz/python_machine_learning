"""check_environment.py: is this computer ready for Advanced Machine Learning?

Run it from a terminal with the course environment active, from your pinecrest-ml project folder:

    python check_environment.py
    (or by its path:  python labs/module-01/l7_0-orientation/check_environment.py)
    python labs/module-01/l7_0-orientation/check_environment.py --download
                       (also downloads the image data and pretrained weights, about 175 MB, once)

It checks five things and tells you exactly what to fix:
  1. Python is 3.12 or newer
  2. a virtual environment is active
  3. every course package imports (and prints its version), including PyTorch and torchvision
  4. the Pinecrest machine-learning data can be found and has every file
  5. the image data (Fashion-MNIST, EuroSAT) and the two pretrained networks are on this computer

The last line is "READY for Advanced Machine Learning", "READY for Sessions 1 to 4" (when only the
image data is missing: run again with --download before Session 5), or "NOT READY" with a numbered fix
list. This file is a tool, not an exercise: there is nothing to edit. The first run takes up to a
minute (PyTorch and MLflow are large and slow to import the first time).
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

DOWNLOAD = "--download" in sys.argv
OFFLINE = os.environ.get("LABS_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")

# (name you import, name on the pip line, where it is first needed, must it be there today?)
PACKAGES = [
    ("numpy", "numpy", "every session", True),
    ("pandas", "pandas", "every session", True),
    ("matplotlib", "matplotlib", "every session", True),
    ("sklearn", "scikit-learn", "every session", True),
    ("scipy", "scipy", "Lessons 7.2 and 8.3", True),
    ("joblib", "joblib", "Lesson 12.1", True),
    ("skops", "skops", "Lesson 12.1", True),
    ("xgboost", "xgboost", "optional comparisons", True),
    ("lightgbm", "lightgbm", "optional comparisons", True),
    ("shap", "shap", "Lesson 7.2 (today)", True),
    ("lime", "lime", "Lesson 7.3 (Session 2)", True),
    ("umap", "umap-learn", "Lesson 9.3", True),
    ("torch", "torch", "Module 10", True),
    ("torchvision", "torchvision", "Lesson 9.3 and Module 10", True),
    ("mlflow", "mlflow", "Lesson 12.1", True),
    ("fastapi", "fastapi", "Lesson 12.2", True),
    ("uvicorn", "uvicorn", "Lesson 12.2", True),
    ("httpx", "httpx", "Lesson 12.2", True),
    ("imblearn", "imbalanced-learn", "review of Foundations", False),
    ("optuna", "optuna", "review of Foundations", False),
    ("seaborn", "seaborn", "optional charts", False),
    ("jupyterlab", "jupyterlab", "notebooks in JupyterLab", False),
]

DATA_FILES = {                      # file -> expected number of data rows
    "customers_churn.csv": 6000, "weekly_demand.csv": 9984, "transactions.csv": 50000,
    "reviews.csv": 4000, "products_ml.csv": 200, "interactions.csv": 32178, "daily_store_sales.csv": 4384,
}
FASHION_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
                 "t10k-labels-idx1-ubyte"]
EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
                   "PermanentCrop", "Residential", "River", "SeaLake"]
WEIGHTS = {"resnet18-f37072fd.pth": "ResNet18 (45 MB)", "mobilenet_v3_small-047dcff4.pth": "MobileNet V3 Small (10 MB)"}

fixes = []      # things that must be fixed before Session 2
images = []     # image data still to download (needed from Session 5 on)
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


def image_roots(data):
    """Where image data may already be: $ML_DATA_DIR, a shared cache, then the course data/ folder."""
    roots = []
    if os.environ.get("ML_DATA_DIR", "").strip():
        roots.append(pathlib.Path(os.environ["ML_DATA_DIR"]).expanduser())
    roots.append(pathlib.Path.home() / ".cache" / "ml-course-data")
    if data is not None:
        roots.append(data)
    return roots


print("Advanced Machine Learning: environment check")
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
    import shap
    if not shap.__version__.startswith("0.53"):
        notes.append(f"shap is {shap.__version__}; the lessons print output from 0.53. "
                     "Plots and some messages may differ.")
except ImportError:
    pass
try:
    import torch
    gpu = torch.cuda.is_available()
    line("ok", "PyTorch device", "GPU found (fine; the course uses the CPU)" if gpu else "CPU only (expected)")
except ImportError:
    torch = None

# 4. Course data
print("4. Course data")
data = find_data()
if data is None or not (data / "customers_churn.csv").is_file():
    line("FIX", "data folder", "not found")
    fixes.append("Unzip the course data so that a data folder sits in your project folder:\n"
                 "       python -m zipfile -e pinecrest-ml-data.zip .\n"
                 "     then run this check from the project folder (the one that contains data/).")
    data = None
else:
    line("ok", "data folder", str(data))
    absent = [name for name in DATA_FILES if not (data / name).is_file()]
    if absent:
        line("FIX", "files missing", ", ".join(absent))
        fixes.append("Unzip pinecrest-ml-data.zip again: these files are missing from the data folder: "
                     + ", ".join(absent))
    else:
        import csv
        wrong = []
        for name, rows in DATA_FILES.items():
            with open(data / name, newline="", encoding="utf-8") as f:
                n = sum(1 for _ in csv.reader(f)) - 1      # data rows, without the header
            if n != rows:
                wrong.append(f"{name} has {n:,} rows, expected {rows:,}")
        if wrong:
            line("FIX", "row counts", "; ".join(wrong))
            fixes.append("A data file has the wrong size. Download pinecrest-ml-data.zip again and unzip it.")
        else:
            line("ok", "7 data files", "all present, row counts as expected")


# 5. Image data and pretrained weights (needed from Session 5 on; downloads with --download)
def find_images(roots):
    found = {}
    for root in roots:
        if "FashionMNIST" not in found and all((root / "FashionMNIST" / "raw" / f).is_file() for f in FASHION_FILES):
            found["FashionMNIST"] = root
        eu = root / "eurosat" / "2750"
        if "eurosat" not in found and all((eu / c).is_dir() and any((eu / c).iterdir()) for c in EUROSAT_CLASSES):
            found["eurosat"] = root
    return found


def weight_dirs(roots):
    dirs = []
    if torch is not None:
        dirs.append(pathlib.Path(torch.hub.get_dir()) / "checkpoints")   # PyTorch's own cache (TORCH_HOME)
    dirs += [r / "torch" / "hub" / "checkpoints" for r in roots]
    return dirs


def download_images(target):
    """Download whatever is missing into the course data/ folder and PyTorch's cache, as the lessons do."""
    from torchvision import datasets, models
    print(f"  Downloading into {target} (about 175 MB; a few minutes on a good connection) ...")
    datasets.FashionMNIST(str(target), train=True, download=True)
    datasets.FashionMNIST(str(target), train=False, download=True)
    datasets.EuroSAT(str(target), download=True)
    models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    print("  Download finished.")


print("5. Image data and pretrained weights (Lesson 9.3 and Module 10)")
roots = image_roots(data)
if DOWNLOAD and not OFFLINE and torch is not None and data is not None:
    try:
        download_images(data)
    except Exception as err:            # network trouble: report it and carry on with the check
        line("note", "download failed", f"{type(err).__name__}: {err}"[:80])
found = find_images(roots)
for name, label, size in [("FashionMNIST", "Fashion-MNIST", "about 30 MB"), ("eurosat", "EuroSAT", "about 90 MB")]:
    if name in found:
        line("ok", label, str(found[name] / name))
    else:
        line("todo", label, f"not downloaded yet ({size})")
        images.append(label)
for fname, label in WEIGHTS.items():
    hit = next((d for d in weight_dirs(roots) if (d / fname).is_file()), None)
    if hit:
        line("ok", label.split(" (")[0], str(hit))
    else:
        line("todo", label.split(" (")[0], f"not downloaded yet ({label.split('(')[1].rstrip(')')})")
        images.append(label.split(" (")[0])

print("=" * 64)
for text in notes:
    print("Note:", text)
if fixes:
    print("NOT READY. Fix before Session 2:")
    for number, text in enumerate(fixes, start=1):
        print(f"  {number}. {text}")
    print("Then run this check again. Stuck? Show this output to your instructor.")
elif images:
    print("Still to download (needed from Session 5 on): " + ", ".join(images) + ".")
    print("At home, on a good connection, run this check again with --download (about 175 MB, once).")
    print("READY for Sessions 1 to 4")
else:
    print("READY for Advanced Machine Learning")
