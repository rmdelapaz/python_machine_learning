"""labs_paths: where the course data lives, where a lab may write files, and where the image data is.

Every lab starts with the same bootstrap (see labs/README.md), which puts this folder on
sys.path, then:

    from labs_paths import DATA                         # pathlib.Path to the folder holding customers_churn.csv
    from labs_paths import DATA, OUTPUT, OFFLINE

    customers = pd.read_csv(DATA / "customers_churn.csv")
    fig.savefig(OUTPUT / "risk_deciles.png")

DATA      The Pinecrest machine-learning data folder (customers_churn.csv and six more CSV files).
          Found, in this order:
            1. the environment variable PINECREST_ML_DATA, if it is set;
            2. a `data/customers_churn.csv` in the lab's own folder or any folder above it
               (also tried: the current folder and this file's folder, and the folders above them).
               Inside the course repository that is the course's own data/ folder;
            3. otherwise a FileNotFoundError that says what to do.
          Labs only READ from DATA.
OUTPUT    A folder the lab may write to. Normally `output/` next to the lab file (created the first
          time it is used). When LABS_OFFLINE=1 it is a temporary folder that is deleted when the
          program ends, so the lab checker never leaves files behind.
OFFLINE   True when the environment variable LABS_OFFLINE is 1 (also true/yes/on). Then no lab
          downloads anything (image datasets, pretrained weights) and no lab opens a window.
image_root(name)
          The folder to pass as `root=` to a torchvision dataset, without downloading anything:
              from labs_paths import image_root, OFFLINE
              train = datasets.FashionMNIST(image_root("FashionMNIST"), train=True,
                                            download=not OFFLINE, transform=transforms.ToTensor())
          `name` is the dataset's own folder name: "FashionMNIST" or "eurosat". The first folder that
          already holds <name>/ wins: $ML_DATA_DIR, ~/.cache/ml-course-data (the instructor's shared
          cache), then a data/ folder next to the course data (where the lessons' root="data" puts the
          download on a learner's machine). When none has it, the data/ folder is returned, so a
          learner who is online downloads into the same place the lessons do; offline, a clear
          FileNotFoundError says how to fetch it.
use_cached_weights()
          Points TORCH_HOME at ~/.cache/ml-course-data/torch (or $ML_DATA_DIR/torch) when pretrained
          torchvision weights are already there, so `models.resnet18(weights=...)` loads without a
          download. Returns the folder it used, or None (then PyTorch uses its normal cache). Call it
          before creating a pretrained model. Offline with no cached weights it raises a clear error.
fixtures(start=None)
          The nearest `_fixtures/` folder at or above `start` (a file or folder; default: the lab file).
          Use it for small saved inputs a lab needs offline (saved features, a saved model, JSON).

DATA and OUTPUT are worked out on first use, so `from labs_paths import OFFLINE` alone never needs
the data folder and never creates a folder.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

__all__ = ["DATA", "OUTPUT", "OFFLINE", "image_root", "use_cached_weights", "fixtures"]

LABS_DIR = Path(__file__).resolve().parent
MARKER = Path("data") / "customers_churn.csv"
ENV_DATA = "PINECREST_ML_DATA"
SHARED_CACHE = Path.home() / ".cache" / "ml-course-data"

OFFLINE = os.environ.get("LABS_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")

_cache: dict[str, Path] = {}


def _lab_dir() -> Path:
    """Folder of the program being run (the lab file); the current folder in a notebook or under pytest."""
    main = sys.modules.get("__main__")
    f = getattr(main, "__file__", None)
    if f and Path(f).suffix == ".py" and Path(f).name not in ("__main__.py",):
        p = Path(f).resolve().parent
        # `python -m pytest` and friends: __main__ is a tool inside site-packages, not a lab
        if "site-packages" not in p.parts and "dist-packages" not in p.parts:
            return p
    return Path.cwd().resolve()


def _walk_up(start: Path):
    start = start.resolve()
    yield start
    yield from start.parents


def _find_data() -> Path:
    env = os.environ.get(ENV_DATA, "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "customers_churn.csv").is_file():
            return p
        raise FileNotFoundError(
            f"\n\n{ENV_DATA} is set to {env!r}, but there is no customers_churn.csv in that folder.\n"
            "Point it at the folder that holds customers_churn.csv and the other course CSV files,\n"
            "or remove the variable so the labs can look for a data/ folder themselves.\n")
    tried = []
    for start in (_lab_dir(), Path.cwd(), LABS_DIR):
        for folder in _walk_up(start):
            if folder in tried:
                continue
            tried.append(folder)
            if (folder / MARKER).is_file():
                return (folder / "data").resolve()
    raise FileNotFoundError(
        "\n\nThe Pinecrest machine-learning data was not found.\n"
        "Fix it in one of two ways:\n"
        "  1. Unzip pinecrest-ml-data.zip next to the labs folder, so that this file exists:\n"
        f"       {LABS_DIR.parent / MARKER}\n"
        f"  2. Or set the environment variable {ENV_DATA} to the folder that holds customers_churn.csv:\n"
        f"       macOS / Linux:  export {ENV_DATA}=/path/to/data\n"
        f"       PowerShell:     $env:{ENV_DATA} = \"C:\\path\\to\\data\"\n"
        f"Looked for data/customers_churn.csv in {_lab_dir()} and every folder above it.\n")


def _make_output() -> Path:
    if OFFLINE:
        p = Path(tempfile.mkdtemp(prefix="pinecrest-ml-labs-"))
        atexit.register(shutil.rmtree, p, ignore_errors=True)
        return p
    p = _lab_dir() / "output"
    p.mkdir(exist_ok=True)
    return p


def _cache_roots() -> list[Path]:
    roots = []
    env = os.environ.get("ML_DATA_DIR", "").strip()
    if env:
        roots.append(Path(env).expanduser())
    roots.append(SHARED_CACHE)
    return roots


def image_root(name: str = "FashionMNIST") -> Path:
    """Folder to pass as root= to a torchvision dataset whose files live in <root>/<name>/."""
    candidates = _cache_roots()
    try:
        candidates.append(__getattr__("DATA"))
    except FileNotFoundError:
        pass
    for root in candidates:
        if (root / name).is_dir() and any((root / name).iterdir()):
            return root.resolve()
    fallback = candidates[-1]
    if OFFLINE:
        raise FileNotFoundError(
            f"\n\nThe image dataset {name!r} is not on this computer, and LABS_OFFLINE=1 forbids downloading it.\n"
            f"Looked in: {', '.join(str(c) for c in candidates)}.\n"
            "Run the lesson that uses it once while online (it downloads into data/), or unset LABS_OFFLINE\n"
            "and run this lab again: it will download into the data/ folder.\n")
    return fallback.resolve()


def use_cached_weights() -> Path | None:
    """Point TORCH_HOME at a cache that already holds pretrained weights; return it (or None)."""
    if os.environ.get("TORCH_HOME") and (Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints").is_dir():
        return Path(os.environ["TORCH_HOME"])
    for root in _cache_roots():
        home = root / "torch"
        if (home / "hub" / "checkpoints").is_dir() and any((home / "hub" / "checkpoints").glob("*.pth")):
            os.environ["TORCH_HOME"] = str(home)
            return home
    if OFFLINE:
        raise FileNotFoundError(
            "\n\nNo cached pretrained weights were found and LABS_OFFLINE=1 forbids downloading them.\n"
            "Run the transfer-learning lesson once while online (PyTorch keeps the weights in its cache),\n"
            "or unset LABS_OFFLINE and run this lab again.\n")
    return None


def fixtures(start: str | os.PathLike | None = None) -> Path:
    """Nearest `_fixtures/` folder at or above `start` (default: the lab file's folder)."""
    base = Path(start).resolve() if start else _lab_dir()
    if base.is_file():
        base = base.parent
    for folder in _walk_up(base):
        if (folder / "_fixtures").is_dir():
            return folder / "_fixtures"
    raise FileNotFoundError(
        f"\n\nNo _fixtures folder was found at or above {base}.\n"
        "Labs keep small saved inputs in labs/module-NN/_fixtures/.\n")


def __getattr__(name: str):
    # DATA and OUTPUT are computed on first use (PEP 562), then remembered.
    if name == "DATA":
        if "DATA" not in _cache:
            _cache["DATA"] = _find_data()
        return _cache["DATA"]
    if name == "OUTPUT":
        if "OUTPUT" not in _cache:
            _cache["OUTPUT"] = _make_output()
        return _cache["OUTPUT"]
    raise AttributeError(f"module 'labs_paths' has no attribute {name!r}")


if __name__ == "__main__":
    print("labs folder :", LABS_DIR)
    print("OFFLINE     :", OFFLINE)
    try:
        print("DATA        :", __getattr__("DATA"))
    except FileNotFoundError as e:
        print("DATA        : NOT FOUND", e)
        sys.exit(1)
    for ds in ("FashionMNIST", "eurosat"):
        try:
            print(f"{ds:<12}:", image_root(ds))
        except FileNotFoundError:
            print(f"{ds:<12}: not on this computer (downloads on first use)")
    try:
        print("weights     :", use_cached_weights() or "PyTorch's default cache")
    except FileNotFoundError:
        print("weights     : none cached")
