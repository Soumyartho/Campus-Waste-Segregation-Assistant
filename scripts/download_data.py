"""Download the Kaggle 'Garbage Classification' dataset (and optionally TrashNet).

Uses the Kaggle API when credentials exist; otherwise prints manual download steps.
Usage: python scripts/download_data.py [--trashnet] [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste.utils import load_yaml, resolve_path  # noqa: E402


def has_images(folder: Path) -> bool:
    """True when the folder already contains at least one image."""
    return folder.exists() and any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in folder.rglob("*")
    )


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing members that would escape ``dest``."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (dest / member).resolve().is_relative_to(dest):
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(dest)


def manual_steps(slug: str, raw_dir: Path) -> str:
    return f"""
Kaggle download did not run. Do ONE of the following, then re-run 'make data':

  A) Set up the Kaggle API (recommended)
     1. Sign in at https://www.kaggle.com -> Settings -> API -> 'Create New Token'.
     2. Save the downloaded kaggle.json to  %USERPROFILE%\\.kaggle\\kaggle.json  (Windows)
        or ~/.kaggle/kaggle.json (Linux/macOS).
     3. Re-run: python scripts/download_data.py

  B) Download manually
     1. Open https://www.kaggle.com/datasets/{slug}
     2. Click 'Download' and unzip the archive into:
        {raw_dir}
     The class folders (battery, biological, brown-glass, ...) can be at any depth below it.
"""


def download_kaggle(slug: str, raw_dir: Path) -> bool:
    """Download and unzip a Kaggle dataset with the kaggle CLI; returns success."""
    if has_images(raw_dir):
        print(f"Kaggle dataset already present in {raw_dir}")
        return True
    if shutil.which("kaggle") is None and not _kaggle_module_available():
        return False
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug,
           "-p", str(raw_dir), "--unzip"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0 and has_images(raw_dir)


def _kaggle_module_available() -> bool:
    try:
        import kaggle  # noqa: F401
    except Exception:
        return False
    return True


def download_trashnet(url: str, dest: Path) -> None:
    """Download the resized TrashNet archive from GitHub and extract it."""
    if has_images(dest):
        print(f"TrashNet already present in {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "trashnet.zip"
    print(f"Downloading TrashNet from {url} ...")
    urllib.request.urlretrieve(url, archive)
    safe_extract(archive, dest)
    archive.unlink()
    print(f"TrashNet extracted to {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--trashnet", action="store_true", help="also download TrashNet")
    args = parser.parse_args()

    src = load_yaml(args.config)["source"]
    raw_dir = resolve_path(src["raw_dir"])
    ok = download_kaggle(src["kaggle_slug"], raw_dir)
    if not ok:
        print(manual_steps(src["kaggle_slug"], raw_dir))
    if args.trashnet:
        download_trashnet(src["trashnet_url"], resolve_path(src["trashnet_dir"]))
    resolve_path(src["campus_dir"]).mkdir(parents=True, exist_ok=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
