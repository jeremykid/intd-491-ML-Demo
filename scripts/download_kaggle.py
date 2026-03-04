#!/usr/bin/env python
"""Download and extract a Kaggle dataset into `data/raw/`."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spam_lightning.config import ProjectConfig
from spam_lightning.data.preprocessing import discover_tabular_files
from spam_lightning.utils.logging import configure_logging
from spam_lightning.utils.paths import ensure_dir

logger = configure_logging()


def ensure_kaggle_credentials() -> Path:
    """Validate that `~/.kaggle/kaggle.json` exists."""

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return kaggle_json
    raise FileNotFoundError(
        "Kaggle credentials not found. Supported options:\n"
        "1) Put kaggle.json at ~/.kaggle/kaggle.json\n"
        "2) Upload kaggle.json inside the notebook, copy it to ~/.kaggle/, and chmod 600"
    )


def extract_nested_zips(root_dir: Path) -> None:
    """Recursively extract nested zip files created by some Kaggle datasets."""

    seen: set[Path] = set()
    while True:
        archives = [path for path in root_dir.rglob("*.zip") if path not in seen]
        if not archives:
            break
        for archive in archives:
            logger.info("Extracting nested archive: %s", archive)
            with zipfile.ZipFile(archive, "r") as zipped:
                zipped.extractall(archive.parent)
            seen.add(archive)


def download_dataset(dataset: str, raw_dir: Path) -> list[Path]:
    """Download a Kaggle dataset and return discovered tabular files."""

    if not dataset or dataset == "REPLACE_ME":
        raise ValueError("Provide a real Kaggle dataset slug with --dataset username/dataset-name.")

    ensure_kaggle_credentials()
    target_dir = ensure_dir(raw_dir / dataset.replace("/", "__"))

    from kaggle.api.kaggle_api_extended import KaggleApi

    logger.info("Downloading Kaggle dataset '%s' into %s", dataset, target_dir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset=dataset, path=str(target_dir), unzip=True, quiet=False)
    extract_nested_zips(target_dir)

    tabular_files = discover_tabular_files(target_dir)
    if not tabular_files:
        logger.warning("Download completed, but no CSV/TSV files were found under %s", target_dir)
    else:
        logger.info("Discovered %d tabular files:", len(tabular_files))
        for path in tabular_files:
            logger.info("  - %s", path)
    return tabular_files


def parse_args() -> argparse.Namespace:
    defaults = ProjectConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. username/spam-email")
    parser.add_argument(
        "--raw_dir",
        default=str(PROJECT_ROOT / defaults.paths.raw_dir),
        help="Directory where the dataset will be downloaded and extracted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tabular_files = download_dataset(dataset=args.dataset, raw_dir=Path(args.raw_dir))
    print("\nDiscovered tabular files:")
    for path in tabular_files:
        print(path)


if __name__ == "__main__":
    main()
