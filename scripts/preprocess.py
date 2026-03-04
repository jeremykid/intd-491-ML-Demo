#!/usr/bin/env python
"""Preprocess a Kaggle spam dataset into train/val/test CSV splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spam_lightning.config import ProjectConfig
from spam_lightning.data.preprocessing import parse_label_map_items, preprocess_dataset
from spam_lightning.utils.logging import configure_logging

logger = configure_logging()


def parse_args() -> argparse.Namespace:
    defaults = ProjectConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default=str(PROJECT_ROOT / defaults.paths.raw_dir))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / defaults.paths.processed_dir))
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--text_col", default=defaults.preprocess.text_col)
    parser.add_argument("--label_col", default=defaults.preprocess.label_col)
    parser.add_argument("--label_map", nargs="*", default=None, help="Mappings like spam=1 ham=0")
    parser.add_argument("--dataset", default=defaults.preprocess.dataset_slug)
    parser.add_argument("--seed", type=int, default=defaults.preprocess.seed)
    parser.add_argument("--train_ratio", type=float, default=defaults.preprocess.train_ratio)
    parser.add_argument("--val_ratio", type=float, default=defaults.preprocess.val_ratio)
    parser.add_argument("--test_ratio", type=float, default=defaults.preprocess.test_ratio)
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", default=defaults.preprocess.lowercase)
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_map = parse_label_map_items(args.label_map)
    result = preprocess_dataset(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        input_csv=Path(args.input_csv) if args.input_csv else None,
        text_col=args.text_col,
        label_col=args.label_col,
        label_map=label_map,
        lowercase=args.lowercase,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dataset_slug=args.dataset,
    )

    logger.info("Wrote processed splits to %s", Path(args.out_dir).resolve())
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
