#!/usr/bin/env python
"""Evaluate a trained spam classifier checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as L

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spam_lightning.config import ProjectConfig
from spam_lightning.data.datamodule import SpamDataModule
from spam_lightning.models.lit_model import SpamLitModule


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    defaults = ProjectConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to artifacts/best.ckpt")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / defaults.paths.processed_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    artifacts_dir = ckpt_path.parent
    config = _load_json(artifacts_dir / "config.json")
    vocab_path = Path(config["artifacts"]["vocab_json"]).resolve()
    data_dir = Path(args.data_dir).resolve()

    datamodule = SpamDataModule(
        data_dir=data_dir,
        batch_size=int(config["trainer"]["batch_size"]),
        num_workers=int(config["trainer"]["num_workers"]),
        pin_memory=bool(config["trainer"]["pin_memory"]),
        lowercase=bool(config["lowercase"]),
        min_freq=int(config["vocab"]["min_freq"]),
        max_vocab_size=int(config["vocab"]["max_vocab_size"]),
        vocab_path=vocab_path,
    )
    datamodule.setup("test")

    model = SpamLitModule.load_from_checkpoint(str(ckpt_path))
    trainer = L.Trainer(accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
    results = trainer.test(model, datamodule=datamodule)
    metrics = results[0] if results else {}

    print("Test metrics")
    print(f"  accuracy: {metrics.get('test_acc')}")
    print(f"  f1: {metrics.get('test_f1')}")


if __name__ == "__main__":
    main()
