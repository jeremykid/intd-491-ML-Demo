#!/usr/bin/env python
"""Train the spam classifier with PyTorch Lightning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spam_lightning.config import ProjectConfig
from spam_lightning.data.datamodule import SpamDataModule
from spam_lightning.models.lit_model import SpamLitModule
from spam_lightning.utils.logging import configure_logging
from spam_lightning.utils.paths import ensure_dir
from spam_lightning.utils.seed import set_global_seed

logger = configure_logging()


def _load_stats(data_dir: Path) -> dict:
    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    return {}


def _require_processed_data(data_dir: Path) -> None:
    for split in ("train.csv", "val.csv", "test.csv"):
        path = data_dir / split
        if not path.exists():
            raise FileNotFoundError(f"Missing processed data file: {path}")


def parse_args() -> argparse.Namespace:
    defaults = ProjectConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / defaults.paths.processed_dir))
    parser.add_argument("--artifacts_dir", default=str(PROJECT_ROOT / defaults.paths.artifacts_dir))
    parser.add_argument("--logs_dir", default=str(PROJECT_ROOT / defaults.paths.logs_dir))
    parser.add_argument("--seed", type=int, default=defaults.preprocess.seed)
    parser.add_argument("--batch_size", type=int, default=defaults.data.batch_size)
    parser.add_argument("--num_workers", type=int, default=defaults.data.num_workers)
    parser.add_argument("--pin_memory", action="store_true", default=defaults.data.pin_memory)
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", default=defaults.preprocess.lowercase)
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    parser.add_argument("--min_freq", type=int, default=defaults.data.min_freq)
    parser.add_argument("--max_vocab_size", type=int, default=defaults.data.max_vocab_size)
    parser.add_argument("--embed_dim", type=int, default=defaults.model.embed_dim)
    parser.add_argument("--lr", type=float, default=defaults.model.learning_rate)
    parser.add_argument("--max_epochs", type=int, default=defaults.train.max_epochs)
    parser.add_argument("--precision", default=defaults.train.precision)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=defaults.train.deterministic)
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--dataset", default=defaults.preprocess.dataset_slug)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    artifacts_dir = ensure_dir(Path(args.artifacts_dir).resolve())
    logs_dir = ensure_dir(Path(args.logs_dir).resolve())
    _require_processed_data(data_dir)

    set_global_seed(args.seed)

    vocab_path = artifacts_dir / "vocab.json"
    datamodule = SpamDataModule(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        lowercase=args.lowercase,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    datamodule.setup("fit")
    datamodule.save_vocab(vocab_path)

    model = SpamLitModule(
        vocab_size=datamodule.vocab_size,
        embed_dim=args.embed_dim,
        learning_rate=args.lr,
        pad_index=datamodule.pad_index,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best",
        dirpath=str(artifacts_dir),
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        deterministic=args.deterministic,
        precision=args.precision,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_f1", mode="max", patience=3),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=CSVLogger(save_dir=str(logs_dir), name="spam_lightning"),
    )
    trainer.fit(model, datamodule=datamodule)

    stats = _load_stats(data_dir)
    config_payload = {
        "dataset_slug": args.dataset,
        "selected_raw_file": stats.get("source_file"),
        "text_col": stats.get("text_col"),
        "label_col": stats.get("label_col"),
        "label_mapping": stats.get("label_mapping", {}),
        "split_ratios": stats.get("split_ratios"),
        "seed": args.seed,
        "lowercase": args.lowercase,
        "vocab": {
            "min_freq": args.min_freq,
            "max_vocab_size": args.max_vocab_size,
            "vocab_path": str(vocab_path.resolve()),
        },
        "model": {
            "embed_dim": args.embed_dim,
            "learning_rate": args.lr,
            "vocab_size": datamodule.vocab_size,
            "pad_index": datamodule.pad_index,
        },
        "trainer": {
            "max_epochs": args.max_epochs,
            "precision": args.precision,
            "deterministic": args.deterministic,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
        },
        "artifacts": {
            "best_ckpt": checkpoint_callback.best_model_path,
            "vocab_json": str(vocab_path.resolve()),
            "config_json": str((artifacts_dir / "config.json").resolve()),
        },
    }
    config_path = artifacts_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    logger.info("Training complete. Best checkpoint: %s", checkpoint_callback.best_model_path)
    print(f"Best checkpoint path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
