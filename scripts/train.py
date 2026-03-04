#!/usr/bin/env python
"""Train the spam classifier with PyTorch Lightning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

VALID_MODEL_NAMES = ("embeddingbag", "lstm", "transformer")


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


def default_run_name(model_name: str, seed: int) -> str:
    """Create a deterministic default run name."""

    return f"{model_name}-seed{seed}"


def build_model_config(args: argparse.Namespace, vocab_size: int, pad_index: int) -> dict[str, Any]:
    """Serialize model settings for config.json and checkpoint reconstruction."""

    return {
        "model_name": args.model_name,
        "embed_dim": args.embed_dim,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "vocab_size": vocab_size,
        "pad_index": pad_index,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "lstm_num_layers": args.lstm_num_layers,
        "lstm_bidirectional": args.lstm_bidirectional,
        "transformer_num_layers": args.transformer_num_layers,
        "transformer_num_heads": args.transformer_num_heads,
        "transformer_ff_dim": args.transformer_ff_dim,
        "transformer_pooling": args.transformer_pooling,
        "transformer_positional_encoding": args.transformer_positional_encoding,
        "max_seq_len": args.max_seq_len,
    }


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
    parser.add_argument("--max_seq_len", type=int, default=defaults.data.max_seq_len)
    parser.add_argument("--model_name", choices=VALID_MODEL_NAMES, default=defaults.model.model_name)
    parser.add_argument("--embed_dim", type=int, default=defaults.model.embed_dim)
    parser.add_argument("--lr", type=float, default=defaults.model.learning_rate)
    parser.add_argument("--dropout", type=float, default=defaults.model.dropout)
    parser.add_argument("--lstm_hidden_dim", type=int, default=defaults.model.lstm_hidden_dim)
    parser.add_argument("--lstm_num_layers", type=int, default=defaults.model.lstm_num_layers)
    parser.add_argument("--lstm_bidirectional", dest="lstm_bidirectional", action="store_true", default=defaults.model.lstm_bidirectional)
    parser.add_argument("--lstm_unidirectional", dest="lstm_bidirectional", action="store_false")
    parser.add_argument("--transformer_num_layers", type=int, default=defaults.model.transformer_num_layers)
    parser.add_argument("--transformer_num_heads", type=int, default=defaults.model.transformer_num_heads)
    parser.add_argument("--transformer_ff_dim", type=int, default=defaults.model.transformer_ff_dim)
    parser.add_argument("--transformer_pooling", default=defaults.model.transformer_pooling)
    parser.add_argument("--transformer_positional_encoding", default=defaults.model.transformer_positional_encoding)
    parser.add_argument("--max_epochs", type=int, default=defaults.train.max_epochs)
    parser.add_argument("--precision", default=defaults.train.precision)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=defaults.train.deterministic)
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--run_name", default=defaults.train.run_name)
    parser.add_argument("--dataset", default=defaults.preprocess.dataset_slug)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    run_name = args.run_name or default_run_name(args.model_name, args.seed)
    artifacts_root = ensure_dir(Path(args.artifacts_dir).resolve())
    artifacts_dir = ensure_dir(artifacts_root / run_name)
    logs_root = ensure_dir(Path(args.logs_dir).resolve())
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
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
    )
    datamodule.setup("fit")
    datamodule.save_vocab(vocab_path)

    model = SpamLitModule(
        vocab_size=datamodule.vocab_size,
        model_name=args.model_name,
        embed_dim=args.embed_dim,
        learning_rate=args.lr,
        pad_index=datamodule.pad_index,
        dropout=args.dropout,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        lstm_bidirectional=args.lstm_bidirectional,
        transformer_num_layers=args.transformer_num_layers,
        transformer_num_heads=args.transformer_num_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_pooling=args.transformer_pooling,
        transformer_positional_encoding=args.transformer_positional_encoding,
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
        logger=CSVLogger(save_dir=str(logs_root), name="spam_lightning", version=run_name),
    )
    trainer.fit(model, datamodule=datamodule)

    stats = _load_stats(data_dir)
    config_payload = {
        "dataset_slug": args.dataset,
        "run_name": run_name,
        "model_name": args.model_name,
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
        "model": build_model_config(args, vocab_size=datamodule.vocab_size, pad_index=datamodule.pad_index),
        "trainer": {
            "max_epochs": args.max_epochs,
            "precision": args.precision,
            "deterministic": args.deterministic,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "max_seq_len": args.max_seq_len,
        },
        "artifacts": {
            "artifact_dir": str(artifacts_dir.resolve()),
            "best_ckpt": checkpoint_callback.best_model_path,
            "vocab_json": str(vocab_path.resolve()),
            "config_json": str((artifacts_dir / "config.json").resolve()),
        },
    }
    config_path = artifacts_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    logger.info("Training complete for %s. Best checkpoint: %s", run_name, checkpoint_callback.best_model_path)
    print(f"Run name: {run_name}")
    print(f"Best checkpoint path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
