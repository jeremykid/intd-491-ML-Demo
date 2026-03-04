"""PyTorch Lightning DataModule for the spam classification demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from spam_lightning.data.text_utils import (
    Vocab,
    build_vocab,
    load_vocab,
    regex_tokenize,
    save_vocab,
)

try:
    import pytorch_lightning as L
except ImportError:  # allow importing for tests without Lightning installed
    import lightning as L  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


class SpamExample(NamedTuple):
    """A single tokenised example."""

    token_ids: list[int]
    label: int


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SpamDataset(Dataset[SpamExample]):
    """In-memory dataset that reads a processed CSV split."""

    def __init__(
        self,
        csv_path: Path,
        vocab: Vocab,
        lowercase: bool = True,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.examples: list[SpamExample] = []
        for _, row in df.iterrows():
            tokens = regex_tokenize(str(row["text"]), lowercase=lowercase)
            ids = [vocab.lookup_index(t) for t in tokens] if tokens else [vocab.unk_index]
            self.examples.append(SpamExample(token_ids=ids, label=int(row["label"])))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SpamExample:
        return self.examples[index]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SpamDataModule(L.LightningDataModule):
    """Lightning DataModule that wraps tokenisation, vocab, and batching."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        lowercase: bool = True,
        min_freq: int = 2,
        max_vocab_size: int = 20_000,
        vocab_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path

        self._vocab: Optional[Vocab] = None
        self.train_ds: Optional[SpamDataset] = None
        self.val_ds: Optional[SpamDataset] = None
        self.test_ds: Optional[SpamDataset] = None

    # -- vocab properties ----------------------------------------------------

    @property
    def vocab(self) -> Vocab:
        if self._vocab is None:
            raise RuntimeError("Vocab not built yet — call .setup('fit') first.")
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_index(self) -> int:
        return self.vocab.pad_index

    # -- setup ---------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        # Build or load vocab from training data
        if self._vocab is None:
            if self.vocab_path is not None and Path(self.vocab_path).exists():
                self._vocab = load_vocab(Path(self.vocab_path))
            else:
                train_df = pd.read_csv(self.data_dir / "train.csv")
                sequences = [
                    regex_tokenize(str(text), lowercase=self.lowercase)
                    for text in train_df["text"]
                ]
                self._vocab = build_vocab(
                    sequences,
                    min_freq=self.min_freq,
                    max_size=self.max_vocab_size,
                )

        if stage in (None, "fit"):
            self.train_ds = SpamDataset(self.data_dir / "train.csv", self.vocab, self.lowercase)
            self.val_ds = SpamDataset(self.data_dir / "val.csv", self.vocab, self.lowercase)
        if stage in (None, "test"):
            self.test_ds = SpamDataset(self.data_dir / "test.csv", self.vocab, self.lowercase)

    # -- collate for EmbeddingBag -------------------------------------------

    def collate_batch(
        self, batch: list[SpamExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate a list of :class:`SpamExample` into EmbeddingBag tensors."""
        all_ids: list[int] = []
        offsets: list[int] = []
        labels: list[float] = []
        for example in batch:
            offsets.append(len(all_ids))
            all_ids.extend(example.token_ids)
            labels.append(float(example.label))
        return (
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float),
        )

    # -- dataloaders ---------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_batch,
        )

    # -- vocab persistence ---------------------------------------------------

    def save_vocab(self, path: Path) -> None:
        """Save the current vocabulary to a JSON file."""
        save_vocab(self.vocab, path)
