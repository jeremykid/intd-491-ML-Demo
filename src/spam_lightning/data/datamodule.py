"""PyTorch Lightning DataModule for the spam classification demo."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from spam_lightning.data.text_utils import Vocab, build_vocab, load_vocab, regex_tokenize, save_vocab

try:
    import pytorch_lightning as L
except ImportError:  # pragma: no cover - fallback for environments using lightning package name
    import lightning as L  # type: ignore[no-redef]


VALID_MODEL_NAMES = {"embeddingbag", "lstm", "transformer"}


class SpamExample(NamedTuple):
    """A single tokenized example."""

    token_ids: list[int]
    label: int


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
            ids = [vocab.lookup_index(token) for token in tokens] if tokens else [vocab.unk_index]
            self.examples.append(SpamExample(token_ids=ids, label=int(row["label"])))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SpamExample:
        return self.examples[index]


class SpamDataModule(L.LightningDataModule):
    """Lightning DataModule that wraps tokenization, vocab, and batching."""

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
        model_name: str = "embeddingbag",
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        if model_name not in VALID_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Expected one of {sorted(VALID_MODEL_NAMES)}."
            )

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path
        self.model_name = model_name
        self.max_seq_len = max_seq_len

        self._vocab: Optional[Vocab] = None
        self.train_ds: Optional[SpamDataset] = None
        self.val_ds: Optional[SpamDataset] = None
        self.test_ds: Optional[SpamDataset] = None

    @property
    def vocab(self) -> Vocab:
        if self._vocab is None:
            raise RuntimeError("Vocab not built yet. Call .setup() first.")
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_index(self) -> int:
        if self._vocab is None:
            return 0
        return self.vocab.pad_index

    def setup(self, stage: Optional[str] = None) -> None:
        """Build or load the vocab and initialize split datasets."""

        if self._vocab is None:
            if self.vocab_path is not None and Path(self.vocab_path).exists():
                self._vocab = load_vocab(Path(self.vocab_path))
            else:
                train_df = pd.read_csv(self.data_dir / "train.csv")
                sequences = [regex_tokenize(str(text), lowercase=self.lowercase) for text in train_df["text"]]
                self._vocab = build_vocab(sequences, min_freq=self.min_freq, max_size=self.max_vocab_size)

        if stage in (None, "fit"):
            self.train_ds = SpamDataset(self.data_dir / "train.csv", self.vocab, self.lowercase)
            self.val_ds = SpamDataset(self.data_dir / "val.csv", self.vocab, self.lowercase)
        if stage in (None, "test", "predict"):
            self.test_ds = SpamDataset(self.data_dir / "test.csv", self.vocab, self.lowercase)

    def collate_bag_batch(self, batch: list[SpamExample]) -> dict[str, torch.Tensor]:
        """Collate a batch for the EmbeddingBag baseline."""

        all_ids: list[int] = []
        offsets: list[int] = []
        labels: list[float] = []

        for example in batch:
            offsets.append(len(all_ids))
            all_ids.extend(example.token_ids)
            labels.append(float(example.label))

        return {
            "tokens": torch.tensor(all_ids, dtype=torch.long),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    def collate_sequence_batch(self, batch: list[SpamExample]) -> dict[str, torch.Tensor]:
        """Collate a padded batch for LSTM and Transformer backbones."""

        truncated = [example.token_ids[: self.max_seq_len] or [self.vocab.unk_index] for example in batch]
        lengths = [len(token_ids) for token_ids in truncated]
        batch_max_len = max(lengths)
        input_ids: list[list[int]] = []
        masks: list[list[int]] = []
        labels: list[float] = []

        for token_ids, example, length in zip(truncated, batch, lengths):
            pad_len = batch_max_len - length
            input_ids.append(token_ids + [self.pad_index] * pad_len)
            masks.append([1] * length + [0] * pad_len)
            labels.append(float(example.label))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    def collate_batch(self, batch: list[SpamExample]) -> dict[str, torch.Tensor]:
        """Dispatch to the correct collator for the selected model type."""

        if self.model_name == "embeddingbag":
            return self.collate_bag_batch(batch)
        return self.collate_sequence_batch(batch)

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

    def save_vocab(self, path: Path) -> None:
        """Save the current vocabulary to disk."""

        save_vocab(self.vocab, path)
