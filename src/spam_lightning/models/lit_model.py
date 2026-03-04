"""LightningModule for EmbeddingBag-based spam classification."""

from __future__ import annotations

import torch
import pytorch_lightning as L
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


class SpamLitModule(L.LightningModule):
    """A lightweight spam classifier for classroom demos."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        learning_rate: float = 1e-3,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            mode="mean",
            padding_idx=pad_index,
        )
        self.classifier = nn.Linear(embed_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Run the EmbeddingBag encoder and return a flat logit tensor."""

        embeddings = self.embedding(tokens, offsets)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        tokens, offsets, labels = batch
        logits = self(tokens, offsets)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        labels_int = labels.int()

        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric = getattr(self, f"{stage}_f1")
        acc_metric.update(probs, labels_int)
        f1_metric.update(probs, labels_int)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        self.log(
            f"{stage}_acc",
            acc_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        self.log(
            f"{stage}_f1",
            f1_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam for the classifier."""

        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
