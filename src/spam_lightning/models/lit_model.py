"""LightningModule for configurable spam classification backbones."""

from __future__ import annotations

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from spam_lightning.models.backbones import (
    BiLSTMBackbone,
    EmbeddingBagBackbone,
    TransformerEncoderBackbone,
)


class SpamLitModule(L.LightningModule):
    """A configurable spam classifier with multiple text backbones."""

    def __init__(
        self,
        vocab_size: int,
        model_name: str = "embeddingbag",
        embed_dim: int = 64,
        learning_rate: float = 1e-3,
        pad_index: int = 0,
        dropout: float = 0.1,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = True,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 4,
        transformer_ff_dim: int = 128,
        transformer_pooling: str = "mean",
        transformer_positional_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name

        if model_name == "embeddingbag":
            self.backbone = EmbeddingBagBackbone(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                pad_index=pad_index,
            )
        elif model_name == "lstm":
            self.backbone = BiLSTMBackbone(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                pad_index=pad_index,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                bidirectional=lstm_bidirectional,
                dropout=dropout,
            )
        elif model_name == "transformer":
            self.backbone = TransformerEncoderBackbone(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                pad_index=pad_index,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                ff_dim=transformer_ff_dim,
                dropout=dropout,
                pooling=transformer_pooling,
                positional_encoding=transformer_positional_encoding,
            )
        else:
            raise ValueError(f"Unsupported model_name '{model_name}'.")

        self.classifier = nn.Linear(self.backbone.output_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the selected backbone and return logits."""

        if self.model_name == "embeddingbag":
            features = self.backbone(batch["tokens"], batch["offsets"])
        elif self.model_name == "lstm":
            features = self.backbone(batch["input_ids"], batch["lengths"])
        else:
            features = self.backbone(batch["input_ids"], batch["attention_mask"])
        return self.classifier(features).squeeze(-1)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        labels = batch["labels"]
        logits = self(batch)
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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
