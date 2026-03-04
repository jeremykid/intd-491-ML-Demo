import pytest
import torch

from spam_lightning.models.backbones import (
    BiLSTMBackbone,
    EmbeddingBagBackbone,
    TransformerEncoderBackbone,
)
from spam_lightning.models.lit_model import SpamLitModule


def test_embeddingbag_backbone_output_shape() -> None:
    backbone = EmbeddingBagBackbone(vocab_size=20, embed_dim=8, pad_index=0)
    features = backbone(
        tokens=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        offsets=torch.tensor([0, 3], dtype=torch.long),
    )
    assert tuple(features.shape) == (2, 8)


def test_bilstm_backbone_output_shape() -> None:
    backbone = BiLSTMBackbone(
        vocab_size=20,
        embed_dim=8,
        pad_index=0,
        hidden_dim=6,
        num_layers=1,
        bidirectional=True,
        dropout=0.1,
    )
    features = backbone(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        lengths=torch.tensor([3, 2], dtype=torch.long),
    )
    assert tuple(features.shape) == (2, 12)


def test_transformer_backbone_output_shape() -> None:
    backbone = TransformerEncoderBackbone(
        vocab_size=20,
        embed_dim=8,
        pad_index=0,
        num_layers=2,
        num_heads=2,
        ff_dim=16,
        dropout=0.1,
        pooling="mean",
        positional_encoding="sinusoidal",
    )
    features = backbone(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    )
    assert tuple(features.shape) == (2, 8)


def test_transformer_rejects_invalid_head_count() -> None:
    with pytest.raises(ValueError):
        TransformerEncoderBackbone(
            vocab_size=20,
            embed_dim=10,
            pad_index=0,
            num_layers=2,
            num_heads=4,
            ff_dim=16,
            dropout=0.1,
            pooling="mean",
            positional_encoding="sinusoidal",
        )


def test_lightning_module_accepts_sequence_batch() -> None:
    model = SpamLitModule(vocab_size=20, model_name="transformer", embed_dim=8, transformer_num_heads=2)
    logits = model(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
            "lengths": torch.tensor([3, 2], dtype=torch.long),
            "labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }
    )
    assert tuple(logits.shape) == (2,)
