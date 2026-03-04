"""Backbone modules for the spam classifier demo."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embed_dim)
        )
        encoding = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class EmbeddingBagBackbone(nn.Module):
    """EmbeddingBag baseline that produces one vector per email."""

    def __init__(self, vocab_size: int, embed_dim: int, pad_index: int) -> None:
        super().__init__()
        self.output_dim = embed_dim
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            mode="mean",
            padding_idx=pad_index,
        )

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens, offsets)


class BiLSTMBackbone(nn.Module):
    """Sequence encoder using an LSTM over token embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_index: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        if self.bidirectional:
            features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            features = hidden[-1]
        return self.dropout(features)


class TransformerEncoderBackbone(nn.Module):
    """Lightweight Transformer encoder with masked mean pooling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_index: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        pooling: str,
        positional_encoding: str,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, received embed_dim={embed_dim}, num_heads={num_heads}."
            )
        if pooling != "mean":
            raise ValueError(f"Unsupported transformer pooling '{pooling}'. Only 'mean' is supported.")
        if positional_encoding != "sinusoidal":
            raise ValueError(
                f"Unsupported positional encoding '{positional_encoding}'. Only 'sinusoidal' is supported."
            )

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.position = SinusoidalPositionalEncoding(embed_dim=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = embed_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        embedded = self.position(embedded)
        key_padding_mask = ~attention_mask.bool()
        encoded = self.encoder(embedded, src_key_padding_mask=key_padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.dropout(pooled)
