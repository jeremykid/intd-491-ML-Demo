from argparse import Namespace
from pathlib import Path

from scripts.evaluate import build_datamodule_from_config
from scripts.predict import build_inference_batch
from scripts.train import build_model_config, default_run_name
from spam_lightning.data.text_utils import Vocab


def test_default_run_name_includes_model_and_seed() -> None:
    assert default_run_name("transformer", 42) == "transformer-seed42"


def test_build_model_config_includes_model_name_and_seq_settings() -> None:
    args = Namespace(
        model_name="lstm",
        embed_dim=64,
        lr=1e-3,
        dropout=0.1,
        lstm_hidden_dim=32,
        lstm_num_layers=1,
        lstm_bidirectional=True,
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_ff_dim=128,
        transformer_pooling="mean",
        transformer_positional_encoding="sinusoidal",
        max_seq_len=256,
    )

    config = build_model_config(args, vocab_size=123, pad_index=0)

    assert config["model_name"] == "lstm"
    assert config["max_seq_len"] == 256
    assert config["vocab_size"] == 123


def test_build_datamodule_from_config_restores_model_name(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    config = {
        "model_name": "transformer",
        "lowercase": True,
        "vocab": {"min_freq": 2, "max_vocab_size": 100},
        "trainer": {"batch_size": 8, "num_workers": 0, "pin_memory": False, "max_seq_len": 32},
        "model": {"max_seq_len": 32},
    }

    datamodule = build_datamodule_from_config(tmp_path, config, vocab_path)

    assert datamodule.model_name == "transformer"
    assert datamodule.max_seq_len == 32


def test_build_inference_batch_switches_by_model_name() -> None:
    vocab = Vocab(token_to_idx={"<pad>": 0, "<unk>": 1, "free": 2}, idx_to_token={0: "<pad>", 1: "<unk>", 2: "free"})

    bag_batch = build_inference_batch(
        "free free",
        config={"model_name": "embeddingbag", "lowercase": True, "trainer": {}, "model": {}},
        vocab=vocab,
    )
    seq_batch = build_inference_batch(
        "free free",
        config={"model_name": "lstm", "lowercase": True, "trainer": {"max_seq_len": 4}, "model": {}},
        vocab=vocab,
    )

    assert "tokens" in bag_batch and "offsets" in bag_batch
    assert "input_ids" in seq_batch and "attention_mask" in seq_batch and "lengths" in seq_batch
