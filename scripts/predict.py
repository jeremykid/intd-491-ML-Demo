#!/usr/bin/env python
"""Run single-text spam prediction with a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spam_lightning.data.text_utils import load_vocab, regex_tokenize
from spam_lightning.models.lit_model import SpamLitModule
from spam_lightning.data.preprocessing import clean_text


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to artifacts/best.ckpt")
    parser.add_argument("--text", required=True, help="A single email text to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    artifacts_dir = ckpt_path.parent
    config = _load_json(artifacts_dir / "config.json")
    vocab = load_vocab(Path(config["artifacts"]["vocab_json"]).resolve())
    model = SpamLitModule.load_from_checkpoint(str(ckpt_path))
    model.eval()

    normalized = clean_text(args.text, lowercase=bool(config["lowercase"]))
    tokens = regex_tokenize(normalized, lowercase=False) or ["<unk>"]
    token_ids = [vocab.lookup_index(token) for token in tokens]
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        logits = model(token_tensor, offsets)
        probability = torch.sigmoid(logits).item()

    predicted_label = "spam" if probability >= 0.5 else "ham"
    print(f"Spam probability: {probability:.4f}")
    print(f"Predicted label: {predicted_label}")


if __name__ == "__main__":
    main()
