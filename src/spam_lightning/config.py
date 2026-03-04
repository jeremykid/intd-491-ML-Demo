"""Project configuration defaults for the spam classification demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

TEXT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "text",
    "message",
    "email",
    "content",
    "body",
    "v2",
)

LABEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "label",
    "spam",
    "target",
    "category",
    "v1",
    "class",
)

DEFAULT_LABEL_SYNONYMS: dict[str, int] = {
    "0": 0,
    "0.0": 0,
    "1": 1,
    "1.0": 1,
    "false": 0,
    "ham": 0,
    "legit": 0,
    "negative": 0,
    "no": 0,
    "nonspam": 0,
    "notspam": 0,
    "safe": 0,
    "true": 1,
    "junk": 1,
    "positive": 1,
    "spam": 1,
    "yes": 1,
}


@dataclass
class PathConfig:
    """Default project paths relative to the repo root."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"


@dataclass
class PreprocessConfig:
    """Preprocessing defaults."""

    dataset_slug: str = "harshsinha1234/email-spam-classification"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    lowercase: bool = True
    text_col: Optional[str] = None
    label_col: Optional[str] = None
    label_map: dict[str, int] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data loading defaults."""

    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    min_freq: int = 2
    max_vocab_size: int = 20000


@dataclass
class ModelConfig:
    """Model defaults."""

    embed_dim: int = 64
    learning_rate: float = 1e-3


@dataclass
class TrainConfig:
    """Training defaults."""

    max_epochs: int = 5
    precision: str = "32-true"
    deterministic: bool = True


@dataclass
class ProjectConfig:
    """Aggregate project config used by scripts and notebooks."""

    paths: PathConfig = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full config tree into a JSON-friendly dictionary."""

        return asdict(self)

    def resolve_path(self, root: Path, relative_path: str) -> Path:
        """Resolve a configured path against a repo root."""

        return (root / relative_path).resolve()
