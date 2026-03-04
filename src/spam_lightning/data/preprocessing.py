"""Dataset preprocessing: column detection, label normalisation, and split writing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from spam_lightning.config import (
    DEFAULT_LABEL_SYNONYMS,
    LABEL_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
)
from spam_lightning.data.text_utils import regex_tokenize


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_TABULAR_SUFFIXES = {".csv", ".tsv"}


def discover_tabular_files(directory: Path) -> list[Path]:
    """Recursively find CSV / TSV files under *directory*."""
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in _TABULAR_SUFFIXES and p.is_file()
    )


def select_input_file(raw_dir: Path, input_csv: Optional[Path] = None) -> Path:
    """Return a single tabular file to use for preprocessing.

    If *input_csv* is supplied and exists, use it directly; otherwise pick the
    first discovered file under *raw_dir*.
    """
    if input_csv is not None:
        path = Path(input_csv)
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified input file does not exist: {path}")

    files = discover_tabular_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No CSV / TSV files found under {raw_dir}")
    return files[0]


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def _best_candidate(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first column whose lowered name matches a candidate."""
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def detect_text_and_label_columns(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> tuple[str, str]:
    """Detect or validate text and label column names.

    Raises :class:`ValueError` when detection fails, listing available columns
    so the user can specify them manually.
    """
    columns = df.columns.tolist()

    if text_col is None:
        text_col = _best_candidate(columns, TEXT_COLUMN_CANDIDATES)
    if label_col is None:
        label_col = _best_candidate(columns, LABEL_COLUMN_CANDIDATES)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect text/label columns. "
            f"Available columns: {columns}. "
            f"Please specify --text_col and --label_col explicitly."
        )
    return text_col, label_col


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------


def normalize_binary_labels(
    series: pd.Series,
    label_map: Optional[dict[str, int]] = None,
) -> pd.Series:
    """Map raw label values to ``{0, 1}`` integers.

    Uses *label_map* when provided; otherwise falls back to
    :data:`DEFAULT_LABEL_SYNONYMS`.
    """
    mapping = label_map if label_map else DEFAULT_LABEL_SYNONYMS
    normalised = series.astype(str).str.strip().str.lower().map(mapping)
    if normalised.isna().any():
        bad = series[normalised.isna()].unique().tolist()
        raise ValueError(f"Unmapped label values: {bad}. Provide an explicit --label_map.")
    return normalised.astype(int)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalise whitespace and optionally lowercase *text*."""
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def parse_label_map_items(items: Optional[list[str]]) -> Optional[dict[str, int]]:
    """Parse ``['spam=1', 'ham=0']`` style CLI arguments into a dict."""
    if not items:
        return None
    result: dict[str, int] = {}
    for item in items:
        key, _, value = item.partition("=")
        result[key.strip().lower()] = int(value.strip())
    return result


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_dataset(
    *,
    raw_dir: Path,
    out_dir: Path,
    input_csv: Optional[Path] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    label_map: Optional[dict[str, int]] = None,
    lowercase: bool = True,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    dataset_slug: Optional[str] = None,
) -> dict:
    """Run the full preprocessing pipeline and write train/val/test CSVs.

    Returns a dict containing ``{"stats": {...}, "files": {...}}``.
    """
    # -- locate & load -------------------------------------------------------
    selected_file = select_input_file(raw_dir, input_csv)
    sep = "\t" if selected_file.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(selected_file, sep=sep)

    text_col, label_col = detect_text_and_label_columns(df, text_col, label_col)

    # -- drop rows whose label cannot be mapped to 0/1 ----------------------
    mapping = label_map if label_map else DEFAULT_LABEL_SYNONYMS
    _raw_labels = df[label_col].astype(str).str.strip().str.lower()
    _valid_mask = _raw_labels.isin(mapping)
    n_bad = (~_valid_mask).sum()
    if n_bad > 0:
        import logging
        logging.getLogger("spam_lightning").warning(
            "Dropped %d rows with unmappable label values: %s",
            n_bad,
            df.loc[~_valid_mask, label_col].unique().tolist(),
        )
        df = df[_valid_mask].reset_index(drop=True)

    # -- clean & normalise ---------------------------------------------------
    df["text"] = df[text_col].astype(str).apply(lambda t: clean_text(t, lowercase=lowercase))
    df["label"] = normalize_binary_labels(df[label_col], label_map)

    # Keep only what we need
    df = df[["text", "label"]].dropna().reset_index(drop=True)

    # -- split ---------------------------------------------------------------
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df["label"],
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), random_state=seed, stratify=temp_df["label"],
    )

    # -- write ---------------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # -- stats ---------------------------------------------------------------
    label_mapping = {str(k): int(v) for k, v in (label_map or DEFAULT_LABEL_SYNONYMS).items()}
    stats = {
        "source_file": str(selected_file),
        "text_col": text_col,
        "label_col": label_col,
        "label_mapping": label_mapping,
        "lowercase": lowercase,
        "seed": seed,
        "total_rows": len(df),
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "spam_ratio": {
            "train": float(train_df["label"].mean()),
            "val": float(val_df["label"].mean()),
            "test": float(test_df["label"].mean()),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return {
        "stats": stats,
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "stats": str(out_dir / "stats.json"),
        },
    }
