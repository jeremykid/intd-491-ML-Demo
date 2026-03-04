import pandas as pd
import pytest

from spam_lightning.data.preprocessing import detect_text_and_label_columns, normalize_binary_labels


def test_detect_text_and_label_columns_with_v_columns() -> None:
    df = pd.DataFrame({"v1": ["ham", "spam"], "v2": ["hello", "buy now"]})
    text_col, label_col = detect_text_and_label_columns(df, text_col=None, label_col=None)
    assert text_col == "v2"
    assert label_col == "v1"


def test_explicit_column_override_wins() -> None:
    df = pd.DataFrame({"body_text": ["hello"], "target_label": [1]})
    text_col, label_col = detect_text_and_label_columns(
        df,
        text_col="body_text",
        label_col="target_label",
    )
    assert text_col == "body_text"
    assert label_col == "target_label"


def test_label_normalization_supports_common_binary_forms() -> None:
    spam_ham = normalize_binary_labels(pd.Series(["spam", "ham"]), label_map=None)
    bool_like = normalize_binary_labels(pd.Series(["true", "false"]), label_map=None)
    numeric = normalize_binary_labels(pd.Series([1, 0]), label_map=None)
    assert spam_ham.tolist() == [1, 0]
    assert bool_like.tolist() == [1, 0]
    assert numeric.tolist() == [1, 0]


def test_detection_failure_reports_available_columns() -> None:
    df = pd.DataFrame({"subject": ["hello"], "outcome": [1]})
    with pytest.raises(ValueError) as exc_info:
        detect_text_and_label_columns(df, text_col=None, label_col=None)

    message = str(exc_info.value)
    assert "subject" in message
    assert "outcome" in message
