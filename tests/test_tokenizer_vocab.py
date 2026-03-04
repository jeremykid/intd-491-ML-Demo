from spam_lightning.data.text_utils import build_vocab, regex_tokenize


def test_regex_tokenizer_lowercases_and_strips_punctuation() -> None:
    tokens = regex_tokenize("Free MONEY!!! It's real.", lowercase=True)
    assert tokens == ["free", "money", "it's", "real"]


def test_vocab_builds_with_special_tokens_and_unknown_lookup() -> None:
    sequences = [["spam", "free"], ["ham", "free"], ["free"]]
    vocab = build_vocab(sequences, min_freq=2, max_size=None)

    assert vocab.pad_index == 0
    assert vocab.unk_index == 1
    assert vocab.lookup_index("free") != vocab.unk_index
    assert vocab.lookup_index("spam") == vocab.unk_index
