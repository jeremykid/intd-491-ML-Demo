from spam_lightning.data.datamodule import SpamDataModule, SpamExample


def test_sequence_collate_pads_and_tracks_lengths(tmp_path) -> None:
    module = SpamDataModule(data_dir=tmp_path, batch_size=2, model_name="lstm", max_seq_len=5)
    batch = [
        SpamExample(token_ids=[2, 3, 4], label=1),
        SpamExample(token_ids=[5, 6], label=0),
    ]

    collated = module.collate_batch(batch)

    assert collated["input_ids"].tolist() == [[2, 3, 4], [5, 6, 0]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert collated["lengths"].tolist() == [3, 2]
    assert collated["labels"].tolist() == [1.0, 0.0]


def test_sequence_collate_truncates_to_max_seq_len(tmp_path) -> None:
    module = SpamDataModule(data_dir=tmp_path, batch_size=1, model_name="transformer", max_seq_len=3)
    batch = [SpamExample(token_ids=[2, 3, 4, 5, 6], label=1)]

    collated = module.collate_batch(batch)

    assert collated["input_ids"].tolist() == [[2, 3, 4]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1]]
    assert collated["lengths"].tolist() == [3]
