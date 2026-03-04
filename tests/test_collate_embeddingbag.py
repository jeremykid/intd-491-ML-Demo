from spam_lightning.data.datamodule import SpamDataModule, SpamExample


def test_collate_embeddingbag_shapes_and_offsets(tmp_path) -> None:
    module = SpamDataModule(data_dir=tmp_path, batch_size=2)
    batch = [
        SpamExample(token_ids=[2, 3, 4], label=1),
        SpamExample(token_ids=[5, 6], label=0),
    ]

    tokens, offsets, labels = module.collate_batch(batch)

    assert tokens.tolist() == [2, 3, 4, 5, 6]
    assert offsets.tolist() == [0, 3]
    assert labels.tolist() == [1.0, 0.0]
    assert tuple(tokens.shape) == (5,)
    assert tuple(offsets.shape) == (2,)
