from spam_lightning.data.datamodule import SpamDataModule, SpamExample


def test_collate_embeddingbag_shapes_and_offsets(tmp_path) -> None:
    module = SpamDataModule(data_dir=tmp_path, batch_size=2, model_name="embeddingbag")
    batch = [
        SpamExample(token_ids=[2, 3, 4], label=1),
        SpamExample(token_ids=[5, 6], label=0),
    ]

    collated = module.collate_batch(batch)

    assert collated["tokens"].tolist() == [2, 3, 4, 5, 6]
    assert collated["offsets"].tolist() == [0, 3]
    assert collated["labels"].tolist() == [1.0, 0.0]
    assert tuple(collated["tokens"].shape) == (5,)
    assert tuple(collated["offsets"].shape) == (2,)
