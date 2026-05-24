from hsi_compression.data import build_dataset


def test_build_dataset_can_resolve_tif_source_from_reference_split(tmp_path):
    dataset_root = tmp_path / "hyspecnet"
    split_dir = dataset_root / "splits" / "easy"
    patch_dir = dataset_root / "patches" / "tile" / "patch"
    split_dir.mkdir(parents=True)
    patch_dir.mkdir(parents=True)

    (split_dir / "train.csv").write_text("tile/patch/patch-DATA.npy\n", encoding="utf-8")
    tif_path = patch_dir / "patch-SPECTRAL_IMAGE.TIF"
    tif_path.touch()

    dataset = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty="easy",
        prefer_npy=False,
    )

    assert dataset.using_npy is False
    assert dataset.paths == [tif_path]
