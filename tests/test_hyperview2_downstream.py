import csv

import numpy as np
import pytest
import torch

from hsi_compression.downstream import (
    HYPERVIEW2_TARGET_COLUMNS,
    Hyperview2CompressionDataset,
    Hyperview2FeatureDataset,
    Hyperview2PixelSetDataset,
    SpectralSetRegressor,
    Standardizer,
    build_hyperview2_samples,
    collate_compression_batch,
    collate_pixel_set_batch,
    compute_regression_metrics,
    hyperview_score,
)
from hsi_compression.downstream.hyperview2 import to_chw


def _write_labels(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *HYPERVIEW2_TARGET_COLUMNS])
        writer.writerows(rows)


def test_build_hyperview2_samples_pairs_labels_with_prisma_arrays(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "prisma").mkdir(parents=True)
    _write_labels(
        root / "train.csv",
        [
            ["sample001", 1, 2, 3, 4, 5, 6],
            ["sample002", 2, 3, 4, 5, 6, 7],
        ],
    )
    np.save(root / "train" / "prisma" / "sample001_prisma.npy", np.ones((230, 2, 3)))
    np.save(root / "train" / "prisma" / "sample002_prisma.npy", np.ones((2, 3, 230)))

    samples = build_hyperview2_samples(root, modality="prisma")

    assert [sample.sample_id for sample in samples] == ["sample001", "sample002"]
    assert samples[0].target.tolist() == [1, 2, 3, 4, 5, 6]


def test_build_hyperview2_samples_pairs_official_numeric_layout(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "HYPERVIEW2" / "train" / "hsi_satellite").mkdir(parents=True)
    (root / "HYPERVIEW2" / "test" / "hsi_satellite").mkdir(parents=True)
    with (root / "HYPERVIEW2" / "train_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", *HYPERVIEW2_TARGET_COLUMNS])
        writer.writerow([0, 1, 2, 3, 4, 5, 6])
        writer.writerow([7, 2, 3, 4, 5, 6, 7])
    np.savez(
        root / "HYPERVIEW2" / "train" / "hsi_satellite" / "0000.npz",
        cube=np.ones((230, 2, 3)),
    )
    np.savez(
        root / "HYPERVIEW2" / "train" / "hsi_satellite" / "0007.npz",
        cube=np.ones((230, 2, 3)),
    )
    np.savez(
        root / "HYPERVIEW2" / "test" / "hsi_satellite" / "0000.npz",
        cube=np.zeros((230, 2, 3)),
    )

    samples = build_hyperview2_samples(
        root,
        modality="prisma",
        labels_csv=root / "HYPERVIEW2" / "train_gt.csv",
    )

    assert [sample.sample_id for sample in samples] == ["0", "7"]
    assert [sample.array_path.name for sample in samples] == ["0000.npz", "0007.npz"]
    assert all("train" in sample.array_path.parts for sample in samples)


def test_hyperview2_feature_dataset_returns_fixed_spectral_stats(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "prisma").mkdir(parents=True)
    _write_labels(root / "train.csv", [["sample001", 1, 2, 3, 4, 5, 6]])
    cube = np.zeros((230, 2, 2), dtype=np.float32)
    cube[:, 0, 0] = 1.0
    np.save(root / "train" / "prisma" / "sample001_prisma.npy", cube)

    samples = build_hyperview2_samples(root, modality="prisma")
    dataset = Hyperview2FeatureDataset(samples, modality="prisma", normalization="none")
    item = dataset[0]

    assert item["features"].shape[0] == 2 * 230 + 1
    assert item["target"].shape[0] == 6
    assert item["features"][-1].item() == pytest.approx(1.0)


def test_hyperview2_pixel_set_dataset_returns_pixel_spectra(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "prisma").mkdir(parents=True)
    _write_labels(root / "train.csv", [["sample001", 1, 2, 3, 4, 5, 6]])
    cube = np.zeros((230, 1, 2), dtype=np.float32)
    cube[:, 0, 0] = 1.0
    np.save(root / "train" / "prisma" / "sample001_prisma.npy", cube)

    samples = build_hyperview2_samples(root, modality="prisma")
    dataset = Hyperview2PixelSetDataset(samples, modality="prisma", normalization="none")
    item = dataset[0]

    assert item["pixels"].shape == (2, 230)
    assert item["valid_mask"].shape == (2,)
    assert item["valid_mask"].all()


def test_hyperview2_compression_dataset_and_collate_pad_spatial(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "prisma").mkdir(parents=True)
    _write_labels(
        root / "train.csv",
        [
            ["sample001", 1, 2, 3, 4, 5, 6],
            ["sample002", 2, 3, 4, 5, 6, 7],
        ],
    )
    np.save(root / "train" / "prisma" / "sample001_prisma.npy", np.ones((230, 1, 2)))
    np.save(root / "train" / "prisma" / "sample002_prisma.npy", np.ones((230, 2, 3)))

    samples = build_hyperview2_samples(root, modality="prisma")
    dataset = Hyperview2CompressionDataset(samples, modality="prisma", normalization="none")
    batch = collate_compression_batch([dataset[0], dataset[1]], pad_multiple=4)

    assert batch["x"].shape == (2, 230, 4, 4)
    assert batch["valid_mask"].shape == (2, 230, 4, 4)
    assert batch["x"][0, 0, -1, -1].item() == pytest.approx(1.0)
    assert batch["valid_mask"][0, :, :1, :2].all()
    assert not batch["valid_mask"][0, :, 1:, :].any()
    assert batch["original_shape"] == [(230, 1, 2), (230, 2, 3)]


def test_collate_pixel_set_batch_pads_variable_pixel_counts():
    batch = [
        {
            "pixels": np.ones((2, 230), dtype=np.float32),
            "valid_mask": np.ones(2, dtype=bool),
            "target": np.ones(6, dtype=np.float32),
            "sample_id": "a",
            "path": "a.npz",
        },
        {
            "pixels": np.ones((3, 230), dtype=np.float32),
            "valid_mask": np.asarray([True, False, True]),
            "target": np.ones(6, dtype=np.float32) * 2,
            "sample_id": "b",
            "path": "b.npz",
        },
    ]
    tensor_batch = [
        {
            "pixels": torch.from_numpy(item["pixels"]),
            "valid_mask": torch.from_numpy(item["valid_mask"]),
            "target": torch.from_numpy(item["target"]),
            "sample_id": item["sample_id"],
            "path": item["path"],
        }
        for item in batch
    ]

    out = collate_pixel_set_batch(tensor_batch)

    assert out["pixels"].shape == (2, 3, 230)
    assert out["valid_mask"].tolist() == [[True, True, False], [True, False, True]]


def test_spectral_set_regressor_forward_shape():
    model = SpectralSetRegressor(in_channels=230, hidden_dim=16, pixel_layers=2, head_layers=2)
    pixels = torch.randn(4, 5, 230)
    valid_mask = torch.ones(4, 5, dtype=torch.bool)

    y = model(pixels, valid_mask)

    assert y.shape == (4, 6)


def test_to_chw_uses_expected_band_axis():
    hwc = np.zeros((4, 5, 230), dtype=np.float32)
    chw = to_chw(hwc, expected_bands=230)

    assert chw.shape == (230, 4, 5)


def test_to_chw_preserves_singleton_spatial_dimensions():
    cube = np.zeros((230, 1, 2), dtype=np.float32)
    chw = to_chw(cube, expected_bands=230)

    assert chw.shape == (230, 1, 2)


def test_hyperview_score_is_zero_for_perfect_predictions():
    y_true = np.asarray([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=np.float32)
    baseline_mse = np.ones(6, dtype=np.float32)

    score, per_target = hyperview_score(y_true, y_true.copy(), baseline_mse)

    assert score == 0.0
    assert np.all(per_target == 0.0)


def test_compute_regression_metrics_reports_targets():
    y_true = np.asarray([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=np.float32)
    y_pred = y_true + 1.0
    baseline_mse = np.ones(6, dtype=np.float32) * 2.0

    metrics = compute_regression_metrics(y_true, y_pred, baseline_mse)

    assert metrics["hyperview_score"] == pytest.approx(0.5)
    assert set(metrics["targets"]) == set(HYPERVIEW2_TARGET_COLUMNS)


def test_standardizer_roundtrip():
    values = np.asarray([[1, 2], [3, 6]], dtype=np.float32)
    standardizer = Standardizer.fit(values)

    recovered = standardizer.inverse_transform(standardizer.transform(values))

    np.testing.assert_allclose(recovered, values)
