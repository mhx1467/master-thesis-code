import csv
import json

import numpy as np
import pytest
import torch

from hsi_compression.downstream import (
    HYPERVIEW2_FEATURE_SETS,
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
    extract_spectral_features,
    hyperview_score,
)
from hsi_compression.downstream.hyperview2 import load_mask, normalize_cube, to_chw
from hsi_compression.downstream.hyperview2_compression_eval import (
    apply_input_spectral_mapping,
    build_spectral_mapping,
    infer_recon_input_normalization,
    invert_output_spectral_mapping,
    make_feature_matrix,
    reconstruct_spectral_resample_passthrough,
)
from hsi_compression.downstream.hyperview2_regressors import available_regressor_names


def _write_labels(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *HYPERVIEW2_TARGET_COLUMNS])
        writer.writerows(rows)


def test_build_hyperview2_samples_pairs_labels_with_prisma_arrays(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(
        root / "train_gt.csv",
        [
            [1, 1, 2, 3, 4, 5, 6],
            [2, 2, 3, 4, 5, 6, 7],
        ],
    )
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=np.ones((230, 2, 3)))
    np.savez(root / "train" / "hsi_satellite" / "0002.npz", data=np.ones((2, 3, 230)))

    samples = build_hyperview2_samples(root, modality="prisma")

    assert [sample.sample_id for sample in samples] == ["1", "2"]
    assert samples[0].target.tolist() == [1, 2, 3, 4, 5, 6]


def test_build_hyperview2_samples_pairs_official_numeric_layout(tmp_path):
    root = tmp_path / "hyperview2"
    # official layout uses numeric ids and split-specific folders.
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
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(root / "train_gt.csv", [[1, 1, 2, 3, 4, 5, 6]])
    cube = np.zeros((230, 2, 2), dtype=np.float32)
    cube[:, 0, 0] = 1.0
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=cube)

    samples = build_hyperview2_samples(root, modality="prisma")
    dataset = Hyperview2FeatureDataset(samples, modality="prisma", normalization="none")
    item = dataset[0]

    assert item["features"].shape[0] == 2 * 230 + 1
    assert item["target"].shape[0] == 6
    assert item["features"][-1].item() == pytest.approx(1.0)


def test_hyperview2_feature_dataset_supports_larger_feature_sets(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(root / "train_gt.csv", [[1, 1, 2, 3, 4, 5, 6]])
    cube = np.ones((230, 2, 2), dtype=np.float32)
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=cube)

    samples = build_hyperview2_samples(root, modality="prisma")
    lengths = []
    for feature_set in HYPERVIEW2_FEATURE_SETS:
        dataset = Hyperview2FeatureDataset(
            samples,
            modality="prisma",
            normalization="none",
            feature_set=feature_set,
        )
        lengths.append(dataset[0]["features"].shape[0])

    assert lengths == [2 * 230 + 1, 4 * 230 + 1, 9 * 230 + 1]


def test_extract_spectral_features_rejects_unknown_feature_set():
    cube = np.ones((230, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="feature_set"):
        extract_spectral_features(cube, feature_set="unknown")


def test_reflectance_normalization_clips_and_zeros_invalid_pixels():
    cube = np.asarray(
        [
            [[-0.5, 0.25], [1.5, np.nan]],
            [[0.2, 0.4], [np.inf, 0.8]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([[True, False], [True, True]])

    normalized = normalize_cube(cube, mask=mask, mode="reflectance_0_1")

    assert normalized.dtype == np.float32
    assert normalized[:, 0, 1].tolist() == [0.0, 0.0]
    assert normalized[0, 0, 0].item() == pytest.approx(0.0)
    assert normalized[0, 1, 0].item() == pytest.approx(1.0)
    assert normalized[1, 1, 0].item() == pytest.approx(1.0)
    assert normalized[0, 1, 1].item() == pytest.approx(0.0)

    value_mask = np.ones_like(cube, dtype=bool)
    value_mask[1, 1, 0] = False
    normalized_with_value_mask = normalize_cube(cube, mask=value_mask, mode="reflectance_0_1")

    assert normalized_with_value_mask[1, 1, 0].item() == pytest.approx(0.0)


def test_infer_recon_input_normalization_detects_reflectance_variant_name():
    assert (
        infer_recon_input_normalization("hyperview2_mamba_input_reflectance_0_1")
        == "reflectance_0_1"
    )
    assert infer_recon_input_normalization("hyperview2_mamba_input_hyspecnet") == "reflectance_0_1"


def test_available_regressor_names_can_report_optional_models():
    names = available_regressor_names(include_unavailable=True)

    assert "extra_trees" in names
    assert "lightgbm" in names


def test_hyperview2_pixel_set_dataset_returns_pixel_spectra(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(root / "train_gt.csv", [[1, 1, 2, 3, 4, 5, 6]])
    cube = np.zeros((230, 1, 2), dtype=np.float32)
    cube[:, 0, 0] = 1.0
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=cube)

    samples = build_hyperview2_samples(root, modality="prisma")
    dataset = Hyperview2PixelSetDataset(samples, modality="prisma", normalization="none")
    item = dataset[0]

    assert item["pixels"].shape == (2, 230)
    assert item["valid_mask"].shape == (2,)
    assert item["valid_mask"].all()


def test_hyperview2_compression_dataset_and_collate_pad_spatial(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(
        root / "train_gt.csv",
        [
            [1, 1, 2, 3, 4, 5, 6],
            [2, 2, 3, 4, 5, 6, 7],
        ],
    )
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=np.ones((230, 1, 2)))
    np.savez(root / "train" / "hsi_satellite" / "0002.npz", data=np.ones((230, 2, 3)))

    samples = build_hyperview2_samples(root, modality="prisma")
    dataset = Hyperview2CompressionDataset(samples, modality="prisma", normalization="none")
    # collate pads variable spatial sizes to a shared model input tensor.
    batch = collate_compression_batch([dataset[0], dataset[1]], pad_multiple=4)

    assert batch["x"].shape == (2, 230, 4, 4)
    assert batch["valid_mask"].shape == (2, 230, 4, 4)
    assert batch["x"][0, 0, -1, -1].item() == pytest.approx(1.0)
    assert batch["valid_mask"][0, :, :1, :2].all()
    assert not batch["valid_mask"][0, :, 1:, :].any()
    assert batch["original_shape"] == [(230, 1, 2), (230, 2, 3)]


def test_hyperview2_compression_dataset_uses_npz_mask(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(root / "train_gt.csv", [[1, 1, 2, 3, 4, 5, 6]])
    cube = np.ones((230, 2, 2), dtype=np.float32)
    mask = np.ones((230, 2, 2), dtype=bool)
    mask[:, 1, 1] = False
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=cube, mask=mask)

    samples = build_hyperview2_samples(root, modality="prisma")
    item = Hyperview2CompressionDataset(samples, modality="prisma", normalization="none")[0]

    assert item["valid_mask"].shape == (230, 2, 2)
    assert item["valid_mask"][:, 0, 0].all()
    assert not item["valid_mask"][:, 1, 1].any()


def test_hyspecnet_202_spectral_mapping_roundtrip_shapes(tmp_path):
    wavelengths = {f"Band {idx}": float(400 + idx * 9) for idx in range(230)}
    (tmp_path / "wavelengths.json").write_text(
        json.dumps({"hsi_satellite_wavelengths": wavelengths}),
        encoding="utf-8",
    )

    mapping = build_spectral_mapping("hyspecnet_202_approx", tmp_path, "prisma", 230)
    x = torch.linspace(0.0, 1.0, steps=230 * 2 * 3).reshape(1, 230, 2, 3)
    mask = torch.ones_like(x, dtype=torch.bool)

    x_model, mask_model = apply_input_spectral_mapping(x, mask, mapping)
    x_out = invert_output_spectral_mapping(x_model, mapping)

    assert mapping is not None
    assert mapping["input_channels"] == 230
    assert mapping["model_input_channels"] == 202
    assert x_model.shape == (1, 202, 2, 3)
    assert mask_model.shape == (1, 202, 2, 3)
    assert x_out.shape == (1, 230, 2, 3)
    assert mask_model.all()


def test_spectral_resample_passthrough_writes_hyperview2_reconstruction(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(
        root / "train_gt.csv",
        [
            [1, 1, 2, 3, 4, 5, 6],
            [2, 2, 3, 4, 5, 6, 7],
        ],
    )
    wavelengths = {f"Band {idx}": float(400 + idx * 9) for idx in range(230)}
    (root / "wavelengths.json").write_text(
        json.dumps({"hsi_satellite_wavelengths": wavelengths}),
        encoding="utf-8",
    )
    cube = np.linspace(0.0, 1.0, num=230 * 1 * 2, dtype=np.float32).reshape(230, 1, 2)
    mask = np.ones((230, 1, 2), dtype=bool)
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=cube, mask=mask)
    np.savez(root / "train" / "hsi_satellite" / "0002.npz", data=cube[::-1].copy(), mask=mask)

    recon_root, summary = reconstruct_spectral_resample_passthrough(
        source_root=root,
        recon_parent=tmp_path / "recons",
        device=torch.device("cpu"),
        variant_name="resample_control",
        batch_size=2,
        num_workers=0,
    )

    assert summary["baseline_type"] == "no_codec_spectral_resample_passthrough"
    assert summary["input_channels"] == 230
    assert summary["model_input_channels"] == 202
    assert summary["output_channels"] == 230
    assert summary["samples"] == 2
    assert (recon_root / "train" / "hsi_satellite" / "0001.npz").exists()


def test_torch_feature_matrix_matches_numpy_feature_matrix(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(
        root / "train_gt.csv",
        [
            [1, 1, 2, 3, 4, 5, 6],
            [2, 2, 3, 4, 5, 6, 7],
        ],
    )
    cube = np.linspace(0.0, 1.0, num=230 * 2 * 3, dtype=np.float32).reshape(230, 2, 3)
    mask = np.ones((2, 3), dtype=bool)
    mask[1, 2] = False
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=cube, mask=mask)
    np.savez(root / "train" / "hsi_satellite" / "0002.npz", data=cube[::-1].copy(), mask=mask)
    samples = build_hyperview2_samples(root, modality="prisma")

    x_numpy, y_numpy, ids_numpy = make_feature_matrix(
        samples,
        modality="prisma",
        normalization="reflectance_0_1",
        feature_set="mean_std_derivatives",
    )
    x_torch, y_torch, ids_torch = make_feature_matrix(
        samples,
        modality="prisma",
        normalization="reflectance_0_1",
        feature_set="mean_std_derivatives",
        feature_device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
    )

    assert ids_torch == ids_numpy
    np.testing.assert_allclose(y_torch, y_numpy)
    np.testing.assert_allclose(x_torch, x_numpy, rtol=1e-6, atol=1e-6)


def test_hyperview2_compression_dataset_accepts_singleton_spatial_mask(tmp_path):
    root = tmp_path / "hyperview2"
    (root / "train" / "hsi_satellite").mkdir(parents=True)
    _write_labels(root / "train_gt.csv", [[1, 1, 2, 3, 4, 5, 6]])
    cube = np.ones((230, 1, 1), dtype=np.float32)
    mask = np.zeros((230, 1, 1), dtype=bool)
    np.savez(root / "train" / "hsi_satellite" / "0001.npz", data=cube, mask=mask)

    samples = build_hyperview2_samples(root, modality="prisma")
    item = Hyperview2CompressionDataset(samples, modality="prisma", normalization="none")[0]

    assert item["valid_mask"].shape == (230, 1, 1)
    assert not item["valid_mask"].any()


def test_load_mask_preserves_singleton_width_before_collapsing_spectral_axis(tmp_path):
    path = tmp_path / "mask.npz"
    mask = np.ones((230, 2, 1), dtype=bool)
    mask[:, 1, 0] = False
    np.savez(path, mask=mask)

    loaded = load_mask(path, shape_hw=(2, 1))

    assert loaded.shape == (2, 1)
    assert loaded[:, 0].tolist() == [True, False]


def test_collate_pixel_set_batch_pads_variable_pixel_counts():
    # pixel-set batches can contain different numbers of valid pixels per sample.
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


def test_spectral_set_regressor_encode_set_matches_head_input():
    model = SpectralSetRegressor(in_channels=230, hidden_dim=16, pixel_layers=2, head_layers=2)
    model.eval()
    pixels = torch.randn(4, 5, 230)
    valid_mask = torch.ones(4, 5, dtype=torch.bool)

    # encode_set output must be exactly what the regression head consumes.
    pooled = model.encode_set(pixels, valid_mask)

    assert pooled.shape == (4, 33)
    torch.testing.assert_close(model(pixels, valid_mask), model.head(pooled))


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

    # standardizing and then reversing should recover the original values.
    recovered = standardizer.inverse_transform(standardizer.transform(values))

    np.testing.assert_allclose(recovered, values)
