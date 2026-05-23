import math

import numpy as np
import pytest
import torch

from hsi_compression.downstream.hyperview2 import compute_regression_metrics, hyperview_score
from hsi_compression.metrics import (
    compute_actual_bpppc_from_strings,
    compute_compression_ratio_from_bpppc,
    compute_true_bpppc,
    invalid_region_mae,
    mae,
    masked_mae,
    masked_mse,
    masked_psnr,
    masked_rmse,
    masked_sam,
    masked_sam_deg,
    masked_sid,
    mse,
    psnr,
    ref_sam,
    ref_sam_deg,
    ref_ssim,
    sam,
    sam_deg,
    sid,
    ssim,
)


def test_mse_mae_and_psnr_match_hand_computed_values():
    x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    x_hat = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]]]])

    expected_mse = (0.0**2 + 0.5**2 + 1.0**2 + 1.5**2) / 4.0
    expected_mae = (0.0 + 0.5 + 1.0 + 1.5) / 4.0
    expected_psnr = 10.0 * math.log10(2.0**2 / expected_mse)

    assert mse(x_hat, x).item() == pytest.approx(expected_mse)
    assert mae(x_hat, x).item() == pytest.approx(expected_mae)
    assert psnr(x_hat, x, data_range=2.0).item() == pytest.approx(expected_psnr)


def test_psnr_uses_epsilon_for_exact_reconstructions():
    x = torch.ones(1, 2, 2, 2)

    expected_psnr = 10.0 * math.log10(1.0 / 1e-12)

    assert psnr(x, x).item() == pytest.approx(expected_psnr)


def test_masked_error_metrics_use_only_true_mask_entries():
    x = torch.zeros(1, 2, 2, 2)
    x_hat = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    mask = torch.tensor(
        [[[[True, False], [True, False]], [[False, True], [False, True]]]]
    )

    selected = [1.0, 3.0, 6.0, 8.0]
    expected_mse = sum(value**2 for value in selected) / len(selected)
    expected_mae = sum(abs(value) for value in selected) / len(selected)
    expected_rmse = math.sqrt(expected_mse)
    expected_psnr = 10.0 * math.log10(10.0**2 / expected_mse)

    assert masked_mse(x_hat, x, mask).item() == pytest.approx(expected_mse)
    assert masked_mae(x_hat, x, mask).item() == pytest.approx(expected_mae)
    assert masked_rmse(x_hat, x, mask).item() == pytest.approx(expected_rmse)
    assert masked_psnr(x_hat, x, mask, data_range=10.0).item() == pytest.approx(
        expected_psnr
    )


def test_masked_error_metrics_support_float_weight_masks():
    x = torch.zeros(1, 1, 1, 2)
    x_hat = torch.tensor([[[[1.0, 3.0]]]])
    mask = torch.tensor([[[[0.25, 0.75]]]])

    expected_mse = 1.0**2 * 0.25 + 3.0**2 * 0.75
    expected_mae = 1.0 * 0.25 + 3.0 * 0.75

    assert masked_mse(x_hat, x, mask).item() == pytest.approx(expected_mse)
    assert masked_mae(x_hat, x, mask).item() == pytest.approx(expected_mae)


def test_masked_error_metrics_return_zero_for_empty_masks():
    x = torch.zeros(1, 1, 2, 2)
    x_hat = torch.ones_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)

    assert masked_mse(x_hat, x, mask).item() == 0.0
    assert masked_mae(x_hat, x, mask).item() == 0.0
    assert masked_rmse(x_hat, x, mask).item() == 0.0


def test_invalid_region_mae_averages_only_invalid_entries():
    x_hat = torch.tensor([[[[1.0, -2.0], [3.0, -5.0]]]])
    mask = torch.tensor([[[[True, False], [True, False]]]])

    expected_mae = (2.0 + 5.0) / 2.0

    assert invalid_region_mae(x_hat, mask).item() == pytest.approx(expected_mae)


def test_invalid_region_mae_returns_zero_when_everything_is_valid():
    x_hat = torch.ones(1, 1, 2, 2)
    mask = torch.ones_like(x_hat, dtype=torch.bool)

    assert invalid_region_mae(x_hat, mask).item() == 0.0


def test_spectral_angle_metrics_match_known_angles():
    x = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
    x_hat = torch.tensor([[[[0.0, 0.5]], [[1.0, math.sqrt(3.0) / 2.0]]]])

    expected_rad = (math.pi / 2.0 + math.pi / 3.0) / 2.0
    expected_deg = expected_rad * 180.0 / math.pi

    assert sam(x_hat, x).item() == pytest.approx(expected_rad)
    assert ref_sam(x_hat, x).item() == pytest.approx(expected_rad)
    assert sam_deg(x_hat, x).item() == pytest.approx(expected_deg)
    assert ref_sam_deg(x_hat, x).item() == pytest.approx(expected_deg)


def test_masked_spectral_angle_averages_only_valid_pixels():
    x = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
    x_hat = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    mask = torch.tensor([[[[True, False]], [[True, False]]]])

    assert masked_sam(x_hat, x, mask).item() == pytest.approx(math.pi / 2.0)
    assert masked_sam_deg(x_hat, x, mask).item() == pytest.approx(90.0)


def test_masked_spectral_angle_returns_nan_for_empty_masks():
    x = torch.ones(1, 2, 1, 2)
    mask = torch.zeros_like(x, dtype=torch.bool)

    assert torch.isnan(masked_sam(x, x, mask))
    assert torch.isnan(masked_sam_deg(x, x, mask))


def test_sid_matches_hand_computed_symmetric_kl_divergence():
    x = torch.tensor([[[[1.0]], [[3.0]]]])
    x_hat = torch.tensor([[[[2.0]], [[2.0]]]])

    p = [0.25, 0.75]
    q = [0.5, 0.5]
    expected_sid = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))
    expected_sid += sum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))

    assert sid(x_hat, x).item() == pytest.approx(expected_sid)


def test_masked_sid_averages_only_valid_pixels():
    x = torch.tensor([[[[1.0, 1.0]], [[3.0, 1.0]]]])
    x_hat = torch.tensor([[[[2.0, 3.0]], [[2.0, 1.0]]]])
    mask = torch.tensor([[[[False, True]], [[False, True]]]])

    p = [0.5, 0.5]
    q = [0.75, 0.25]
    expected_sid = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))
    expected_sid += sum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))

    assert masked_sid(x_hat, x, mask).item() == pytest.approx(expected_sid)


def test_sid_clamps_non_positive_values_to_keep_result_finite():
    x = torch.tensor([[[[0.0]], [[1.0]]]])
    x_hat = torch.tensor([[[[-1.0]], [[2.0]]]])

    assert torch.isfinite(sid(x_hat, x))


def test_masked_sid_returns_nan_for_empty_masks():
    x = torch.ones(1, 2, 1, 1)
    mask = torch.zeros_like(x, dtype=torch.bool)

    assert torch.isnan(masked_sid(x, x, mask))


def test_ref_ssim_returns_nan_for_spatial_inputs_smaller_than_window():
    x = torch.ones(1, 3, 2, 2)

    assert torch.isnan(ref_ssim(x, x, channels=3))
    assert torch.isnan(ssim(x, x))


def test_ref_ssim_is_near_one_for_identical_images():
    x = torch.linspace(0.0, 1.0, steps=11 * 11).reshape(1, 1, 11, 11)

    assert ref_ssim(x, x, channels=1).item() == pytest.approx(1.0)


def test_compute_true_bpppc_converts_likelihoods_to_bits_per_value():
    likelihoods = torch.tensor([0.5, 0.25, 1.0, 0.125])

    assert compute_true_bpppc(likelihoods, (1, 1, 2, 2)) == pytest.approx(1.5)


def test_compute_actual_bpppc_from_strings_counts_nested_bitstreams():
    strings = [[b"ab", b""], (bytearray(b"cde"), [b"f"])]

    assert compute_actual_bpppc_from_strings(strings, (1, 3, 2, 1)) == pytest.approx(8.0)


def test_compute_actual_bpppc_from_strings_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="must not be None"):
        compute_actual_bpppc_from_strings(None, (1, 1, 1, 1))

    with pytest.raises(ValueError, match="Expected original_shape"):
        compute_actual_bpppc_from_strings(b"abc", (1, 1, 1))

    with pytest.raises(ValueError, match="Invalid original_shape"):
        compute_actual_bpppc_from_strings(b"abc", (1, 0, 1, 1))

    with pytest.raises(TypeError, match="Unsupported strings container"):
        compute_actual_bpppc_from_strings({"payload": b"abc"}, (1, 1, 1, 1))


def test_compute_compression_ratio_from_bpppc_handles_valid_and_missing_rates():
    assert compute_compression_ratio_from_bpppc(4.0) == pytest.approx(4.0)
    assert compute_compression_ratio_from_bpppc(2.0, original_bits_per_channel=12.0) == (
        pytest.approx(6.0)
    )
    assert compute_compression_ratio_from_bpppc(None) is None
    assert compute_compression_ratio_from_bpppc(0.0) is None
    assert compute_compression_ratio_from_bpppc(-1.0) is None


def test_hyperview_score_normalizes_mse_by_baseline_per_target():
    y_true = np.array([[0.0, 1.0], [2.0, 4.0], [4.0, 7.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 1.0], [0.0, 7.0], [4.0, 4.0]], dtype=np.float32)
    baseline_mse = np.array([2.0, 4.0], dtype=np.float32)

    score, per_target = hyperview_score(y_true, y_pred, baseline_mse)

    expected_per_target = np.array([5.0 / 6.0, 1.5], dtype=np.float32)
    assert score == pytest.approx(float(expected_per_target.mean()))
    np.testing.assert_allclose(per_target, expected_per_target)


def test_hyperview_score_uses_eps_when_baseline_mse_is_zero():
    y_true = np.array([[0.0], [2.0]], dtype=np.float32)
    y_pred = np.array([[1.0], [4.0]], dtype=np.float32)
    baseline_mse = np.array([0.0], dtype=np.float32)

    score, per_target = hyperview_score(y_true, y_pred, baseline_mse, eps=0.5)

    expected_score = 2.5 / 0.5
    assert score == pytest.approx(expected_score)
    np.testing.assert_allclose(per_target, np.array([expected_score], dtype=np.float32))


def test_compute_regression_metrics_reports_per_target_values():
    y_true = np.array([[0.0, 1.0], [2.0, 4.0], [4.0, 7.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 1.0], [0.0, 7.0], [4.0, 4.0]], dtype=np.float32)
    baseline_mse = np.array([2.0, 4.0], dtype=np.float32)

    metrics = compute_regression_metrics(
        y_true,
        y_pred,
        baseline_mse,
        target_columns=("target_a", "target_b"),
    )

    assert metrics["hyperview_score"] == pytest.approx((5.0 / 6.0 + 1.5) / 2.0)
    assert metrics["mean_mse"] == pytest.approx((5.0 / 3.0 + 6.0) / 2.0)
    assert metrics["mean_mae"] == pytest.approx((1.0 + 2.0) / 2.0)
    assert metrics["targets"]["target_a"] == {
        "mse": pytest.approx(5.0 / 3.0),
        "mae": pytest.approx(1.0),
        "rmse": pytest.approx(math.sqrt(5.0 / 3.0)),
        "relative_mse": pytest.approx(5.0 / 6.0),
        "baseline_mse": pytest.approx(2.0),
    }
    assert metrics["targets"]["target_b"] == {
        "mse": pytest.approx(6.0),
        "mae": pytest.approx(2.0),
        "rmse": pytest.approx(math.sqrt(6.0)),
        "relative_mse": pytest.approx(1.5),
        "baseline_mse": pytest.approx(4.0),
    }
