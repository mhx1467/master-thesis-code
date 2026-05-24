import torch

from hsi_compression.losses import SymbolCodeLengthLoss, build_loss
from hsi_compression.metrics import ref_ssim


def test_symbol_code_length_loss_prefers_zero_integer_residual():
    loss_fn = SymbolCodeLengthLoss(symbol_scale=10000, code_weight=1.0, mse_weight=0.0)
    target = torch.zeros(1, 2, 1, 1)

    # zero residual should be cheaper than a one-symbol prediction error.
    zero_residual = loss_fn(torch.zeros_like(target), target, None)
    one_symbol_residual = loss_fn(torch.full_like(target, 0.0001), target, None)

    assert zero_residual.item() == 0.0
    assert one_symbol_residual.item() > zero_residual.item()


def test_symbol_code_length_loss_rounding_keeps_gradient_path():
    loss_fn = SymbolCodeLengthLoss(symbol_scale=10000, code_weight=1.0, mse_weight=0.0)
    prediction = torch.full((1, 2, 1, 1), 0.01, requires_grad=True)
    target = torch.zeros_like(prediction)

    loss = loss_fn(prediction, target, None)
    loss.backward()

    # straight-through rounding should still allow gradients to reach predictions.
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0.0


def test_symbol_code_length_loss_respects_mask():
    loss_fn = SymbolCodeLengthLoss(symbol_scale=10000, code_weight=1.0, mse_weight=0.0)
    target = torch.zeros(1, 2, 1, 1)
    prediction = torch.tensor([[[[0.0]], [[0.0001]]]])
    mask = torch.tensor([[[[True]], [[False]]]])

    loss = loss_fn(prediction, target, mask)

    assert loss.item() == 0.0


def test_build_loss_accepts_symbol_code_length_kwargs():
    loss_fn = build_loss("symbol_code_length", symbol_scale=10000, code_weight=0.0001)

    assert isinstance(loss_fn, SymbolCodeLengthLoss)
    assert loss_fn.select_by_loss is True


def test_ref_ssim_handles_small_spatial_inputs():
    x = torch.ones(1, 4, 4, 4)

    score = ref_ssim(x, x, channels=4)

    assert torch.isfinite(score)
