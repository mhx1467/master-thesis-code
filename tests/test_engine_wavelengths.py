import torch

from hsi_compression.engine.train import train_one_epoch
from hsi_compression.engine.validate import validate_one_epoch


class _WavelengthAwareToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.seen_train_wavelengths = None
        self.seen_eval_wavelengths = None

    def forward(self, x, valid_mask=None, wavelengths=None, output_wavelengths=None):
        _ = valid_mask
        if wavelengths is None:
            raise AssertionError("wavelengths were not forwarded to the model")
        if output_wavelengths is None:
            raise AssertionError("output_wavelengths were not forwarded to the model")
        if self.training:
            self.seen_train_wavelengths = wavelengths.detach().clone()
        else:
            self.seen_eval_wavelengths = wavelengths.detach().clone()
        return {"x_hat": x * self.scale}


def _masked_mse_loss(x_hat, x, mask):
    if mask is None:
        return torch.mean((x_hat - x) ** 2)
    return torch.mean((x_hat - x) ** 2 * mask.float())


def _batch():
    return {
        "x": torch.ones(1, 3, 16, 16),
        "valid_mask": torch.ones(1, 3, 16, 16, dtype=torch.bool),
        "wavelengths": torch.tensor([400.0, 500.0, 600.0]),
        "output_wavelengths": torch.tensor([400.0, 500.0, 600.0]),
    }


def test_train_one_epoch_forwards_wavelength_metadata():
    model = _WavelengthAwareToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_one_epoch(
        model=model,
        loader=[_batch()],
        optimizer=optimizer,
        loss_fn=_masked_mse_loss,
        device=torch.device("cpu"),
        show_progress=False,
    )

    torch.testing.assert_close(
        model.seen_train_wavelengths,
        torch.tensor([400.0, 500.0, 600.0]),
    )


def test_validate_one_epoch_forwards_wavelength_metadata():
    model = _WavelengthAwareToyModel()

    validate_one_epoch(
        model=model,
        loader=[_batch()],
        loss_fn=_masked_mse_loss,
        device=torch.device("cpu"),
        show_progress=False,
    )

    torch.testing.assert_close(
        model.seen_eval_wavelengths,
        torch.tensor([400.0, 500.0, 600.0]),
    )
