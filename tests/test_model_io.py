import pytest
import torch

from hsi_compression.engine.model_io import (
    call_model_compress,
    call_model_decompress,
    call_model_forward,
    exact_reconstruction_target,
    model_proxy_bpppc,
    validate_packed_output,
)


class MaskAwareModel(torch.nn.Module):
    proxy_bpppc = 0.25

    def forward(self, x, valid_mask=None):
        return {"x_hat": x + valid_mask.float()}

    def compress(self, x, valid_mask=None):
        return {"strings": [b"abc"], "shape": tuple(x.shape), "mask_sum": int(valid_mask.sum())}

    def decompress(self, strings, shape):
        return {"x_hat": torch.zeros(shape), "strings": strings}

    def exact_reconstruction_target(self, x):
        return torch.round(x)


class LegacyModel(torch.nn.Module):
    def forward(self, x):
        return {"x_hat": x}

    def compress(self, x):
        return {"latent": x, "z_shape": tuple(x.shape[-2:])}

    def decompress(self, latent, z_shape=None):
        assert z_shape is not None
        return {"x_hat": latent}


def test_model_io_uses_mask_when_supported():
    model = MaskAwareModel()
    x = torch.zeros(1, 1, 2, 2)
    mask = torch.ones_like(x, dtype=torch.bool)

    assert torch.equal(call_model_forward(model, x, mask)["x_hat"], torch.ones_like(x))
    packed = call_model_compress(model, x, mask)
    assert packed["mask_sum"] == 4
    assert call_model_decompress(model, packed)["x_hat"].shape == x.shape
    assert model_proxy_bpppc(model) == pytest.approx(0.25)
    assert torch.equal(exact_reconstruction_target(model, x + 0.4), torch.zeros_like(x))


def test_model_io_falls_back_to_legacy_signatures():
    model = LegacyModel()
    x = torch.ones(1, 1, 2, 2)
    mask = torch.zeros_like(x, dtype=torch.bool)

    assert torch.equal(call_model_forward(model, x, mask)["x_hat"], x)
    packed = call_model_compress(model, x, mask)
    assert torch.equal(call_model_decompress(model, packed)["x_hat"], x)


def test_validate_packed_output_rejects_incomplete_payloads():
    with pytest.raises(RuntimeError, match="must return a mapping"):
        validate_packed_output(None)
    with pytest.raises(RuntimeError, match="strings"):
        validate_packed_output({"shape": (1, 1, 1, 1)})
    with pytest.raises(RuntimeError, match="shape"):
        validate_packed_output({"strings": [b"abc"]})
