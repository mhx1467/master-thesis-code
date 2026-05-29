from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def model_proxy_bpppc(model: torch.nn.Module) -> float | None:
    model_raw = unwrap_model(model)
    proxy = getattr(model_raw, "proxy_bpppc", None)
    if proxy is not None:
        return float(proxy)
    legacy = getattr(model_raw, "bpppc", None)
    return float(legacy) if legacy is not None else None


def supports_actual_compression(model: torch.nn.Module) -> bool:
    return bool(getattr(unwrap_model(model), "supports_actual_compression", False))


def exact_reconstruction_target(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    target_fn = getattr(unwrap_model(model), "exact_reconstruction_target", None)
    return target_fn(x) if callable(target_fn) else x


def call_model_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    wavelengths: Sequence[float] | torch.Tensor | None = None,
    output_wavelengths: Sequence[float] | torch.Tensor | None = None,
) -> Any:
    try:
        return model(
            x,
            valid_mask=mask,
            wavelengths=wavelengths,
            output_wavelengths=output_wavelengths,
        )
    except TypeError:
        try:
            return model(x, valid_mask=mask, wavelengths=wavelengths)
        except TypeError:
            try:
                return model(x, valid_mask=mask)
            except TypeError:
                return model(x)


def call_model_compress(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    wavelengths: Sequence[float] | torch.Tensor | None = None,
) -> Any:
    try:
        return model.compress(x, valid_mask=mask, wavelengths=wavelengths)
    except TypeError:
        try:
            return model.compress(x, valid_mask=mask)
        except TypeError:
            return model.compress(x)


def call_model_decompress(
    model: torch.nn.Module,
    packed: Mapping[str, Any],
    output_wavelengths: Sequence[float] | torch.Tensor | None = None,
) -> Any:
    if "latent" in packed:
        return model.decompress(latent=packed["latent"], z_shape=packed.get("z_shape"))

    kwargs: dict[str, Any] = {"strings": packed["strings"], "shape": packed["shape"]}
    if packed.get("z_shape") is not None:
        kwargs["z_shape"] = packed["z_shape"]
    if packed.get("output_channels") is not None:
        kwargs["output_channels"] = packed["output_channels"]
    if output_wavelengths is not None:
        kwargs["output_wavelengths"] = output_wavelengths

    try:
        return model.decompress(**kwargs)
    except TypeError:
        kwargs.pop("output_wavelengths", None)
        return model.decompress(**kwargs)


def validate_packed_output(packed: Mapping[str, Any]) -> None:
    if not isinstance(packed, Mapping):
        raise RuntimeError("model.compress() must return a mapping")
    if "latent" in packed:
        return
    if "strings" not in packed:
        raise RuntimeError("model.compress() output must contain 'strings'")
    if "shape" not in packed:
        raise RuntimeError("model.compress() output must contain 'shape'")
    if packed["strings"] is None:
        raise RuntimeError("model.compress() returned strings=None")
