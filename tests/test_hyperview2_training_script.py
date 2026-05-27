import torch

from scripts.train_hyperview2_compressor import (
    _configure_trainable_parameters,
    _limit_samples,
)


def test_limit_samples_is_seeded_and_nested():
    samples = [f"sample-{index}" for index in range(20)]

    small = _limit_samples(samples, 5, seed=123)
    larger = _limit_samples(samples, 10, seed=123)

    assert small == _limit_samples(samples, 5, seed=123)
    assert set(small).issubset(set(larger))
    assert small != samples[:5]


def test_configure_trainable_parameters_supports_regex_filtering():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 3)
    model.decoder = torch.nn.Linear(3, 2)
    model.entropy_bottleneck = torch.nn.Linear(3, 3)

    report = _configure_trainable_parameters(
        model,
        trainable_regex=["decoder", "entropy_bottleneck"],
        freeze_regex=[],
    )

    assert report["trainable_parameters"] < report["total_parameters"]
    assert model.encoder.weight.requires_grad is False
    assert model.decoder.weight.requires_grad is True
    assert model.entropy_bottleneck.weight.requires_grad is True


def test_configure_trainable_parameters_freeze_overrides_trainable_filter():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 3)
    model.decoder = torch.nn.Linear(3, 2)

    _configure_trainable_parameters(
        model,
        trainable_regex=["encoder", "decoder"],
        freeze_regex=["decoder"],
    )

    assert model.encoder.weight.requires_grad is True
    assert model.decoder.weight.requires_grad is False
