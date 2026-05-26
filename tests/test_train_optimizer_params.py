import torch

from scripts.train import _load_pretrained_weights, _split_main_aux_parameters


class _EntropyLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.quantiles = torch.nn.Parameter(torch.zeros(1))


def test_split_main_aux_parameters_excludes_quantiles_from_main_optimizer():
    model = torch.nn.Sequential(_EntropyLikeModule(), torch.nn.Linear(1, 1))

    main_parameters, aux_parameters = _split_main_aux_parameters(model)
    main_ids = {id(parameter) for parameter in main_parameters}
    aux_ids = {id(parameter) for parameter in aux_parameters}

    assert main_ids.isdisjoint(aux_ids)
    assert len(aux_parameters) == 1
    assert id(model[0].quantiles) in aux_ids
    assert id(model[0].weight) in main_ids
    assert id(model[1].weight) in main_ids


def test_load_pretrained_weights_compatible_only_skips_shape_mismatches():
    source = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    target = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 3))
    with torch.no_grad():
        source[0].weight.fill_(3.0)
        source[0].bias.fill_(4.0)
        source[1].weight.fill_(5.0)

    report = _load_pretrained_weights(
        target,
        {"model_state_dict": source.state_dict()},
        compatible_only=True,
    )

    assert report["loaded"] == 2
    assert report["skipped"] == 2
    assert torch.allclose(target[0].weight, torch.full_like(target[0].weight, 3.0))
    assert torch.allclose(target[0].bias, torch.full_like(target[0].bias, 4.0))
    assert target[1].weight.shape == (3, 2)
