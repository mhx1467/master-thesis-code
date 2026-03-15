from hsi_compression.metrics import masked_mse


LOSS_REGISTRY = {
    "masked_mse": masked_mse,
}


def build_loss(loss_name: str):
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_name: {loss_name}. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[loss_name]