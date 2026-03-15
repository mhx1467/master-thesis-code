from pathlib import Path
import torch


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model,
    optimizer,
    config: dict,
    best_val_loss: float,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
    }

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    map_location="cpu",
):
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint