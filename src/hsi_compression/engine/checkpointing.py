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
    scheduler=None,
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
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        checkpoint["extra"] = extra

    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def save_last_checkpoint(
    checkpoint_path: str | Path,
    epoch: int,
    model,
    optimizer,
    config: dict,
    val_metrics: dict,
    scheduler=None,
) -> None:
    last_path = Path(checkpoint_path).parent / (
        Path(checkpoint_path).stem.replace("_best", "") + "_last.pt"
    )
    save_checkpoint(
        path=last_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        config=config,
        best_val_loss=val_metrics.get("loss", float("inf")),
        scheduler=scheduler,
        extra={
            "val_psnr": val_metrics.get("psnr", 0.0),
            "val_sam_deg": val_metrics.get("sam_deg", 0.0),
            "val_bpppc": val_metrics.get("bpppc", 0.0),
            "is_last": True,
        },
    )


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def find_resume_checkpoint(checkpoint_path: str | Path) -> Path | None:
    ckpt = Path(checkpoint_path)
    last = ckpt.parent / (ckpt.stem.replace("_best", "") + "_last.pt")
    return last if last.exists() else None
