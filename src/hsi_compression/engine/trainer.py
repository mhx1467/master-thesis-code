from pathlib import Path

from hsi_compression.engine.train import train_one_epoch
from hsi_compression.engine.validate import validate_one_epoch
from hsi_compression.engine.checkpointing import save_checkpoint


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs: int,
    checkpoint_path: str | Path,
    config: dict,
    logger=None,
    scheduler=None,
):
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/rmse": train_metrics["rmse"],
            "val/loss": val_metrics["loss"],
            "val/rmse": val_metrics["rmse"],
            "val/sam_deg": val_metrics["sam_deg"],
        }

        latent_shape = val_metrics.get("latent_shape")
        if latent_shape is not None:
            record["model/latent_c"] = latent_shape[0]
            record["model/latent_h"] = latent_shape[1]
            record["model/latent_w"] = latent_shape[2]

        # Compression ratio proxy if the model provides it
        if hasattr(model, "compression_ratio_proxy") and latent_shape is not None:
            input_shape = config.get("input_shape")
            if input_shape is not None:
                ratio = model.compression_ratio_proxy(
                    input_shape=input_shape,
                    latent_shape=latent_shape,
                )
                record["model/compression_ratio_proxy"] = ratio

        history.append(record)

        if logger is not None:
            logger.log(record, step=epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={record['train/loss']:.6f} | "
            f"train_rmse={record['train/rmse']:.6f} | "
            f"val_loss={record['val/loss']:.6f} | "
            f"val_rmse={record['val/rmse']:.6f} | "
            f"val_sam_deg={record['val/sam_deg']:.6f}"
        )

        if record["val/loss"] < best_val_loss:
            best_val_loss = record["val/loss"]

            extra = {
                "latent_shape": latent_shape,
            }
            if "model/compression_ratio_proxy" in record:
                extra["compression_ratio_proxy"] = record["model/compression_ratio_proxy"]

            save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                best_val_loss=best_val_loss,
                extra=extra,
            )

            if logger is not None:
                logger.summary["best_val_loss"] = best_val_loss
                logger.summary["best_epoch"] = epoch
                logger.summary["best_checkpoint_path"] = str(checkpoint_path)

                if "model/compression_ratio_proxy" in record:
                    logger.summary["compression_ratio_proxy"] = record["model/compression_ratio_proxy"]

    return {
        "best_val_loss": best_val_loss,
        "history": history,
    }