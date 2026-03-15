from pathlib import Path

from hsi_compression.engine.train import train_one_epoch
from hsi_compression.engine.validate import validate_one_epoch
from hsi_compression.engine.checkpointing import save_checkpoint
from hsi_compression.utils.distributed import is_main_process


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs: int,
    checkpoint_path,
    config: dict,
    logger=None,
    scheduler=None,
    show_progress=True,
    train_sampler=None,
):
    best_val_loss = float("inf")
    history = []

    training_cfg = config.get("training", {})
    early_cfg = training_cfg.get("early_stopping", {})

    early_enabled = early_cfg.get("enabled", False)
    early_patience = early_cfg.get("patience", 5)
    early_min_delta = early_cfg.get("min_delta", 0.0)

    epochs_without_improvement = 0    

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            show_progress=show_progress,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            show_progress=show_progress,
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

        model_for_ratio = model.module if hasattr(model, "module") else model

        if hasattr(model_for_ratio, "compression_ratio_proxy") and latent_shape is not None:
            input_shape = config.get("input_shape")
            if input_shape is not None:
                ratio = model_for_ratio.compression_ratio_proxy(
                    input_shape=input_shape,
                    latent_shape=latent_shape,
                )
                record["model/compression_ratio_proxy"] = ratio

        history.append(record)

        if logger is not None and is_main_process():
            logger.log(record, step=epoch)

        if is_main_process():
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={record['train/loss']:.6f} | "
                f"train_rmse={record['train/rmse']:.6f} | "
                f"val_loss={record['val/loss']:.6f} | "
                f"val_rmse={record['val/rmse']:.6f} | "
                f"val_sam_deg={record['val/sam_deg']:.6f}"
            )

        if record["val/loss"] < best_val_loss - early_min_delta:
            best_val_loss = record["val/loss"]
            epochs_without_improvement = 0

            extra = {
                "latent_shape": latent_shape,
            }
            if "model/compression_ratio_proxy" in record:
                extra["compression_ratio_proxy"] = record["model/compression_ratio_proxy"]

            if is_main_process():
                model_to_save = model.module if hasattr(model, "module") else model

                save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    model=model_to_save,
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
        else:
            epochs_without_improvement += 1

        if early_enabled and epochs_without_improvement >= early_patience:
            if is_main_process():
                print(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience={early_patience})."
                )
            break        

    return {
        "best_val_loss": best_val_loss,
        "history": history,
    }