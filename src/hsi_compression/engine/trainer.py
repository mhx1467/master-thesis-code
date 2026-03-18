from hsi_compression.engine.checkpointing import save_checkpoint
from hsi_compression.engine.train import train_one_epoch
from hsi_compression.engine.validate import validate_one_epoch
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
    show_progress: bool = True,
    train_sampler=None,
    grad_clip_max_norm: float = 1.0,
    num_input_bands: int = 202,
    quantization_bits: int = 8,
):
    best_val_psnr = float("-inf")
    history = []

    training_cfg = config.get("training", {})
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", False)
    early_patience = early_cfg.get("patience", 20)
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
            grad_clip_max_norm=grad_clip_max_norm,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            num_input_bands=num_input_bands,
            quantization_bits=quantization_bits,
            epoch=epoch,
            total_epochs=epochs,
            show_progress=show_progress,
        )

        if scheduler is not None:
            scheduler.step()

        latent_shape = val_metrics.get("latent_shape")
        model_for_ratio = model.module if hasattr(model, "module") else model

        cr_proxy = None
        if hasattr(model_for_ratio, "compression_ratio_proxy") and latent_shape:
            input_shape = (num_input_bands, 128, 128)
            cr_proxy = model_for_ratio.compression_ratio_proxy(
                input_shape=input_shape,
                latent_shape=latent_shape,
            )

        record = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/rmse": train_metrics["rmse"],
            "train/psnr": train_metrics["psnr"],
            "val/loss": val_metrics["loss"],
            "val/rmse": val_metrics["rmse"],
            "val/psnr": val_metrics["psnr"],
            "val/sam_deg": val_metrics["sam_deg"],
            "val/bpppc": val_metrics["bpppc"],
        }
        if latent_shape:
            record.update(
                {
                    "model/latent_c": latent_shape[0],
                    "model/latent_h": latent_shape[1],
                    "model/latent_w": latent_shape[2],
                }
            )
        if cr_proxy is not None:
            record["model/cr_proxy"] = cr_proxy

        history.append(record)

        if logger is not None and is_main_process():
            logger.log(record, step=epoch)

        if is_main_process():
            print(
                f"  train_loss={record['train/loss']:.5f} | "
                f"train_psnr={record['train/psnr']:.2f}dB | "
                f"val_loss={record['val/loss']:.5f} | "
                f"val_psnr={record['val/psnr']:.2f}dB | "
                f"val_sam={record['val/sam_deg']:.2f}° | "
                f"bpppc={record['val/bpppc']:.4f}"
            )

        val_psnr = record["val/psnr"]
        if val_psnr > best_val_psnr + early_min_delta:
            best_val_psnr = val_psnr
            epochs_without_improvement = 0

            if is_main_process():
                extra = {
                    "latent_shape": latent_shape,
                    "best_val_psnr": best_val_psnr,
                    "best_val_bpppc": record["val/bpppc"],
                }
                if cr_proxy is not None:
                    extra["cr_proxy"] = cr_proxy

                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    model=model_to_save,
                    optimizer=optimizer,
                    config=config,
                    best_val_loss=val_metrics["loss"],
                    extra=extra,
                )

                if logger is not None:
                    logger.summary["best_val_psnr"] = best_val_psnr
                    logger.summary["best_val_bpppc"] = record["val/bpppc"]
                    logger.summary["best_epoch"] = epoch
        else:
            epochs_without_improvement += 1

        if early_enabled and epochs_without_improvement >= early_patience:
            if is_main_process():
                print(
                    f"\nEarly stopping after {epoch} epochs "
                    f"(patience={early_patience}, no improvement in PSNR)."
                )
            break

    return {
        "best_val_psnr": best_val_psnr,
        "history": history,
    }
