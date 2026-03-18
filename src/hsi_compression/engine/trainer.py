from pathlib import Path

from hsi_compression.engine.checkpointing import (
    find_resume_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_last_checkpoint_async,
)
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
    resume: bool = False,
    sam_every_n_epochs: int = 10,
):
    checkpoint_path = Path(checkpoint_path)
    best_val_psnr   = float("-inf")
    last_sam_deg    = None
    start_epoch     = 1
    history         = []
    _last_save_thread = None 

    training_cfg    = config.get("training", {})
    early_cfg       = training_cfg.get("early_stopping", {})
    early_enabled   = early_cfg.get("enabled", False)
    early_patience  = early_cfg.get("patience", 20)
    early_min_delta = early_cfg.get("min_delta", 0.0)
    epochs_without_improvement = 0

    if resume:
        last_path = find_resume_checkpoint(checkpoint_path)
        if last_path is not None and is_main_process():
            print(f"\nWznawianie z: {last_path}")
            ckpt = load_checkpoint(
                path=last_path, model=model,
                optimizer=optimizer, scheduler=scheduler,
                map_location=device,
            )
            start_epoch   = ckpt["epoch"] + 1
            best_val_psnr = ckpt.get("extra", {}).get("val_psnr", float("-inf"))
            print(f"Wznowiono od epoki {start_epoch} | Best PSNR: {best_val_psnr:.2f} dB\n")
        elif is_main_process():
            print("Brak last.pt — trening od początku.")

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            loss_fn=loss_fn, device=device, epoch=epoch,
            total_epochs=epochs, show_progress=show_progress,
            grad_clip_max_norm=grad_clip_max_norm,
        )

        compute_sam = (epoch % sam_every_n_epochs == 0) or (epoch == epochs)

        val_metrics = validate_one_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn,
            device=device, num_input_bands=num_input_bands,
            quantization_bits=quantization_bits,
            epoch=epoch, total_epochs=epochs,
            show_progress=show_progress,
            compute_sam=compute_sam,
        )

        if scheduler is not None:
            scheduler.step()

        if val_metrics["sam_deg"] is not None:
            last_sam_deg = val_metrics["sam_deg"]

        latent_shape = val_metrics.get("latent_shape")
        model_raw    = model.module if hasattr(model, "module") else model

        cr_proxy = None
        if hasattr(model_raw, "compression_ratio_proxy") and latent_shape:
            cr_proxy = model_raw.compression_ratio_proxy(
                input_shape=(num_input_bands, 128, 128),
                latent_shape=latent_shape,
            )

        record = {
            "epoch":       epoch,
            "train/loss":  train_metrics["loss"],
            "train/rmse":  train_metrics["rmse"],
            "train/psnr":  train_metrics["psnr"],
            "val/loss":    val_metrics["loss"],
            "val/rmse":    val_metrics["rmse"],
            "val/psnr":    val_metrics["psnr"],
            "val/bpppc":   val_metrics["bpppc"],
        }
        if last_sam_deg is not None:
            record["val/sam_deg"] = last_sam_deg
        if latent_shape:
            record.update({
                "model/latent_c": latent_shape[0],
                "model/latent_h": latent_shape[1],
                "model/latent_w": latent_shape[2],
            })
        if cr_proxy is not None:
            record["model/cr_proxy"] = cr_proxy

        history.append(record)

        if logger is not None and is_main_process():
            logger.log(record, step=epoch)

        if is_main_process():
            sam_str = f"sam={last_sam_deg:.2f}°" if last_sam_deg else ""
            sam_tag = " ← SAM computed" if compute_sam else ""
            print(
                f"  train={record['train/psnr']:.2f}dB | "
                f"val={record['val/psnr']:.2f}dB | "
                f"{sam_str} | "
                f"bpppc={record['val/bpppc']:.4f}"
                f"{sam_tag}"
            )

        if is_main_process():
            if _last_save_thread is not None:
                _last_save_thread.join()
            _last_save_thread = save_last_checkpoint_async(
                checkpoint_path=checkpoint_path, epoch=epoch,
                model=model_raw, optimizer=optimizer, config=config,
                val_metrics=val_metrics, scheduler=scheduler,
            )

            val_psnr = record["val/psnr"]
            if val_psnr > best_val_psnr + early_min_delta:
                best_val_psnr = val_psnr
                epochs_without_improvement = 0
                save_checkpoint(
                    path=checkpoint_path, epoch=epoch,
                    model=model_raw, optimizer=optimizer,
                    config=config, best_val_loss=val_metrics["loss"],
                    scheduler=scheduler,
                    extra={
                        "latent_shape":   latent_shape,
                        "best_val_psnr":  best_val_psnr,
                        "best_val_bpppc": record["val/bpppc"],
                        "val_sam_deg":    last_sam_deg,
                        "cr_proxy":       cr_proxy,
                    },
                )
                print(f"New best (PSNR={best_val_psnr:.2f} dB)")

                if logger is not None:
                    logger.summary["best_val_psnr"]  = best_val_psnr
                    logger.summary["best_val_bpppc"] = record["val/bpppc"]
                    logger.summary["best_epoch"]     = epoch
            else:
                epochs_without_improvement += 1

        if early_enabled and epochs_without_improvement >= early_patience:
            if is_main_process():
                print(f"\nEarly stopping po {epoch} epokach.")
            break

    if _last_save_thread is not None:
        _last_save_thread.join()

    return {"best_val_psnr": best_val_psnr, "history": history}
