from pathlib import Path

import torch

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
    use_amp: bool = True,
):
    checkpoint_path = Path(checkpoint_path)
    best_val_loss = float("inf")
    best_val_psnr = float("-inf")
    last_sam_deg = None
    start_epoch = 1
    history = []
    _last_save_thread = None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    training_cfg = config.get("training", {})
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", False)
    early_patience = early_cfg.get("patience", 20)
    early_psnr_min_delta = early_cfg.get("psnr_min_delta", early_cfg.get("min_delta", 0.0))
    epochs_without_improvement = 0

    if resume:
        last_path = find_resume_checkpoint(checkpoint_path)
        if last_path is not None and is_main_process():
            print(f"\nResuming from: {last_path}")
            ckpt = load_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_val_psnr = ckpt.get("extra", {}).get("best_val_psnr", float("-inf"))
            print(
                f"Resumed {start_epoch} | Best loss: {best_val_loss:.6f} | Best PSNR: {best_val_psnr:.2f} dB\n"
            )
        elif is_main_process():
            print("No last.pt found — starting training from scratch.")

    model_raw = model.module if hasattr(model, "module") else model
    num_params = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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
            scaler=scaler,
            use_amp=use_amp,
        )

        compute_sam = (epoch % sam_every_n_epochs == 0) or (epoch == epochs)

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
            compute_sam=compute_sam,
            use_amp=use_amp,
        )

        if scheduler is not None:
            scheduler.step()

        if val_metrics["masked_sam_deg"] is not None:
            last_sam_deg = val_metrics["masked_sam_deg"]

        latent_shape = val_metrics.get("latent_shape")
        cr_proxy = None
        if hasattr(model_raw, "compression_ratio_proxy") and latent_shape:
            cr_proxy = model_raw.compression_ratio_proxy(
                input_shape=(num_input_bands, 128, 128),
                latent_shape=latent_shape,
            )

        peak_vram_gb = None
        if device.type == "cuda":
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3

        record = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/masked_mse": train_metrics["masked_mse"],
            "train/masked_mae": train_metrics["masked_mae"],
            "train/masked_psnr": train_metrics["masked_psnr"],
            "train/masked_sam_deg": train_metrics["masked_sam_deg"],
            "train/mse": train_metrics["mse"],
            "train/mae": train_metrics["mae"],
            "train/psnr": train_metrics["psnr"],
            "train/sam_deg": train_metrics["sam_deg"],
            "train/invalid_mae": train_metrics["invalid_mae"],
            "train/epoch_time_sec": train_metrics["epoch_time_sec"],
            "val/loss": val_metrics["loss"],
            "val/masked_mse": val_metrics["masked_mse"],
            "val/masked_mae": val_metrics["masked_mae"],
            "val/masked_psnr": val_metrics["masked_psnr"],
            "val/masked_sam_deg": val_metrics["masked_sam_deg"],
            "val/mse": val_metrics["mse"],
            "val/mae": val_metrics["mae"],
            "val/psnr": val_metrics["psnr"],
            "val/sam_deg": val_metrics["sam_deg"],
            "val/invalid_mae": val_metrics["invalid_mae"],
            "val/bpppc": val_metrics["bpppc"],
            "val/epoch_time_sec": val_metrics["epoch_time_sec"],
            "model/num_params": num_params,
        }
        if latent_shape:
            for i, dim in enumerate(latent_shape):
                record[f"model/latent_dim_{i}"] = dim
        if cr_proxy is not None:
            record["model/cr_proxy"] = cr_proxy
        if peak_vram_gb is not None:
            record["system/peak_vram_gb"] = peak_vram_gb

        history.append(record)

        if logger is not None and is_main_process():
            logger.log(record, step=epoch)

        if is_main_process():
            masked_sam_val = record["val/masked_sam_deg"]
            masked_sam_str = f"{masked_sam_val:.2f}°" if masked_sam_val is not None else "n/a"
            print(
                f"  train mPSNR={record['train/masked_psnr']:.2f}dB | "
                f"val mPSNR={record['val/masked_psnr']:.2f}dB | "
                f"val PSNR={record['val/psnr']:.2f}dB | "
                f"val mSAM={masked_sam_str} | "
                f"bpppc={record['val/bpppc']:.4f}"
            )

        if is_main_process():
            if _last_save_thread is not None:
                _last_save_thread.join()
            _last_save_thread = save_last_checkpoint_async(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                model=model_raw,
                optimizer=optimizer,
                config=config,
                val_metrics=val_metrics,
                scheduler=scheduler,
            )

            val_loss = record["val/loss"]
            val_psnr = record["val/masked_psnr"]
            if val_psnr > best_val_psnr + early_psnr_min_delta:
                best_val_loss = val_loss
                best_val_psnr = val_psnr
                epochs_without_improvement = 0
                save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    model=model_raw,
                    optimizer=optimizer,
                    config=config,
                    best_val_loss=best_val_loss,
                    scheduler=scheduler,
                    extra={
                        "latent_shape": latent_shape,
                        "best_val_psnr": best_val_psnr,
                        "best_val_bpppc": record["val/bpppc"],
                        "val_masked_sam_deg": last_sam_deg,
                        "cr_proxy": cr_proxy,
                    },
                )
                print(f"New best masked PSNR ({best_val_psnr:.2f} dB, loss={best_val_loss:.6f})")

                if logger is not None:
                    logger.summary["best_val_loss"] = best_val_loss
                    logger.summary["best_val_masked_psnr"] = best_val_psnr
                    logger.summary["best_val_bpppc"] = record["val/bpppc"]
                    logger.summary["best_epoch"] = epoch

                    import wandb

                    artifact = wandb.Artifact(
                        name=f"model-{logger.id}",
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "val_loss": best_val_loss,
                            "val_masked_psnr": best_val_psnr,
                            "val_masked_sam_deg": last_sam_deg,
                            "val_bpppc": record["val/bpppc"],
                            "cr_proxy": cr_proxy,
                            "latent_shape": str(latent_shape),
                        },
                    )
                    artifact.add_file(str(checkpoint_path))
                    logger.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
            else:
                epochs_without_improvement += 1

        if early_enabled and epochs_without_improvement >= early_patience:
            if is_main_process():
                print(f"\nEarly stopping po {epoch} epokach.")
            break

    if _last_save_thread is not None:
        _last_save_thread.join()

    return {"best_val_loss": best_val_loss, "best_val_psnr": best_val_psnr, "history": history}
