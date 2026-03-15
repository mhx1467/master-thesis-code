import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import masked_rmse, masked_sam_deg
from hsi_compression.utils.distributed import is_main_process, reduce_mean


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
):
    model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_sam_deg = 0.0
    num_batches = 0
    latent_shape = None

    use_progress = show_progress and is_main_process()

    progress = loader
    if use_progress:
        desc = "Val"
        if epoch is not None and total_epochs is not None:
            desc = f"Val {epoch}/{total_epochs}"
        progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)

        outputs = model(x)
        x_hat = outputs["x_hat"]
        z = outputs.get("z")

        loss = loss_fn(x_hat, x, mask)
        rmse = masked_rmse(x_hat, x, mask)
        sam_deg = masked_sam_deg(x_hat, x, mask)

        total_loss += loss.item()
        total_rmse += rmse.item()
        total_sam_deg += sam_deg.item()
        num_batches += 1

        if latent_shape is None and z is not None:
            latent_shape = tuple(z.shape[1:])

        if use_progress:
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rmse": f"{rmse.item():.4f}",
                "sam": f"{sam_deg.item():.2f}",
            })

    avg_loss = total_loss / max(num_batches, 1)
    avg_rmse = total_rmse / max(num_batches, 1)
    avg_sam_deg = total_sam_deg / max(num_batches, 1)

    avg_loss = reduce_mean(avg_loss, device)
    avg_rmse = reduce_mean(avg_rmse, device)
    avg_sam_deg = reduce_mean(avg_sam_deg, device)

    return {
        "loss": avg_loss,
        "rmse": avg_rmse,
        "sam_deg": avg_sam_deg,
        "latent_shape": latent_shape,
    }