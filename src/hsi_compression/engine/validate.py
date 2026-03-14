import torch

from hsi_compression.metrics import masked_mse, masked_rmse, masked_sam_deg


@torch.no_grad()
def validate(
    model,
    loader,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_sam_deg = 0.0
    num_batches = 0

    last_latent_shape = None

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["valid_mask"].to(device)

        x_hat, z = model(x)

        loss = masked_mse(x_hat, x, mask)
        rmse = masked_rmse(x_hat, x, mask)
        sam_deg = masked_sam_deg(x_hat, x, mask)

        total_loss += loss.item()
        total_rmse += rmse.item()
        total_sam_deg += sam_deg.item()
        num_batches += 1

        if last_latent_shape is None:
            # z shape is (N, C, H, W)
            last_latent_shape = tuple(z.shape[1:])

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_rmse": total_rmse / max(num_batches, 1),
        "val_sam_deg": total_sam_deg / max(num_batches, 1),
        "latent_shape": last_latent_shape,
    }