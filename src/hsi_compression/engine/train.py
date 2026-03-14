import torch

from hsi_compression.metrics import masked_mse, masked_rmse


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
):
    model.train()

    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["valid_mask"].to(device)

        optimizer.zero_grad()

        x_hat, z = model(x)
        loss = masked_mse(x_hat, x, mask)
        loss.backward()
        optimizer.step()

        rmse = masked_rmse(x_hat.detach(), x, mask)

        total_loss += loss.item()
        total_rmse += rmse.item()
        num_batches += 1

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_rmse": total_rmse / max(num_batches, 1),
    }