import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import masked_rmse


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
):
    model.train()

    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0

    progress = loader
    if show_progress:
        desc = "Train"
        if epoch is not None and total_epochs is not None:
            desc = f"Train {epoch}/{total_epochs}"
        progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        x = batch["x"].to(device)
        mask = batch["valid_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(x)
        x_hat = outputs["x_hat"]

        loss = loss_fn(x_hat, x, mask)
        loss.backward()
        optimizer.step()

        rmse = masked_rmse(x_hat.detach(), x, mask)

        total_loss += loss.item()
        total_rmse += rmse.item()
        num_batches += 1

        if show_progress:
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rmse": f"{rmse.item():.4f}",
            })

    return {
        "loss": total_loss / max(num_batches, 1),
        "rmse": total_rmse / max(num_batches, 1),
    }