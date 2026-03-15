import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import masked_rmse
from hsi_compression.utils.distributed import is_main_process, reduce_mean


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

    use_progress = show_progress and is_main_process()

    progress = loader
    if use_progress:
        desc = "Train"
        if epoch is not None and total_epochs is not None:
            desc = f"Train {epoch}/{total_epochs}"
        progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)

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

        if use_progress:
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rmse": f"{rmse.item():.4f}",
            })

    avg_loss = total_loss / max(num_batches, 1)
    avg_rmse = total_rmse / max(num_batches, 1)

    avg_loss = reduce_mean(avg_loss, device)
    avg_rmse = reduce_mean(avg_rmse, device)

    return {
        "loss": avg_loss,
        "rmse": avg_rmse,
    }