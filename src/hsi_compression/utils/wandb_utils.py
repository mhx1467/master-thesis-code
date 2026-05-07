import os

import wandb


def init_wandb(
    project: str,
    run_name: str,
    config: dict,
    run_id: str | None = None,
    resume: str | None = None,
):
    api_key = os.environ.get("WANDB_API_KEY")

    if api_key is None:
        print("WARNING: WANDB_API_KEY not set. Running W&B in offline mode.")
        os.environ["WANDB_MODE"] = "offline"
    else:
        wandb.login(key=api_key, relogin=True)

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        id=run_id,
        resume=resume or ("allow" if run_id is not None else None),
    )

    return run
