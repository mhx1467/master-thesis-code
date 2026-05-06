import argparse
from pathlib import Path

import wandb

DEFAULT_ENTITY = "hmadej-silesian-university-of-technology"
DEFAULT_PROJECT = "hsi-compression-paper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the best model artifact for a W&B run.")
    parser.add_argument("run", help="Run id, run name, or full entity/project/run path.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--alias", default="best")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--filename", default=None)
    return parser.parse_args()


def resolve_run(api: wandb.Api, run_ref: str, entity: str, project: str):
    if run_ref.count("/") == 2:
        return api.run(run_ref)

    try:
        return api.run(f"{entity}/{project}/{run_ref}")
    except Exception:
        pass

    runs = [
        run for run in api.runs(f"{entity}/{project}") if run.id == run_ref or run.name == run_ref
    ]
    if not runs:
        raise SystemExit(f"No W&B run found for '{run_ref}' in {entity}/{project}")
    if len(runs) > 1:
        names = ", ".join(f"{run.id}:{run.name}" for run in runs)
        raise SystemExit(f"Run reference is ambiguous: {names}")
    return runs[0]


def select_artifact(run, alias: str):
    candidates = [
        artifact
        for artifact in run.logged_artifacts()
        if artifact.type == "model" and alias in artifact.aliases
    ]
    if not candidates:
        raise SystemExit(f"Run {run.id} has no model artifact with alias '{alias}'")
    candidates.sort(key=lambda artifact: artifact.created_at or "", reverse=True)
    return candidates[0]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    run = resolve_run(api, args.run, args.entity, args.project)
    artifact = select_artifact(run, args.alias)

    download_dir = Path(artifact.download(root=str(output_dir / f".wandb_{run.id}_{args.alias}")))
    checkpoint_files = sorted(download_dir.glob("*.pt"))
    if not checkpoint_files:
        raise SystemExit(f"Artifact {artifact.name} did not contain a .pt checkpoint")

    source = checkpoint_files[0]
    target = output_dir / (args.filename or source.name)
    target.write_bytes(source.read_bytes())
    print(target)


if __name__ == "__main__":
    main()
