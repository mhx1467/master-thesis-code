from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    return project_root() / "artifacts"


def stats_dir() -> Path:
    return artifacts_root() / "stats"


def checkpoints_dir() -> Path:
    return artifacts_root() / "checkpoints"


def logs_dir() -> Path:
    return artifacts_root() / "logs"


def figures_dir() -> Path:
    return artifacts_root() / "figures"


def ensure_artifact_dirs() -> None:
    for p in [stats_dir(), checkpoints_dir(), logs_dir(), figures_dir()]:
        p.mkdir(parents=True, exist_ok=True)


def default_stats_path(difficulty: str = "easy") -> Path:
    return stats_dir() / f"band_stats_{difficulty}_train_full.pt"
