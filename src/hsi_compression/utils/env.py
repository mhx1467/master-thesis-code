from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> Path | None:
    """Load .env from repository root if present, without overriding existing vars."""
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        return env_path

    load_dotenv(override=False)
    return None
