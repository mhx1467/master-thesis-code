from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PIP = [sys.executable, "-m", "pip"]
CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/"
    "causal_conv1d-1.6.2.post1%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
MAMBA_SSM_WHEEL = (
    "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/"
    "mamba_ssm-2.3.2.post1%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)


def run(cmd, *, cwd: str | Path | None = None, required: bool = True, tail: int = 6000) -> bool:
    print("Running:", " ".join(map(str, cmd)))
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout[-tail:])
    if result.returncode == 0:
        return True
    message = f"Command failed with exit code {result.returncode}: {' '.join(map(str, cmd))}"
    if required:
        raise RuntimeError(message)
    print("Optional command failed:", message)
    return False


def sync_repo(repo_url: str, repo_dir: str | Path, repo_ref: str = "main") -> Path:
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        run(["git", "clone", repo_url, str(repo_dir)])
    run(["git", "fetch", "origin"], cwd=repo_dir)
    run(["git", "checkout", repo_ref], cwd=repo_dir)
    run(["git", "pull", "--ff-only"], cwd=repo_dir, required=False)

    for module_name in list(sys.modules):
        if module_name == "hsi_compression" or module_name.startswith("hsi_compression."):
            del sys.modules[module_name]

    os.chdir(repo_dir)
    print("Repo:", Path.cwd())
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    print("Git:", head)
    return repo_dir


def _marker_path(marker_name: str | Path) -> Path:
    marker = Path(marker_name)
    return marker if marker.is_absolute() else Path("/content") / marker


def _editable_spec(extras: tuple[str, ...]) -> str:
    return f".[{','.join(extras)}]" if extras else "."


def install_base_env(
    *,
    marker_name: str | Path,
    force_reinstall: bool = False,
    project_extras: tuple[str, ...] = ("downstream",),
    extra_packages: tuple[str, ...] = ("eotdl", "tqdm", "matplotlib"),
    optional_boosting: bool = False,
) -> Path:
    marker = _marker_path(marker_name)
    if force_reinstall and marker.exists():
        marker.unlink()
    if marker.exists():
        print("Dependency marker exists, skipping reinstall:", marker)
        return marker

    run(PIP + ["install", "-q", "--upgrade", "pip", "setuptools<82", "wheel", "packaging"])
    run(
        PIP
        + [
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "numpy==1.26.4",
            "pandas==2.2.2",
            "scipy>=1.12,<1.15",
            "scikit-learn>=1.6,<1.8",
        ]
    )
    run(PIP + ["install", "-q", "-e", _editable_spec(project_extras), *extra_packages])
    if optional_boosting:
        run(PIP + ["install", "-q", "lightgbm", "catboost", "xgboost"])

    marker.write_text("installed\n", encoding="utf-8")
    print("Dependencies installed. Restarting runtime to reload binary modules.")
    os.kill(os.getpid(), 9)
    return marker


def install_mamba_env(
    *,
    marker_name: str | Path,
    force_reinstall: bool = False,
    require_cuda: bool = True,
    require_mamba: bool = True,
    project_extras: tuple[str, ...] = ("downstream",),
    extra_packages: tuple[str, ...] = ("eotdl", "tqdm", "matplotlib"),
    error_message: str = "mamba-ssm is required for this notebook.",
) -> Path:
    marker = _marker_path(marker_name)
    if force_reinstall and marker.exists():
        marker.unlink()
    if not marker.exists():
        run(
            PIP
            + [
                "install",
                "-q",
                "--upgrade",
                "pip",
                "setuptools<82",
                "wheel",
                "packaging",
                "pybind11",
                "ninja",
            ]
        )
        run(
            PIP
            + [
                "install",
                "-q",
                "--force-reinstall",
                "torch==2.7.1",
                "torchvision==0.22.1",
                "torchaudio==2.7.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ]
        )
        run(
            PIP
            + [
                "install",
                "-q",
                "--upgrade",
                "--force-reinstall",
                "numpy==1.26.4",
                "pandas==2.2.2",
                "scipy>=1.12,<1.15",
                "scikit-learn>=1.6,<1.8",
            ]
        )
        run(PIP + ["install", "-q", "-e", _editable_spec(project_extras), *extra_packages])
        _verify_prebuilt_mamba_runtime()
        run(PIP + ["install", "-q", "--force-reinstall", "--no-deps", CAUSAL_CONV1D_WHEEL])
        run(PIP + ["install", "-q", "--force-reinstall", "--no-deps", MAMBA_SSM_WHEEL])
        marker.write_text("installed\n", encoding="utf-8")
        print("Dependencies installed. Restarting runtime to reload binary modules.")
        os.kill(os.getpid(), 9)
    else:
        print("Dependency marker exists, skipping reinstall:", marker)

    verify_runtime(require_cuda=require_cuda, require_torch27=True)
    if require_mamba:
        verify_mamba(error_message=error_message)
    return marker


def _verify_prebuilt_mamba_runtime() -> None:
    import torch

    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if python_tag != "cp312":
        raise RuntimeError(f"Prebuilt Mamba wheels expect Python 3.12, got {python_tag}.")
    cxx11_abi = "TRUE" if getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True) else "FALSE"
    if cxx11_abi != "TRUE":
        raise RuntimeError(f"Prebuilt Mamba wheels expect Torch CXX11 ABI TRUE, got {cxx11_abi}.")
    print("Torch CXX11 ABI:", cxx11_abi)


def verify_runtime(*, require_cuda: bool = False, require_torch27: bool = False) -> None:
    import numpy as np
    import torch

    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if require_torch27 and not torch.__version__.startswith("2.7."):
        raise RuntimeError(f"Mamba prebuilt wheels require Torch 2.7.x, got {torch.__version__}.")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("GPU runtime is required. In Colab choose Runtime -> GPU.")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        subprocess.run(["nvidia-smi"], check=False)


def verify_mamba(*, error_message: str) -> None:
    try:
        from mamba_ssm import Mamba  # noqa: F401

        print("mamba-ssm import: ok")
    except Exception as exc:
        raise RuntimeError(error_message) from exc


def is_hyperview2_root(path: str | Path) -> bool:
    path = Path(path)
    required = [
        path / "train_gt.csv",
        path / "submission.csv",
        path / "train/hsi_satellite",
        path / "test/hsi_satellite",
    ]
    return all(item.exists() for item in required)


def find_hyperview2_root(search_root: str | Path) -> Path | None:
    search_root = Path(search_root)
    if is_hyperview2_root(search_root):
        return search_root
    if search_root.exists():
        for candidate in sorted(search_root.rglob("HYPERVIEW2")):
            if is_hyperview2_root(candidate):
                return candidate
    return None
