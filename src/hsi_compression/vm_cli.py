from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from hsi_compression.paths import project_root

DEFAULT_VM_CONFIG = project_root() / "configs" / "vms.yaml"
COMMAND_SCRIPTS = {
    "copy-dataset": project_root() / "scripts" / "commands" / "copy-data-set-to-vm.sh",
    "prepare-environment": project_root() / "scripts" / "commands" / "prepare-environment-on-vm.sh",
}


def _expand_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _load_config(config_path: Path) -> list[dict]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"VM config does not exist: {config_path}. Create it from configs/vms.yaml template."
        )

    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict) or "vms" not in raw:
        raise ValueError("VM config must contain a top-level 'vms' list")

    vms = raw["vms"]
    if not isinstance(vms, list):
        raise ValueError("The 'vms' field must be a list")

    names = set()
    normalized: list[dict] = []
    for entry in vms:
        if not isinstance(entry, dict):
            raise ValueError("Each VM entry must be a mapping")

        name = entry.get("name")
        host = entry.get("host")
        key_path = entry.get("ssh_key_path")

        if not name or not host or not key_path:
            raise ValueError("Each VM requires name, host, and ssh_key_path")
        if name in names:
            raise ValueError(f"Duplicate VM name in config: {name}")

        names.add(name)

        port = entry.get("port", 22)
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError(f"Invalid port for VM '{name}': {port}")

        normalized.append(
            {
                "name": str(name),
                "host": str(host),
                "ssh_key_path": str(key_path),
                "port": port,
                "user": str(entry.get("user", os.environ.get("USER", ""))).strip(),
                "repo_url": str(entry.get("repo_url", "")).strip(),
                "dataset_path": entry.get("dataset_path"),
                "remote_project_dir": str(entry.get("remote_project_dir", "/workspace/hsi")),
            }
        )

    return normalized


def _find_vm(vms: list[dict], name: str) -> dict:
    for vm in vms:
        if vm["name"] == name:
            return vm
    raise KeyError(f"VM '{name}' not found in config")


def _print_vm(vm: dict) -> None:
    print(f"name: {vm['name']}")
    print(f"host: {vm['host']}")
    print(f"port: {vm['port']}")
    print(f"user: {vm['user'] or '(empty)'}")
    print(f"ssh_key_path: {vm['ssh_key_path']}")
    print(f"dataset_path: {vm['dataset_path'] or '(not set)'}")
    print(f"remote_project_dir: {vm['remote_project_dir']}")
    print(f"repo_url: {vm['repo_url']}")


def _run_shell(cmd: list[str], env: dict[str, str], dry_run: bool) -> int:
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0

    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


def _build_env(port: int) -> dict[str, str]:
    env = os.environ.copy()
    env["SSH_PORT"] = str(port)
    return env


def cmd_list(args: argparse.Namespace) -> int:
    vms = _load_config(_expand_path(args.config))
    if not vms:
        print("No VMs configured")
        return 0

    for vm in vms:
        print(f"{vm['name']}: {vm['user']}@{vm['host']}:{vm['port']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    vms = _load_config(_expand_path(args.config))
    vm = _find_vm(vms, args.name)
    _print_vm(vm)
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    vms = _load_config(_expand_path(args.config))
    vm = _find_vm(vms, args.name)

    key_path = _expand_path(vm["ssh_key_path"])
    user = args.user or vm["user"]

    if not user:
        raise ValueError("Remote user is empty. Set VM 'user' in config or pass --user.")

    cmd = [
        "ssh",
        "-i",
        str(key_path),
        "-p",
        str(vm["port"]),
        f"{user}@{vm['host']}",
    ]

    if args.dry_run:
        print("Running:", " ".join(cmd))
        return 0

    return int(subprocess.run(cmd, check=False).returncode)


def cmd_run(args: argparse.Namespace) -> int:
    vms = _load_config(_expand_path(args.config))
    vm = _find_vm(vms, args.name)

    script = COMMAND_SCRIPTS[args.command]
    if not script.exists():
        raise FileNotFoundError(f"Command script does not exist: {script}")

    key_path = _expand_path(vm["ssh_key_path"])
    user = args.user or vm["user"]

    if not user:
        raise ValueError("Remote user is empty. Set VM 'user' in config or pass --user.")

    env = _build_env(vm["port"])

    if args.command == "copy-dataset":
        dataset_path = args.dataset_path or vm["dataset_path"]
        if not dataset_path:
            raise ValueError(
                "Dataset path is required for copy-dataset. Set it in config or pass --dataset-path."
            )
        if not vm["repo_url"]:
            raise ValueError("repo_url is required for copy-dataset. Set it in VM config.")
        env["REPO_URL"] = vm["repo_url"]
        cmd = [
            str(script),
            vm["host"],
            str(key_path),
            str(_expand_path(str(dataset_path))),
            user,
        ]
        return _run_shell(cmd, env=env, dry_run=args.dry_run)

    if args.command == "prepare-environment":
        remote_project_dir = args.remote_project_dir or vm["remote_project_dir"]
        if not vm["repo_url"]:
            raise ValueError("repo_url is required for prepare-environment. Set it in VM config.")
        env["REPO_URL"] = vm["repo_url"]
        cmd = [
            str(script),
            vm["host"],
            str(key_path),
            user,
            remote_project_dir,
        ]
        if args.python_version:
            cmd.append(args.python_version)
        return _run_shell(cmd, env=env, dry_run=args.dry_run)

    raise ValueError(f"Unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage training VMs and run predefined commands")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_VM_CONFIG),
        help="Path to VM config YAML (default: configs/vms.yaml)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    list_parser = subparsers.add_parser("list", help="List all configured VMs")
    list_parser.set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show details for one VM")
    show_parser.add_argument("name", help="VM name")
    show_parser.set_defaults(func=cmd_show)

    ssh_parser = subparsers.add_parser("ssh", help="Open an SSH session to a VM")
    ssh_parser.add_argument("name", help="VM name")
    ssh_parser.add_argument("--user", help="Override remote user")
    ssh_parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )
    ssh_parser.set_defaults(func=cmd_ssh)

    run_parser = subparsers.add_parser("run", help="Run a predefined command for a VM")
    run_parser.add_argument("name", help="VM name")
    run_parser.add_argument(
        "command",
        choices=sorted(COMMAND_SCRIPTS.keys()),
        help="Predefined command to execute",
    )
    run_parser.add_argument("--user", help="Override remote user")
    run_parser.add_argument("--dataset-path", help="Required for copy-dataset if not set in config")
    run_parser.add_argument("--remote-project-dir", help="Override remote project directory")
    run_parser.add_argument("--python-version", help="Python version for prepare-environment")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
