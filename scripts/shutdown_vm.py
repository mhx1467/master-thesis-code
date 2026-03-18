import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from hsi_compression.utils import load_project_env

DEFAULT_TARGET_NAME = "Thesis GPU"
DEFAULT_INSTANCES_PATH = "/api/v2/instances"
DEFAULT_BASE_URL = "https://dashboard.tensordock.com"


def _build_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"

    return headers


def _request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = Request(url=url, data=body, headers=headers, method=method)

    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if not raw.strip():
                return {}
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object from {url}, got: {type(parsed)}")
            return parsed
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} while calling {method} {url}. Response: {details}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


def _extract_instance_id(instances_payload: dict[str, Any], target_name: str) -> str:
    data = instances_payload.get("data")

    if isinstance(data, dict):
        instances = data.get("instances", [])
    elif isinstance(data, list):
        instances = data
    else:
        instances = instances_payload.get("instances", [])

    if not isinstance(instances, list):
        raise ValueError("Unexpected payload format: could not find instances list")

    for instance in instances:
        if not isinstance(instance, dict):
            continue

        attributes = instance.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        name = attributes.get("name") or instance.get("name")
        instance_id = instance.get("id")
        if name == target_name and isinstance(instance_id, str) and instance_id.strip():
            return instance_id

    raise RuntimeError(f"VM named '{target_name}' not found in instances list")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a VM by name and stop it via API",
    )
    parser.add_argument(
        "--vm-name",
        default=DEFAULT_TARGET_NAME,
        help=f"VM display name to look for (default: {DEFAULT_TARGET_NAME})",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TENSOR_DOCK_API_BASE_URL", DEFAULT_BASE_URL),
        help=(
            "API base URL (default: TENSOR_DOCK_API_BASE_URL env var or "
            f"{DEFAULT_BASE_URL})"
        ),
    )
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("TENSOR_DOCK_API_KEY_SECRET"),
        help="Bearer token (default: TENSOR_DOCK_API_KEY_SECRET env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stop URL without making the stop request",
    )
    return parser.parse_args()


def main() -> int:
    load_project_env()
    args = parse_args()

    headers = _build_headers(args)
    if "Authorization" not in headers:
        print(
            "Error: provide bearer token via --bearer-token or TENSOR_DOCK_API_KEY_SECRET.",
            file=sys.stderr,
        )
        return 2

    instances_url = urljoin(args.base_url.rstrip("/") + "/", DEFAULT_INSTANCES_PATH.lstrip("/"))

    print(f"Fetching instances: {instances_url}")
    instances_payload = _request_json("GET", instances_url, headers=headers)
    vm_id = _extract_instance_id(instances_payload, args.vm_name)

    stop_url = urljoin(
        args.base_url.rstrip("/") + "/",
        f"{DEFAULT_INSTANCES_PATH.lstrip('/')}/{vm_id}/stop",
    )

    print(f"Found VM '{args.vm_name}' with id: {vm_id}")

    if args.dry_run:
        print(f"Dry run enabled. Would call: POST {stop_url}")
        return 0

    print(f"Stopping VM via: {stop_url}")
    _request_json("POST", stop_url, headers=headers, payload={})
    print("Stop request sent successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
