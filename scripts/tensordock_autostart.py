#!/usr/bin/env python3
"""Retry-start a TensorDock instance until it becomes available.

The script intentionally reads credentials from environment variables only, so
tokens do not end up in shell history, process lists, or repository files.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, parse, request

DEFAULT_API_BASE = "https://dashboard.tensordock.com/api/v2"
ONLINE_STATUSES = ("running", "online")
STARTING_STATUSES = ("starting", "pending", "provisioning")


class ApiError(RuntimeError):
    def __init__(self, status: int | None, body: str):
        self.status = status
        self.body = body
        super().__init__(f"API request failed with status {status}: {body[:500]}")


@dataclass(frozen=True)
class InstanceView:
    id: str | None
    name: str | None
    status: str | None
    ip_address: str | None
    port_forwards: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regularly try to start a TensorDock instance and notify when it is online."
        )
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("TENSORDOCK_API_BASE", DEFAULT_API_BASE),
        help=f"TensorDock API base URL. Default: {DEFAULT_API_BASE}",
    )
    parser.add_argument(
        "--instance-id",
        default=os.environ.get("TENSORDOCK_INSTANCE_ID"),
        help="Instance UUID. Prefer TENSORDOCK_INSTANCE_ID instead of CLI args.",
    )
    parser.add_argument(
        "--instance-name",
        default=os.environ.get("TENSORDOCK_INSTANCE_NAME"),
        help="Optional instance name used with --list-instances or discovery.",
    )
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=int(os.environ.get("TENSORDOCK_RETRY_INTERVAL_SEC", "300")),
        help="Seconds between retry attempts. Default: 300.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.environ.get("TENSORDOCK_HTTP_TIMEOUT_SEC", "30")),
        help="HTTP timeout per request. Default: 30.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("TENSORDOCK_MAX_ATTEMPTS", "0")),
        help="Maximum attempts before exit. 0 means unlimited.",
    )
    parser.add_argument(
        "--auth-test",
        action="store_true",
        default=os.environ.get("TENSORDOCK_AUTH_TEST", "").lower() in {"1", "true", "yes"},
        help="Run POST /auth/test before polling.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one check/start attempt and exit. Useful for cron or systemd timers.",
    )
    parser.add_argument(
        "--list-instances",
        action="store_true",
        help="List visible instances and exit.",
    )
    parser.add_argument(
        "--notify-webhook-url",
        default=os.environ.get("TENSORDOCK_NOTIFY_WEBHOOK_URL"),
        help="Optional webhook URL. Receives JSON with text/content/message fields.",
    )
    parser.add_argument(
        "--ntfy-topic",
        default=os.environ.get("TENSORDOCK_NTFY_TOPIC"),
        help="Optional ntfy.sh topic for push notifications.",
    )
    parser.add_argument(
        "--notify-cmd",
        default=os.environ.get("TENSORDOCK_NOTIFY_CMD"),
        help=(
            "Optional local command with placeholders: {message}, {status}, {instance_id}. "
            "Example: notify-send 'TensorDock' '{message}'"
        ),
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Continue polling after the first online notification.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def load_token() -> str:
    token = os.environ.get("TENSORDOCK_API_TOKEN")
    if not token:
        raise SystemExit("Missing TENSORDOCK_API_TOKEN environment variable.")
    return token


def request_json(
    method: str,
    base_url: str,
    path: str,
    token: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    data = None
    request_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    req = request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, sanitize_api_body(payload)) from exc
    except error.URLError as exc:
        raise ApiError(None, str(exc.reason)) from exc

    if not payload.strip():
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"raw": payload}


def sanitize_api_body(body: str) -> str:
    """Keep API errors readable without dumping huge HTML bodies."""
    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()[:1000]
    return json.dumps(parsed_body, ensure_ascii=True)[:1000]


def unwrap_resource(resource: dict[str, Any]) -> dict[str, Any]:
    data = resource.get("data", resource)
    if isinstance(data, dict) and "attributes" in data and isinstance(data["attributes"], dict):
        merged = dict(data["attributes"])
        merged.setdefault("id", data.get("id"))
        merged.setdefault("type", data.get("type"))
        return merged
    if isinstance(data, dict):
        return data
    return {}


def normalize_instance(resource: dict[str, Any]) -> InstanceView:
    instance = unwrap_resource(resource)
    status = pick_first(instance, "status", "state", "powerState", "power_state")
    ip_address = pick_first(
        instance,
        "ipAddress",
        "ip_address",
        "publicIpAddress",
        "public_ip_address",
        "publicIp",
        "public_ip",
    )
    port_forwards = instance.get("portForwards") or instance.get("port_forwards") or []
    if not isinstance(port_forwards, list):
        port_forwards = []
    return InstanceView(
        id=pick_first(instance, "id", "uuid"),
        name=pick_first(instance, "name", "hostname"),
        status=str(status) if status is not None else None,
        ip_address=str(ip_address) if ip_address is not None else None,
        port_forwards=port_forwards,
    )


def pick_first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def extract_instances(payload: dict[str, Any]) -> list[InstanceView]:
    data = payload.get("data", payload)
    candidates: Any = None
    if isinstance(data, dict):
        candidates = data.get("instances")
        if candidates is None and isinstance(data.get("attributes"), dict):
            candidates = data["attributes"].get("instances")
    if candidates is None:
        candidates = payload.get("instances")
    if not isinstance(candidates, list):
        return []
    return [normalize_instance(item) for item in candidates if isinstance(item, dict)]


def status_key(status: str | None) -> str:
    return (status or "").strip().lower()


def describe_instance(instance: InstanceView) -> str:
    parts = []
    if instance.name:
        parts.append(instance.name)
    if instance.id:
        parts.append(instance.id)
    if instance.status:
        parts.append(f"status={instance.status}")
    if instance.ip_address:
        parts.append(f"ip={instance.ip_address}")
    if instance.port_forwards:
        ports = ", ".join(describe_port_forward(port) for port in instance.port_forwards)
        parts.append(f"ports=[{ports}]")
    return " ".join(parts) if parts else "<unknown instance>"


def describe_port_forward(port: dict[str, Any]) -> str:
    internal = port.get("internal_port") or port.get("internalPort")
    external = port.get("external_port") or port.get("externalPort")
    if internal and external:
        return f"{external}->{internal}"
    if external:
        return str(external)
    return json.dumps(port, sort_keys=True)


def get_instance(base_url: str, token: str, instance_id: str, timeout: int) -> InstanceView:
    payload = request_json("GET", base_url, f"/instances/{instance_id}", token, timeout=timeout)
    return normalize_instance(payload)


def list_instances(base_url: str, token: str, timeout: int) -> list[InstanceView]:
    payload = request_json("GET", base_url, "/instances", token, timeout=timeout)
    return extract_instances(payload)


def find_instance_by_name(instances: list[InstanceView], name: str) -> InstanceView | None:
    for instance in instances:
        if instance.name == name:
            return instance
    normalized_name = name.lower()
    for instance in instances:
        if instance.name and instance.name.lower() == normalized_name:
            return instance
    return None


def start_instance(base_url: str, token: str, instance_id: str, timeout: int) -> dict[str, Any]:
    return request_json("POST", base_url, f"/instances/{instance_id}/start", token, timeout=timeout)


def auth_test(base_url: str, token: str, timeout: int) -> None:
    request_json("POST", base_url, "/auth/test", token, timeout=timeout)


def send_notification(args: argparse.Namespace, instance: InstanceView) -> None:
    message = f"TensorDock instance is online: {describe_instance(instance)}"
    status = instance.status or ""
    instance_id = instance.id or args.instance_id or ""
    failures = []

    if args.ntfy_topic:
        topic = parse.quote(args.ntfy_topic.strip(), safe="")
        try:
            send_ntfy(topic, message, args.timeout_sec)
        except ApiError as exc:
            failures.append(f"ntfy failed: {exc.body}")

    if args.notify_webhook_url:
        try:
            send_webhook(args.notify_webhook_url, message, args.timeout_sec)
        except ApiError as exc:
            failures.append(f"webhook failed: {exc.body}")

    if args.notify_cmd:
        command = args.notify_cmd.format(
            message=message,
            status=status,
            instance_id=instance_id,
        )
        try:
            subprocess.run(shlex.split(command), timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            failures.append(f"notify command failed: {exc}")

    log(message)
    for failure in failures:
        log(failure)


def send_ntfy(topic: str, message: str, timeout: int) -> None:
    url = f"https://ntfy.sh/{topic}"
    req = request.Request(
        url,
        data=message.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Title": "TensorDock instance online",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout):
            return
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, sanitize_api_body(payload)) from exc
    except error.URLError as exc:
        raise ApiError(None, str(exc.reason)) from exc


def send_webhook(url: str, message: str, timeout: int) -> None:
    body = {
        "text": message,
        "content": message,
        "message": message,
    }
    data = json.dumps(body).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout):
            return
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, sanitize_api_body(payload)) from exc
    except error.URLError as exc:
        raise ApiError(None, str(exc.reason)) from exc


def resolve_instance_id(args: argparse.Namespace, token: str) -> str:
    if args.instance_id:
        return args.instance_id
    if not args.instance_name:
        raise SystemExit("Set TENSORDOCK_INSTANCE_ID, pass --instance-id, or pass --instance-name.")

    instances = list_instances(args.api_base, token, args.timeout_sec)
    instance = find_instance_by_name(instances, args.instance_name)
    if not instance or not instance.id:
        raise SystemExit(f"No TensorDock instance named {args.instance_name!r} was found.")
    log(f"Resolved {args.instance_name!r} to instance id {instance.id}.")
    return instance.id


def print_instances(instances: list[InstanceView]) -> None:
    if not instances:
        print("No instances returned by the API.")
        return
    for instance in instances:
        print(describe_instance(instance))


def main() -> int:
    args = parse_args()
    token = load_token()

    if args.interval_sec < 1:
        raise SystemExit("--interval-sec must be >= 1.")
    if args.timeout_sec < 1:
        raise SystemExit("--timeout-sec must be >= 1.")
    if args.max_attempts < 0:
        raise SystemExit("--max-attempts must be >= 0.")

    try:
        if args.auth_test:
            auth_test(args.api_base, token, args.timeout_sec)
            log("Authentication test succeeded.")

        if args.list_instances:
            print_instances(list_instances(args.api_base, token, args.timeout_sec))
            return 0

        instance_id = resolve_instance_id(args, token)
        notified_online = False
        attempt = 0

        while True:
            attempt += 1
            if args.max_attempts and attempt > args.max_attempts:
                log(f"Reached max attempts ({args.max_attempts}); exiting.")
                return 2

            try:
                instance = get_instance(args.api_base, token, instance_id, args.timeout_sec)
                current_status = status_key(instance.status)
                log(f"Attempt {attempt}: {describe_instance(instance)}")

                if current_status in ONLINE_STATUSES:
                    if not notified_online:
                        send_notification(args, instance)
                        notified_online = True
                    if not args.keep_running:
                        return 0
                elif current_status in STARTING_STATUSES:
                    log("Instance is already starting; waiting for it to become online.")
                else:
                    log("Requesting instance start.")
                    start_instance(args.api_base, token, instance_id, args.timeout_sec)
                    log("Start request accepted by API.")
            except ApiError as exc:
                status = f"HTTP {exc.status}" if exc.status else "network error"
                log(f"Attempt {attempt} failed ({status}): {exc.body}")

            if args.once:
                return 0
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        log("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
