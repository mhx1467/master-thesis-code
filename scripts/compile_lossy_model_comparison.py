import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

FIELDNAMES = [
    "slug",
    "label",
    "family",
    "status",
    "model_name",
    "variant",
    "objective",
    "split",
    "difficulty",
    "num_samples",
    "num_input_bands",
    "num_params",
    "psnr",
    "ssim",
    "sa_deg",
    "likelihood_bpppc",
    "actual_bpppc",
    "actual_compression_ratio",
    "actual_psnr",
    "actual_ssim",
    "actual_sa_deg",
    "encode_ms_per_batch",
    "decode_ms_per_batch",
    "protocol_warnings",
    "checkpoint",
    "eval_json",
    "notes",
]

REFERENCE_RESULT_STATUSES = {"reference_comparable", "needs_eval"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile lossy baseline and Mamba evaluation records into comparison tables."
    )
    parser.add_argument(
        "--manifest",
        default="configs/eval/lossy_model_comparison.yaml",
        help="YAML manifest describing checkpoints and available eval JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis/lossy_model_comparison",
        help="Directory for comparison CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        default="$DATASET_ROOT",
        help="Dataset root string used when printing missing evaluation commands.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(path)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_raw(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _matches_int(actual: Any, expected: Any) -> bool:
    try:
        return int(actual) == int(expected)
    except (TypeError, ValueError):
        return False


def _metrics_from_eval(data: dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}

    if "metrics" in data:
        metrics = data.get("metrics") or {}
        return {
            "model_name": data.get("model"),
            "variant": data.get("variant"),
            "objective": data.get("objective"),
            "split": data.get("split"),
            "difficulty": data.get("difficulty"),
            "num_samples": data.get("num_samples"),
            "num_input_bands": data.get("num_input_bands"),
            "num_params": data.get("params"),
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "sa_deg": metrics.get("sa_deg"),
            "likelihood_bpppc": metrics.get("likelihood_bpppc"),
            "actual_bpppc": metrics.get("actual_bpppc"),
            "actual_compression_ratio": metrics.get("actual_compression_ratio"),
            "actual_psnr": metrics.get("actual_psnr"),
            "actual_ssim": metrics.get("actual_ssim"),
            "actual_sa_deg": metrics.get("actual_sa_deg"),
            "encode_ms_per_batch": metrics.get("encode_ms_per_batch"),
            "decode_ms_per_batch": metrics.get("decode_ms_per_batch"),
        }

    return {
        "model_name": data.get("model_name"),
        "variant": data.get("variant"),
        "objective": data.get("objective"),
        "split": data.get("split"),
        "difficulty": data.get("difficulty"),
        "num_samples": data.get("num_samples"),
        "num_input_bands": data.get("num_input_bands"),
        "num_params": data.get("num_params"),
        "psnr": data.get("psnr"),
        "ssim": data.get("ssim"),
        "sa_deg": _first_value(data.get("sam_deg"), data.get("sa_deg")),
        "likelihood_bpppc": data.get("likelihood_bpppc"),
        "actual_bpppc": data.get("actual_bpppc"),
        "actual_compression_ratio": data.get("actual_compression_ratio"),
        "actual_psnr": data.get("actual_psnr"),
        "actual_ssim": data.get("actual_ssim"),
        "actual_sa_deg": _first_value(data.get("actual_sam_deg"), data.get("actual_sa_deg")),
        "encode_ms_per_batch": data.get("encode_ms_per_batch"),
        "decode_ms_per_batch": data.get("decode_ms_per_batch"),
    }


def _protocol_warnings(row: dict[str, Any], protocol: dict[str, Any]) -> str | None:
    warnings = []
    expected_bands = protocol.get("expected_num_input_bands")
    if (
        expected_bands is not None
        and row.get("num_input_bands") is not None
        and not _matches_int(row["num_input_bands"], expected_bands)
    ):
        warnings.append(f"num_input_bands={row['num_input_bands']} expected={expected_bands}")

    expected_samples = protocol.get("expected_num_samples")
    if (
        expected_samples is not None
        and row.get("num_samples") is not None
        and not _matches_int(row["num_samples"], expected_samples)
    ):
        warnings.append(f"num_samples={row['num_samples']} expected={expected_samples}")

    if row.get("status") == "reference_comparable" and _as_float(row["actual_bpppc"]) is None:
        warnings.append("reference row is missing actual_bpppc")

    if not warnings:
        return None
    return "; ".join(warnings)


def _row_from_entry(entry: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    eval_path = _resolve(entry.get("eval_json")) or _resolve(entry.get("legacy_eval_json"))
    eval_data = _load_json(eval_path)
    metrics = _metrics_from_eval(eval_data)

    row = dict.fromkeys(FIELDNAMES)
    row.update(
        {
            "slug": entry["slug"],
            "label": entry["label"],
            "family": entry.get("family"),
            "status": entry.get("status", "unknown"),
            "checkpoint": entry.get("checkpoint"),
            "eval_json": str(eval_path) if eval_path is not None else None,
            "notes": entry.get("notes"),
        }
    )
    for key, value in metrics.items():
        if key in row:
            row[key] = value
    row["protocol_warnings"] = _protocol_warnings(row, protocol)
    return row


def _format_float(value: Any, places: int = 4) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.{places}f}"


def _format_bpppc(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.6f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _eval_command(
    entry: dict[str, Any],
    protocol: dict[str, Any],
    dataset_root: str,
    batch_size: int,
    num_workers: int,
) -> str | None:
    if entry.get("status") != "needs_eval":
        return None
    checkpoint = entry.get("checkpoint")
    if not checkpoint:
        return None
    split = protocol.get("split", "test")
    difficulty = protocol.get("difficulty", "easy")
    run_name = f"lossy_compare_{entry['slug']}_{difficulty}_{split}"
    return (
        "python scripts/evaluate.py "
        f"{checkpoint} {dataset_root} "
        f"--split {split} --difficulty {difficulty} "
        f"--batch-size {batch_size} --num-workers {num_workers} "
        f"--run-name {run_name} --save-json --disable-wandb --no-progress"
    )


def _has_any_metric(row: dict[str, Any]) -> bool:
    metric_keys = [
        "psnr",
        "ssim",
        "sa_deg",
        "likelihood_bpppc",
        "actual_bpppc",
        "actual_psnr",
        "actual_ssim",
        "actual_sa_deg",
    ]
    return any(row.get(key) is not None for key in metric_keys)


def _is_reference_comparable_result(row: dict[str, Any]) -> bool:
    return (
        row.get("status") in REFERENCE_RESULT_STATUSES
        and _as_float(row.get("actual_bpppc")) is not None
        and not row.get("protocol_warnings")
    )


def _reference_comparable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_reference_comparable_result(row)]


def _decoded_metric(row: dict[str, Any], actual_key: str, forward_key: str) -> Any:
    return _first_value(row.get(actual_key), row.get(forward_key))


def _notes_with_warnings(row: dict[str, Any]) -> str:
    notes = row.get("notes") or ""
    warnings = row.get("protocol_warnings")
    if warnings:
        if notes:
            return f"{notes} Protocol warning: {warnings}."
        return f"Protocol warning: {warnings}."
    return notes


def _write_summary(
    path: Path,
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    eval_commands: list[str],
) -> None:
    reference_rows = _reference_comparable_rows(rows)
    reference_rows = sorted(
        reference_rows,
        key=lambda row: (_first_float(row["actual_bpppc"], 1e9) or 1e9, row["label"]),
    )
    nonreference_metric_rows = [
        row for row in rows if not _is_reference_comparable_result(row) and _has_any_metric(row)
    ]

    lines = [
        "# Lossy Model Comparison",
        "",
        f"Protocol: {protocol.get('dataset_protocol', 'n/a')}",
        f"Input source: {protocol.get('dataset_source', 'n/a')}",
        f"Split: {protocol.get('difficulty', 'easy')}/{protocol.get('split', 'test')}",
        f"Metric policy: {protocol.get('metric_policy', 'actual_bpppc is primary')}",
        "",
        "## Reference-Comparable Results",
        "",
        "| model | family | decoded PSNR dB | decoded SSIM | decoded SA deg | actual bpppc | actual CR | likelihood bpppc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if reference_rows:
        for row in reference_rows:
            lines.append(
                "| {label} | {family} | {psnr} | {ssim} | {sa} | {bpppc} | {cr} | {likelihood} |".format(
                    label=row["label"],
                    family=row["family"],
                    psnr=_format_float(_decoded_metric(row, "actual_psnr", "psnr")),
                    ssim=_format_float(_decoded_metric(row, "actual_ssim", "ssim")),
                    sa=_format_float(_decoded_metric(row, "actual_sa_deg", "sa_deg")),
                    bpppc=_format_bpppc(row["actual_bpppc"]),
                    cr=_format_float(row["actual_compression_ratio"], places=2),
                    likelihood=_format_bpppc(row["likelihood_bpppc"]),
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Gaps And Non-Comparable Rows",
            "",
            "| model | family | status | reason / notes |",
            "|---|---|---|---|",
        ]
    )
    for row in gap_rows:
        lines.append(
            f"| {row['label']} | {row['family']} | {row['status']} | {_notes_with_warnings(row)} |"
        )

    if nonreference_metric_rows:
        lines.extend(
            [
                "",
                "## Legacy / Non-Comparable Metrics",
                "",
                "These rows are useful for orientation only. They are excluded from the "
                "reference-comparable table when they violate the active protocol or miss measured "
                "bitstream bitrate.",
                "",
                "| model | status | decoded PSNR dB | decoded SSIM | decoded SA deg | actual bpppc | samples | bands | warnings |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in nonreference_metric_rows:
            lines.append(
                "| {label} | {status} | {psnr} | {ssim} | {sa} | {bpppc} | "
                "{samples} | {bands} | {warnings} |".format(
                    label=row["label"],
                    status=row["status"],
                    psnr=_format_float(_decoded_metric(row, "actual_psnr", "psnr")),
                    ssim=_format_float(_decoded_metric(row, "actual_ssim", "ssim")),
                    sa=_format_float(_decoded_metric(row, "actual_sa_deg", "sa_deg")),
                    bpppc=_format_bpppc(row["actual_bpppc"]),
                    samples=_format_raw(row["num_samples"]),
                    bands=_format_raw(row["num_input_bands"]),
                    warnings=row.get("protocol_warnings") or "",
                )
            )

    if eval_commands:
        lines.extend(["", "## Missing Evaluation Commands", ""])
        lines.extend(["```bash", *eval_commands, "```"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    entries = manifest.get("models", [])

    rows = [_row_from_entry(entry, protocol) for entry in entries]
    gap_rows = [row for row in rows if not _is_reference_comparable_result(row)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "comparison_all.csv", rows)
    _write_csv(
        output_dir / "comparison_reference_comparable.csv",
        _reference_comparable_rows(rows),
    )
    _write_csv(output_dir / "comparison_gaps.csv", gap_rows)

    eval_commands = []
    for entry, row in zip(entries, rows, strict=True):
        if _is_reference_comparable_result(row):
            continue
        command = _eval_command(
            entry=entry,
            protocol=protocol,
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if command:
            eval_commands.append(command)

    _write_summary(
        output_dir / "summary.md",
        protocol=protocol,
        rows=rows,
        gap_rows=gap_rows,
        eval_commands=eval_commands,
    )

    manifest_copy = output_dir / "manifest_used.yaml"
    manifest_copy.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved: {output_dir / 'summary.md'}")
    print(f"Saved: {output_dir / 'comparison_all.csv'}")
    print(f"Saved: {output_dir / 'comparison_reference_comparable.csv'}")
    print(f"Saved: {output_dir / 'comparison_gaps.csv'}")


if __name__ == "__main__":
    main()
