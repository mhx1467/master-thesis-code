import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from hsi_compression.downstream import (
    DEFAULT_REGRESSORS,
    HYPERVIEW2_FEATURE_SETS,
    HYPERVIEW2_MODALITY_DIRS,
    HYPERVIEW2_TARGET_COLUMNS,
    Hyperview2FeatureDataset,
    available_regressor_names,
    build_hyperview2_regressor,
    build_hyperview2_samples,
    compute_regression_metrics,
    split_samples,
)
from hsi_compression.downstream.hyperview2_regressors import (
    BOOSTING_REGRESSORS,
    REGRESSOR_SPECS,
)
from hsi_compression.utils.git import get_git_commit_hash, is_git_dirty


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ready-made tabular regressors on HYPERVIEW2 spectral features."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="data/hyperview2/HYPERVIEW2",
        help="Path to the canonical HYPERVIEW2 root containing train_gt.csv and train/.",
    )
    parser.add_argument(
        "--modality",
        default="prisma",
        choices=tuple(HYPERVIEW2_MODALITY_DIRS),
        help="Input modality directory to use.",
    )
    parser.add_argument(
        "--feature-set",
        default="mean_std_derivatives",
        choices=HYPERVIEW2_FEATURE_SETS,
        help="Spectral feature family extracted before fitting tabular regressors.",
    )
    parser.add_argument(
        "--normalization",
        default="none",
        choices=("none", "minmax", "percentile", "reflectance_0_1", "hyspecnet"),
        help="Per-sample cube normalization before feature extraction.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_REGRESSORS),
        help="Regressor names, or 'all' / 'boosting'.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--output-dir",
        default="artifacts/downstream/hyperview2_regressors",
        help="Directory for JSON and CSV summaries.",
    )
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="Fail instead of skipping regressors whose optional dependencies are missing.",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _expand_model_names(names: list[str]) -> list[str]:
    expanded: list[str] = []
    for name in names:
        if name == "all":
            expanded.extend(REGRESSOR_SPECS)
        elif name == "boosting":
            expanded.extend(BOOSTING_REGRESSORS)
        else:
            expanded.append(name)
    return list(dict.fromkeys(expanded))


def _extract_features(
    samples,
    *,
    modality: str,
    normalization: str,
    feature_set: str,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = Hyperview2FeatureDataset(
        samples,
        modality=modality,
        normalization=normalization,
        feature_set=feature_set,
    )
    iterator = range(len(dataset))
    if show_progress:
        iterator = tqdm(iterator, desc=f"Extract {modality} features")

    features = []
    targets = []
    sample_ids = []
    for idx in iterator:
        item = dataset[idx]
        features.append(item["features"].numpy())
        targets.append(item["target"].numpy())
        sample_ids.append(str(item["sample_id"]))

    return (
        np.stack(features, axis=0).astype(np.float32),
        np.stack(targets, axis=0).astype(np.float32),
        sample_ids,
    )


def _baseline_mse(y_train: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    train_mean = y_train.mean(axis=0, keepdims=True)
    return ((y_val - train_mean) ** 2).mean(axis=0).astype(np.float32)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "status",
        "hyperview_score",
        "mean_mse",
        "mean_mae",
        "fit_time_sec",
        "predict_time_sec",
        *[f"{target}_rmse" for target in HYPERVIEW2_TARGET_COLUMNS],
        *[f"{target}_relative_mse" for target in HYPERVIEW2_TARGET_COLUMNS],
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = _parse_args()
    if args.list_models:
        for name, spec in REGRESSOR_SPECS.items():
            availability = "available" if name in available_regressor_names() else "missing"
            print(f"{name}\t{availability}\t{spec.dependency}\t{spec.description}")
        return

    samples = build_hyperview2_samples(
        args.dataset_root,
        modality=args.modality,
        split="train",
        max_samples=args.max_samples,
    )
    train_samples, val_samples = split_samples(
        samples, val_fraction=args.val_fraction, seed=args.seed
    )
    x_train, y_train, train_ids = _extract_features(
        train_samples,
        modality=args.modality,
        normalization=args.normalization,
        feature_set=args.feature_set,
        show_progress=not args.no_progress,
    )
    x_val, y_val, val_ids = _extract_features(
        val_samples,
        modality=args.modality,
        normalization=args.normalization,
        feature_set=args.feature_set,
        show_progress=not args.no_progress,
    )
    baseline_mse = _baseline_mse(y_train, y_val)
    model_names = _expand_model_names(args.models)

    rows = []
    results = {
        "protocol": {
            "dataset": "HYPERVIEW2",
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "split_source": "train_gt.csv fixed internal train/validation split",
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "modality": args.modality,
            "normalization": args.normalization,
            "feature_set": args.feature_set,
            "target_columns": list(HYPERVIEW2_TARGET_COLUMNS),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_sample_ids": train_ids,
            "val_sample_ids": val_ids,
            "git_commit": get_git_commit_hash(),
            "git_dirty": is_git_dirty(),
        },
        "baseline_mse": baseline_mse.tolist(),
        "models": {},
    }

    for name in model_names:
        start_fit = time.perf_counter()
        row: dict[str, Any] = {"model": name}
        try:
            regressor = build_hyperview2_regressor(
                name,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                n_features=x_train.shape[1],
                n_samples=x_train.shape[0],
                n_targets=y_train.shape[1],
            )
            regressor.fit(x_train, y_train)
            fit_time = time.perf_counter() - start_fit
            start_predict = time.perf_counter()
            y_pred = np.asarray(regressor.predict(x_val), dtype=np.float32)
            predict_time = time.perf_counter() - start_predict
            metrics = compute_regression_metrics(y_val, y_pred, baseline_mse)
            row.update(
                {
                    "status": "ok",
                    "hyperview_score": metrics["hyperview_score"],
                    "mean_mse": metrics["mean_mse"],
                    "mean_mae": metrics["mean_mae"],
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                }
            )
            for target, target_metrics in metrics["targets"].items():
                row[f"{target}_rmse"] = target_metrics["rmse"]
                row[f"{target}_relative_mse"] = target_metrics["relative_mse"]
            results["models"][name] = {
                "status": "ok",
                "family": REGRESSOR_SPECS[name].family,
                "description": REGRESSOR_SPECS[name].description,
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
                "metrics": metrics,
            }
        except Exception as exc:
            if args.fail_on_unavailable:
                raise
            row.update(
                {
                    "status": "failed",
                    "fit_time_sec": time.perf_counter() - start_fit,
                    "error": str(exc),
                }
            )
            results["models"][name] = {
                "status": "failed",
                "family": REGRESSOR_SPECS.get(name).family if name in REGRESSOR_SPECS else None,
                "error": str(exc),
            }
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(output_dir / "summary.csv", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved: {output_dir / 'summary.csv'}")
    print(f"Saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
