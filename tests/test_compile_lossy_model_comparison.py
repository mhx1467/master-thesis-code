import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compile_lossy_model_comparison.py"
SPEC = importlib.util.spec_from_file_location("compile_lossy_model_comparison", MODULE_PATH)
assert SPEC is not None
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def test_row_from_archived_eval_summary_extracts_actual_codec_metrics(tmp_path: Path) -> None:
    eval_path = tmp_path / "mamba_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "model": "hierarchical_spectral_mamba_ae",
                "variant": "K4 + spatial",
                "objective": "RD lambda=0.0003",
                "split": "test",
                "difficulty": "easy",
                "num_samples": 1149,
                "num_input_bands": 202,
                "params": 778490,
                "metrics": {
                    "psnr": 45.0,
                    "ssim": 0.983,
                    "sa_deg": 3.3,
                    "likelihood_bpppc": 0.111,
                    "actual_bpppc": 0.112,
                    "actual_compression_ratio": 142.0,
                    "actual_psnr": 45.1,
                    "actual_ssim": 0.982,
                    "actual_sa_deg": 3.4,
                },
            }
        ),
        encoding="utf-8",
    )

    row = comparison._row_from_entry(
        {
            "slug": "hierarchical_mamba_k4_rd_0_0003",
            "label": "Hierarchical Mamba K4 RD 3e-4",
            "family": "mamba",
            "status": "reference_comparable",
            "eval_json": str(eval_path),
        },
        {"expected_num_samples": 1149, "expected_num_input_bands": 202},
    )

    assert row["model_name"] == "hierarchical_spectral_mamba_ae"
    assert row["actual_bpppc"] == 0.112
    assert row["actual_psnr"] == 45.1
    assert row["protocol_warnings"] is None


def test_row_from_legacy_eval_keeps_protocol_warning(tmp_path: Path) -> None:
    eval_path = tmp_path / "legacy_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "model_name": "baseline_1d_ae_v2",
                "split": "test",
                "difficulty": "easy",
                "num_samples": 1148,
                "num_input_bands": 180,
                "psnr": 43.3,
                "sam_deg": 4.0,
            }
        ),
        encoding="utf-8",
    )

    row = comparison._row_from_entry(
        {
            "slug": "legacy_baseline_1d_ae_v2",
            "label": "Legacy 1D AE v2",
            "family": "legacy_baseline",
            "status": "legacy_not_reference_comparable",
            "legacy_eval_json": str(eval_path),
        },
        {"expected_num_samples": 1149, "expected_num_input_bands": 202},
    )

    assert row["sa_deg"] == 4.0
    assert "num_input_bands=180 expected=202" in row["protocol_warnings"]
    assert "num_samples=1148 expected=1149" in row["protocol_warnings"]


def test_eval_command_is_generated_only_for_needs_eval_entries() -> None:
    protocol = {"split": "test", "difficulty": "easy"}

    assert (
        comparison._eval_command(
            {
                "slug": "missing_training",
                "status": "needs_training_or_checkpoint",
                "checkpoint": "artifacts/checkpoints/missing.pt",
            },
            protocol,
            "$DATASET_ROOT",
            batch_size=4,
            num_workers=4,
        )
        is None
    )

    command = comparison._eval_command(
        {
            "slug": "baseline_2d_patch_lic_recon",
            "status": "needs_eval",
            "checkpoint": "artifacts/checkpoints/baseline.pt",
        },
        protocol,
        "$DATASET_ROOT",
        batch_size=4,
        num_workers=4,
    )

    assert command is not None
    assert "python scripts/evaluate.py artifacts/checkpoints/baseline.pt $DATASET_ROOT" in command
    assert "--run-name lossy_compare_baseline_2d_patch_lic_recon_easy_test" in command


def test_completed_needs_eval_row_is_reference_comparable(tmp_path: Path) -> None:
    eval_path = tmp_path / "baseline_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "model_name": "baseline_3d_patch_ae",
                "split": "test",
                "difficulty": "easy",
                "num_samples": 1149,
                "num_input_bands": 202,
                "num_params": 217601,
                "psnr": 45.3,
                "ssim": 0.982,
                "sa_deg": 3.0,
                "likelihood_bpppc": 1.50,
                "actual_bpppc": 1.51,
                "actual_compression_ratio": 10.6,
                "actual_psnr": 45.4,
                "actual_ssim": 0.981,
                "actual_sa_deg": 2.98,
            }
        ),
        encoding="utf-8",
    )

    row = comparison._row_from_entry(
        {
            "slug": "baseline_3d_patch_recon",
            "label": "3D patch baseline",
            "family": "active_baseline",
            "status": "needs_eval",
            "eval_json": str(eval_path),
        },
        {"expected_num_samples": 1149, "expected_num_input_bands": 202},
    )

    assert comparison._is_reference_comparable_result(row)

    summary_path = tmp_path / "summary.md"
    comparison._write_summary(
        summary_path,
        {"dataset_protocol": "HySpecNet-11k easy test split"},
        rows=[row],
        gap_rows=[],
        eval_commands=[],
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "| 3D patch baseline | active_baseline | 45.4000 | 0.9810 | 2.9800 |" in text
    assert "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" not in text


def test_summary_prefers_decoded_actual_metrics(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.md"
    row = dict.fromkeys(comparison.FIELDNAMES) | {
        "slug": "hierarchical_mamba_k4_rd_0_0003",
        "label": "Hierarchical Mamba K4 RD 3e-4",
        "family": "mamba",
        "status": "reference_comparable",
        "psnr": 44.0,
        "ssim": 0.98,
        "sa_deg": 3.5,
        "actual_psnr": 45.0,
        "actual_ssim": 0.97,
        "actual_sa_deg": 3.4,
        "actual_bpppc": 0.112,
        "actual_compression_ratio": 142.0,
        "likelihood_bpppc": 0.111,
    }

    comparison._write_summary(
        summary_path,
        {"dataset_protocol": "HySpecNet-11k easy test split"},
        rows=[row],
        gap_rows=[],
        eval_commands=[],
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "| Hierarchical Mamba K4 RD 3e-4 | mamba | 45.0000 | 0.9700 | 3.4000 |" in text
    assert "| Hierarchical Mamba K4 RD 3e-4 | mamba | 44.0000 | 0.9800 | 3.5000 |" not in text
