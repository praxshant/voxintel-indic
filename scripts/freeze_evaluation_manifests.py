"""Freeze synthetic evaluation manifests and their reproducibility metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def condition(row: dict[str, str]) -> str:
    return json.loads(row["degradation"])["condition"]


def with_clean_condition(row: dict[str, str]) -> dict[str, str]:
    clean = dict(row)
    clean["variant_type"] = "synthetic_clean"
    clean["audio_root"] = "dataset/synthetic/v0.3"
    clean["degradation"] = json.dumps(
        {"condition": "clean", "source_audio_path": clean["audio_path"]},
        sort_keys=True,
    )
    clean["condition"] = "clean"
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "dataset" / "synthetic" / "v0.3_robustness" / "manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset" / "manifests" / "evaluation",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=root / "dataset" / "indicvoices" / "filtered" / "manifest.csv",
    )
    parser.add_argument(
        "--public-curated-manifest",
        type=Path,
        default=root / "dataset" / "indicvoices" / "curated" / "manifest.csv",
    )
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    if not rows:
        raise ValueError("Cannot freeze empty manifest")
    for row in rows:
        row["condition"] = condition(row)
    clean_manifest = args.manifest.parent.parent / "v0.3" / "manifest.csv"
    clean_rows = read_rows(clean_manifest)
    clean_by_id = {row["id"]: row for row in clean_rows}
    if len(clean_by_id) != len(clean_rows):
        raise ValueError("Clean manifest contains duplicate source IDs")
    train_conditions = {
        "gaussian_snr10", "gaussian_snr20", "traffic_snr15",
        "office_snr15", "telephone_channel",
    }
    calibration_conditions = train_conditions
    final_conditions = set(train_conditions) | {"unseen_combined"}
    if any(condition(row) == "unseen_combined" and row["split"] != "test" for row in rows):
        raise ValueError("unseen_combined is not test-only")

    selections: dict[str, list[dict[str, str]]] = {
        "synthetic_train": [
            row for row in rows
            if row["split"] == "train" and condition(row) in train_conditions
        ],
        "synthetic_calibration": [
            row for row in rows
            if row["split"] == "dev" and condition(row) in calibration_conditions
        ],
        "synthetic_final_test": [
            row for row in rows
            if row["split"] == "test" and condition(row) in final_conditions
        ],
        "synthetic_clean_final_test": [
            with_clean_condition(clean_by_id[row["id"]])
            for row in clean_rows
            if row["split"] == "test"
        ],
    }
    fields = sorted({key for selected in selections.values() for row in selected for key in row})
    manifest_records: dict[str, Any] = {}
    for name, selected in selections.items():
        output_path = args.output / f"{name}.csv"
        write_rows(output_path, selected, fields)
        manifest_records[name] = {
            "path": str(output_path.relative_to(root)).replace("\\", "/"),
            "sha256": digest(output_path),
            "rows": len(selected),
            "split": sorted({row["split"] for row in selected}),
            "conditions": sorted({condition(row) for row in selected}),
            "unseen_combined_allowed": name == "synthetic_final_test",
        }
    all_path = args.output / "synthetic_asr_pool.csv"
    pool_rows = rows + [
        with_clean_condition(row) for row in clean_rows
    ]
    pool_fields = sorted({key for row in pool_rows for key in row})
    write_rows(all_path, pool_rows, pool_fields)
    manifest_records["synthetic_asr_pool"] = {
        "path": str(all_path.relative_to(root)).replace("\\", "/"),
        "sha256": digest(all_path),
        "rows": len(pool_rows),
        "purpose": "ASR transcript generation only; classifier roles use the three frozen manifests",
    }
    public_rows = read_rows(args.public_manifest)
    public_fields = list(public_rows[0]) if public_rows else []
    trusted_public = [
        row for row in public_rows
        if row.get("review_status") == "reviewed"
        and row.get("label_provenance") not in {"keyword_derived_candidate", ""}
    ]
    public_path = args.output / "public_trusted.csv"
    write_rows(public_path, trusted_public, public_fields)
    manifest_records["public_trusted"] = {
        "path": str(public_path.relative_to(root)).replace("\\", "/"),
        "sha256": digest(public_path),
        "rows": len(trusted_public),
        "source_rows": len(public_rows),
        "policy": (
            "Only explicitly reviewed labels with non-derived provenance are "
            "eligible. Current count is zero until review is completed."
        ),
    }
    curated_public = read_rows(args.public_curated_manifest)
    curated_path = args.output / "public_curated_candidates.csv"
    curated_fields = list(curated_public[0]) if curated_public else []
    write_rows(curated_path, curated_public, curated_fields)
    manifest_records["public_curated_candidates"] = {
        "path": str(curated_path.relative_to(root)).replace("\\", "/"),
        "sha256": digest(curated_path),
        "rows": len(curated_public),
        "purpose": (
            "Audit and human-review queue only; excluded from trusted metrics "
            "until review_status is explicitly reviewed."
        ),
    }
    freeze = {
        "version": "v0.3-evaluation-freeze-2026-10-14",
        "source_manifest": str(args.manifest.relative_to(root)).replace("\\", "/"),
        "source_manifest_sha256": digest(args.manifest),
        "split_rule": (
            "Paraphrase clusters are assigned to exactly one source split before "
            "TTS. Synthetic voices are disjoint by split: shubh/train, "
            "simran/dev, aditya/test."
        ),
        "training_policy": (
            "Train only on synthetic_train. Do not use synthetic_calibration or "
            "synthetic_final_test for fitting."
        ),
        "calibration_policy": (
            "Tune thresholds and reject-gate parameters only on "
            "synthetic_calibration."
        ),
        "final_test_policy": (
            "Evaluate once on synthetic_final_test after model and threshold "
            "selection. unseen_combined is test-only."
        ),
        "public_data_policy": (
            "IndicVoices records with review_status=unreviewed and "
            "label_provenance=keyword_derived_candidate are excluded from "
            "trusted metrics until human review."
        ),
        "manifests": manifest_records,
    }
    freeze_path = args.output / "FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
