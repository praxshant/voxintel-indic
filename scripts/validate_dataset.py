"""Validate text and generated dataset invariants before training or evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INTENTS = {
    "balance", "recent_transactions", "block_card", "emi_due", "report_fraud",
    "complaint_status", "update_details", "branch_timing", "pay_bill",
    "transfer_to_agent",
}
OTHER_TYPES = {
    "out_of_domain", "chitchat", "ambiguous_in_domain",
    "unsupported_near_miss", "non_intent_audio",
}
SPLITS = {"train", "dev", "test"}


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return load_csv(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    has_v03_split = "intended_split" in rows[0] if rows else False
    ids = [row.get("id", "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate utterance IDs")
    seen_text: dict[str, str] = {}
    clusters: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        intent = row.get("intent", "")
        other_type = row.get("other_type", "")
        if intent not in INTENTS and intent != "other":
            errors.append(f"invalid intent {intent!r} in {row.get('id')}")
        if intent == "other" and other_type not in OTHER_TYPES:
            errors.append(f"invalid other_type {other_type!r} in {row.get('id')}")
        if has_v03_split and row.get("intended_split") not in SPLITS:
            errors.append(f"invalid intended_split in {row.get('id')}")
        key = normalize(row.get("text", ""))
        if not key and row.get("source") == "tts":
            errors.append(f"empty text in {row.get('id')}")
        if key and key in seen_text:
            errors.append(f"duplicate normalized text: {row.get('id')} and {seen_text[key]}")
        if key:
            seen_text[key] = row.get("id", "")
        cluster = row.get("paraphrase_cluster", "")
        if cluster:
            clusters[cluster].add(row.get("intended_split", ""))
    for cluster, split_names in clusters.items():
        if len(split_names) > 1:
            errors.append(f"paraphrase cluster crosses splits: {cluster}")
    return errors


def validate_manifest(path: Path, root: Path) -> tuple[list[str], dict[str, Any]]:
    rows = load_manifest(path)
    errors: list[str] = []
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    clusters_by_split: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for row in rows:
        split = str(row.get("split", ""))
        if split not in SPLITS:
            errors.append(f"invalid manifest split {split!r} for {row.get('id')}")
        counts[split] += 1
        degradation = row.get("degradation", "")
        if degradation:
            try:
                condition = json.loads(str(degradation)).get("condition")
            except json.JSONDecodeError:
                errors.append(f"invalid degradation metadata for {row.get('id')}")
                condition = None
            if condition == "unseen_combined" and split != "test":
                errors.append(
                    f"unseen_combined must be test-only: {row.get('id')} is {split}"
                )
        speaker = str(row.get("speaker", ""))
        if speaker:
            speakers_by_split[split].add(speaker)
        cluster = str(row.get("paraphrase_cluster", ""))
        if cluster:
            clusters_by_split[split].add(cluster)
        audio_path = root / str(row.get("audio_path", ""))
        if not audio_path.exists():
            errors.append(f"missing audio: {audio_path}")
            continue
        expected = str(row.get("sha256", ""))
        if expected and sha256(audio_path) != expected:
            errors.append(f"checksum mismatch: {audio_path}")
        try:
            with wave.open(str(audio_path), "rb") as audio:
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                    errors.append(f"non-mono/16-bit audio: {audio_path}")
                if int(row.get("sample_rate", audio.getframerate())) != audio.getframerate():
                    errors.append(f"sample rate mismatch: {audio_path}")
                if audio.getnframes() == 0:
                    errors.append(f"empty audio: {audio_path}")
        except (OSError, EOFError, wave.Error):
            errors.append(f"unreadable WAV: {audio_path}")
    split_values = list(speakers_by_split.values())
    if len(split_values) >= 2 and set.intersection(*split_values):
        errors.append("voice leakage across splits")
    cluster_values = list(clusters_by_split.values())
    if len(cluster_values) >= 2 and set.intersection(*cluster_values):
        errors.append("manifest paraphrase-cluster leakage across splits")
    return errors, {
        "rows": len(rows),
        "split_counts": dict(counts),
        "voices_by_split": {key: sorted(value) for key, value in speakers_by_split.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path,
                        default=repo_root / "dataset" / "source" / "utterances_v0.3.csv")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--report", type=Path,
                        default=repo_root / "dataset" / "reports" / "data_readiness_validation.json")
    args = parser.parse_args()

    errors = validate_source(load_csv(args.source))
    report: dict[str, Any] = {"source": str(args.source), "source_rows": len(load_csv(args.source))}
    if args.manifest:
        if not args.manifest_root:
            parser.error("--manifest-root is required with --manifest")
        manifest_errors, manifest_report = validate_manifest(args.manifest, args.manifest_root)
        errors.extend(manifest_errors)
        report["manifest"] = manifest_report
    report["errors"] = errors
    report["passed"] = not errors
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
