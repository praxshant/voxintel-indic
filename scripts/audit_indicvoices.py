"""Audit weak IndicVoices assignment labels without rewriting source records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("indicvoices_filtered/manifest.csv"))
    parser.add_argument("--report", type=Path, default=Path("indicvoices_audit.json"))
    parser.add_argument("--annotate", action="store_true",
                        help="Add explicit weak-label provenance and review fields to the manifest.")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.annotate:
        for row in rows:
            row["label_provenance"] = "keyword_derived_candidate"
            row["review_status"] = row.get("review_status") or "unreviewed"
        fieldnames = sorted({key for row in rows for key in row})
        with args.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    labels = Counter(row.get("intent", "") for row in rows)
    review = Counter(row.get("review_status", "missing") for row in rows)
    speakers = Counter(row.get("speaker_id", "") for row in rows if row.get("speaker_id"))
    suspicious = [
        row.get("id", "") for row in rows
        if row.get("task_name", "").casefold() in {"banking", "bank jobs"}
        or not row.get("text", "").strip()
    ]
    report = {
        "manifest": str(args.manifest),
        "rows": len(rows),
        "label_counts": dict(labels),
        "review_status_counts": dict(review),
        "unique_speakers": len(speakers),
        "duplicate_speaker_records": sum(count > 1 for count in speakers.values()),
        "suspicious_record_count": len(suspicious),
        "suspicious_record_ids_sample": suspicious[:50],
        "label_policy": "Keyword-derived labels are candidates, not verified ground truth.",
        "evaluation_policy": "Keep public records separate from synthetic evaluation and do not tune thresholds on held-out public records.",
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
