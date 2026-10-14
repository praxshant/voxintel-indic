"""Create a deterministic, stratified language/intent spot-check report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "dataset" / "source" / "utterances_v0.3.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset" / "reports" / "synthetic_language_spot_check.json",
    )
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()
    with args.source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["language_code"], row["intent"])].append(row)
    samples = []
    for key in sorted(grouped):
        for row in grouped[key][:args.per_group]:
            samples.append({
                "id": row["id"],
                "language_code": row["language_code"],
                "intent": row["intent"],
                "other_type": row["other_type"],
                "mix": row["mix"],
                "paraphrase_cluster": row["paraphrase_cluster"],
                "intended_split": row["intended_split"],
                "text": row["text"],
                "review_status": row["review_status"],
            })
    report = {
        "source": str(args.source.relative_to(root)).replace("\\", "/"),
        "policy": (
            "Machine-generated text is spot-check evidence only. Human review "
            "must approve language appropriateness before trusted model claims."
        ),
        "groups": len(grouped),
        "sample_count": len(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(grouped), "sample_count": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
