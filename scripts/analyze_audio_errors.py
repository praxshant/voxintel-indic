"""Create a small, reproducible error analysis for a direct-audio report.

This script reads the already frozen final-test metrics. It does not retrain a
model, tune a threshold, or inspect test audio to make a modelling decision.

Usage:
  conda run -n torch26 python scripts/analyze_audio_errors.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def class_metrics(confusion_matrix: list[list[int]], labels: list[str]) -> list[dict]:
    """Return per-class precision, recall, and support from a confusion matrix."""
    rows: list[dict] = []
    for index, label in enumerate(labels):
        true_count = sum(confusion_matrix[index])
        predicted_count = sum(row[index] for row in confusion_matrix)
        true_positive = confusion_matrix[index][index]
        rows.append(
            {
                "label": label,
                "support": true_count,
                "precision": round(true_positive / predicted_count, 4) if predicted_count else 0.0,
                "recall": round(true_positive / true_count, 4) if true_count else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (row["recall"], row["label"]))


def top_confusions(confusion_matrix: list[list[int]], labels: list[str], limit: int = 10) -> list[dict]:
    """Return the most frequent off-diagonal errors."""
    errors = []
    for actual_idx, row in enumerate(confusion_matrix):
        for predicted_idx, count in enumerate(row):
            if actual_idx != predicted_idx and count:
                errors.append(
                    {
                        "actual": labels[actual_idx],
                        "predicted": labels[predicted_idx],
                        "count": int(count),
                    }
                )
    return sorted(errors, key=lambda item: (-item["count"], item["actual"], item["predicted"]))[:limit]


def condition_summary(per_condition: dict) -> list[dict]:
    """Produce a compact condition-level table sorted by accepted accuracy."""
    rows = []
    for condition, metrics in per_condition.items():
        rows.append(
            {
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "coverage": metrics["coverage"],
                "false_accept_rate_other": metrics["false_accept_rate_other"],
            }
        )
    return sorted(rows, key=lambda row: (row["accuracy"], row["condition"]))


def markdown_report(summary: dict) -> str:
    lines = [
        "# Direct Audio Error Analysis",
        "",
        "This is descriptive analysis of a frozen final-test report. It must not be used to retune the evaluated model.",
        "",
        "## Robustness summary",
        "",
        f"- Accuracy on accepted predictions: {summary['robustness']['accuracy'] * 100:.2f}%",
        f"- Coverage: {summary['robustness']['coverage'] * 100:.2f}%",
        f"- False-accept rate for `other`: {summary['robustness']['false_accept_rate_other'] * 100:.2f}%",
        "",
        "## Conditions",
        "",
        "| Condition | Accuracy | Coverage | Other false accepts |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        lines.append(
            f"| {row['condition']} | {row['accuracy'] * 100:.2f}% | "
            f"{row['coverage'] * 100:.2f}% | {row['false_accept_rate_other'] * 100:.2f}% |"
        )

    lines.extend(["", "## Lowest-recall labels", "", "| Label | Support | Precision | Recall |", "|---|---:|---:|---:|"])
    for row in summary["class_metrics"][:5]:
        lines.append(
            f"| {row['label']} | {row['support']} | {row['precision'] * 100:.2f}% | {row['recall'] * 100:.2f}% |"
        )

    lines.extend(["", "## Most common confusions", "", "| Actual | Predicted | Count |", "|---|---|---:|"])
    for row in summary["top_confusions"]:
        lines.append(f"| {row['actual']} | {row['predicted']} | {row['count']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use the weakest conditions and labels to guide qualitative inspection, not final-test retuning.",
            "- The `other` false-accept rate remains the principal safety limitation of this direct-audio probe.",
            "- The next model decision must be selected on the frozen calibration manifest before another final-test run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repo_root / "dataset/reports/audio_classifier_v2.json")
    parser.add_argument("--output", type=Path, default=repo_root / "dataset/reports/audio_classifier_v2_error_analysis.json")
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    evaluation = report["evaluation"]
    robust = evaluation["full_robustness_test"]
    labels = robust["label_order"]
    summary = {
        "schema_version": "audio_error_analysis_v1",
        "source_report": str(args.input),
        "selection_note": "Descriptive frozen-test analysis only; do not use for model or threshold selection.",
        "robustness": {
            key: robust[key]
            for key in ("accuracy", "coverage", "false_accept_rate_other", "n_total", "n_accepted")
        },
        "conditions": condition_summary(evaluation["per_condition"]),
        "class_metrics": class_metrics(robust["confusion_matrix"], labels),
        "top_confusions": top_confusions(robust["confusion_matrix"], labels),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(summary), encoding="utf-8")
    print(f"Error analysis JSON written to {args.output}")
    print(f"Error analysis Markdown written to {markdown_path}")


if __name__ == "__main__":
    main()
