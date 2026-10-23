"""Compare ASR-cascade (text classifier) vs direct audio-to-intent classifier.

Loads both evaluation reports and produces a side-by-side comparison table
in JSON and Markdown.

Usage:
  conda run -n torch26 python scripts/compare_architectures.py \\
      --baseline  dataset/reports/hybrid_baseline_v2.json \\
      --audio     dataset/reports/audio_classifier_v2.json \\
      --output    dataset/reports/architecture_comparison.json

Outputs:
  dataset/reports/architecture_comparison.json  (machine-readable)
  dataset/reports/architecture_comparison.md    (human-readable table)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(d: dict, *keys, default="N/A"):
    """Safely get nested value."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d if d is not None else default


def _fmt(v, pct: bool = False, digits: int = 4) -> str:
    if v == "N/A" or v is None:
        return "N/A"
    try:
        f = float(v)
        if pct:
            return f"{f*100:.2f}%"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def extract_metrics(report: dict, system: str) -> dict:
    """Extract comparable metrics from either report schema."""
    schema = report.get("schema_version", "")

    if schema.startswith("audio_classifier_v"):
        # Direct audio classifier report
        clean = _safe(report, "evaluation", "clean_final_test")
        robust = _safe(report, "evaluation", "full_robustness_test")
        unseen = _safe(report, "evaluation", "per_condition", "unseen_combined")
        aurc = _safe(report, "evaluation", "aurc_full_test")
        rc80 = _safe(report, "evaluation", "selective_acc_at_80pct_coverage")
        rc90 = _safe(report, "evaluation", "selective_acc_at_90pct_coverage")
        rc95 = _safe(report, "evaluation", "selective_acc_at_95pct_coverage")
        threshold = _safe(report, "calibration", "threshold")
        n_train = _safe(report, "data", "n_train")
        encoder = _safe(report, "model", "encoder")
        n_trainable = _safe(report, "model", "n_params_trainable")

        latency_note = "GPU inference (frozen encoder + MLP head)"

        per_condition = {
            k: _safe(v_dict, "accuracy")
            for k, v_dict in _safe(report, "evaluation", "per_condition", default={}).items()
            if isinstance(v_dict, dict)
        }

    else:
        # ASR cascade (hybrid_baseline_v2) report
        clean = _safe(report, "conditions", "clean_test")
        robust = _safe(report, "conditions", "robust_test")
        unseen = _safe(report, "per_condition", "unseen_combined")
        aurc = _safe(report, "conditions", "robust_test", "risk_coverage", "max_prob", "aurc")
        rc80 = _safe(report, "conditions", "robust_test", "risk_coverage", "max_prob", "selective_accuracy_80")
        rc90 = _safe(report, "conditions", "robust_test", "risk_coverage", "max_prob", "selective_accuracy_90")
        rc95 = _safe(report, "conditions", "robust_test", "risk_coverage", "max_prob", "selective_accuracy_95")
        threshold = _safe(report, "calibration_threshold")
        n_train = _safe(report, "train_rows")
        encoder = "char-ngram TF-IDF (3-5gram) + LogReg"
        n_trainable = "N/A (sklearn)"
        latency_note = "Hybrid ASR (IndicConformer/Whisper) + TF-IDF + LogReg"

        per_condition = {}
        for k, v_dict in _safe(report, "per_condition", default={}).items():
            if isinstance(v_dict, dict):
                per_condition[k] = _safe(v_dict, "accuracy")

    return {
        "system": system,
        "encoder": encoder,
        "n_trainable_params": n_trainable,
        "n_train": n_train,
        "threshold": threshold,
        "latency_note": latency_note,
        # Clean test
        "clean_accuracy": _safe(clean, "accuracy"),
        "clean_coverage": _safe(clean, "coverage"),
        "clean_in_scope_acc": _safe(clean, "in_scope_accuracy"),
        "clean_false_accept_other": _safe(clean, "false_accept_rate_other"),
        "clean_ci": _safe(clean, "accuracy_ci95", default=_safe(clean, "confidence_interval_95")),
        # Robust test
        "robust_accuracy": _safe(robust, "accuracy"),
        "robust_coverage": _safe(robust, "coverage"),
        "robust_in_scope_acc": _safe(robust, "in_scope_accuracy"),
        "robust_false_accept_other": _safe(robust, "false_accept_rate_other"),
        "robust_ci": _safe(robust, "accuracy_ci95", default=_safe(robust, "confidence_interval_95")),
        # unseen_combined
        "unseen_accuracy": _safe(unseen, "accuracy"),
        "unseen_coverage": _safe(unseen, "coverage"),
        # Risk-coverage
        "aurc": aurc,
        "selective_acc_80pct": rc80,
        "selective_acc_90pct": rc90,
        "selective_acc_95pct": rc95,
        # Per condition
        "per_condition": per_condition,
    }


def build_markdown_table(m1: dict, m2: dict) -> str:
    """Build a Markdown comparison table."""
    s1 = m1["system"]
    s2 = m2["system"]

    rows = [
        ("**Architecture**", "", ""),
        ("Encoder / features", m1["encoder"], m2["encoder"]),
        ("Trainable params", str(m1["n_trainable_params"]), str(m2["n_trainable_params"])),
        ("Train samples", str(m1["n_train"]), str(m2["n_train"])),
        ("Reject threshold", _fmt(m1["threshold"]), _fmt(m2["threshold"])),
        ("Latency / pipeline", m1["latency_note"], m2["latency_note"]),
        ("**Clean Final Test**", "", ""),
        ("Accuracy", _fmt(m1["clean_accuracy"], pct=True), _fmt(m2["clean_accuracy"], pct=True)),
        ("Coverage", _fmt(m1["clean_coverage"], pct=True), _fmt(m2["clean_coverage"], pct=True)),
        ("In-scope accuracy", _fmt(m1["clean_in_scope_acc"], pct=True), _fmt(m2["clean_in_scope_acc"], pct=True)),
        ("False-accept (other)", _fmt(m1["clean_false_accept_other"], pct=True), _fmt(m2["clean_false_accept_other"], pct=True)),
        ("95% CI", str(m1["clean_ci"]), str(m2["clean_ci"])),
        ("**Robustness Final Test**", "", ""),
        ("Accuracy", _fmt(m1["robust_accuracy"], pct=True), _fmt(m2["robust_accuracy"], pct=True)),
        ("Coverage", _fmt(m1["robust_coverage"], pct=True), _fmt(m2["robust_coverage"], pct=True)),
        ("In-scope accuracy", _fmt(m1["robust_in_scope_acc"], pct=True), _fmt(m2["robust_in_scope_acc"], pct=True)),
        ("False-accept (other)", _fmt(m1["robust_false_accept_other"], pct=True), _fmt(m2["robust_false_accept_other"], pct=True)),
        ("**Unseen Combined (test-only)**", "", ""),
        ("Accuracy", _fmt(m1["unseen_accuracy"], pct=True), _fmt(m2["unseen_accuracy"], pct=True)),
        ("Coverage", _fmt(m1["unseen_coverage"], pct=True), _fmt(m2["unseen_coverage"], pct=True)),
        ("**Risk-Coverage**", "", ""),
        ("AURC (↓ better)", _fmt(m1["aurc"], digits=6), _fmt(m2["aurc"], digits=6)),
        ("Sel. accuracy @80% cov", _fmt(m1["selective_acc_80pct"], pct=True), _fmt(m2["selective_acc_80pct"], pct=True)),
        ("Sel. accuracy @90% cov", _fmt(m1["selective_acc_90pct"], pct=True), _fmt(m2["selective_acc_90pct"], pct=True)),
        ("Sel. accuracy @95% cov", _fmt(m1["selective_acc_95pct"], pct=True), _fmt(m2["selective_acc_95pct"], pct=True)),
    ]

    # Header
    lines = [
        f"| Metric | {s1} | {s2} |",
        "|--------|-------|-------|",
    ]
    for metric, v1, v2 in rows:
        if metric.startswith("**"):
            lines.append(f"| {metric} | | |")
        else:
            lines.append(f"| {metric} | {v1} | {v2} |")

    # Per-condition table
    all_conds = sorted(set(m1["per_condition"]) | set(m2["per_condition"]))
    if all_conds:
        lines.append("")
        lines.append("### Per-condition accuracy (accepted predictions)")
        lines.append(f"| Condition | {s1} | {s2} |")
        lines.append("|-----------|-------|-------|")
        for cond in all_conds:
            v1 = _fmt(m1["per_condition"].get(cond, "N/A"), pct=True)
            v2 = _fmt(m2["per_condition"].get(cond, "N/A"), pct=True)
            lines.append(f"| {cond} | {v1} | {v2} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--baseline", type=Path,
        default=repo_root / "dataset/reports/hybrid_baseline_v2.json",
        help="ASR-cascade baseline report (hybrid_baseline_v2.json).",
    )
    parser.add_argument(
        "--audio", type=Path,
        default=repo_root / "dataset/reports/audio_classifier_v2.json",
        help="Direct audio classifier report (audio_classifier_v2.json).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "dataset/reports/architecture_comparison.json",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        raise FileNotFoundError(f"Baseline report not found: {args.baseline}")
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio classifier report not found: {args.audio}")

    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    audio_report = json.loads(args.audio.read_text(encoding="utf-8"))

    m1 = extract_metrics(baseline_report, "ASR-cascade (char n-gram)")
    m2 = extract_metrics(audio_report, "Direct audio (wav2vec2+MLP)")

    comparison = {
        "schema_version": "architecture_comparison_v1",
        "systems": [m1, m2],
        "notes": [
            "ASR-cascade: hybrid IndicConformer+Whisper ASR -> char 3-5gram TF-IDF -> Logistic Regression",
            f"Direct audio: frozen {m2['encoder']} encoder -> mean pool -> 2-layer MLP",
            "Both systems use the same frozen splits (train/dev/test), identical threshold selection on dev only.",
            "Audio is synthetic TTS at 8 kHz, resampled to 16 kHz (does not restore missing high-frequency content).",
            "IndicVoices public labels excluded (unverified). Only synthetic ground truth used.",
            "unseen_combined (390 rows) is strictly test-only in both systems.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Comparison JSON written to {args.output}")

    # Also write Markdown
    md_path = args.output.with_suffix(".md")
    md = f"# Architecture Comparison: VoxIntel Indic\n\n"
    md += build_markdown_table(m1, m2)
    md += "\n\n---\n\n**Notes:**\n"
    for note in comparison["notes"]:
        md += f"- {note}\n"
    md_path.write_text(md, encoding="utf-8")
    print(f"Comparison Markdown written to {md_path}")

    # Print summary
    print("\n=== COMPARISON SUMMARY ===")
    for key in ["clean_accuracy", "robust_accuracy", "unseen_accuracy", "aurc"]:
        v1 = _fmt(m1[key], pct=key != "aurc")
        v2 = _fmt(m2[key], pct=key != "aurc")
        print(f"  {key:30s}: {m1['system'][:25]:25s}={v1:10s}  {m2['system'][:25]:25s}={v2}")


if __name__ == "__main__":
    main()
