"""Create a conservative, auditable IndicVoices intent-candidate subset.

The downloaded manifest and audio remain unchanged. This script writes a
curated candidate manifest plus an exclusion audit. Labels are still not
ground truth: every included record remains pending human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


INTENTS = (
    "balance", "recent_transactions", "block_card", "emi_due",
    "report_fraud", "complaint_status", "update_details", "branch_timing",
    "pay_bill", "transfer_to_agent",
)

EVIDENCE: dict[str, tuple[str, ...]] = {
    "balance": (
        "account balance", "available balance", "current balance", "balance",
        "बैलेंस", "शिल्लक", "অ্যাকাউন্ট ব্যালেন্স", "பேலன்ஸ்", "డబ్బు",
        "પૈસા", "ಹಣ",
    ),
    "recent_transactions": (
        "recent transaction", "transaction history", "mini statement",
        "account statement", "last transaction", "लेनदेन", "व्यवहार",
        "লেনদেন", "பரிவர்த்தனை", "లావాదేవీ", "વ્યવહાર", "ವಹಿವಾಟು",
    ),
    "block_card": (
        "block card", "freeze card", "card is lost", "card lost",
        "card stolen", "कार्ड ब्लॉक", "कार्ड खो", "कार्ड चोरी",
        "কার্ড ব্লক", "கார்டு பிளாக்", "కార్డ్ బ్లాక్",
    ),
    "emi_due": (
        "emi due", "emi date", "next emi", "installment due", "loan installment",
        "ईएमआई", "किस्त", "हप्ता", "ঋণ", "கடன்", "లోన్",
    ),
    "report_fraud": (
        "fraud", "unauthorized transaction", "unauthorised transaction",
        "unknown charge", "suspicious transaction", "फ्रॉड", "धोखाधड़ी",
        "अनधिकृत", "অন্যায়", "মোসডি", "மோசடி", "మోసం", "फसवणूक",
        "છેતરપિંડી", "ಮೋಸ",
    ),
    "complaint_status": (
        "complaint status", "complaint number", "track complaint",
        "complaint update", "शिकायत की स्थिति", "অভিযোগের অবস্থা",
        "புகார் நிலை", "ఫిర్యాదు స్థితి", "તકરારની સ્થિતિ",
    ),
    "update_details": (
        "update mobile", "change mobile", "update phone", "change phone",
        "update address", "change address", "update email", "change email",
        "update name", "change name", "मोबाइल नंबर बदल", "पता बदल",
        "फोन नंबर", "மொபைல்", "మొబైల్", "મોબાઇલ", "ಮೊಬೈಲ್",
    ),
    "branch_timing": (
        "branch timing", "branch hours", "branch opening", "branch closing",
        "शाखा का समय", "ब्रांच टाइम", "শাখার সময়", "கிளை நேரம்",
        "శాఖ సమయం", "શાખાના સમય", "ಶಾಖೆಯ ಸಮಯ",
    ),
    "pay_bill": (
        "pay bill", "bill payment", "electricity bill", "mobile bill",
        "gas bill", "utility bill", "बिल भुगतान", "बिजली बिल", "বিল",
        "பில்", "బిల్లు", "ચૂકવણી", "ಪಾವತಿ",
    ),
    "transfer_to_agent": (
        "talk to agent", "speak to agent", "connect agent", "customer care",
        "customer support", "representative", "executive", "प्रतिनिधि",
        "ग्राहक सेवा", "কাস্টমার কেয়ার", "வாடிக்கையாளர் சேவை",
        "కస్టమర్ కేర్", "ગ્રાહક સેવા", "ಗ್ರಾಹಕ ಸೇವೆ",
    ),
}

OTHER_EVIDENCE = {
    "out_of_domain": (
        "weather", "cricket", "movie", "recipe", "song", "train ticket",
        "শিক্ষা", "school", "नौकरी", "job", "விவசாயம்", "కళాశాల",
    ),
    "chitchat": ("hello", "how are you", "good morning", "ধন্যবাদ", "धन्यवाद"),
}


def normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def text_for(row: dict[str, str]) -> str:
    fields = (
        "text", "verbatim", "normalized", "unsanitized_verbatim",
        "unsanitized_normalized", "task_name",
    )
    return normalise(" ".join(row.get(field, "") for field in fields))


def match_intents(text: str) -> dict[str, list[str]]:
    return {
        intent: [phrase for phrase in phrases if normalise(phrase) in text]
        for intent, phrases in EVIDENCE.items()
        if any(normalise(phrase) in text for phrase in phrases)
    }


def curate(row: dict[str, str]) -> tuple[bool, str, str, dict[str, list[str]]]:
    text = text_for(row)
    matches = match_intents(text)
    if len(matches) == 1:
        intent, evidence = next(iter(matches.items()))
        return True, intent, "single_intent_evidence", {intent: evidence}
    if len(matches) > 1:
        return False, "other", "conflicting_intent_evidence", matches
    for other_type, phrases in OTHER_EVIDENCE.items():
        found = [phrase for phrase in phrases if normalise(phrase) in text]
        if found:
            return True, "other", other_type, {"other": found}
    return False, "other", "no_actionable_intent_evidence", {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path, default=root / "dataset" / "indicvoices" / "filtered" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=root / "dataset" / "indicvoices" / "curated")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input manifest is empty")

    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        keep, intent, reason, evidence = curate(row)
        output = dict(row)
        output.update({
            "curated_intent": intent,
            "curation_decision": "candidate_included" if keep else "excluded",
            "curation_reason": reason,
            "curation_evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            "label_provenance": "deterministic_transcript_evidence_candidate",
            "review_status": "unreviewed",
        })
        (included if keep else excluded).append(output)

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(included[0] if included else excluded[0])
    for name, selected in (("manifest.csv", included), ("excluded.csv", excluded)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    summary: dict[str, Any] = {
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256(args.manifest),
        "source_rows": len(rows),
        "included_rows": len(included),
        "excluded_rows": len(excluded),
        "included_by_intent": dict(Counter(row["curated_intent"] for row in included)),
        "excluded_by_reason": dict(Counter(row["curation_reason"] for row in excluded)),
        "policy": (
            "Included rows are deterministic candidates only. They are not "
            "trusted evaluation labels until human review approves the transcript "
            "and intent. Original manifest and audio are preserved."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
