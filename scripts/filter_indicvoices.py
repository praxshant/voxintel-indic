"""Download and filter IndicVoices records relevant to the VoxIntel intents.

The Hugging Face IndicVoices dataset is gated. Set HF_TOKEN in the project
.env or environment, then run this script with the torch26 environment.
Audio is written from the dataset's compressed bytes and annotations are
stored verbatim in JSONL/CSV manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset


LANGUAGE_CONFIGS = {
    "bn-IN": "bengali",
    "gu-IN": "gujarati",
    "hi-IN": "hindi",
    "kn-IN": "kannada",
    "mr-IN": "marathi",
    "ta-IN": "tamil",
    "te-IN": "telugu",
}

KEYWORDS: dict[str, tuple[str, ...]] = {
    "account_balance": (
        "bank", "बैंक", "ব্যাংক", "வங்கி", "బ్యాంక్", "बँक", "બેંક", "ಬ್ಯಾಂಕ್",
        "account", "खाता", "खाते", "अकाउंट", "অ্যাকাউন্ট", "கணக்கு", "ఖాతా",
        "खाते", "ખાતું", "ಖಾತೆ",
        "balance", "बैलेंस", "शिल्लक", "টাকা", "பணம்", "డబ్బు", "પૈસા", "ಹಣ",
    ),
    "transactions": (
        "transaction", "लेनदेन", "व्यवहार", "লেনদেন", "பரிவர்த்தனை", "లావాదేవీ",
        "व्यवहार", "વ્યવહાર", "ವಹಿವಾಟು", "statement", "स्टेटमेंट", "मिनी स्टेटमेंट",
    ),
    "card": (
        "card", "कार्ड", "কার্ড", "கார்டு", "కార్డ్", "कार्ड", "કાર્ડ", "ಕಾರ್ಡ್",
        "credit", "क्रेडिट", "debit", "डेबिट", "block", "ब्लॉक", "ब्लॉक करा",
        "freeze", "खो गया", "lost",
    ),
    "payment_bill_emi": (
        "payment", "भुगतान", "पेमेंट", "পরিশোধ", "செலுத்த", "చెల్లించ", "देय",
        "ચૂકવણી", "ಪಾವತಿ", "pay", "paid", "bill", "बिल", "বিল", "பில்", "బిల్లు",
        "वीज", "electricity", "mobile bill", "gas bill", "emi", "ईएमआई", "किस्त",
        "हप्ता", "ঋণ", "கடன்", "లోన్", "loan", "हफ्ता",
    ),
    "fraud": (
        "fraud", "फ्रॉड", "धोखा", "धोखाधड़ी", "प्रतारण", "प्रतिरूप", "অন্যায়",
        "மோசடி", "మోసం", "फसवणूक", "છેતરપિંડી", "ಮೋಸ", "suspicious", "unknown charge",
        "unauthorized", "अनधिकृत", "बिना बताए", "न केलेला",
    ),
    "complaint_customer_support": (
        "complaint", "शिकायत", "तक्रार", "অভিযোগ", "புகார்", "ఫిర్యాదు", "ફરિયાદ",
        "ದೂರು", "status", "स्थिति", "स्थिती", "நிலை", "స్థితి", "ticket", "ग्राहक",
        "customer", "कस्टमर", "सेवा", "service", "प्रतिनिधि", "agent", "executive",
        "அதிகாரி", "ప్రతినిధి",
    ),
    "branch": (
        "branch", "शाखा", "ब्रांच", "শাখা", "கிளை", "శాఖ", "शाखा", "શાખા", "ಶಾಖೆ",
    ),
    "update_details": (
        "update", "अपडेट", "बदल", "मोबाइल नंबर", "mobile number", "address", "पता",
        "email", "ईमेल", "नाव", "name", "नाम", "फोन नंबर", "phone number",
    ),
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--utterances", type=Path,
                        default=repo_root / "dataset" / "source" / "utterances_v0.2.csv")
    parser.add_argument("--output", type=Path,
                        default=repo_root / "dataset" / "indicvoices" / "filtered")
    parser.add_argument("--splits", nargs="+", default=["valid"], choices=["valid", "train"])
    parser.add_argument("--limit-per-language", type=int, default=0)
    parser.add_argument("--min-keyword-hits", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dotenv_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def source_languages(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        return sorted({row["language_code"] for row in rows if row["language_code"]})


def normalized_text(record: dict[str, Any]) -> str:
    fields = ("text", "verbatim", "normalized", "unsanitized_verbatim",
              "unsanitized_normalized", "task_name")
    return " ".join(str(record.get(field, "")) for field in fields).casefold()


def keyword_matches(text: str) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for category, words in KEYWORDS.items():
        found = [word for word in words if word.casefold() in text]
        if found:
            matches[category] = sorted(set(found))
    return matches


def assignment_label(text: str, matches: dict[str, list[str]]) -> tuple[str, str]:
    """Map a matched record to the frozen assignment taxonomy conservatively."""
    if "fraud" in matches:
        return "report_fraud", ""
    if "transactions" in matches:
        return "recent_transactions", ""
    if "branch" in matches:
        return "branch_timing", ""
    if "update_details" in matches:
        return "update_details", ""
    if "complaint_customer_support" in matches:
        support_words = ("agent", "executive", "प्रतिनिधि", "customer", "ग्राहक", "सेवा")
        if any(word in text for word in support_words):
            return "transfer_to_agent", ""
        return "complaint_status", ""
    if "card" in matches and any(
        word in text for word in ("block", "ब्लॉक", "freeze", "खो गया", "lost")
    ):
        return "block_card", ""
    if "payment_bill_emi" in matches:
        if any(word in text for word in ("emi", "ईएमआई", "किस्त", "हप्ता", "loan", "ऋण", "கடன்", "లోన్")):
            return "emi_due", ""
        return "pay_bill", ""
    if "account_balance" in matches:
        balance_words = ("balance", "बैलेंस", "शिल्लक", "টাকা", "பணம்", "డబ్బు", "પૈસા", "ಹಣ")
        if any(word.casefold() in text for word in balance_words):
            return "balance", ""
        return "other", "ambiguous_in_domain"
    if "card" in matches:
        return "other", "unsupported_near_miss"
    return "other", "ambiguous_in_domain"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    options = args()
    repo_root = Path(__file__).resolve().parents[1]
    requested_codes = source_languages(options.utterances)
    configs = {code: LANGUAGE_CONFIGS[code] for code in requested_codes if code in LANGUAGE_CONFIGS}
    skipped = [code for code in requested_codes if code not in LANGUAGE_CONFIGS]
    token = os.environ.get("HF_TOKEN") or dotenv_value(repo_root / ".env", "HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment or project .env file.")

    print(f"Requested CSV languages: {', '.join(requested_codes)}")
    print(f"IndicVoices configs: {', '.join(configs.values()) or 'none'}")
    if skipped:
        print(f"Skipped unavailable configs: {', '.join(skipped)}")
    if options.dry_run:
        return

    audio_dir = options.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    counts = {code: 0 for code in configs}
    for code, config in configs.items():
        for split in options.splits:
            print(f"Scanning {config}/{split}...")
            dataset = load_dataset(
                "ai4bharat/IndicVoices", config, split=split,
                streaming=True, token=token,
            ).cast_column("audio_filepath", Audio(decode=False))
            for index, record in enumerate(dataset):
                if options.limit_per_language and index >= options.limit_per_language:
                    break
                matches = keyword_matches(normalized_text(record))
                if sum(len(values) for values in matches.values()) < options.min_keyword_hits:
                    continue
                searchable_text = normalized_text(record)
                intent, other_type = assignment_label(searchable_text, matches)
                audio = record.get("audio_filepath") or {}
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if not isinstance(audio_bytes, bytes) or not audio_bytes:
                    raise RuntimeError(f"Missing audio bytes in {config}/{split} record {index}.")
                source_id = f"{config}_{split}_{index:08d}_{record.get('speaker_id', 'unknown')}"
                filename = safe_name(source_id) + ".flac"
                path = audio_dir / filename
                path.write_bytes(audio_bytes)
                record_without_audio = {
                    key: value for key, value in record.items() if key != "audio_filepath"
                }
                output = {
                    **record_without_audio,
                    "id": source_id,
                    "intent": intent,
                    "label_provenance": "keyword_derived_candidate",
                    "review_status": "unreviewed",
                    "text": record.get("normalized") or record.get("text") or "",
                    "other_type": other_type,
                    "source": "indicvoices",
                    "script": "native",
                    "mix": "mono",
                    "source_dataset": "ai4bharat/IndicVoices",
                    "config": config,
                    "language_code": code,
                    "split": split,
                    "source_index": index,
                    "matched_categories": sorted(matches),
                    "matched_keywords": matches,
                    "audio_path": str(path.relative_to(options.output)).replace("\\", "/"),
                    "audio_format": "flac",
                    "audio_sha256": sha256_bytes(audio_bytes),
                }
                records.append(output)
                counts[code] += 1
                if len(records) % 100 == 0:
                    print(f"kept {len(records)} records")

    options.output.mkdir(parents=True, exist_ok=True)
    with (options.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    fieldnames = sorted({key for record in records for key in record})
    with (options.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                   for key, value in record.items()}
            writer.writerow(row)
    summary = {
        "source_dataset": "ai4bharat/IndicVoices",
        "source_utterances": str(options.utterances),
        "requested_language_codes": requested_codes,
        "downloaded_language_configs": configs,
        "skipped_language_codes": skipped,
        "splits": options.splits,
        "min_keyword_hits": options.min_keyword_hits,
        "records_kept": len(records),
        "records_by_language": counts,
        "audio_directory": "audio",
    }
    (options.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
