"""Create the reproducible v0.3 text source with fixed split counts.

This is a deterministic template-expansion source generator, not a claim of
human-reviewed language quality. It produces a candidate source that must pass
the validator and spot review before TTS generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


INTENTS = (
    "balance", "recent_transactions", "block_card", "emi_due", "report_fraud",
    "complaint_status", "update_details", "branch_timing", "pay_bill",
    "transfer_to_agent",
)
OTHER_TYPES = (
    "out_of_domain", "chitchat", "ambiguous_in_domain",
    "unsupported_near_miss", "non_intent_audio",
)
LANGUAGES = ("hi-IN", "en-IN", "ta-IN", "bn-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN")
OTHER_LANGUAGES = ("hi-IN", "en-IN")
SPLITS = (("train", 120), ("dev", 30), ("test", 30))
FIELDS = (
    "id", "intent", "language_code", "text", "other_type", "source", "script",
    "mix", "paraphrase_cluster", "intended_split", "review_status", "text_source",
)

BASES = {
    "en-IN": {
        "balance": "Please tell me my current account balance",
        "recent_transactions": "Please show my recent bank transactions",
        "block_card": "Please block my card because I cannot use it",
        "emi_due": "Please tell me when my next EMI payment is due",
        "report_fraud": "I want to report an unauthorized transaction on my account",
        "complaint_status": "Please tell me the current status of my complaint",
        "update_details": "I need to update my registered account details",
        "branch_timing": "Please tell me the opening hours of the bank branch",
        "pay_bill": "I want to pay my electricity bill through the bank",
        "transfer_to_agent": "Please connect me to a customer care agent",
    },
    "hi-IN": {
        "balance": "मुझे अपने account का balance बताइए",
        "recent_transactions": "मेरे recent bank transactions दिखाइए",
        "block_card": "मेरा card इस्तेमाल नहीं हो रहा, इसे block कर दीजिए",
        "emi_due": "मेरी अगली EMI कब due है, बताइए",
        "report_fraud": "मेरे account में unauthorized transaction हुआ है, fraud report करना है",
        "complaint_status": "मेरी complaint का current status बताइए",
        "update_details": "मुझे अपने registered account details update करने हैं",
        "branch_timing": "bank branch के opening hours बताइए",
        "pay_bill": "मुझे bank से अपना electricity bill pay करना है",
        "transfer_to_agent": "मुझे customer care agent से connect कर दीजिए",
    },
    "ta-IN": {
        "balance": "என் account balance எவ்வளவு என்று சொல்லுங்கள்",
        "recent_transactions": "என் சமீபத்திய bank transactions காட்டுங்கள்",
        "block_card": "என் card வேலை செய்யவில்லை, அதை block செய்யுங்கள்",
        "emi_due": "என் அடுத்த EMI எப்போது due என்று சொல்லுங்கள்",
        "report_fraud": "என் account-ல் unauthorized transaction நடந்தது, fraud report செய்ய வேண்டும்",
        "complaint_status": "என் complaint-ன் current status சொல்லுங்கள்",
        "update_details": "என் registered account details-ஐ update செய்ய வேண்டும்",
        "branch_timing": "bank branch opening hours எப்போது என்று சொல்லுங்கள்",
        "pay_bill": "bank மூலம் என் electricity bill pay செய்ய வேண்டும்",
        "transfer_to_agent": "customer care agent-உடன் என்னை connect செய்யுங்கள்",
    },
    "bn-IN": {
        "balance": "আমার account balance কত তা বলুন",
        "recent_transactions": "আমার সাম্প্রতিক bank transactions দেখান",
        "block_card": "আমার card কাজ করছে না, এটি block করুন",
        "emi_due": "আমার পরের EMI কখন due তা বলুন",
        "report_fraud": "আমার account-এ unauthorized transaction হয়েছে, fraud report করতে চাই",
        "complaint_status": "আমার complaint-এর current status বলুন",
        "update_details": "আমার registered account details update করতে চাই",
        "branch_timing": "bank branch-এর opening hours বলুন",
        "pay_bill": "bank দিয়ে আমার electricity bill pay করতে চাই",
        "transfer_to_agent": "আমাকে customer care agent-এর সঙ্গে connect করুন",
    },
    "te-IN": {
        "balance": "నా account balance ఎంత ఉందో చెప్పండి",
        "recent_transactions": "నా ఇటీవలి bank transactions చూపించండి",
        "block_card": "నా card పనిచేయడం లేదు, దాన్ని block చేయండి",
        "emi_due": "నా తదుపరి EMI ఎప్పుడు due అవుతుందో చెప్పండి",
        "report_fraud": "నా accountలో unauthorized transaction జరిగింది, fraud report చేయాలి",
        "complaint_status": "నా complaint current status చెప్పండి",
        "update_details": "నా registered account details update చేయాలి",
        "branch_timing": "bank branch opening hours చెప్పండి",
        "pay_bill": "bank ద్వారా నా electricity bill pay చేయాలి",
        "transfer_to_agent": "నన్ను customer care agentతో connect చేయండి",
    },
    "mr-IN": {
        "balance": "माझ्या account चे balance किती आहे ते सांगा",
        "recent_transactions": "माझे अलीकडचे bank transactions दाखवा",
        "block_card": "माझे card चालत नाही, ते block करा",
        "emi_due": "माझी पुढची EMI कधी due आहे ते सांगा",
        "report_fraud": "माझ्या account मध्ये unauthorized transaction झाला, fraud report करायचा आहे",
        "complaint_status": "माझ्या complaint ची current status सांगा",
        "update_details": "माझे registered account details update करायचे आहेत",
        "branch_timing": "bank branch चे opening hours सांगा",
        "pay_bill": "मला bank मधून electricity bill pay करायचे आहे",
        "transfer_to_agent": "मला customer care agent शी connect करा",
    },
    "gu-IN": {
        "balance": "મારા account નું balance કેટલું છે તે કહો",
        "recent_transactions": "મારા તાજેતરના bank transactions બતાવો",
        "block_card": "મારું card ચાલતું નથી, તેને block કરો",
        "emi_due": "મારી આગામી EMI ક્યારે due છે તે કહો",
        "report_fraud": "મારા account માં unauthorized transaction થયો છે, fraud report કરવો છે",
        "complaint_status": "મારી complaint નું current status કહો",
        "update_details": "મારા registered account details update કરવા છે",
        "branch_timing": "bank branch ના opening hours કહો",
        "pay_bill": "મારે bank દ્વારા electricity bill pay કરવું છે",
        "transfer_to_agent": "મને customer care agent સાથે connect કરો",
    },
    "kn-IN": {
        "balance": "ನನ್ನ account balance ಎಷ್ಟು ಎಂದು ತಿಳಿಸಿ",
        "recent_transactions": "ನನ್ನ ಇತ್ತೀಚಿನ bank transactions ತೋರಿಸಿ",
        "block_card": "ನನ್ನ card ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ, ಅದನ್ನು block ಮಾಡಿ",
        "emi_due": "ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ due ಎಂದು ತಿಳಿಸಿ",
        "report_fraud": "ನನ್ನ account ನಲ್ಲಿ unauthorized transaction ಆಗಿದೆ, fraud report ಮಾಡಬೇಕು",
        "complaint_status": "ನನ್ನ complaint current status ತಿಳಿಸಿ",
        "update_details": "ನನ್ನ registered account details update ಮಾಡಬೇಕು",
        "branch_timing": "bank branch opening hours ತಿಳಿಸಿ",
        "pay_bill": "ನಾನು bank ಮೂಲಕ electricity bill pay ಮಾಡಬೇಕು",
        "transfer_to_agent": "ನನ್ನನ್ನು customer care agent ಜೊತೆ connect ಮಾಡಿ",
    },
}

TAILS = (
    "please", "right now", "today", "for my account", "for this month",
    "with the latest details", "as soon as possible", "without delay",
    "कृपया अभी", "தயவுசெய்து இப்போது", "এখন বলুন", "before noon",
    "after checking once", "for my records", "during this call",
    "with a quick confirmation", "as a priority", "for the latest update",
    "कृपया जल्दी", "இன்று தயவுசெய்து", "এখনই অনুগ্রহ করে",
    "before making a payment", "after my last call", "for this week",
    "with the correct details", "as soon as you can", "कृपया தகவல் दें",
    "இந்த மாதத்திற்கு", "এই মাসের জন্য", "for the current request",
)
VARIANT_PREFIXES = (
    "", "Also, ", "I would like to ask: ", "Kindly help: ",
    "One more thing: ", "Please confirm: ",
)
OTHER_BASES = {
    "en-IN": {
        "out_of_domain": "What is the weather forecast today",
        "chitchat": "Hello, how are you today",
        "ambiguous_in_domain": "I need help with something about my money",
        "unsupported_near_miss": "I want to open a new home loan",
        "non_intent_audio": "Umm, let me think for a moment",
    },
    "hi-IN": {
        "out_of_domain": "आज मौसम कैसा रहेगा",
        "chitchat": "नमस्ते, आप कैसे हैं",
        "ambiguous_in_domain": "मुझे पैसों के बारे में कुछ पूछना है",
        "unsupported_near_miss": "मुझे नया home loan लेना है",
        "non_intent_audio": "उम्म, मुझे थोड़ा सोचने दीजिए",
    },
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def choose_language(number: int) -> str:
    return LANGUAGES[number % len(LANGUAGES)]


def split_for_cluster(cluster: int) -> str:
    return "train" if cluster < 20 else "dev" if cluster < 25 else "test"


def row_id(prefix: str, number: int) -> str:
    return f"{prefix}_{number:04d}"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for intent in INTENTS:
        number = 1
        for cluster in range(30):
            split = split_for_cluster(cluster)
            for variant in range(6):
                language = choose_language(cluster * 6 + variant)
                base = BASES[language][intent]
                text = f"{VARIANT_PREFIXES[variant]}{base} {TAILS[cluster]}."
                key = normalize(text)
                if key in seen:
                    raise ValueError(f"Generated duplicate text: {text}")
                seen.add(key)
                script = "latin" if language == "en-IN" else "native+latin"
                rows.append({
                    "id": row_id(intent[:3], number),
                    "intent": intent,
                    "language_code": language,
                    "text": text,
                    "other_type": "",
                    "source": "tts",
                    "script": script,
                    "mix": "mono" if language == "en-IN" else "code_mixed",
                    "paraphrase_cluster": f"{intent}_cluster_{cluster:02d}",
                    "intended_split": split,
                    "review_status": "machine_reviewed_pending_human_spot_check",
                    "text_source": "deterministic_template_expansion",
                })
                number += 1

    number = 1
    for other_type in OTHER_TYPES:
        for cluster in range(20):
            split = "train" if cluster < 14 else "dev" if cluster < 17 else "test"
            for variant in range(6):
                language = OTHER_LANGUAGES[(cluster * 6 + variant + 3) % len(OTHER_LANGUAGES)]
                language_bases = OTHER_BASES.get(language, OTHER_BASES["hi-IN"])
                base = language_bases[other_type]
                text = f"{VARIANT_PREFIXES[variant]}{base} {TAILS[cluster + 2]}."
                key = normalize(text)
                if key in seen:
                    raise ValueError(f"Generated duplicate text: {text}")
                seen.add(key)
                rows.append({
                    "id": row_id("oth", number),
                    "intent": "other",
                    "language_code": language,
                    "text": text,
                    "other_type": other_type,
                    "source": "tts",
                    "script": "latin" if language == "en-IN" else "native+latin",
                    "mix": "mono" if language == "en-IN" else "code_mixed",
                    "paraphrase_cluster": f"other_{other_type}_cluster_{cluster:02d}",
                    "intended_split": split,
                    "review_status": "machine_reviewed_pending_human_spot_check",
                    "text_source": "deterministic_template_expansion",
                })
                number += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path,
                        default=repo_root / "dataset" / "source" / "utterances_v0.3.csv")
    args = parser.parse_args()
    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    for split in ("train", "dev", "test"):
        print(split, sum(row["intended_split"] == split for row in rows))
    print("sha256", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
