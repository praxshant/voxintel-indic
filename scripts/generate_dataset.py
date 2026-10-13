"""Generate a reproducible Sarvam Bulbul speech-to-intent dataset.

The script keeps the source CSV as the label source of truth. TTS rows are
sent to Sarvam's REST API, while explicit silence/noise rows are generated
locally. API keys are read from SARVAM_API_KEY or a local .env file and are
never written to output files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import random
import struct
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


API_URL = "https://api.sarvam.ai/text-to-speech"
SUPPORTED_LANGUAGES = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
}
SPEAKERS = {
    "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja",
    "rohan", "simran", "kavya", "amit", "dev", "ishita", "shreya",
    "ratan", "varun", "manan", "sumit", "roopa", "kabir", "aayan",
    "ashutosh", "advait", "anand", "tanya", "tarun", "sunny", "mani",
    "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan",
    "soham", "rupali",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("utterances_v0.2.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset_v0.2"))
    parser.add_argument("--model", default="bulbul:v3")
    parser.add_argument("--speakers", nargs="+", default=["shubh", "simran"])
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "intent", "language_code", "text", "other_type", "source", "script", "mix"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate utterance IDs found.")
    for row in rows:
        if row["source"] == "tts" and not row["text"].strip():
            raise ValueError(f"TTS row {row['id']} has empty text.")
        if row["source"] == "tts" and row["language_code"] not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code in {row['id']}: {row['language_code']}")
        if row.get("intended_split") and row["intended_split"] not in {"train", "dev", "test"}:
            raise ValueError(f"Invalid intended_split in {row['id']}: {row['intended_split']}")
    return rows


def wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    pcm = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, sample))
        pcm.extend(struct.pack("<h", round(clipped * 32767)))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(pcm))
    return output.getvalue()


def create_special_audio(row_id: str, sample_rate: int) -> bytes:
    duration = 2.0 if row_id == "non_005" else 2.0
    count = round(sample_rate * duration)
    if row_id == "non_005":
        samples = [0.0] * count
    else:
        rng = random.Random(row_id)
        samples = [rng.uniform(-0.08, 0.08) for _ in range(count)]
    return wav_bytes(samples, sample_rate)


def sarvam_audio(
    api_key: str,
    text: str,
    language_code: str,
    model: str,
    speaker: str,
    sample_rate: int,
    pace: float,
    retries: int,
) -> bytes:
    payload = json.dumps({
        "text": text,
        "language_code": language_code,
        "speaker": speaker,
        "model": model,
        "speech_sample_rate": sample_rate,
        "pace": pace,
    }).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "api-subscription-key": api_key,
        },
        method="POST",
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            audios = body.get("audios")
            if not isinstance(audios, list) or not audios or not isinstance(audios[0], str):
                raise ValueError("Sarvam response did not contain an audio payload.")
            return base64.b64decode(audios[0])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Sarvam request failed after {retries} attempts: {exc}") from exc
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def is_valid_wav(path: Path, expected_rate: int) -> bool:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                wav_file.getnchannels() == 1
                and wav_file.getsampwidth() == 2
                and wav_file.getframerate() == expected_rate
                and wav_file.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_split(row_id: str, variant: int, speaker: str, speakers: list[str]) -> str:
    # Keep the first speaker in training and reserve the second for evaluation.
    if len(speakers) > 1 and speaker == speakers[1]:
        return "test"
    bucket = int(hashlib.sha256(f"{row_id}:{variant}".encode()).hexdigest()[:8], 16) % 10
    return "dev" if bucket == 0 else "train"


def split_for_row(row: dict[str, str], variant: int, speaker: str, speakers: list[str]) -> str:
    intended = row.get("intended_split", "").strip()
    return intended or assign_split(row["id"], variant, speaker, speakers)


def speaker_for_row(row: dict[str, str], variant: int, speakers: list[str]) -> str:
    intended = row.get("intended_split", "").strip()
    if intended and len(speakers) >= 3:
        return speakers[{"train": 0, "dev": 1, "test": 2}[intended]]
    return speakers[variant % len(speakers)]


def main() -> None:
    args = parse_args()
    if args.sample_rate not in {8000, 16000, 22050, 24000, 32000, 44100, 48000}:
        raise ValueError("Unsupported sample rate.")
    if args.variants < 1:
        raise ValueError("--variants must be at least 1.")
    if any(speaker not in SPEAKERS for speaker in args.speakers):
        raise ValueError(f"Unknown speaker. Supported v3 examples: {', '.join(sorted(SPEAKERS))}")

    rows = read_rows(args.csv)
    dotenv = load_dotenv(args.csv.parent / ".env")
    api_key = os.environ.get("SARVAM_API_KEY") or dotenv.get("SARVAM_API_KEY", "")
    tts_rows = [row for row in rows if row["source"] == "tts"]
    local_rows = [row for row in rows if row["source"] != "tts"]
    print(f"Rows: {len(rows)} total; {len(tts_rows)} TTS; {len(local_rows)} local audio")
    print(f"Languages: {', '.join(sorted({row['language_code'] for row in tts_rows}))}")
    print(f"Output: {args.output.resolve()}")
    if args.dry_run:
        return
    if not api_key:
        raise RuntimeError("Set SARVAM_API_KEY in the environment or .env before generating audio.")

    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    errors_path = args.output / "errors.jsonl"
    manifest: list[dict[str, Any]] = []
    for row in rows:
        variants = range(args.variants) if row["source"] == "tts" else range(1)
        for variant in variants:
            speaker = speaker_for_row(row, variant, args.speakers) if row["source"] == "tts" else "local"
            filename = f"{row['id']}__v{variant + 1}__{speaker}.wav"
            path = audio_dir / filename
            if path.exists() and not args.force and is_valid_wav(path, args.sample_rate):
                print(f"exists {filename}")
            else:
                print(f"generate {filename}")
                try:
                    if row["source"] == "tts":
                        audio = sarvam_audio(
                            api_key, row["text"], row["language_code"], args.model,
                            speaker, args.sample_rate, args.pace, args.retries,
                        )
                    else:
                        audio = create_special_audio(row["id"], args.sample_rate)
                    path.write_bytes(audio)
                except (OSError, RuntimeError, ValueError) as exc:
                    with errors_path.open("a", encoding="utf-8") as error_handle:
                        error_handle.write(json.dumps({
                            "id": row["id"], "variant": variant + 1,
                            "speaker": speaker, "error": str(exc),
                        }, ensure_ascii=False) + "\n")
                    raise
                if not is_valid_wav(path, args.sample_rate):
                    raise RuntimeError(f"Generated file failed WAV validation: {path}")
                if row["source"] == "tts":
                    time.sleep(args.sleep_seconds)
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
            manifest.append({
                **row,
                "variant": variant + 1,
                "speaker": speaker,
                "model": args.model if row["source"] == "tts" else "local_generator",
                "audio_path": str(path.relative_to(args.output)).replace("\\", "/"),
                "sample_rate": rate,
                "channels": 1,
                "duration_seconds": round(frames / rate, 4),
                "split": split_for_row(row, variant + 1, speaker, args.speakers),
                "paraphrase_cluster": row.get("paraphrase_cluster", ""),
                "source_split": row.get("intended_split", ""),
                "sha256": sha256(path),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    fieldnames = list(manifest[0])
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "source_csv": str(args.csv),
        "model": args.model,
        "sample_rate": args.sample_rate,
        "voices": args.speakers,
        "variants": args.variants,
        "rows": len(manifest),
        "splits": {split: sum(item["split"] == split for item in manifest) for split in ("train", "dev", "test")},
        "languages": sorted({item["language_code"] for item in manifest}),
    }
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
