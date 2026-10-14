"""Run the Saaras ASR -> character n-gram logistic regression baseline.

The script supports a cached transcript CSV so ASR and classifier evaluation
are reproducible and API calls are never repeated unnecessarily. Saaras
transcription is opt-in with --transcribe; classifier-only runs require a
transcript CSV containing id, transcript, split, intent, and condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ASR_URL = "https://api.sarvam.ai/speech-to-text"


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def multipart(file_path: Path, language_code: str | None) -> tuple[bytes, str]:
    boundary = f"----VoxIntel{uuid.uuid4().hex}"
    chunks = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{file_path.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
        file_path.read_bytes(),
        f"\r\n--{boundary}\r\n".encode(),
    ]
    if language_code:
        chunks.append(
            f'Content-Disposition: form-data; name="language_code"\r\n\r\n'
            f"{language_code}\r\n--{boundary}\r\n".encode()
        )
    chunks.append(b"--\r\n")
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def transcribe(audio_path: Path, api_key: str, language_code: str | None) -> str:
    body, content_type = multipart(audio_path, language_code)
    request = urllib.request.Request(
        ASR_URL,
        data=body,
        headers={
            "api-subscription-key": api_key,
            "Content-Type": content_type,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Saaras request failed for {audio_path}: {exc}") from exc
    transcript = payload.get("transcript")
    if not isinstance(transcript, str):
        raise ValueError(f"Saaras response lacked a transcript for {audio_path}")
    return transcript


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["id", "transcript", "split", "intent", "condition"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def transcribe_manifest(manifest: Path, audio_root: Path, cache: Path, api_key: str) -> None:
    rows = read_csv(manifest)
    cached = {row["id"]: row for row in read_csv(cache)} if cache.exists() else {}
    for row in rows:
        if row["id"] in cached:
            continue
        audio_path = audio_root / row["audio_path"]
        cached[row["id"]] = {
            "id": row["id"],
            "transcript": transcribe(audio_path, api_key, row.get("language_code")),
            "split": row["split"],
            "intent": row["intent"],
            "other_type": row.get("other_type", ""),
            "condition": json.loads(row["degradation"])["condition"],
        }
        write_csv(cache, list(cached.values()))
    write_csv(cache, list(cached.values()))


def evaluate(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train" and
             row["condition"] != "unseen_combined"]
    calibration = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    model = Pipeline([
        ("features", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                     min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit([row["transcript"] for row in train], [row["intent"] for row in train])
    calibration_accuracy = model.score(
        [row["transcript"] for row in calibration], [row["intent"] for row in calibration]
    )
    test_predictions = model.predict([row["transcript"] for row in test])
    test_accuracy = float(sum(
        prediction == row["intent"] for prediction, row in zip(test_predictions, test)
    ) / max(1, len(test)))
    report = {
        "model": "character_tfidf_3_5_logistic_regression",
        "asr": "Sarvam Saaras API",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "final_test_rows": len(test),
        "calibration_accuracy": calibration_accuracy,
        "final_test_accuracy": test_accuracy,
        "note": (
            "This is a baseline result only. Synthetic and public data must "
            "never be merged, and public unreviewed labels are excluded."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path,
                        default=root / "dataset" / "manifests" / "evaluation" /
                        "synthetic_final_test.csv")
    parser.add_argument("--all-manifests", type=Path,
                        help="CSV containing train, dev, and test ASR transcripts.")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--transcripts", type=Path,
                        default=root / "dataset" / "manifests" / "evaluation" /
                        "saaras_transcripts.csv")
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=root / "dataset" / "reports" / "saaras_baseline.json")
    args = parser.parse_args()
    dotenv = load_dotenv(root / ".env")
    if args.transcribe:
        if not args.all_manifests or not args.audio_root:
            parser.error("--transcribe requires --all-manifests and --audio-root")
        api_key = os.environ.get("SARVAM_API_KEY") or dotenv.get("SARVAM_API_KEY")
        if not api_key:
            parser.error("SARVAM_API_KEY is required for --transcribe")
        transcribe_manifest(args.all_manifests, args.audio_root, args.transcripts, api_key)
    if not args.transcripts.exists():
        parser.error("Transcript cache not found; run with --transcribe first")
    evaluate(read_csv(args.transcripts), args.output)


if __name__ == "__main__":
    main()
