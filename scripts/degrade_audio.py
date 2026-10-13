"""Create deterministic telephony-style noisy variants from clean WAV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import wave
from pathlib import Path


def read_wav(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit WAV: {path}")
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = [int.from_bytes(raw[index:index + 2], "little", signed=True) / 32768.0
               for index in range(0, len(raw), 2)]
    return rate, samples


def write_wav(path: Path, rate: int, samples: list[float]) -> None:
    raw = b"".join(int(max(-1.0, min(1.0, sample)) * 32767).to_bytes(2, "little", signed=True)
                   for sample in samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(raw)


def rms(samples: list[float]) -> float:
    return math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))


def degrade(samples: list[float], rate: int, seed: int, snr_db: float) -> list[float]:
    rng = random.Random(seed)
    signal_rms = max(rms(samples), 1e-5)
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noisy = [value + rng.gauss(0.0, noise_rms) for value in samples]
    clipped = [max(-0.82, min(0.82, value)) for value in noisy]
    delay = max(1, round(rate * 0.035))
    reverberated = clipped[:]
    for index in range(delay, len(clipped)):
        reverberated[index] += 0.12 * clipped[index - delay]
    dropout = max(1, round(rate * 0.018))
    for _ in range(max(1, len(samples) // (rate * 3))):
        start = rng.randrange(0, max(1, len(samples) - dropout))
        for index in range(start, min(len(reverberated), start + dropout)):
            reverberated[index] *= 0.05
    return reverberated


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[10.0, 20.0])
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output_audio = args.output / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, str | int | float]] = []
    for row in rows[:args.max_files]:
        source_path = args.manifest.parent / row["audio_path"]
        rate, samples = read_wav(source_path)
        for snr_db in args.snr_db:
            seed = int(hashlib.sha256(f"{row['id']}:{snr_db}".encode()).hexdigest()[:8], 16)
            filename = f"{row['id']}__{row.get('speaker', 'unknown')}__snr{int(snr_db)}.wav"
            output_path = output_audio / filename
            if not output_path.exists():
                write_wav(output_path, rate, degrade(samples, rate, seed, snr_db))
            generated.append({
                **row,
                "audio_path": str(output_path.relative_to(args.output)).replace("\\", "/"),
                "variant_type": "synthetic_degraded",
                "degradation": json.dumps({
                    "snr_db": snr_db, "clipping_limit": 0.82,
                    "reverb_delay_ms": 35, "dropout_ms": 18, "seed": seed,
                    "source_audio_path": row["audio_path"],
                }, sort_keys=True),
                "sha256": digest(output_path),
                "sample_rate": rate,
            })
    if generated:
        fieldnames = list(generated[0])
        with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(generated)
    summary = {"rows": len(generated), "snr_db": args.snr_db, "source_manifest": str(args.manifest)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
