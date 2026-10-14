"""Create deterministic, documented robustness variants from clean WAV files.

The default suite is a controlled simulation, not real-call validation. It uses
colored-noise environmental proxies and simple telephone/channel transforms
without external audio dependencies or unlicensed background recordings.
"""

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


def normalize_noise(noise: list[float], target_rms: float) -> list[float]:
    scale = target_rms / max(rms(noise), 1e-5)
    return [value * scale for value in noise]


def colored_noise(length: int, rng: random.Random, kind: str) -> list[float]:
    values = [rng.gauss(0.0, 1.0) for _ in range(length)]
    if kind == "traffic":
        smoothed: list[float] = []
        previous = 0.0
        for value in values:
            previous = 0.92 * previous + 0.08 * value
            smoothed.append(previous)
        return smoothed
    if kind == "office":
        return [value + 0.35 * math.sin(index / 31.0) for index, value in enumerate(values)]
    if kind == "fan":
        return [value + 0.8 * math.sin(index / 120.0) for index, value in enumerate(values)]
    raise ValueError(f"Unknown environmental proxy: {kind}")


def telephone_bandlimit(samples: list[float], rate: int) -> list[float]:
    """Approximate a 300-3400 Hz telephone band with deterministic FIR stages."""
    low_pass_window = max(2, round(rate / 6800))
    high_pass_window = max(2, round(rate / 300))
    low_pass: list[float] = []
    running = 0.0
    for index, value in enumerate(samples):
        running += value
        if index >= low_pass_window:
            running -= samples[index - low_pass_window]
        low_pass.append(running / min(index + 1, low_pass_window))
    high_pass: list[float] = []
    baseline = 0.0
    for index, value in enumerate(low_pass):
        baseline += value
        if index >= high_pass_window:
            baseline -= low_pass[index - high_pass_window]
        high_pass.append(value - baseline / min(index + 1, high_pass_window))
    return high_pass


def packet_loss(samples: list[float], rate: int, rng: random.Random) -> list[float]:
    output = samples[:]
    burst = max(1, round(rate * 0.035))
    for _ in range(max(1, len(samples) // (rate * 2))):
        start = rng.randrange(0, max(1, len(output) - burst))
        for index in range(start, min(len(output), start + burst)):
            output[index] = 0.0
    return output


def quantize(samples: list[float], bits: int = 8) -> list[float]:
    levels = (2 ** bits) - 1
    return [round(max(-1.0, min(1.0, value)) * levels) / levels for value in samples]


def degrade(
    samples: list[float],
    rate: int,
    seed: int,
    snr_db: float,
    environment: str | None = None,
    channel: bool = False,
    loss: bool = False,
) -> list[float]:
    rng = random.Random(seed)
    signal_rms = max(rms(samples), 1e-5)
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    if environment:
        noise = normalize_noise(colored_noise(len(samples), rng, environment), noise_rms)
    else:
        noise = [rng.gauss(0.0, noise_rms) for _ in samples]
    noisy = [value + noise[index] for index, value in enumerate(samples)]
    if channel:
        noisy = telephone_bandlimit(noisy, rate)
        noisy = quantize(noisy, bits=8)
    clipped = [max(-0.82, min(0.82, value)) for value in noisy]
    delay = max(1, round(rate * 0.035))
    reverberated = clipped[:]
    for index in range(delay, len(clipped)):
        reverberated[index] += 0.12 * clipped[index - delay]
    dropout = max(1, round(rate * 0.018))
    for _ in range(max(1, len(samples) // (rate * 3))):
        start = rng.randrange(0, max(1, len(reverberated) - dropout))
        for index in range(start, min(len(reverberated), start + dropout)):
            reverberated[index] *= 0.05
    if loss:
        reverberated = packet_loss(reverberated, rate, rng)
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
    parser.add_argument("--suite", choices=("baseline", "robustness"), default="baseline")
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
        conditions = [
            {"name": f"gaussian_snr{int(snr_db)}", "snr_db": snr_db}
            for snr_db in args.snr_db
        ]
        if args.suite == "robustness":
            conditions.extend([
                {"name": "traffic_snr15", "snr_db": 15.0, "environment": "traffic"},
                {"name": "office_snr15", "snr_db": 15.0, "environment": "office"},
                {"name": "telephone_channel", "snr_db": 35.0, "channel": True},
                {"name": "unseen_combined", "snr_db": 15.0, "environment": "fan",
                 "channel": True, "loss": True},
            ])
        for condition in conditions:
            if condition["name"] == "unseen_combined" and row.get("split") != "test":
                continue
            seed = int(hashlib.sha256(
                f"{row['id']}:{condition['name']}".encode()
            ).hexdigest()[:8], 16)
            filename = f"{row['id']}__{row.get('speaker', 'unknown')}__{condition['name']}.wav"
            output_path = output_audio / filename
            if not output_path.exists():
                write_wav(output_path, rate, degrade(
                    samples, rate, seed, condition["snr_db"],
                    condition.get("environment"), condition.get("channel", False),
                    condition.get("loss", False),
                ))
            generated.append({
                **row,
                "audio_path": str(output_path.relative_to(args.output)).replace("\\", "/"),
                "variant_type": "synthetic_degraded",
                "degradation": json.dumps({
                    "condition": condition["name"], "snr_db": condition["snr_db"],
                    "environment_proxy": condition.get("environment"),
                    "telephone_channel": condition.get("channel", False),
                    "packet_loss": condition.get("loss", False),
                    "clipping_limit": 0.82, "reverb_delay_ms": 35,
                    "dropout_ms": 35 if condition.get("loss") else 0, "seed": seed,
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
    summary = {
        "rows": len(generated), "snr_db": args.snr_db, "suite": args.suite,
        "source_manifest": str(args.manifest),
        "conditions": sorted({
            json.loads(str(row["degradation"]))["condition"] for row in generated
        }),
        "note": "Synthetic proxies only; not real-call validation.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
