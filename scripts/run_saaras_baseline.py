"""Run the ASR → character n-gram logistic-regression baseline.

Supports providers:
  --provider saaras          Original Saaras/Sarvam cloud ASR (unchanged behaviour)
  --provider indicconformer  Local AI4Bharat IndicConformer-600M via HuggingFace
  --provider whisper         Local OpenAI Whisper-base.en via HuggingFace (English en-IN)
  --provider hybrid          Hybrid local ensemble: IndicConformer for Indic languages,
                             Whisper-base.en for English (en-IN). Reuses valid cached transcripts.

ASR output is cached per-provider in separate CSV files:
  saaras_transcripts.csv
  indicconformer_transcripts.csv
  whisper_transcripts.csv
  hybrid_transcripts.csv

Failed requests remain explicit in the cache and are never converted into
fabricated transcripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Saaras constants
# ---------------------------------------------------------------------------
ASR_URL = "https://api.sarvam.ai/speech-to-text"
MODEL_NAME = "saarika:v2.5"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Shared cache schema
# ---------------------------------------------------------------------------
REQUIRED_CACHE_FIELDS = {
    "record_key", "id", "audio_path", "split", "intent", "condition",
    "transcript", "status", "provider",
}

# ---------------------------------------------------------------------------
# Language-code mapping for IndicConformer
# IndicConformer expects bare ISO 639-1 codes without country suffix.
# ---------------------------------------------------------------------------
INDICCONFORMER_LANG_MAP: dict[str, str] = {
    "bn-IN": "bn",
    "en-IN": "en",
    "gu-IN": "gu",
    "hi-IN": "hi",
    "kn-IN": "kn",
    "mr-IN": "mr",
    "ta-IN": "ta",
    "te-IN": "te",
}
INDICCONFORMER_SUPPORTED_LANGUAGES = frozenset(INDICCONFORMER_LANG_MAP) - {"en-IN"}
WHISPER_SUPPORTED_LANGUAGES = frozenset({"en-IN"})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else sorted(REQUIRED_CACHE_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def record_key(row: dict[str, str]) -> str:
    """Cache key is deterministic across source ID, audio path, and condition."""
    return hashlib.sha256(
        f"{row['id']}|{row['audio_path']}|{row['condition']}".encode()
    ).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Saaras provider
# ---------------------------------------------------------------------------

def _multipart(file_path: Path, language_code: str | None) -> tuple[bytes, str]:
    boundary = f"----VoxIntel{uuid.uuid4().hex}"
    parts = [
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{file_path.name}\"\r\nContent-Type: audio/wav\r\n\r\n"
        ).encode(),
        file_path.read_bytes(),
        b"\r\n",
    ]
    for name, value in (("model", MODEL_NAME), ("language_code", language_code)):
        if value:
            parts.extend([
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n".encode(),
            ])
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _saaras_transcribe_once(
    audio_path: Path,
    api_key: str,
    language_code: str | None,
) -> tuple[str, dict[str, Any]]:
    body, content_type = _multipart(audio_path, language_code)
    request = urllib.request.Request(
        ASR_URL,
        data=body,
        headers={"api-subscription-key": api_key, "Content-Type": content_type},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            status = response.status
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"http_{exc.code}: {body_text}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"transport_error: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Saaras response was not a JSON object")
    transcript = payload.get("transcript")
    if not isinstance(transcript, str):
        raise ValueError(
            f"Saaras response schema missing string transcript; keys={sorted(payload)}"
        )
    if not transcript.strip():
        raise ValueError("Saaras returned empty transcript")
    return transcript, {
        "http_status": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "response_keys": sorted(payload),
    }


def _saaras_transcribe_with_retries(
    audio_path: Path,
    api_keys: list[str],
    language_code: str | None,
    retries: int,
    sleep_seconds: float,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        retryable_failure = True
        for api_key in api_keys:
            try:
                transcript, meta = _saaras_transcribe_once(audio_path, api_key, language_code)
                meta["attempt"] = attempt + 1
                return transcript, meta
            except RuntimeError as exc:
                last_error = exc
                if not any(f"http_{status}" in str(exc) for status in RETRYABLE_STATUS):
                    retryable_failure = False
        if attempt < retries and retryable_failure:
            time.sleep(sleep_seconds * (2 ** attempt) + random.random() * 0.25)
    raise RuntimeError(str(last_error) if last_error else "Saaras request failed")


# ---------------------------------------------------------------------------
# IndicConformer provider
# ---------------------------------------------------------------------------

_IC_MODEL = None  # module-level singleton
_IC_HF_TOKEN: str | None = None  # set by main() before first load


def _load_indicconformer(device: str = "cuda") -> Any:
    """Load the IndicConformer model once and cache it.

    Uses _IC_HF_TOKEN (populated from .env HF_TOKEN / HF_TOKEN_2) so the
    gated HuggingFace repo can be accessed without exposing credentials.
    """
    global _IC_MODEL
    if _IC_MODEL is not None:
        return _IC_MODEL
    import torch
    from transformers import AutoModel

    resolved_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        print("[IndicConformer] CUDA not available; falling back to CPU.")
    print(f"[IndicConformer] Loading model (device={resolved_device}) …")
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
        token=_IC_HF_TOKEN or None,
    )
    model = model.to(resolved_device)
    model.eval()
    _IC_MODEL = model
    print("[IndicConformer] Model ready.")
    return _IC_MODEL


def _indicconformer_transcribe_once(
    audio_path: Path,
    language_code: str,
    device: str = "cuda",
) -> tuple[str, dict[str, Any]]:
    """Transcribe one audio file with IndicConformer.

    Args:
        audio_path: Path to a WAV file (any sample rate; will be resampled to 16 kHz).
        language_code: Bare ISO 639-1 code (e.g. 'hi', 'ta').
        device: torch device string.

    Returns:
        (transcript, metadata_dict)
    """
    import torch
    import torchaudio

    model = _load_indicconformer(device)

    # IndicConformer keeps TorchScript components in a plain dictionary, so
    # it does not expose parameters through the standard PyTorch iterator.
    actual_device = getattr(model, "d", torch.device("cpu"))

    started = time.perf_counter()
    wav, sr = torchaudio.load(str(audio_path))

    # Resample from 8 kHz (telephone) to 16 kHz (model requirement).
    # Note: resampling does NOT restore missing high-frequency content.
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)

    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.to(actual_device)

    with torch.no_grad():
        result = model(wav, language_code, "ctc")

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    # result may be a string or list
    if isinstance(result, (list, tuple)):
        transcript = result[0] if result else ""
    else:
        transcript = str(result) if result is not None else ""

    transcript = transcript.strip()
    if not transcript:
        raise ValueError(
            f"IndicConformer returned empty transcript for {audio_path.name} (lang={language_code})"
        )

    return transcript, {
        "latency_ms": latency_ms,
        "language_used": language_code,
        "sample_rate_input": sr,
        "resampled": sr != 16000,
    }


def _indicconformer_transcribe_with_retries(
    audio_path: Path,
    manifest_language_code: str,
    device: str,
    retries: int,
    sleep_seconds: float,
) -> tuple[str, dict[str, Any]]:
    """Retry wrapper for IndicConformer."""
    bare_lang = INDICCONFORMER_LANG_MAP.get(manifest_language_code)
    if manifest_language_code not in INDICCONFORMER_SUPPORTED_LANGUAGES:
        raise RuntimeError(
            f"Unsupported language_code {manifest_language_code!r} for IndicConformer. "
            f"Supported: {sorted(INDICCONFORMER_SUPPORTED_LANGUAGES)}"
        )
    if bare_lang is None:
        raise ValueError(
            f"Unsupported language_code '{manifest_language_code}' for IndicConformer. "
            f"Supported: {sorted(INDICCONFORMER_LANG_MAP)}"
        )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            transcript, meta = _indicconformer_transcribe_once(audio_path, bare_lang, device)
            meta["attempt"] = attempt + 1
            return transcript, meta
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds + random.random() * 0.25)

    raise RuntimeError(str(last_error) if last_error else "IndicConformer transcription failed")


# ---------------------------------------------------------------------------
# Whisper provider (for English en-IN)
# ---------------------------------------------------------------------------

_WHISPER_MODEL = None
_WHISPER_PROCESSOR = None


def _load_whisper(device: str = "cuda") -> tuple[Any, Any]:
    global _WHISPER_MODEL, _WHISPER_PROCESSOR
    if _WHISPER_MODEL is not None and _WHISPER_PROCESSOR is not None:
        return _WHISPER_MODEL, _WHISPER_PROCESSOR

    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    resolved_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch_dtype = torch.float16 if resolved_device != "cpu" else torch.float32

    print(f"[Whisper] Loading openai/whisper-base.en (device={resolved_device}) …")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-base.en",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(resolved_device)
    model.eval()

    _WHISPER_MODEL = model
    _WHISPER_PROCESSOR = processor
    print("[Whisper] Model ready.")
    return _WHISPER_MODEL, _WHISPER_PROCESSOR


def _whisper_transcribe_once(
    audio_path: Path,
    device: str = "cuda",
) -> tuple[str, dict[str, Any]]:
    import torch
    import torchaudio

    model, processor = _load_whisper(device)
    actual_device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    started = time.perf_counter()
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    input_features = processor(
        wav.squeeze().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(actual_device, dtype=dtype)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    transcript = transcript.strip()
    if not transcript:
        raise ValueError(f"Whisper returned empty transcript for {audio_path.name}")

    return transcript, {
        "latency_ms": latency_ms,
        "sample_rate_input": sr,
        "resampled": sr != 16000,
    }


def _whisper_transcribe_with_retries(
    audio_path: Path,
    device: str = "cuda",
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            transcript, meta = _whisper_transcribe_once(audio_path, device)
            meta["attempt"] = attempt + 1
            return transcript, meta
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds + random.random() * 0.25)

    raise RuntimeError(str(last_error) if last_error else "Whisper transcription failed")


# ---------------------------------------------------------------------------
# Unified transcription loop
# ---------------------------------------------------------------------------

def transcribe_manifest(
    manifest: Path,
    audio_root: Path,
    cache: Path,
    errors_path: Path,
    provider: str,
    # Saaras-specific
    api_keys: list[str] | None = None,
    # Local/IndicConformer/Whisper-specific
    ic_device: str = "cuda",
    whisper_device: str = "cuda",
    retries: int = 3,
    sleep_seconds: float = 2.0,
    max_new: int | None = None,
) -> dict[str, int]:
    """Transcribe rows in the manifest, resuming from existing cache.

    Args:
        provider: 'saaras', 'indicconformer', 'whisper', or 'hybrid'
    """
    source_rows = read_csv(manifest)
    cached = {row["record_key"]: row for row in read_csv(cache)} if cache.exists() else {}
    errors = read_csv(errors_path) if errors_path.exists() else []
    exhausted_keys: set[int] = set()
    attempted = 0
    unsaved_count = 0

    # Hybrid mode automatically seeds from already-successful rows in provider caches
    if provider == "hybrid":
        seeded = False
        ic_cache_path = cache.parent / "indicconformer_transcripts.csv"
        if ic_cache_path.exists():
            for r in read_csv(ic_cache_path):
                k = r.get("record_key")
                if k and k not in cached and r.get("status") == "success" and r.get("transcript", "").strip():
                    cached[k] = r
                    seeded = True
        wh_cache_path = cache.parent / "whisper_transcripts.csv"
        if wh_cache_path.exists():
            for r in read_csv(wh_cache_path):
                k = r.get("record_key")
                if k and k not in cached and r.get("status") == "success" and r.get("transcript", "").strip():
                    cached[k] = r
                    seeded = True
        if seeded and not cache.exists():
            write_csv(cache, list(cached.values()))

    for row in source_rows:
        key = record_key(row)
        existing = cached.get(key)
        if existing and existing.get("status") == "success" and existing.get("transcript", "").strip():
            continue

        lang = row.get("language_code", "")

        # Skip known permanent language incompatibilities for single-model providers
        if (
            provider == "indicconformer"
            and lang not in INDICCONFORMER_SUPPORTED_LANGUAGES
            and existing
            and existing.get("status") == "failed"
            and existing.get("error", "").startswith("Unsupported language_code")
        ):
            continue
        if (
            provider == "whisper"
            and lang not in WHISPER_SUPPORTED_LANGUAGES
            and existing
            and existing.get("status") == "failed"
        ):
            continue

        if max_new is not None and attempted >= max_new:
            break
        attempted += 1

        # Resolve audio path
        raw_audio_root = row.get("audio_root", "").strip()
        if raw_audio_root:
            row_audio_root = Path(raw_audio_root)
            resolved_audio_root = (
                row_audio_root if row_audio_root.is_absolute()
                else Path(__file__).resolve().parents[1] / row_audio_root
            )
        else:
            resolved_audio_root = audio_root
        audio_path = resolved_audio_root / row["audio_path"]

        condition = row.get("condition") or ""
        if not condition:
            try:
                condition = json.loads(row.get("degradation", "{}")).get("condition", "")
            except (json.JSONDecodeError, AttributeError):
                condition = ""

        output: dict[str, str] = {
            "record_key": key,
            "id": row["id"],
            "audio_path": row["audio_path"],
            "split": row["split"],
            "intent": row["intent"],
            "other_type": row.get("other_type", ""),
            "language_code": lang,
            "condition": condition,
            "transcript": "",
            "status": "failed",
            "provider": provider,
            "error": "",
            "attempts": "0",
            "latency_ms": "",
        }

        try:
            if provider == "saaras":
                if not api_keys:
                    raise RuntimeError("No Saaras API keys configured")
                active_keys = [k for i, k in enumerate(api_keys) if i not in exhausted_keys]
                if not active_keys:
                    raise RuntimeError("All Saaras API keys exhausted")
                transcript, meta = _saaras_transcribe_with_retries(
                    audio_path, active_keys,
                    lang,
                    retries, sleep_seconds,
                )
                actual_prov = "saaras"
            elif provider == "indicconformer":
                transcript, meta = _indicconformer_transcribe_with_retries(
                    audio_path,
                    lang,
                    ic_device,
                    retries, sleep_seconds,
                )
                actual_prov = "indicconformer"
            elif provider == "whisper":
                if lang not in WHISPER_SUPPORTED_LANGUAGES:
                    raise RuntimeError(f"Unsupported language_code {lang!r} for Whisper. Supported: {sorted(WHISPER_SUPPORTED_LANGUAGES)}")
                transcript, meta = _whisper_transcribe_with_retries(
                    audio_path,
                    whisper_device,
                    retries, sleep_seconds,
                )
                actual_prov = "whisper"
            elif provider == "hybrid":
                if lang in WHISPER_SUPPORTED_LANGUAGES:
                    transcript, meta = _whisper_transcribe_with_retries(
                        audio_path,
                        whisper_device,
                        retries, sleep_seconds,
                    )
                    actual_prov = "whisper"
                elif lang in INDICCONFORMER_SUPPORTED_LANGUAGES:
                    transcript, meta = _indicconformer_transcribe_with_retries(
                        audio_path,
                        lang,
                        ic_device,
                        retries, sleep_seconds,
                    )
                    actual_prov = "indicconformer"
                else:
                    raise RuntimeError(f"Unsupported language_code {lang!r} for hybrid pipeline.")
            else:
                raise ValueError(f"Unknown provider: {provider!r}")

            output.update({
                "transcript": transcript,
                "status": "success",
                "provider": actual_prov,
                "attempts": str(meta.get("attempt", 1)),
                "latency_ms": str(meta.get("latency_ms", "")),
            })

        except Exception as exc:
            output["error"] = str(exc)
            errors.append({**output, "error": str(exc)})
            if provider == "saaras" and "http_402" in str(exc) and api_keys:
                exhausted_keys.update(range(len(api_keys)))

        cached[key] = output
        unsaved_count += 1

        # Periodic checkpoint every 25 rows
        if unsaved_count >= 25:
            write_csv(cache, list(cached.values()))
            write_csv(errors_path, errors)
            unsaved_count = 0
            success_count = sum(r.get("status") == "success" for r in cached.values())
            print(f"[{provider}] Checkpoint: {success_count}/{len(source_rows)} successful (attempted {attempted} this session)")

        if provider == "saaras" and api_keys and len(exhausted_keys) == len(api_keys):
            print("[Saaras] All API keys quota-exhausted (HTTP 402). Stopping.")
            break

    # Final write on loop termination
    if unsaved_count > 0:
        write_csv(cache, list(cached.values()))
        write_csv(errors_path, errors)

    return {
        "requested": len(source_rows),
        "cached_success": sum(
            row.get("status") == "success" for row in cached.values()
        ),
        "failed": sum(row.get("status") == "failed" for row in cached.values()),
    }


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(manifest: Path, transcripts: Path) -> dict[str, Any]:
    expected = {record_key(row) for row in read_csv(manifest)}
    cached_rows = read_csv(transcripts) if transcripts.exists() else []
    successful = {
        row["record_key"] for row in cached_rows
        if row.get("status") == "success" and row.get("transcript", "").strip()
    }
    missing = expected - successful

    # Per-split and per-condition breakdown
    manifest_rows = read_csv(manifest)
    key_to_meta = {record_key(r): r for r in manifest_rows}

    split_counts: dict[str, dict[str, int]] = {}
    condition_counts: dict[str, dict[str, int]] = {}
    lang_counts: dict[str, dict[str, int]] = {}

    for k, r in key_to_meta.items():
        split = r.get("split", "unknown")
        cond = r.get("condition", "unknown")
        lang = r.get("language_code", "unknown")
        ok = k in successful

        split_counts.setdefault(split, {"total": 0, "success": 0})
        split_counts[split]["total"] += 1
        split_counts[split]["success"] += int(ok)

        condition_counts.setdefault(cond, {"total": 0, "success": 0})
        condition_counts[cond]["total"] += 1
        condition_counts[cond]["success"] += int(ok)

        lang_counts.setdefault(lang, {"total": 0, "success": 0})
        lang_counts[lang]["total"] += 1
        lang_counts[lang]["success"] += int(ok)

    # Provider breakdown and latency statistics
    prov_counts: dict[str, dict[str, int]] = {}
    latencies: list[float] = []
    for r in cached_rows:
        prov = r.get("provider") or "unknown"
        ok = r.get("status") == "success" and bool(r.get("transcript", "").strip())
        prov_counts.setdefault(prov, {"total": 0, "success": 0})
        prov_counts[prov]["total"] += 1
        if ok:
            prov_counts[prov]["success"] += 1
            lat = r.get("latency_ms")
            if lat:
                try:
                    latencies.append(float(lat))
                except ValueError:
                    pass

    lat_summary: dict[str, Any] = {}
    if latencies:
        latencies.sort()
        lat_summary = {
            "count": len(latencies),
            "median_ms": round(latencies[len(latencies) // 2], 2),
            "p95_ms": round(latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)], 2),
            "mean_ms": round(sum(latencies) / len(latencies), 2),
        }

    return {
        "expected_rows": len(expected),
        "successful_rows": len(expected & successful),
        "failed_or_missing_rows": len(missing),
        "complete": not missing,
        "by_split": split_counts,
        "by_condition": condition_counts,
        "by_language": lang_counts,
        "by_provider": prov_counts,
        "latency_summary": lat_summary,
    }


# ---------------------------------------------------------------------------
# Rejection scoring functions
# ---------------------------------------------------------------------------
# We expose three complementary signals derived directly from the LogReg
# softmax posterior, all of which are mathematically well-defined for this
# classifier.  We do NOT add an "energy score" because scikit-learn LogReg
# logits are not appropriate neural energy scores (they are not free energies
# from a density-estimation perspective).
#
#   max_prob  : Maximum softmax probability (existing threshold signal).
#   margin    : P_top1 - P_top2.  Measures decision-boundary separation.
#   norm_entropy : Normalised Shannon entropy H / log(K), in [0,1].
#                  High entropy → uncertain prediction → good abstain signal.
#                  We report (1 - norm_entropy) as the "confidence" so all
#                  three signals are "higher = more confident".

def compute_rejection_scores(
    model: Pipeline, transcripts: list[str]
) -> dict[str, list[float]]:
    """Return per-transcript rejection scores from a fitted Pipeline.

    Returns a dict with keys:
      - ``max_prob``      : maximum class probability (existing signal)
      - ``margin``        : P_top1 minus P_top2
      - ``norm_entropy``  : 1 - H / log(K)  (higher = more confident)
    """
    probabilities = model.predict_proba(transcripts)   # (N, K)
    n_classes = probabilities.shape[1]
    log_k = math.log(n_classes) if n_classes > 1 else 1.0

    max_prob: list[float] = []
    margin: list[float] = []
    norm_entropy: list[float] = []

    for row in probabilities:
        sorted_row = sorted(row, reverse=True)
        p1 = float(sorted_row[0])
        p2 = float(sorted_row[1]) if len(sorted_row) > 1 else 0.0
        max_prob.append(p1)
        margin.append(p1 - p2)
        # Clamp tiny negatives from floating point
        h = -sum(p * math.log(max(p, 1e-15)) for p in row)
        norm_entropy.append(float(1.0 - h / log_k))

    return {
        "max_prob": max_prob,
        "margin": margin,
        "norm_entropy": norm_entropy,
    }


# ---------------------------------------------------------------------------
# Calibration utilities (fit on calibration split only)
# ---------------------------------------------------------------------------

def _expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 15
) -> float:
    """Compute ECE using equal-width bins.  Lower is better.

    Only call with calibration-split data.
    """
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(confidences)
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        in_bin = [
            i for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == n_bins - 1 and lo <= c <= hi + 1e-9)
        ]
        if not in_bin:
            continue
        acc = sum(correct[i] for i in in_bin) / len(in_bin)
        conf = sum(confidences[i] for i in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(acc - conf)
    return round(ece, 6)


def calibrate_temperature(
    model: Pipeline,
    cal_rows: list[dict[str, str]],
    temps: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
) -> dict[str, Any]:
    """Grid-search temperature scaling on the calibration split.

    Temperature scaling divides logits by T before softmax.  We approximate
    it by raising raw probabilities to (1/T) and renormalising.  This is
    equivalent only when the softmax is applied after scaling — a commonly
    acceptable approximation for post-hoc calibration of this type of model.

    Returns the best T and ECE for uncalibrated and calibrated predictions.
    Only uses calibration data.  Never touches test rows.
    """
    transcripts = [r["transcript"] for r in cal_rows]
    labels = [r["intent"] for r in cal_rows]
    raw_probs = model.predict_proba(transcripts)
    class_list = list(model.classes_)

    def _apply_temp(probs_matrix, T: float):
        scaled = []
        for row in probs_matrix:
            scaled_row = [p ** (1.0 / T) for p in row]
            s = sum(scaled_row) or 1.0
            scaled.append([v / s for v in scaled_row])
        return scaled

    raw_max_prob = [float(max(r)) for r in raw_probs]
    raw_correct = [
        class_list[int(r.argmax())] == lbl
        for r, lbl in zip(raw_probs, labels)
    ]
    ece_uncal = _expected_calibration_error(raw_max_prob, raw_correct)

    best_T = 1.0
    best_ece = ece_uncal
    for T in temps:
        scaled = _apply_temp(raw_probs, T)
        conf = [max(r) for r in scaled]
        ece_t = _expected_calibration_error(conf, raw_correct)
        if ece_t < best_ece:
            best_ece = ece_t
            best_T = T

    return {
        "ece_uncalibrated": ece_uncal,
        "best_temperature": best_T,
        "ece_after_temperature": best_ece,
        "improvement": round(ece_uncal - best_ece, 6),
    }


def calibrate_isotonic(
    model: Pipeline,
    cal_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Fit isotonic regression calibrator on the calibration split.

    Uses sklearn's CalibratedClassifierCV-style approach via
    IsotonicRegression on max_prob vs. binary correct signal.

    Returns ECE before/after and a flag if isotonic actually helped.
    Only uses calibration data.
    """
    from sklearn.isotonic import IsotonicRegression

    transcripts = [r["transcript"] for r in cal_rows]
    labels = [r["intent"] for r in cal_rows]
    raw_probs = model.predict_proba(transcripts)
    class_list = list(model.classes_)

    raw_max_prob = [float(max(r)) for r in raw_probs]
    raw_correct = [
        int(class_list[int(r.argmax())] == lbl)
        for r, lbl in zip(raw_probs, labels)
    ]
    ece_before = _expected_calibration_error(raw_max_prob, [bool(c) for c in raw_correct])

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_max_prob, raw_correct)
    cal_conf = list(ir.predict(raw_max_prob))
    ece_after = _expected_calibration_error(cal_conf, [bool(c) for c in raw_correct])

    return {
        "ece_before_isotonic": ece_before,
        "ece_after_isotonic": round(ece_after, 6),
        "improvement": round(ece_before - ece_after, 6),
        "isotonic_helped": bool(ece_after < ece_before),  # explicit Python bool (not numpy.bool_)
    }


# ---------------------------------------------------------------------------
# Multi-signal threshold selection (calibration data only)
# ---------------------------------------------------------------------------

def choose_threshold_signal(
    model: Pipeline,
    cal_rows: list[dict[str, str]],
    signal: str = "max_prob",
    min_recall: float = 0.90,
) -> dict[str, Any]:
    """Select the reject threshold for a given signal on calibration data.

    ``signal`` must be one of ``max_prob``, ``margin``, or ``norm_entropy``
    (all from :func:`compute_rejection_scores`).

    Returns the threshold and the precision/recall at that operating point.
    Never uses test rows.
    """
    transcripts = [r["transcript"] for r in cal_rows]
    scores_dict = compute_rejection_scores(model, transcripts)
    scores = scores_dict[signal]
    labels_list = list(model.classes_)
    proba = model.predict_proba(transcripts)
    predictions = [labels_list[int(r.argmax())] for r in proba]
    intents = [r["intent"] for r in cal_rows]

    candidates = sorted(set(scores))
    best: dict[str, Any] = {"threshold": 0.0, "recall": 0.0, "precision": 0.0}
    for t in candidates:
        accepted = [pred if sc >= t else "other"
                    for pred, sc in zip(predictions, scores)]
        in_scope_true = sum(i != "other" for i in intents)
        recall = sum(p == i for p, i in zip(accepted, intents) if i != "other") / max(1, in_scope_true)
        accepted_non_other = [p for p, i in zip(accepted, intents) if p != "other"]
        precision = sum(p != "other" for p in accepted_non_other) / max(1, len(accepted_non_other))
        if recall >= min_recall and precision > best["precision"]:
            best = {"threshold": t, "recall": round(recall, 4), "precision": round(precision, 4)}
    return {"signal": signal, **best}


# ---------------------------------------------------------------------------
# Risk–coverage evaluation
# ---------------------------------------------------------------------------

def risk_coverage_curve(
    model: Pipeline,
    rows: list[dict[str, str]],
    signal: str = "max_prob",
) -> dict[str, Any]:
    """Compute the risk–coverage curve for selective classification.

    Sorts rows by descending confidence signal, then sweeps the coverage
    threshold from 0 to 1 and records risk (error rate) at each coverage level.

    Risk = fraction of accepted predictions that are wrong.
    Coverage = fraction of examples accepted (not abstained from).

    Returns arrays suitable for plotting and scalar AURC.
    """
    transcripts = [r["transcript"] for r in rows]
    intents = [r["intent"] for r in rows]

    scores_dict = compute_rejection_scores(model, transcripts)
    scores = scores_dict[signal]
    labels_list = list(model.classes_)
    proba = model.predict_proba(transcripts)
    predictions = [labels_list[int(r.argmax())] for r in proba]
    correct = [int(p == a) for p, a in zip(predictions, intents)]

    # Sort by descending confidence (most confident first)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    sorted_correct = [correct[o] for o in order]

    coverages: list[float] = []
    risks: list[float] = []
    n = len(sorted_correct)
    for k in range(1, n + 1):
        accepted = sorted_correct[:k]
        cov = k / n
        risk = 1.0 - sum(accepted) / k
        coverages.append(round(cov, 6))
        risks.append(round(risk, 6))

    # AURC (trapezoidal) — lower is better (computed on exact dense curve)
    aurc = 0.0
    for i in range(1, len(coverages)):
        aurc += 0.5 * (risks[i] + risks[i - 1]) * (coverages[i] - coverages[i - 1])

    # Sample at most 50 curve points for compact, clean JSON serialization
    step = max(1, n // 50)
    sample_indices = list(range(0, n, step))
    if (n - 1) not in sample_indices:
        sample_indices.append(n - 1)
    sampled_coverages = [coverages[i] for i in sample_indices]
    sampled_risks = [risks[i] for i in sample_indices]

    # Selective accuracy at fixed coverage targets
    def _selective_acc_at(target_cov: float) -> dict[str, Any]:
        k = max(1, round(target_cov * n))
        accepted = sorted_correct[:k]
        return {
            "target_coverage": target_cov,
            "actual_coverage": round(k / n, 4),
            "selective_accuracy": round(sum(accepted) / max(1, len(accepted)), 4),
            "n_accepted": k,
        }

    return {
        "signal": signal,
        "n_rows": n,
        "aurc": round(aurc, 6),
        "coverages": sampled_coverages,
        "risks": sampled_risks,
        "selective_acc_at_80pct": _selective_acc_at(0.80),
        "selective_acc_at_90pct": _selective_acc_at(0.90),
        "selective_acc_at_95pct": _selective_acc_at(0.95),
    }


def false_accept_by_other_type_at_coverage(
    model: Pipeline,
    rows: list[dict[str, str]],
    signal: str = "max_prob",
    coverage_targets: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[str, Any]:
    """False-accept rates broken down by other_type at fixed coverage levels.

    For each coverage target, accepts the top-k% most confident rows and
    reports how many ground-truth ``other`` rows were incorrectly accepted
    (false-accepted) per other_type.
    """
    transcripts = [r["transcript"] for r in rows]
    scores = compute_rejection_scores(model, transcripts)[signal]
    labels_list = list(model.classes_)
    proba = model.predict_proba(transcripts)
    predictions = [labels_list[int(r.argmax())] for r in proba]

    n = len(rows)
    order = sorted(range(n), key=lambda i: -scores[i])

    results: dict[str, Any] = {}
    for target in coverage_targets:
        k = max(1, round(target * n))
        accepted_indices = set(order[:k])
        other_types = sorted({r.get("other_type", "") for r in rows if r["intent"] == "other"})
        fa_by_type: dict[str, Any] = {}
        for ot in other_types:
            ot_indices = [i for i, r in enumerate(rows)
                          if r["intent"] == "other" and r.get("other_type", "") == ot]
            if not ot_indices:
                continue
            fa = sum(1 for i in ot_indices
                     if i in accepted_indices and predictions[i] != "other")
            fa_by_type[ot] = {
                "total": len(ot_indices),
                "false_accepted": fa,
                "false_accept_rate": round(fa / len(ot_indices), 4),
            }
        results[str(target)] = {
            "coverage_target": target,
            "n_accepted": k,
            "false_accept_by_other_type": fa_by_type,
        }
    return results


# ---------------------------------------------------------------------------
# Classifier training, threshold selection, evaluation
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], seed: int = 13) -> list[float]:

    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    samples = []
    for _ in range(1000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    return [round(samples[25], 4), round(samples[975], 4)]


def choose_threshold(model: Pipeline, rows: list[dict[str, str]]) -> float:
    scores = model.predict_proba([row["transcript"] for row in rows])
    labels = list(model.classes_)
    confidences = [float(max(row)) for row in scores]
    predictions = [labels[int(row.argmax())] for row in scores]
    candidates = sorted(set(confidences))
    best = (0.0, 0.0)
    for threshold in candidates:
        accepted = [pred if conf >= threshold else "other"
                    for pred, conf in zip(predictions, confidences)]
        recall = sum(p == r["intent"] for p, r in zip(accepted, rows)
                     if r["intent"] != "other") / max(
                         1, sum(r["intent"] != "other" for r in rows)
                     )
        accepted_non_other = [
            p for p, r in zip(accepted, rows) if r["intent"] != "other"
        ]
        precision = sum(p != "other" for p in accepted_non_other) / max(
            1, len(accepted_non_other)
        )
        if recall >= 0.90 and precision > best[1]:
            best = (threshold, precision)
    return best[0] if best[0] else 0.0


def evaluate_group(
    model: Pipeline, rows: list[dict[str, str]], threshold: float
) -> dict[str, Any]:
    probabilities = model.predict_proba([row["transcript"] for row in rows])
    labels = list(model.classes_)
    predictions = []
    for scores in probabilities:
        index = int(scores.argmax())
        predictions.append(labels[index] if float(scores[index]) >= threshold else "other")
    actual = [row["intent"] for row in rows]
    in_scope = [i != "other" for i in actual]
    accepted = [p != "other" for p in predictions]
    false_accept = [
        int(pred != "other") for pred, truth in zip(predictions, actual)
        if truth == "other"
    ]
    per_other: dict[str, float] = {}
    for other_type in sorted({row.get("other_type", "") for row in rows if row["intent"] == "other"}):
        values = [
            int(pred != "other") for pred, row in zip(predictions, rows)
            if row["intent"] == "other" and row.get("other_type", "") == other_type
        ]
        if values:
            per_other[other_type] = round(sum(values) / len(values), 4)

    # Risk–coverage curves for all three reject signals
    rc_signals = {
        sig: risk_coverage_curve(model, rows, signal=sig)
        for sig in ("max_prob", "margin", "norm_entropy")
    }
    fa_at_cov = false_accept_by_other_type_at_coverage(model, rows, signal="max_prob")

    # Per-language breakdown if language_code present
    languages = sorted({r.get("language_code", "") for r in rows if r.get("language_code")})
    by_lang: dict[str, Any] = {}
    if languages:
        for lang in languages:
            lang_rows = [r for r in rows if r.get("language_code") == lang]
            lang_actual = [r["intent"] for r in lang_rows]
            lang_preds = [p for p, r in zip(predictions, rows) if r.get("language_code") == lang]
            lang_acc = round(float(accuracy_score(lang_actual, lang_preds)), 4) if lang_actual else 0.0
            lang_cov = round(sum(p != "other" for p in lang_preds) / max(1, len(lang_rows)), 4)
            by_lang[lang] = {
                "rows": len(lang_rows),
                "accuracy": lang_acc,
                "coverage": lang_cov,
                "abstention_rate": round(1.0 - lang_cov, 4),
            }

    return {
        "rows": len(rows),
        "accuracy": round(float(accuracy_score(actual, predictions)), 4),
        "in_scope_accuracy": round(sum(
            p == a for p, a, flag in zip(predictions, actual, in_scope) if flag
        ) / max(1, sum(in_scope)), 4),
        "coverage": round(sum(accepted) / max(1, len(rows)), 4),
        "abstention_rate": round(1 - sum(accepted) / max(1, len(rows)), 4),
        "false_accept_rate_other": round(sum(false_accept) / max(1, len(false_accept)), 4),
        "false_accept_rate_by_other_type": per_other,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(actual, predictions, labels=labels).tolist(),
        "accuracy_ci95": bootstrap_ci([float(p == a) for p, a in zip(predictions, actual)]),
        "risk_coverage": rc_signals,
        "false_accept_by_coverage": fa_at_cov,
        "by_language": by_lang,
    }


def evaluate(transcripts: Path, output: Path, provider: str) -> dict[str, Any]:
    rows = [row for row in read_csv(transcripts)
            if row.get("status", "success") == "success" and row["transcript"].strip()]
    train = [row for row in rows if row["split"] == "train" and row.get("condition") != "unseen_combined"]
    calibration = [row for row in rows if row["split"] == "dev"]
    model = Pipeline([
        ("features", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                     min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    if not train or not calibration:
        raise ValueError("Training and calibration transcripts are required")
    model.fit([row["transcript"] for row in train], [row["intent"] for row in train])

    # Threshold selection uses ONLY calibration (dev) rows — never test rows
    threshold = choose_threshold(model, calibration)

    # Calibration analysis on calibration (dev) split only — never test rows
    temp_cal = calibrate_temperature(model, calibration)
    iso_cal = calibrate_isotonic(model, calibration)
    signal_thresholds = {
        sig: choose_threshold_signal(model, calibration, signal=sig)
        for sig in ("max_prob", "margin", "norm_entropy")
    }

    groups = {
        "clean_test": [row for row in rows if row["split"] == "test" and row.get("condition") == "clean"],
        "robust_test": [row for row in rows if row["split"] == "test" and row.get("condition") != "clean"],
    }
    report: dict[str, Any] = {
        "model": "character_tfidf_3_5_logistic_regression",
        "asr_provider": provider,
        "asr_model": (
            "ai4bharat/indic-conformer-600m-multilingual" if provider == "indicconformer"
            else "openai/whisper-base.en" if provider == "whisper"
            else "Hybrid (AI4Bharat IndicConformer-600M + OpenAI Whisper-base.en)" if provider == "hybrid"
            else "Sarvam Saaras saarika:v2.5"
        ),
        "transcript_rows": len(rows),
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "calibration_threshold": threshold,
        "calibration_analysis": {
            "temperature_scaling": temp_cal,
            "isotonic_regression": iso_cal,
            "signal_thresholds": signal_thresholds,
        },
        "conditions": {name: evaluate_group(model, group, threshold)
                       for name, group in groups.items() if group},
    }
    for condition in sorted({row.get("condition", "") for row in rows if row["split"] == "test"}):
        group = [row for row in rows if row["split"] == "test" and row.get("condition") == condition]
        if group:
            report.setdefault("per_condition", {})[condition] = evaluate_group(model, group, threshold)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]

    parser.add_argument("--provider", choices=["saaras", "indicconformer", "whisper", "hybrid"],
                        default="saaras",
                        help="ASR backend to use (default: saaras).")
    parser.add_argument("--manifest", type=Path,
                        default=root / "dataset/manifests/evaluation/synthetic_asr_pool.csv")
    parser.add_argument("--audio-root", type=Path,
                        default=root / "dataset/synthetic/v0.3_robustness")

    # Provider-specific cache paths (kept separate — never merged)
    parser.add_argument(
        "--transcripts", type=Path, default=None,
        help="Path to transcript cache CSV. Defaults to provider-named file in manifests/evaluation/.",
    )
    parser.add_argument(
        "--errors", type=Path, default=None,
        help="Path to error log CSV. Defaults to provider-named file in reports/.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to baseline report JSON. Defaults to provider-named file in reports/.",
    )

    parser.add_argument("--transcribe", action="store_true",
                        help="Run ASR transcription (skip if only evaluating).")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-new", type=int,
        help="Stop after this many uncached/failed rows in one resumable batch.",
    )

    # Local model devices
    parser.add_argument("--ic-device", default="cuda",
                        help="Torch device for IndicConformer ('cuda' or 'cpu').")
    parser.add_argument("--whisper-device", default="cuda",
                        help="Torch device for Whisper ('cuda' or 'cpu').")
    parser.add_argument(
        "--ic-endpoint", default=None,
        help="(Reserved) URL for a self-hosted IndicConformer HTTP wrapper.",
    )

    args = parser.parse_args()
    dotenv = load_dotenv(root / ".env")

    # Resolve default paths based on provider
    provider = args.provider
    if args.transcripts is None:
        args.transcripts = root / f"dataset/manifests/evaluation/{provider}_transcripts.csv"
    if args.errors is None:
        args.errors = root / f"dataset/reports/{provider}_transcription_errors.csv"
    if args.output is None:
        args.output = root / f"dataset/reports/{provider}_baseline.json"

    # ------------------------------------------------------------------
    # Transcription phase
    # ------------------------------------------------------------------
    if args.transcribe:
        if provider == "saaras":
            keys = [
                value for name, value in sorted({**dotenv, **os.environ}.items())
                if name == "SARVAM_API_KEY" or name.startswith("SARVAM_API_KEY_")
                if value
            ]
            if not keys:
                parser.error("No SARVAM_API_KEY configuration found")
            keys = list(reversed(keys))
            summary = transcribe_manifest(
                args.manifest, args.audio_root, args.transcripts, args.errors,
                provider=provider,
                api_keys=keys,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                max_new=args.max_new,
            )
        elif provider == "indicconformer":
            # Resolve HF token (HF_TOKEN or HF_TOKEN_2) for gated model access.
            hf_candidates = [
                value for name, value in sorted({**dotenv, **os.environ}.items())
                if name.startswith("HF_TOKEN") and value
            ]
            globals()["_IC_HF_TOKEN"] = hf_candidates[0] if hf_candidates else None
            if globals()["_IC_HF_TOKEN"]:
                print("[IndicConformer] HF token found — gated model access enabled.")
            else:
                print("[IndicConformer] WARNING: No HF_TOKEN in .env — gated model may be inaccessible.")
            summary = transcribe_manifest(
                args.manifest, args.audio_root, args.transcripts, args.errors,
                provider=provider,
                ic_device=args.ic_device,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                max_new=args.max_new,
            )
        elif provider == "whisper":
            summary = transcribe_manifest(
                args.manifest, args.audio_root, args.transcripts, args.errors,
                provider=provider,
                whisper_device=args.whisper_device,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                max_new=args.max_new,
            )
        elif provider == "hybrid":
            hf_candidates = [
                value for name, value in sorted({**dotenv, **os.environ}.items())
                if name.startswith("HF_TOKEN") and value
            ]
            globals()["_IC_HF_TOKEN"] = hf_candidates[0] if hf_candidates else None
            summary = transcribe_manifest(
                args.manifest, args.audio_root, args.transcripts, args.errors,
                provider=provider,
                ic_device=args.ic_device,
                whisper_device=args.whisper_device,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                max_new=args.max_new,
            )
        else:
            parser.error(f"Unknown provider: {provider}")

        print(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------
    # Coverage check (always run after transcription or as standalone)
    # ------------------------------------------------------------------
    if not args.transcripts.exists():
        parser.error(
            f"Transcript cache not found at {args.transcripts}; run with --transcribe first"
        )

    cov = coverage_report(args.manifest, args.transcripts)
    if not cov["complete"]:
        blocked = {
            "status": "blocked_incomplete_transcripts",
            "provider": provider,
            "coverage": cov,
            "errors": str(args.errors),
            "note": (
                "Final metrics are withheld until every frozen ASR-pool row has a "
                "successful cached transcript."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(blocked, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(blocked, indent=2))
        raise SystemExit(2)

    evaluate(args.transcripts, args.output, provider)


if __name__ == "__main__":
    main()
