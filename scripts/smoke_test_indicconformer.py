"""
Phase 2: IndicConformer smoke test.

Selects a deterministic set of rows from the ASR pool:
- One clean file per language (8 languages)
- 1 extra noisy row (gaussian_snr10)
- 1 extra 'other' intent row
Total: ~10-12 rows.

Transcribes them using the IndicConformer provider and writes results to
indicconformer_transcripts.csv for inspection.
"""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# ── repo root ──────────────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic")
sys.path.insert(0, str(ROOT))

# ── load HF token from .env ────────────────────────────────────────────────
env_path = ROOT / ".env"
hf_token = None
for line in env_path.read_text(encoding="utf-8-sig").splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        if k.strip().startswith("HF_TOKEN"):
            hf_token = v.strip().strip("\"'")
            break

if not hf_token:
    print("ERROR: No HF_TOKEN found in .env")
    sys.exit(1)

os.environ["HF_TOKEN"] = hf_token
print("HF_TOKEN loaded (not shown)")

# ── set module-level token before importing provider ───────────────────────
import scripts.run_saaras_baseline as baseline
baseline._IC_HF_TOKEN = hf_token

# ── load ASR pool ──────────────────────────────────────────────────────────
POOL = ROOT / "dataset/manifests/evaluation/synthetic_asr_pool.csv"
CACHE = ROOT / "dataset/manifests/evaluation/indicconformer_transcripts.csv"
ERRORS = ROOT / "dataset/reports/indicconformer_transcription_errors.csv"
AUDIO_ROOT = ROOT / "dataset/synthetic/v0.3_robustness"

with POOL.open(newline="", encoding="utf-8-sig") as f:
    all_rows = list(csv.DictReader(f))

# ── select deterministic smoke rows ────────────────────────────────────────
# The model has no English decoder; keep en-IN explicitly unsupported.
LANGS = ["bn-IN", "gu-IN", "hi-IN", "kn-IN", "mr-IN", "ta-IN", "te-IN"]
seen_langs = set()
smoke = []

# Sort rows for determinism: by id then condition
sorted_rows = sorted(all_rows, key=lambda r: (r["id"], r.get("condition", "")))

# 1. One clean row per language
for r in sorted_rows:
    lang = r.get("language_code", "")
    if lang in LANGS and lang not in seen_langs and r.get("condition", "") == "clean":
        smoke.append(r)
        seen_langs.add(lang)
    if len(seen_langs) == len(LANGS):
        break

# 2. One noisy row (gaussian_snr10, any lang)
for r in sorted_rows:
    if r.get("condition") == "gaussian_snr10" and r not in smoke:
        smoke.append(r)
        break

# 3. One 'other' intent row (clean)
for r in sorted_rows:
    if (
        r.get("intent") == "other"
        and r.get("condition") == "clean"
        and r.get("language_code") in baseline.INDICCONFORMER_SUPPORTED_LANGUAGES
        and r not in smoke
    ):
        smoke.append(r)
        break

print(f"\nSmoke set: {len(smoke)} rows")
for r in smoke:
    print(f"  {r['id']:20s}  lang={r.get('language_code','?'):6s}  "
          f"cond={r.get('condition','?'):20s}  intent={r['intent']}")

# ── verify audio files exist ───────────────────────────────────────────────
print("\nVerifying audio paths...")
missing = []
for r in smoke:
    raw = r.get("audio_root", "").strip()
    ar = (Path(raw) if Path(raw).is_absolute() else ROOT / raw) if raw else AUDIO_ROOT
    p = ar / r["audio_path"]
    if not p.exists():
        missing.append(str(p))
    else:
        print(f"  OK  {p.name}")
if missing:
    print(f"MISSING FILES: {missing}")
    sys.exit(1)

# ── run IndicConformer on smoke set ────────────────────────────────────────
print("\nLoading IndicConformer model (first run downloads ~2.4 GB)...")
t0 = time.perf_counter()
model = baseline._load_indicconformer(device="cuda")
load_time = time.perf_counter() - t0
print(f"Model loaded in {load_time:.1f}s")

print("\nTranscribing smoke set...")
results = []
for r in smoke:
    raw = r.get("audio_root", "").strip()
    ar = (Path(raw) if Path(raw).is_absolute() else ROOT / raw) if raw else AUDIO_ROOT
    audio_path = ar / r["audio_path"]
    lang_code = r.get("language_code", "")
    bare_lang = baseline.INDICCONFORMER_LANG_MAP.get(lang_code)
    key = baseline.record_key(r)

    try:
        if bare_lang is None or lang_code not in baseline.INDICCONFORMER_SUPPORTED_LANGUAGES:
            raise ValueError(
                f"IndicConformer does not support manifest language {lang_code!r}"
            )
        transcript, meta = baseline._indicconformer_transcribe_once(
            audio_path, bare_lang, device="cuda"
        )
        status = "success"
        error = ""
        print(f"  OK {r['id']:20s} [{lang_code:6s}] -> {transcript[:60]!r}  ({meta['latency_ms']:.0f}ms)")
    except Exception as exc:
        transcript = ""
        status = "failed"
        error = str(exc)
        meta = {"latency_ms": 0}
        print(f"  FAIL {r['id']:20s} [{lang_code:6s}] FAILED: {exc}")

    results.append({
        "record_key": key,
        "id": r["id"],
        "audio_path": r["audio_path"],
        "split": r["split"],
        "intent": r["intent"],
        "other_type": r.get("other_type", ""),
        "language_code": lang_code,
        "condition": r.get("condition", ""),
        "transcript": transcript,
        "status": status,
        "provider": "indicconformer",
        "error": error,
        "attempts": "1",
        "latency_ms": str(meta.get("latency_ms", "")),
    })

# ── save smoke results ─────────────────────────────────────────────────────
smoke_out = ROOT / "dataset/reports/indicconformer_smoke_results.csv"
baseline.write_csv(smoke_out, results)
print(f"\nSmoke results saved → {smoke_out}")

# ── summary ───────────────────────────────────────────────────────────────
success = sum(1 for r in results if r["status"] == "success")
failed  = sum(1 for r in results if r["status"] == "failed")
latencies = [float(r["latency_ms"]) for r in results if r["status"] == "success" and r["latency_ms"]]
avg_lat = sum(latencies) / len(latencies) if latencies else 0

print(f"\n{'='*60}")
print(f"SMOKE TEST SUMMARY")
print(f"{'='*60}")
print(f"  Total rows:   {len(results)}")
print(f"  Successful:   {success}")
print(f"  Failed:       {failed}")
print(f"  Avg latency:  {avg_lat:.0f}ms")
print(f"  Model load:   {load_time:.1f}s")
print(f"{'='*60}")

if failed > 0:
    print("\nFailed rows:")
    for r in results:
        if r["status"] == "failed":
            print(f"  {r['id']} [{r['language_code']}]: {r['error'][:120]}")
    sys.exit(1)
else:
    print("\nSMOKE TEST PASSED — proceeding to full transcription is safe.")
