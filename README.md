# VoxIntel Indic

VoxIntel Indic is a reproducible research project for noisy, code-mixed Indic
speech-to-intent classification. It compares a Saaras ASR-to-text classifier
with a direct audio-to-intent model, using an explicit `other` class and reject
gate.

## Repository layout

- [`dataset/`](./dataset/): source text, clean/noisy synthetic audio,
  IndicVoices data, manifests, reports, documentation, and assignment plan.
- [`scripts/`](./scripts/): reproducible data-generation, filtering,
  degradation, audit, and validation tools.
- [`reports/`](./reports/): reserved for future research reports.
- [`notebooks/`](./notebooks/): reserved for reproducible analysis notebooks.
- [`CLAUDE.md`](./CLAUDE.md): project instructions and assignment context.
- [`requirements.txt`](./requirements.txt): Python package requirements.

The root intentionally contains only repository configuration and setup
documentation. `.env` is local-only and ignored by Git.

## Evaluation boundary

This project uses synthetic speech and curated public data only. It does not
recruit speakers or require a real-speaker test set. Synthetic and public-data
results must remain separate and cannot establish real-world caller
performance. Direct audio-to-intent superiority remains an unproven hypothesis.

## Current state

See [`dataset/README.md`](./dataset/README.md) for measured counts and
validation details. In brief:

- v0.2 source: 100 rows, preserved unchanged.
- v0.3 source: 2,400 candidate rows with cluster-disjoint intended splits.
- v0.2 synthetic audio: 198 validated 8 kHz mono WAV files.
- IndicVoices source subset: 1,915 FLAC records and original annotations,
  preserved as an audit source.
- IndicVoices curated candidate subset: 346 records retained by conservative
  intent-specific evidence; all remain unreviewed and excluded from trusted
  metrics.
- v0.3 clean audio: 2,400/2,400 files generated and validated across seven
  configured key slots.
- v0.3 noisy audio: 4,800 deterministic variants generated and validated at
  10 dB and 20 dB SNR.
- v0.3 robustness suite: 12,390/12,390 deterministic variants generated and
  validated across six documented conditions; `unseen_combined` is test-only
  in the regenerated manifest.
- Evaluation manifests are frozen under `dataset/manifests/evaluation/` (14,790 pool rows).
- Baseline ASR & Classifier Status:
  - **Saaras (Sarvam ASR)**: Cloud API quota exhausted with HTTP 402 on all configured key slots (`dataset/reports/saaras_baseline.json`). 2,860/14,790 transcripts preserved in `saaras_transcripts.csv`; 11,930 remaining rows blocked.
  - **Local Multi-Model ASR (`hybrid`)**: Fully completed across all 14,790 pool rows (100.0% coverage) in `dataset/manifests/evaluation/hybrid_transcripts.csv` without fabricating transcripts:
    - **IndicConformer-600M** (`ai4bharat/indic-conformer-600m-multilingual`, MIT): 11,525 rows across all 7 Indic languages (`hi-IN`, `ta-IN`, `bn-IN`, `te-IN`, `mr-IN`, `gu-IN`, `kn-IN`).
    - **Whisper-base.en** (`openai/whisper-base.en`, MIT): 3,265 rows for English (`en-IN`).
    - **Latency**: Median 849.5 ms, Mean 1,167.2 ms, p95 2,286.4 ms on NVIDIA RTX 3050 GPU.
  - **Classifier & Reject Gate** (`dataset/reports/hybrid_baseline_v2.json`):
    - Character 3–5-gram TF-IDF (`sublinear_tf=True`, `min_df=2`) + Balanced Logistic Regression trained on 9,720 synthetic training rows.
    - **Reject threshold** $T = 0.1234$ selected strictly on 2,340 calibration (`dev`) rows at $\ge 90\%$ recall (max-prob signal). Never uses test data.
    - **Calibration analysis** (calibration split only):
      - ECE (uncalibrated): 0.0985; Temperature scaling (T=0.5): ECE → 0.0062 (Δ=0.0923); Isotonic regression: ECE → 0.000 (Δ=0.0985).
      - All three reject signals (max_prob / margin / norm_entropy) achieve 99.94% recall at precision 1.0 at their respective calibration-selected thresholds.
  - **Measured Evaluation Metrics** (`dataset/reports/hybrid_baseline_v2.json`):
    - **Clean Final Test** (390 rows): Accuracy = 1.0000, In-Scope Accuracy = 1.0000, 95% CI: [1.0, 1.0].
    - **Robust Final Test** (2,340 rows): Accuracy = 0.9996, In-Scope Accuracy = 0.9994, 95% CI: [0.9987, 1.0].
    - **Unseen Combined Condition** (`unseen_combined`, strictly test-only, 390 rows): Accuracy = 0.9974, In-Scope Accuracy = 0.9967, 95% CI: [0.9923, 1.0].
    - **Rejection & Other False Accept Rate**: Coverage = 0.7692, Abstention = 0.2308, False Accept Rate on `other` = 0.0000 across all five `other_type` categories.
    - **Risk–Coverage (clean test)**: AURC ≈ 0.000 for all three signals. Selective accuracy: 80% cov → 1.0000, 90% cov → 1.0000, 95% cov → 1.0000.
    - **False-accept by `other_type` at fixed coverage**: 0.0000 for all five categories at 80%, 90%, and 95% coverage targets.
    - **Per-language (clean test)**: bn-IN 97.5% acc/100% cov; en-IN 100% acc/47.1% cov; gu-IN, kn-IN, mr-IN, ta-IN, te-IN all 100% acc/100% cov; hi-IN 100% acc/47.1% cov.
- The frozen public trusted manifest has zero rows until IndicVoices labels
  receive explicit human review; unreviewed keyword candidates are not metrics.
- Public candidates currently cover `balance`, `pay_bill`, `emi_due`,
  `recent_transactions`, `report_fraud`, `update_details`, `transfer_to_agent`,
  and explicit `other`. There are no defensible public candidates yet for
  `block_card`, `complaint_status`, or `branch_timing`.

## Reproduce checks

The scripts resolve their default paths from the repository root, so commands
are reproducible even when launched from another directory:

```powershell
python scripts\generate_utterances_v03.py
python scripts\validate_dataset.py

python scripts\validate_dataset.py `
  --source dataset\source\utterances_v0.2.csv `
  --manifest dataset\synthetic\v0.2\manifest.csv `
  --manifest-root dataset\synthetic\v0.2 `
  --report dataset\reports\v0.2_validation.json

conda run -n torch26 python scripts\audit_indicvoices.py `
  --manifest dataset\indicvoices\filtered\manifest.csv `
  --report dataset\reports\indicvoices_audit.json
```

## Synthetic audio generation

Set `SARVAM_API_KEY` and any numbered `SARVAM_API_KEY_N` variables only in the
local root `.env`. The generator discovers numbered keys automatically, but
does not treat them as independent quotas. Run a small smoke batch
before any larger operation:

```powershell
python scripts\generate_dataset.py `
  --csv dataset\source\utterances_v0.3.csv `
  --output dataset\synthetic\v0.3_smoke `
  --speakers shubh simran aditya `
  --variants 1 `
  --max-rows 8
```

Then create deterministic degraded variants:

```powershell
python scripts\degrade_audio.py `
  --manifest dataset\synthetic\v0.3_smoke\manifest.csv `
  --output dataset\synthetic\v0.3_smoke_noisy `
  --snr-db 10 20

python scripts\degrade_audio.py `
  --manifest dataset\synthetic\v0.3\manifest.csv `
  --output dataset\synthetic\v0.3_robustness `
  --suite robustness

python scripts\validate_dataset.py `
  --source dataset\source\utterances_v0.3.csv `
  --manifest dataset\synthetic\v0.3_robustness\manifest.csv `
  --manifest-root dataset\synthetic\v0.3_robustness `
  --report dataset\reports\v0.3_robustness_validation.json

python scripts\freeze_evaluation_manifests.py

python scripts\spot_check_synthetic.py

python scripts\curate_indicvoices.py
```

## ASR baseline & testing

Run the deterministic IndicConformer smoke test, resumable transcription, or unit tests:

```powershell
# 1. Deterministic smoke test across languages & conditions (~9 files)
conda run -n torch26 python scripts\smoke_test_indicconformer.py

# 2. Resumable hybrid transcription (IndicConformer + Whisper) in bounded batches
conda run -n torch26 python scripts\run_saaras_baseline.py `
  --provider hybrid `
  --transcribe `
  --max-new 500

# 3. Fit classifier, run calibration analysis & risk-coverage evaluation
conda run -n torch26 python scripts\run_saaras_baseline.py `
  --provider hybrid `
  --output dataset\reports\hybrid_baseline_v2.json

# 4. Run test suite and dataset validator
conda run -n torch26 python -m pytest tests -q
conda run -n torch26 python scripts\validate_dataset.py
```

Do not commit `.env`, API keys, or large generated audio unless project policy
explicitly changes.
