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
- Evaluation manifests are frozen under `dataset/manifests/evaluation/`.
- The Saaras plus character n-gram logistic-regression baseline is implemented
  in `scripts/run_saaras_baseline.py`; actual metrics remain pending ASR
  transcript generation.
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

Do not commit `.env`, API keys, or large generated audio unless project policy
explicitly changes.
