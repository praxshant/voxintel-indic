# VoxIntel Indic

VoxIntel Indic is a reproducible speech-to-intent research pipeline for noisy,
code-mixed Indic telephone speech. It compares a Saaras ASR-to-text classifier
against a direct audio-to-intent model, with an explicit `other` class and
reject gate.

## Evaluation boundary

This repository uses synthetic speech and curated public data only. It does not
recruit speakers or require a real-speaker test set. Results must be reported
separately for synthetic and public data and cannot establish performance on
real-world callers. Direct audio-to-intent superiority remains an unproven
hypothesis.

## Current data

- `utterances_v0.2.csv`: preserved 100-row pilot source.
- `utterances_v0.3.csv`: 2,400-row candidate source with 1,800 in-scope and
  600 `other` rows; source validation passes.
- `dataset_v0.2/`: 198 validated Bulbul v3 WAV files.
- `indicvoices_filtered/`: 1,915 IndicVoices FLAC files with weak,
  keyword-derived assignment labels. These labels are not verified ground
  truth, and the CSV manifest explicitly records `label_provenance` and
  `review_status`. English is not covered by this subset.
- v0.3 bulk audio is intentionally not committed or generated until a small
  TTS smoke batch is reviewed.

## Reproduce the checks

```powershell
& 'C:\Users\ACER\anaconda3\python.exe' scripts\generate_utterances_v03.py `
  --output utterances_v0.3.csv

& 'C:\Users\ACER\anaconda3\python.exe' scripts\validate_dataset.py `
  --source utterances_v0.3.csv `
  --report v0.3_source_validation.json

& 'C:\Users\ACER\anaconda3\python.exe' scripts\validate_dataset.py `
  --source utterances_v0.2.csv `
  --manifest dataset_v0.2\manifest.csv `
  --manifest-root dataset_v0.2 `
  --report v0.2_validation.json
```

## Generate synthetic audio

Set `SARVAM_API_KEY` in the local `.env` file. Never commit `.env`, generated
audio, or API credentials. Run a small smoke batch first:

```powershell
& 'C:\Users\ACER\anaconda3\python.exe' scripts\generate_dataset.py `
  --csv utterances_v0.3.csv `
  --output dataset_v0.3_smoke `
  --speakers shubh simran aditya `
  --variants 1
```

The generator is resumable, writes checksums and provenance manifests, and
assigns v0.3 splits from the precomputed paraphrase-cluster split.

Audit the public manifest without changing audio:

```powershell
& 'C:\Users\ACER\anaconda3\envs\torch26\python.exe' scripts\audit_indicvoices.py `
  --manifest indicvoices_filtered\manifest.csv `
  --report indicvoices_audit.json `
  --annotate
```

## Create noisy variants

After clean WAV generation:

```powershell
& 'C:\Users\ACER\anaconda3\python.exe' scripts\degrade_audio.py `
  --manifest dataset_v0.3_smoke\manifest.csv `
  --output dataset_v0.3_smoke_noisy `
  --snr-db 10 20
```

Each degraded record stores its source path, SNR, clipping, reverb, dropout,
and deterministic seed.

## IndicVoices

IndicVoices is gated on Hugging Face and is downloaded with the `torch26`
environment using `HF_TOKEN`. The local filter preserves original annotations
and audio provenance, then adds clearly marked weak assignment labels. See
[`indicvoices_filtered/README.md`](./indicvoices_filtered/README.md).

## Documentation

- [`CLAUDE.md`](./CLAUDE.md): assignment interpretation and project rules.
- [`DATASET_CARD.md`](./DATASET_CARD.md): provenance, intended use, and limits.
- [`DATA_READINESS_REPORT.md`](./DATA_READINESS_REPORT.md): current measured
  counts and validation status.
