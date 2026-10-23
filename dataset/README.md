# VoxIntel Indic Dataset

This directory contains the versioned source text, synthetic audio, filtered
IndicVoices public data, manifests, metadata, validation reports, and the
assignment plan. Counts below are measured from the current files and reports.

## Layout

- `source/`: `utterances_v0.2.csv` and `utterances_v0.3.csv`.
- `synthetic/v0.2/`: clean Bulbul v3 WAV files and manifests.
- `synthetic/v0.3/`: complete clean Bulbul v3 WAV files and manifest.
- `synthetic/v0.3_noisy/`: validated baseline Gaussian-noise variants.
- `synthetic/v0.3_robustness/`: expanded controlled robustness suite.
- `synthetic/noisy_smoke/`: small deterministic degradation smoke test.
- `indicvoices/filtered/`: filtered IndicVoices FLAC files and annotations.
- `reports/`: validation and public-label audit JSON reports.
- `manifests/evaluation/`: frozen training, calibration, final-test, and ASR
  input manifests with hashes in `FREEZE.json`.
- `docs/`: dataset card and readiness report.
- `plan/`: assignment plan PDF.

## Source text

| File | Rows | Status |
|---|---:|---|
| `source/utterances_v0.2.csv` | 100 | Original pilot, preserved unchanged |
| `source/utterances_v0.3.csv` | 2,400 | Deterministic candidate source |

The v0.3 source has 1,800 in-scope rows and 600 `other` rows. Its split totals
are 1,620 train, 390 dev, and 390 test. The source validator reports no
duplicate normalized text or paraphrase-cluster leakage. Text is marked as
pending human spot-check; it is not human-verified language data.

## Synthetic audio

`synthetic/v0.2/` contains 198 clean WAV files generated from 98 TTS rows plus
two local silence/noise rows:

- Sarvam Bulbul v3
- voices: `shubh` for train/dev and `simran` for test
- mono, 16-bit PCM, 8 kHz
- splits: 93 train, 7 dev, 98 test
- checksums and durations are in `manifest.csv` and `manifest.jsonl`

`synthetic/v0.3/` contains all 2,400 clean WAV files: 1,620 train, 390 dev,
and 390 test. The complete manifest passed validation. Earlier bulk attempts
encountered HTTP 429 rate limiting and HTTP 402 payment/quota responses; the
additional configured key slots allowed the resumable run to finish.

`synthetic/noisy_smoke/` contains four degraded files derived from two v0.2
manifest rows at 10 dB and 20 dB SNR. The smoke manifest passed path, checksum, WAV-format, and voice-leakage
validation; the result is recorded at
`reports/noisy_smoke_validation.json` records the smoke test. The complete
`synthetic/v0.3_noisy/` dataset contains 4,800 WAV files: two deterministic
variants (10 dB and 20 dB SNR) for each of the 2,400 clean files. Its
manifest and validation report are stored alongside the clean dataset.

The degradation script records SNR, condition, clipping, reverb, dropout,
source path, and random seed in each manifest. The baseline `v0.3_noisy`
dataset contains 4,800 files: two variants per clean file at 10 dB and 20 dB
Gaussian SNR.

The expanded `synthetic/v0.3_robustness/` suite contains 12,390 files. The
five train/dev-eligible conditions cover all 2,400 clean sources; the
`unseen_combined` condition is generated only for the 390 test sources:

| Condition | Files | Role |
|---|---:|---|
| `gaussian_snr10` | 2,400 | controlled baseline |
| `gaussian_snr20` | 2,400 | controlled baseline |
| `traffic_snr15` | 2,400 | synthetic colored-noise proxy |
| `office_snr15` | 2,400 | synthetic colored-noise proxy |
| `telephone_channel` | 2,400 | approximate band-limit/quantization |
| `unseen_combined` | 390 | held-out combined-condition candidate |

All six conditions passed the manifest and audio validator. The environmental
conditions are deterministic colored-noise proxies, not recordings of traffic,
office, fans, or competing speakers. The telephone condition is an
approximation, not a standards-validated codec simulation. These data test
controlled robustness only and do not establish performance on real calls.
The frozen evaluation manifests are in `manifests/evaluation/`: 8,100 training
rows, 1,950 calibration rows, and 2,340 final-test rows. The final-test
manifest is the only one containing `unseen_combined`. The public trusted
manifest currently contains zero rows because the 346 curated candidates and
the remaining source labels are still unreviewed.

## Baseline ASR and transcription status

The synthetic ASR pool contains 14,790 rows (`dataset/manifests/evaluation/synthetic_asr_pool.csv`).
ASR transcription is completed across the full pool via `scripts/run_saaras_baseline.py`:

| Provider | Backend / Model | License | Audio Rate | Cache Path | Success Rows | Status |
|---|---|---|---|---|---:|---|
| `saaras` | Sarvam Saaras API (`saarika:v2.5`) | Proprietary API | 8 kHz | `dataset/manifests/evaluation/saaras_transcripts.csv` | 2,860 / 14,790 | Quota exhausted (HTTP 402) |
| `indicconformer` | AI4Bharat IndicConformer-600M | MIT | 8 kHz → 16 kHz (resampled) | `dataset/manifests/evaluation/indicconformer_transcripts.csv` | 11,525 / 11,525 | 100% of Indic languages completed |
| `whisper` | OpenAI Whisper-base.en | MIT | 8 kHz → 16 kHz (resampled) | `dataset/manifests/evaluation/whisper_transcripts.csv` | 3,265 / 3,265 | 100% of English (`en-IN`) completed |
| `hybrid` | Multi-Model Local ASR (IndicConformer + Whisper) | MIT | 8 kHz → 16 kHz (resampled) | `dataset/manifests/evaluation/hybrid_transcripts.csv` | 14,790 / 14,790 | **100.0% Complete (Full Pool Evaluated)** |

Key architectural rules and measured results:
- **Cache separation**: Providers maintain separate caches (`saaras_transcripts.csv`, `indicconformer_transcripts.csv`, `whisper_transcripts.csv`, and `hybrid_transcripts.csv`). Each row explicitly records its actual model and provider.
- **Hardware environment**: Local inference ran on NVIDIA GeForce RTX 3050 Laptop GPU (4.3 GB VRAM), CUDA 12.4, PyTorch 2.6.0+cu124.
- **Latency performance**: Across all 14,790 audio clips, transcription latency achieved: Median = 849.5 ms, Mean = 1,167.2 ms, p95 = 2,286.4 ms, p99 = 2,982.5 ms.
- **Evaluation completion**: Rejection threshold $T = 0.1234$ was calibrated on `synthetic_calibration.csv` (dev split only). Character 3–5-gram TF-IDF + Logistic Regression achieved:
  - **Clean Final Test** (390 rows): Accuracy = 1.0000, In-Scope Accuracy = 1.0000, 95% CI: [1.0, 1.0]
  - **Robust Final Test** (2,340 rows): Accuracy = 0.9996, In-Scope Accuracy = 0.9994, 95% CI: [0.9987, 1.0]
  - **Held-Out Combined** (`unseen_combined` test-only, 390 rows): Accuracy = 0.9974, In-Scope Accuracy = 0.9967, 95% CI: [0.9923, 1.0]
  - **Rejection & Other False Accept Rate**: False accept rate on `other` = 0.0000 across all 5 other types. Abstention rate = 0.2308 (all 90 out-of-scope examples rejected).
  - Machine-readable report saved to `dataset/reports/hybrid_baseline.json`.

## Direct audio-to-intent probe

`scripts/train_audio_classifier.py` trains a frozen `facebook/wav2vec2-base-960h`
encoder with a 200,203-parameter MLP intent head. The v2 run consumes the frozen
manifests directly: 8,100 training rows, 1,950 calibration rows, a separate
390-row clean final test, and a separate 2,340-row robustness final test. It
joins each transcript to audio using the full condition-specific record key,
so an utterance's clean and degraded clips cannot be conflated.

The v2 report (`reports/audio_classifier_v2.json`) records 64.80% clean-test
accuracy at 96.15% coverage and 69.87% robustness-test accuracy at 96.15%
coverage. `unseen_combined` remains test-only and scores 73.02% at 96.92%
coverage. The direct model's 36.56% false-accept rate on `other` means it is
not suitable as the operational routing system; the ASR cascade remains the
synthetic-study reference. The older v1 report is retained for audit but is
superseded because it did not consume the frozen role manifests directly.

## IndicVoices

`indicvoices/filtered/` contains 1,915 FLAC records from the gated
`ai4bharat/IndicVoices` dataset, using Bengali, Gujarati, Hindi, Kannada,
Marathi, Tamil, and Telugu configurations from the `valid` split. English is
not covered by IndicVoices and is not claimed here.

Current derived candidate-label counts:

| Label | Count |
|---|---:|
| `pay_bill` | 414 |
| `update_details` | 362 |
| `other` | 348 |
| `balance` | 264 |
| `complaint_status` | 256 |
| `emi_due` | 94 |
| `transfer_to_agent` | 83 |
| `recent_transactions` | 43 |
| `branch_timing` | 23 |
| `report_fraud` | 23 |
| `block_card` | 5 |

These labels are keyword-derived candidates, not verified ground truth. The
manifest preserves original annotation fields and records
`label_provenance=keyword_derived_candidate` and
`review_status=unreviewed`. The audit found 963 speaker IDs, with 349 speaker
IDs repeated among retained records. This does not prove official split
independence.

## Public-data curation

The original 1,915-record filtered subset is preserved unchanged as an audit
source. A conservative deterministic curation pass writes
`indicvoices/curated/manifest.csv` and `excluded.csv` without deleting audio or
annotations:

| Result | Records |
|---|---:|
| Source records audited | 1,915 |
| Candidate records retained | 346 |
| Records excluded | 1,569 |
| Excluded: no actionable intent evidence | 1,560 |
| Excluded: conflicting intent evidence | 9 |

Retained records have one intent-specific evidence match across the available
transcript, normalized/verbatim variants, or task annotation. Generic banking
mentions are not enough. The retained set is still marked `unreviewed` and
`deterministic_transcript_evidence_candidate`; it is not trusted ground truth.
Current candidate coverage is uneven and does not include defensible examples
for `block_card`, `complaint_status`, or `branch_timing`. Those intents must be
reported as unavailable rather than scored as zero on public data.

The curated candidate set currently has no defensible examples for
`block_card`, `complaint_status`, or `branch_timing`; report those public
classes as unavailable rather than zero.

Recreate the curation with:

```powershell
python scripts\curate_indicvoices.py
```

## Licences and attribution

The filtered public data is derived from IndicVoices, documented as CC BY 4.0.
Retain the dataset attribution and cite the IndicVoices paper when distributing
or reporting results. Sarvam API/TTS use remains subject to Sarvam's current
service terms. The assignment PDF is included for project context.

## Validation commands

Run from any working directory; script defaults resolve from the repository:

```powershell
python scripts\validate_dataset.py

python scripts\validate_dataset.py `
  --source 'dataset\source\utterances_v0.2.csv' `
  --manifest 'dataset\synthetic\v0.2\manifest.csv' `
  --manifest-root 'dataset\synthetic\v0.2' `
  --report 'dataset\reports\v0.2_validation.json'

conda run -n torch26 python scripts\audit_indicvoices.py `
  --manifest 'dataset\indicvoices\filtered\manifest.csv' `
  --report 'dataset\reports\indicvoices_audit.json'

python scripts\validate_dataset.py `
  --source 'dataset\source\utterances_v0.3.csv' `
  --manifest 'dataset\synthetic\v0.3_robustness\manifest.csv' `
  --manifest-root 'dataset\synthetic\v0.3_robustness' `
  --report 'dataset\reports\v0.3_robustness_validation.json'
```

## Limitations

There is no real-speaker dataset in this project. Synthetic audio may contain
TTS artefacts, public labels are weak and imbalanced, and the current public
subset has not been reviewed as supervised ground truth. Results must remain
separate for synthetic and public data and cannot establish real-world caller
performance. The clean, baseline-noisy, and expanded-robustness synthetic-audio gates are
passed. Overall v0.3 training readiness is not passed until public-label review
and final evaluation-manifest freezing are complete. Keep `unseen_combined`
held out from training if it is used as the final robustness test condition.
