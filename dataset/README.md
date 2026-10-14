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
