# VoxIntel Indic Data Readiness Report

Generated from the current workspace state; counts below are measured rather
than planned.

## Text source

| Source | Rows | Status |
|---|---:|---|
| `utterances_v0.2.csv` | 100 | preserved |
| `utterances_v0.3.csv` | 2,400 | source validator passed |

The v0.3 intended split counts are 1,620 train, 390 dev, and 390 test. The in-scope
portion is 1,200/300/300 per split (ten intents, 120/30/30 each). The `other`
portion is 420/90/90 per split (five `other_type` categories, 84/18/18 each).
No duplicate normalized text or paraphrase-cluster leakage was found.

## Audio

The v0.2 manifest contains 198 audio files: 93 train, 7 dev, and 98 test.
All files are readable mono 16-bit 8 kHz WAV and checksum validation passed.
`shubh` is used for train/dev and `simran` for test.

The v0.3 clean manifest currently contains all 2,400 readable WAV files: 1,620
train, 390 dev, and 390 test. All are mono 16-bit 8 kHz WAVs and the complete
manifest validator passed.

A multilingual eight-row smoke batch succeeded. The subsequent resumable bulk
operation used seven configured key slots and completed all rows. Earlier
attempts observed HTTP 429 rate limiting and HTTP 402 payment/quota responses;
the additional key slots allowed the remaining requests to finish.

## Public dataset

The local IndicVoices subset contains 1,915 FLAC files and 1,915 annotation
records from seven supported Indic language configurations. Current weak
candidate labels are:

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

Verified public labels: 0. These are derived keyword candidates requiring
review. The current output was downloaded from the `valid` split only, so
official train/valid speaker-overlap analysis has not been established. The
manifest contains 963 unique speaker IDs; 349 speaker IDs occur in more than
one retained record. This is a record-level audit, not proof of split
independence.

## Validation status

- v0.3 source labels, IDs, duplicate text, and paraphrase clusters: **passed**
- v0.2 manifest paths, checksums, WAV format, and split voice leakage: **passed**
- v0.3 clean audio generation: **passed** (2,400/2,400)
- noisy audio generation and validation: **passed** (4,800 files; 10 dB and
  20 dB SNR variants)
- public weak-label review: **not complete**; 346 conservative candidates are
  available for review, while 0 public rows are trusted
- public manifest provenance/review fields: **annotated** (`unreviewed`,
  `keyword_derived_candidate`)
- real-speaker evaluation: **not applicable and intentionally excluded**

The clean, baseline-noisy, and expanded-robustness synthetic-audio gates are
passed, and the evaluation manifests are frozen. Overall training readiness
remains pending until the public candidate review is completed.

## Expanded robustness suite

The expanded suite was generated from the same 2,400 clean sources into a new
directory, without overwriting the validated baseline noisy dataset. It
contains 12,390 readable mono 16-bit 8 kHz WAV files:

| Condition | Files | Train | Dev | Test |
|---|---:|---:|---:|---:|
| `gaussian_snr10` | 2,400 | 1,620 | 390 | 390 |
| `gaussian_snr20` | 2,400 | 1,620 | 390 | 390 |
| `traffic_snr15` | 2,400 | 1,620 | 390 | 390 |
| `office_snr15` | 2,400 | 1,620 | 390 | 390 |
| `telephone_channel` | 2,400 | 1,620 | 390 | 390 |
| `unseen_combined` | 390 | 0 | 0 | 390 |
| **Total** | **12,390** | **8,100** | **1,950** | **2,340** |

The full robustness manifest validator passed with zero errors. `traffic` and
`office` are synthetic colored-noise proxies rather than licensed environmental
recordings. `telephone_channel` is an approximate band-limit and quantization
simulation, not a validated telephone codec implementation. The suite is
controlled augmentation/evaluation data and cannot establish real-world caller
performance. The `unseen_combined` condition should remain test-only if used as
the final robustness generalization check.

## Evaluation freeze

The freeze record is
[`dataset/manifests/evaluation/FREEZE.json`](../manifests/evaluation/FREEZE.json).
It records SHA-256 hashes, split rules, and the eligible role for each
manifest:

- `synthetic_train.csv`: 8,100 rows; fitting only.
- `synthetic_calibration.csv`: 1,950 rows; threshold and reject-gate tuning
  only.
- `synthetic_final_test.csv`: 2,340 rows; one final evaluation only, including
  the 390 `unseen_combined` rows.
- `public_trusted.csv`: 0 rows; all current public labels remain unreviewed
  keyword-derived candidates and are excluded from trusted metrics.
- `public_curated_candidates.csv`: 346 rows; conservative review queue only.

## Public-data curation audit

The original 1,915 downloaded IndicVoices records and annotations remain
unchanged. A separate conservative curation pass audited all available
transcript, normalized/verbatim, unsanitized, and task-name fields:

| Curation result | Records |
|---|---:|
| Source records audited | 1,915 |
| Candidate records retained | 346 |
| Records excluded | 1,569 |
| No actionable intent evidence | 1,560 |
| Conflicting intent evidence | 9 |

The retained rows contain one intent-specific deterministic evidence match.
They remain unreviewed candidates, not verified labels. The retained candidate
distribution is `balance` 113, `pay_bill` 102, `emi_due` 32,
`recent_transactions` 27, `report_fraud` 22, `update_details` 9,
`transfer_to_agent` 5, and explicit `other` 36. There are no defensible
curated candidates for `block_card`, `complaint_status`, or `branch_timing` in
this downloaded public slice. Public metrics for those intents are therefore
unavailable, not zero.

The curation outputs are
`dataset/indicvoices/curated/manifest.csv`,
`dataset/indicvoices/curated/excluded.csv`, and
`dataset/indicvoices/curated/summary.json`.
