# VoxIntel Indic Data Readiness Report

Generated from the current workspace state; counts below are measured rather
than planned.

## Text source

| Source | Rows | Status |
|---|---:|---|
| `utterances_v0.2.csv` | 100 | preserved |
| `utterances_v0.3.csv` | 2,400 | source validator passed |

The v0.3 split counts are 1,620 train, 390 dev, and 390 test. The in-scope
portion is 1,200/300/300 per split (ten intents, 120/30/30 each). The `other`
portion is 420/90/90 per split (five `other_type` categories, 84/18/18 each).
No duplicate normalized text or paraphrase-cluster leakage was found.

## Audio

The v0.2 manifest contains 198 audio files: 93 train, 7 dev, and 98 test.
All files are readable mono 16-bit 8 kHz WAV and checksum validation passed.
`shubh` is used for train/dev and `simran` for test.

No v0.3 bulk audio has been generated yet. This is intentional: the source
must be smoke-tested with the Sarvam API before scaling.

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
- v0.3 clean audio generation: **not run**
- noisy audio validation: **not run**
- public weak-label review: **not complete**
- public manifest provenance/review fields: **annotated** (`unreviewed`,
  `keyword_derived_candidate`)
- real-speaker evaluation: **not applicable and intentionally excluded**

The readiness gate is therefore **not passed for model training**. The next
required action is a small API smoke batch, audio inspection, public-label
review sampling, and then full generation only if those checks succeed.
