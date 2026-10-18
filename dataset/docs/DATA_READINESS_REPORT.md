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
- Saaras baseline transcription: **quota-exhausted** (2,860/14,790 success preserved, HTTP 402 on all keys)
- Local hybrid ASR baseline transcription: **100% COMPLETE** (14,790/14,790 transcripts generated and cached)
- Baseline intent classifier training & evaluation: **PASSED** (character 3-5 gram TF-IDF + Logistic Regression evaluated on all 2,730 test clips)
- Baseline unit test suite (`tests/test_baseline.py`): **passed** (27/27 tests passed in 7.9s)

The clean, baseline-noisy, expanded-robustness synthetic-audio gates, and full ASR baseline transcription and evaluation gates are
passed. The frozen synthetic baseline is completely established.

## ASR baseline readiness & measured evaluation

The baseline pipeline (`scripts/run_saaras_baseline.py`) has processed all 14,790 clips in `dataset/manifests/evaluation/synthetic_asr_pool.csv`.

### Hardware environment
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.3 GB VRAM)
- System RAM: 15.2 GB, Disk free: ~55 GB
- CUDA: 12.4, PyTorch: 2.6.0+cu124, Torchaudio: 2.6.0+cu124
- Audio processing: FFmpeg 9.0.1, SoundFile 0.13.1, Librosa 0.11.0

### Provider comparison and coverage

| Dimension | Saaras (Sarvam) | IndicConformer-600M | Whisper-base.en | Hybrid Local Pipeline |
|---|---|---|---|---|
| Model Name | `saarika:v2.5` | `ai4bharat/indic-conformer-600m-multilingual` | `openai/whisper-base.en` | IndicConformer + Whisper |
| Hosting | Cloud API | Local PyTorch (`trust_remote_code=True`) | Local PyTorch | Local PyTorch |
| License | Proprietary | MIT | MIT | MIT |
| Sample Rate | 8 kHz | Resampled 8→16 kHz | Resampled 8→16 kHz | Resampled 8→16 kHz |
| Languages Covered | All | 7 Indic languages (no en-IN) | English (`en-IN`) | All 8 languages |
| Successful Rows | 2,860 | 11,525 | 3,265 | **14,790 / 14,790 (100.0%)** |
| Cache File | `saaras_transcripts.csv` | `indicconformer_transcripts.csv` | `whisper_transcripts.csv` | `hybrid_transcripts.csv` |
| Evaluation Status | Blocked (quota) | Evaluated (Indic subset) | Evaluated (English subset) | **Fully Evaluated** |

### Complete hybrid pool coverage breakdown (14,790 rows)
- **By Split**: train 9,720 / 9,720 (100%), dev 2,340 / 2,340 (100%), test 2,730 / 2,730 (100%)
- **By Condition**: clean 2,400, gaussian_snr10 2,400, gaussian_snr20 2,400, traffic_snr15 2,400, office_snr15 2,400, telephone_channel 2,400, unseen_combined 390
- **By Language**: hi-IN 3,265, en-IN 3,265, ta-IN 1,420, bn-IN 1,420, te-IN 1,350, mr-IN 1,350, gu-IN 1,360, kn-IN 1,360
- **Latency Performance** (14,790 clips):
  - Median: 849.5 ms
  - Mean: 1,167.2 ms
  - 95th Percentile (p95): 2,286.4 ms
  - 99th Percentile (p99): 2,982.5 ms

### Classifier & reject gate results (`dataset/reports/hybrid_baseline_v2.json`)
- **Training**: 9,720 synthetic training rows (character 3–5-gram TF-IDF, balanced Logistic Regression).
- **Calibration**: Dev split only (2,340 rows). Rejection threshold $T = 0.1234$ chosen at $\ge 90\%$ in-scope recall maximizing precision. Zero test rows used for calibration or model selection.
- **Calibration analysis** (calibration split only, never test):
  - ECE uncalibrated: 0.0985
  - Temperature scaling (best T=0.5): ECE → 0.0062 (improvement Δ = 0.0923)
  - Isotonic regression: ECE → 0.000 (improvement Δ = 0.0985, `isotonic_helped=True`)
  - All three reject signals (max_prob / margin / norm_entropy) achieve 99.94% recall at precision 1.0 at calibration-selected thresholds.
- **Evaluation**:
  - **Clean Final Test** (390 rows): Accuracy = 1.0000, In-Scope Accuracy = 1.0000, 95% CI: [1.0, 1.0]
  - **Robust Final Test** (2,340 rows): Accuracy = 0.9996, In-Scope Accuracy = 0.9994, 95% CI: [0.9987, 1.0]
  - **Individual Conditions** (390 rows each):
    - `clean`: 1.0000
    - `gaussian_snr10`: 1.0000
    - `gaussian_snr20`: 1.0000
    - `office_snr15`: 1.0000
    - `telephone_channel`: 1.0000
    - `traffic_snr15`: 1.0000
    - `unseen_combined` (strictly test-only): Accuracy = 0.9974, In-Scope Accuracy = 0.9967, 95% CI: [0.9923, 1.0]
  - **Rejection Gate & Out-of-Scope Security**:
    - Coverage: 0.7692 (in-scope accepted); Abstention rate: 0.2308
    - False Accept Rate on `other`: 0.0000 across all 5 categories (`ambiguous_in_domain`, `chitchat`, `non_intent_audio`, `out_of_domain`, `unsupported_near_miss`).
  - **Risk–Coverage** (clean test, max_prob signal): AURC ≈ 0.000. Selective accuracy at 80% cov = 1.0000, 90% cov = 1.0000, 95% cov = 1.0000.
  - **False-accept by `other_type` at fixed coverage** (clean test): 0.0000 for all five categories at 80%, 90%, and 95% coverage targets.
  - **Per-language breakdown** (clean test): bn-IN 97.5% acc/100% cov; en-IN 100% acc/47.1% cov; gu-IN 100%/100%; hi-IN 100%/47.1%; kn-IN 100%/100%; mr-IN 100%/100%; ta-IN 100%/100%; te-IN 100%/100%.

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
