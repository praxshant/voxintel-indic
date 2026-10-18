# VoxIntel Indic — Assignment Overview

## Problem statement

This assignment is to build an end-to-end speech-to-intent system for noisy, code-mixed Indic call audio. The model must map short 8 kHz phone-audio utterances directly to a fixed set of intents, with an explicit `other` outcome for out-of-scope requests and a reject gate to reduce false triggers.

The project is not framed as a general-purpose ASR benchmark. The goal is operational intent routing for real-world call handling, with emphasis on:

- noisy Indic speech
- mixed-language input (for example Hinglish plus another Indic language)
- roughly 10 intents plus `other`
- minimal false acceptances
- handling unknown or out-of-vocabulary entities
- real-time or near real-time inference

## Core objective

The main task is to compare two approaches:

1. Baseline: ASR transcript -> text intent classifier
2. Proposed path: audio directly -> intent classifier

The end-to-end approach is pursued because it may reduce ASR error cascading, cut latency, and optimize performance directly for the downstream intent task. However, these are hypotheses that must be tested against the cascade, not assumptions to be treated as proven.

## Scope and constraints

### Minimum seven-day experiment
- The project is a 7-working-day core effort.
- Data collection and preparation happen before Day 1 and are excluded from that clock.
- The plan requires fixed evaluation sets to be ready before modelling starts.
- This is a readiness condition, not an optional admin step.

### Intended domains
- One inbound domain is chosen first (for example banking or insurance).
- The domain defines a fixed intent taxonomy and entity slots.
- A fixed `other` class is required to capture unrelated or unsupported requests.

### Evaluation discipline
The work must report results under strict conditions:

- synthetic table
- public-dataset table

These tables are never merged. Claims should match the dataset behind them.

Speaker and voice disjoint splits are required. Overall accuracy is not enough; the work must report:

- in-scope accuracy
- false accept rate per `other` type
- comparison at fixed recall (for example 90% and 95%)
- confidence intervals

## Architecture

### Baseline
The baseline uses an ASR transcript followed by a character n-gram text intent classifier and a rejection gate. ASR providers supported via `scripts/run_saaras_baseline.py`:

1. **Saaras (Sarvam ASR `saarika:v2.5`)**: Cloud API. Quota-blocked across all configured keys (HTTP 402; 2,860/14,790 transcripts preserved).
2. **IndicConformer-600M (AI4Bharat `indic-conformer-600m-multilingual`)**: Local PyTorch backend via Hugging Face (`trust_remote_code=True`, MIT license). Transcribes the 7 Indic languages (11,525 rows).
3. **Whisper-base.en (OpenAI `whisper-base.en`)**: Local PyTorch backend (MIT license). Transcribes English (`en-IN`, 3,265 rows).
4. **Hybrid Multi-Model ASR**: Combined local pipeline that routes each language to its designated model. 100% completed across all 14,790 frozen pool rows (`hybrid_transcripts.csv`).

The text classifier uses character n-grams (range 3–5, TF-IDF, `sublinear_tf=True`, `min_df=2`) with balanced class-weight Logistic Regression across the 10 banking intents plus `other`. The reject threshold ($T = 0.1234$) is chosen strictly on calibration (`dev`, 2,340 rows) data at $\ge 90\%$ recall, never using final test rows.

Measured Results (`dataset/reports/hybrid_baseline_v2.json`):
- **Clean Final Test** (390 rows): Accuracy = 1.0000, In-Scope Accuracy = 1.0000, 95% CI: [1.0, 1.0].
- **Robust Final Test** (2,340 rows): Accuracy = 0.9996, In-Scope Accuracy = 0.9994, 95% CI: [0.9987, 1.0].
- **Held-Out Combined** (`unseen_combined` test-only, 390 rows): Accuracy = 0.9974, In-Scope Accuracy = 0.9967, 95% CI: [0.9923, 1.0].
- **Rejection & Other Security**: Coverage = 0.7692, Abstention = 0.2308, False Accept Rate on `other` = 0.0000 across all five `other_type` categories.
- **Calibration** (calibration split, never test): ECE uncalibrated = 0.0985; Temperature scaling (T=0.5) → ECE 0.0062; Isotonic regression → ECE 0.000.
- **Risk–Coverage (clean test, max_prob signal)**: AURC ≈ 0.000. Selective accuracy: 80% cov = 1.0000, 90% cov = 1.0000, 95% cov = 1.0000.
- **False-accept at fixed coverage targets**: 0.0000 for all five `other_type` categories at 80%, 90%, and 95% coverage.
- **Signal comparison (calibration split)**: max_prob / margin / norm_entropy each select threshold achieving 99.94% recall at precision 1.0.

### End-to-end path
The proposed end-to-end model uses:

- an audio encoder
- pooled embeddings
- an intent head
- a reject gate
- an auxiliary text/character decoder for entity handling (stretch goal)

The encoder sees telephone-band audio; 8 kHz audio is upsampled before processing, but the model still operates on telephone-band content. The reject gate decides whether the predicted intent is safe to act on, or whether the input should be routed to clarify/fallback instead.

### Reject gate (implemented for baseline)
The reject gate is implemented for the ASR+classifier baseline. Current signals:

- **max_prob**: Maximum softmax probability. Current operating threshold: T=0.1234.
- **margin**: P_top1 − P_top2 (decision-boundary margin). Threshold: 0.0137 on calibration split.
- **norm_entropy**: 1 − H/log(K) (normalised Shannon entropy, higher = more confident). Threshold: 0.0065.

Energy score is explicitly **not implemented** here — scikit-learn LogReg logits are not neural free-energy scores and cannot be responsibly interpreted as such.

For the future end-to-end audio model, additional gate signals may include entropy, energy score from neural logits, distance from embedding centroids, and agreement across noisy copies.

A calibrated risk model may be trained on these features.

## Data strategy

The plan assumes a small but reproducible core dataset first, followed by scaling only after the readiness gate passes.

### Core data buckets
- Synthetic in-scope training data using text-to-speech and degradation pipeline
- Synthetic dev/test data with unseen voices and paraphrase clusters
- Curated public speech from IndicVoices, kept separate from synthetic results
- Optional held-out entity values and public benchmark evaluation

### Evaluation boundary
This implementation uses synthetic speech and curated public datasets only. Do not
recruit speakers, create a real-speaker test set, or describe real-speaker
evaluation as a readiness gate. Results cannot establish performance on
real-world callers.

### Ready gate before training
The following evaluation sets are required before modelling begins:

- T1: clean synthetic, unseen voices and paraphrase clusters, in-scope
- T2: noisy synthetic, in-scope and `other`
- T3: held-out public IndicVoices records, with speaker metadata audited
- T4: held-out entity values (stretch)
- T5: MASSIVE hi-IN audio benchmark (stretch)

This gate prevents the team from starting the modelling phase without a fixed test setup.

## v0.3 readiness status

`dataset/source/utterances_v0.3.csv` is a deterministic candidate text source with 2,400 rows:

- 1,800 in-scope rows: 1,200 train, 300 dev, 300 test
- 600 `other` rows: 420 train, 90 dev, 90 test
- 30 paraphrase clusters per in-scope intent, assigned to one split before TTS
- five explicit `other_type` categories
- all text is marked `machine_reviewed_pending_human_spot_check`; this is not a
  claim of human language review

The source validator currently passes duplicate-text, ID, label, and cluster
leakage checks. A multilingual eight-row Sarvam smoke batch passed. The current
v0.3 clean manifest contains 2,400/2,400 readable 8 kHz WAV files (1,620 train,
390 dev, 390 test). The resumable bulk generation used seven configured
Sarvam key slots and completed after earlier HTTP 429 and HTTP 402 responses.
The clean, baseline-noisy, and expanded-robustness synthetic-audio gates are
passed; do not mark overall readiness passed until public-label review and
evaluation-manifest freezing are complete. The expanded suite contains 12,390
files across Gaussian, synthetic colored-noise, approximate telephone-channel,
and unseen-combination conditions. Its environmental conditions are proxies,
not real recordings, and it does not establish real-world caller performance.

New reproducibility tools:

- `scripts/validate_dataset.py` checks source invariants, manifest references,
  WAV format, checksums, split counts, voice leakage, and cluster leakage.
- `scripts/degrade_audio.py` creates deterministic 8 kHz noisy variants with
  documented condition, SNR, clipping, reverb, dropout, and random seed
  metadata. Keep the `unseen_combined` condition test-only when using it as a
  final robustness check.

The direct audio-to-intent model remains a hypothesis. Do not claim superiority
over the Saaras ASR cascade until both systems are evaluated on the same frozen
synthetic/public-only test sets.

The modelling phase is executed with `scripts/run_saaras_baseline.py` using `--provider hybrid` (IndicConformer for Indic languages, Whisper-base.en for English). All 14,790 transcripts in the frozen ASR pool are generated and cached (`dataset/manifests/evaluation/hybrid_transcripts.csv`). The character 3-5 gram TF-IDF logistic-regression classifier was trained on `synthetic_train.csv` (9,720 rows), calibrated at threshold $T = 0.1234$ on `synthetic_calibration.csv` (2,340 dev rows), and evaluated on `synthetic_final_test.csv` (2,730 test rows across clean and all 6 degradation conditions). Clean test accuracy is 1.0000, robust test accuracy is 0.9996, and unseen combined test accuracy is 0.9974 (`dataset/reports/hybrid_baseline.json`).

## Current data assets and completed work

### Source utterance CSV

[dataset/source/utterances_v0.2.csv](./dataset/source/utterances_v0.2.csv) is the current text-label source:

- 100 rows total
- 70 banking-intent examples
- 30 `other` examples
- 98 TTS rows
- 2 explicit local-audio rows:
  - `non_005`: silence
  - `non_006`: noise
- no exact duplicate utterance text was found in the v0.2 audit

The CSV language set is `bn-IN`, `en-IN`, `gu-IN`, `hi-IN`, `kn-IN`, `mr-IN`,
`ta-IN`, `te-IN`, and `und`.

### Generated synthetic audio

[dataset/synthetic/v0.2/](./dataset/synthetic/v0.2) contains the generated Bulbul pilot:

- 198 WAV files from 98 TTS rows with two voice variants
- Sarvam Bulbul v3
- mono, 16-bit PCM, 8 kHz
- `shubh` is used for train/dev
- `simran` is held out for test
- 93 train, 7 dev, and 98 test files
- `manifest.csv`, `manifest.jsonl`, and `dataset_summary.json`
- SHA-256 checksum for each file

This is a TTS pilot, not proof of natural telephone-speech performance. The
held-out voice split is voice-disjoint, but the small v0.2 text set is not yet a
full paraphrase-cluster split.

### IndicVoices filtered subset

[dataset/indicvoices/filtered/](./dataset/indicvoices/filtered) contains the current
language-restricted, assignment-filtered IndicVoices subset:

- source: `ai4bharat/IndicVoices`
- split downloaded: `valid`
- 1,915 audio files and 1,915 manifest records
- original compressed audio retained as FLAC
- source annotations and speaker/provenance metadata preserved
- SHA-256 checksum for each audio file

The available language mapping is:

| Current CSV code | IndicVoices config | Records |
|---|---|---:|
| `bn-IN` | Bengali | 241 |
| `gu-IN` | Gujarati | 154 |
| `hi-IN` | Hindi | 496 |
| `kn-IN` | Kannada | 152 |
| `mr-IN` | Marathi | 340 |
| `ta-IN` | Tamil | 329 |
| `te-IN` | Telugu | 203 |

`en-IN` is skipped because IndicVoices has no English config, and `und` is
skipped because its language is unspecified. Do not describe this subset as
covering English.

Every retained IndicVoices record has assignment-compatible fields:

- `id`
- `intent`
- `language_code`
- `text`
- `other_type`
- `source`
- `script`
- `mix`

The current intent mapping uses conservative keyword-based matching over
IndicVoices transcript, normalized transcript, verbatim transcript,
unsanitized transcript, and task-name annotations. Supported labels are:

`balance`, `recent_transactions`, `block_card`, `emi_due`, `report_fraud`,
`complaint_status`, `update_details`, `branch_timing`, `pay_bill`,
`transfer_to_agent`, and `other`.

Current label counts:

| Intent | Count |
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

These are weak labels, not human-verified intent labels. They must be reviewed
before being used as supervised training targets. Preserve the original
IndicVoices fields for audit and do not silently replace them with the derived
assignment label.

### Reproduction scripts

- [scripts/generate_dataset.py](./scripts/generate_dataset.py): creates the
  Sarvam TTS WAV dataset, local silence/noise rows, manifests, checksums, and
  voice-disjoint splits.
- [scripts/filter_indicvoices.py](./scripts/filter_indicvoices.py): streams
  gated IndicVoices annotations and audio, keeps only configured languages and
  assignment-related records, writes audio plus labeled manifests.
- [scripts/curate_indicvoices.py](./scripts/curate_indicvoices.py): creates a
  separate conservative candidate manifest using intent-specific evidence and
  explicit exclusion reasons; it never overwrites the source manifest.

Run the IndicVoices filter with the `torch26` conda environment and a local
`HF_TOKEN` in `.env`:

```powershell
$env:HF_TOKEN = (Get-Content .env | `
  Where-Object { $_ -match '^HF_TOKEN=' }) -replace '^HF_TOKEN=', ''

conda run -n torch26 python -X utf8 `
  scripts\filter_indicvoices.py `
  --utterances dataset/source/utterances_v0.2.csv `
  --output dataset/indicvoices/filtered `
  --splits valid
```

The Hugging Face token and Sarvam API key are local secrets. Never commit them,
print them, or place them in manifests.

## Current data-quality rules

- Keep language, script, mix, TTS voice, and source dataset as separate metadata.
- Keep the original CSV and original IndicVoices annotations unchanged.
- Treat keyword-derived IndicVoices intents as candidates requiring review.
- Treat the curated IndicVoices manifest as unreviewed candidates, not ground
  truth. The current pass retains 346/1,915 records and has no defensible
  candidates for `block_card`, `complaint_status`, or `branch_timing`; do not
  score those public intents as zero.
- Do not use `other` records as positive intent examples.
- Do not collect or require real-speaker data in this project scope.
- Maintain speaker-disjoint and voice-disjoint evaluation splits.
- Do not claim that IndicVoices provides English data for this project.
- Use separate synthetic and public-dataset result tables; do not merge their scores.
- Verify audio paths, sample rates/formats, labels, and checksums before training.

## Model pipeline and stages

### Stage 0 — Baseline
- Run Saaras on each test clip
- Feed ASR output to a text intent classifier
- Establish the reference result for all later comparisons

### Stage 1 — Frozen encoder probe
- Cache per-layer pooled embeddings
- Probe models such as XLS-R, Whisper-small encoder, and IndicWav2Vec
- Compare layer-wise performance and select the encoder to fine-tune

### Stage 2 — Fine-tuning
- Fine-tune the top layers of the selected encoder
- Use attentive statistics pooling and a small MLP head
- Include `other` as a class
- Use SpecAugment, class weighting, and mixed precision
- Early stop on dev data with unseen voices/speakers

### Stage 3 — Multi-task / stretch
- Add auxiliary CTC decoder for character transcription
- Add distillation from the text teacher
- Measure the benefit of each additional objective

## Entity handling

Entity handling is a stretch goal, not a minimum requirement.

Planned components:
- auxiliary character decoder
- phonetic gazetteer matching
- handling of common identity/amount/date tokens
- unknown-span flagging for unsupported values

If the auxiliary decoder is weak, Saaras may be used as a fallback for entity spans only, and this must be disclosed clearly in the report.

## Reject gate versus single encoder pass

The summary should not treat the entire end-to-end pipeline as a single encoder pass in all configurations.

The intended inference path is direct audio -> intent, but:

- the reject gate may require extra scoring logic
- the optional agreement-based check may evaluate multiple noisy copies
- the optional entity decode path adds extra computation

So the design is direct end-to-end intent routing, but not necessarily one-pass inference in every operating configuration.

## Seven-day work plan

### Before Day 1
- define task/domain and taxonomy
- generate text data
- build synthetic audio set
- define fixed evaluation sets
- confirm dataset card stub, licenses, terms, and consent

### Day 1
- build evaluation harness
- implement baseline on synthetic/public T1-T3
- create text teacher

### Day 2
- cache per-layer features
- probe candidate encoders
- select encoder/layer choice

### Day 3
- fine-tune best encoder
- compare against baseline on synthetic/public T1-T3

### Day 4
- reject gate and calibration
- false accept rate by `other` type
- operating-point analysis

### Day 5
- stretch: CTC, distillation, entity matcher, T5

### Day 6
- error analysis
- latency analysis
- remaining ablations
- public benchmark table (T6) as a separate row

### Day 7
- final report
- dataset card
- model card
- README
- optional demo

## Cut line

If time slips, the planned cut order is:

1. optional demo
2. T6 public benchmark
3. entity handling
4. CTC and distillation ablation

If fine-tuning fails to beat the frozen probe, Day 5 should be spent on error analysis instead of multi-task training.

## Key risks and assumptions

Main risks are scope and data, not modelling alone.

- Scope and deadline are not fully confirmed; the plan assumes Problem 1 only and a seven-day core.
- Data collection overruns are a primary operational risk.
- TTS artifacts can inflate performance; mitigation requires many voices, heavy augmentation, and held-out public-data evaluation.
- End-to-end may underperform the cascade; if so, this must be stated honestly.
- Entity decoding may be weak; fallback to Saaras for spans only should be documented.

## Deliverables

The project should produce:

- a GitHub repository with scripts, training code, notebooks, and results
- a 5–6 page report with limitations stated upfront
- a dataset card and a short model card
- optional demo for recording/file input and risk scoring

## Working interpretation for VoxIntel Indic

This plan should be treated as the minimum viable research framework for the assignment, not as a claim that direct end-to-end intent routing has already been proven for Indic calls.

The operational summary is:

- Build the cascade baseline first.
- Compare it against a direct audio-to-intent model.
- Use a reject gate to control false triggers.
- Evaluate strictly on disjoint splits with separate synthetic and public-dataset tables.
- Keep all research extensions (entity matching, CTC, distillation, public benchmark) clearly marked as stretch goals unless the core experiment proves them necessary.

## Future assignment guidance

For future work on this assignment, the project should keep the following distinctions clear:

- minimum seven-day core vs. optional research extensions
- fixed data-readiness gate vs. data collection tasks
- direct pipeline goal vs. proven end-to-end advantage
- core metrics vs. stretch metrics
- measured false-trigger control vs. claimed operational readiness

This keeps the project faithful to the 11-page plan while preventing overstatement of results or capabilities.
