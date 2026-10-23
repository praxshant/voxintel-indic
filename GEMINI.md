You are working on the VoxIntel Indic assignment.

## Primary repository

Use this repository as the source of truth:

C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic

GitHub repository:

https://github.com/praxshant/voxintel-indic

Current branch: master

Do not use C:\Users\ACER\OneDrive\Desktop\VoxIntel as the primary repository for this task. That is a separate VoxIntel-R project containing an English SLURP reliability experiment and a completed Whisper phase. Its Whisper work must remain untouched and is not the Indic assignment.

Do not commit or push changes automatically.

---

## Actual assignment objective

Build and evaluate a speech-to-intent system for noisy, code-mixed Indic telephone audio.

The required comparison is:

1. ASR transcript → text intent classifier
2. Future direct audio → intent classifier

The cascade baseline is complete and remains the operational reference. A
direct audio-to-intent probe is also implemented, but it is a synthetic-only
research comparison and is not an operational replacement:

Saaras or an alternative Indic ASR → character n-gram intent classifier → reject/abstain gate

The system must support:

- Ten fixed banking intents:
  - balance
  - recent_transactions
  - block_card
  - emi_due
  - report_fraud
  - complaint_status
  - update_details
  - branch_timing
  - pay_bill
  - transfer_to_agent
- An explicit `other` class
- `other_type` metadata
- No manual labelling
- No real-speaker data collection
- Separate synthetic and public-data reporting
- Voice-disjoint and paraphrase-cluster-disjoint evaluation
- `unseen_combined` strictly test-only
- Calibration and threshold selection only on calibration/dev data
- No test data for training, calibration, model selection, or threshold tuning

The immediate goal is to preserve the frozen synthetic evaluation and make only
small, reproducible improvements that satisfy the rules below.

## Simple research guide

Use this guide only within the assignment requirements; it is not permission to
expand the project scope.

1. **Compare, do not replace.** Keep the hybrid ASR cascade as the reference
   system. Recent Speech-MASSIVE work evaluates cascaded and end-to-end SLU
   together, which matches this assignment's comparison requirement.
2. **Prefer an Indic/multilingual encoder for the next direct-audio probe.**
   The current frozen English encoder is a weak match for code-mixed Indic
   audio. At most, test one locally available multilingual/Indic encoder on the
   same frozen manifests. Do not add a new large download or service unless it
   is necessary and practical.
3. **Keep TTS claims modest.** Research on synthetic speech commands supports
   basic audio-quality filtering but still finds a synthetic-to-real mismatch.
   Preserve the validator, inspect a small deterministic sample, and never turn
   synthetic performance into a real-caller claim.
4. **Use calibration data for every decision.** Select encoder settings,
   fine-tuning depth, rejection threshold, and early stopping from the frozen
   dev/calibration split only. Run final tests after selection, not during it.
5. **Use the required safety metrics.** Report coverage, in-scope accuracy,
   confidence intervals, and false accepts by `other_type`. High false accepts
   mean the model remains research-only.
6. **Stay within the cut line.** Do not add CTC, distillation, entity decoding,
   custom multilingual transfer modules, public-label metrics, or a new data
   collection effort unless the user explicitly expands scope.

### Current direct-audio analysis and blocker

Read `dataset/reports/audio_classifier_v2_error_analysis.md` before proposing a
new direct-audio run. It is descriptive final-test analysis only, not a tuning
source. The key issue is unsafe `other` routing: the largest errors are `other`
to `emi_due` and `other` to `pay_bill`; traffic and 10 dB Gaussian conditions
are the weakest evaluated conditions.

`facebook/wav2vec2-xls-r-300m` is listed in the local Hugging Face cache but
has no local weights, so an offline Indic/multilingual direct probe cannot run.
Do not start a large download without explicit authorization. If authorized,
use one small calibration-only smoke probe first; keep the hybrid ASR cascade
as the supported reference until a new frozen evaluation is completed.

### Reading list for context

- [Speech-MASSIVE (Lee et al., 2024)](https://arxiv.org/abs/2408.03900)
- [Synthetic Speech Commands Quality Checks (Quintas et al., 2024)](https://arxiv.org/abs/2409.12745)
- [Multilingual Intent Transfer (Wu et al., 2025)](https://aclanthology.org/2025.findings-acl.783/)
- [AI4Bharat IndicConformer](https://ai4bharat.iitm.ac.in/areas/model/ASR/IndicConformer/)

---

## Read these files first, in this order

### 1. Project instructions and assignment context

Read the entire files:

- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\CLAUDE.md
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\README.md
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\README.md
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\docs\DATA_READINESS_REPORT.md
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\docs\DATASET_CARD.md

These documents define the assignment scope, data limitations, evaluation rules, public-label restrictions, and readiness status.

### 2. Authoritative frozen evaluation policy

Read:

- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\FREEZE.json

Then inspect the frozen manifests:

- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\synthetic_train.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\synthetic_calibration.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\synthetic_clean_final_test.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\synthetic_final_test.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\synthetic_asr_pool.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\public_trusted.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\public_curated_candidates.csv

Verify:

- Row counts
- SHA-256 hashes
- Split membership
- Source IDs
- Paraphrase clusters
- Voice separation
- Conditions
- `unseen_combined` test-only status
- Public trusted set remains empty
- Public candidate labels remain untrusted

Do not regenerate or change frozen manifests unless a validator proves they are invalid.

### 3. Existing ASR baseline implementation

Read completely:

- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\scripts\run_saaras_baseline.py

Also read:

- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\tests\test_baseline.py
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\manifests\evaluation\saaras_transcripts.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\reports\saaras_transcription_errors.csv
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\dataset\reports\saaras_baseline.json

Understand these functions and behaviors before editing:

- Cache-key generation
- Saaras request schema
- Audio-path resolution
- Retry behavior
- API-key fallback behavior
- HTTP 402 quota stopping
- Cache preservation
- Coverage reporting
- Classifier training
- Calibration threshold selection
- Final evaluation blocking
- Error logging

The cache key must distinguish the same source utterance across clean/noisy conditions using:

- source ID
- audio path
- condition

Never replace missing transcripts with source text or fabricated text.

### 4. Data-generation and validation logic

Read:

- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\scripts\freeze_evaluation_manifests.py
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\scripts\degrade_audio.py
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\scripts\validate_dataset.py
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\scripts\spot_check_synthetic.py
- C:\Users\ACER\OneDrive\Desktop\VoxIntel Indic\scripts\curate_indicvoices.py

Do not expand the dataset or download new datasets unless absolutely necessary.

---

## Current known state

The frozen synthetic ASR pool contains 14,790 rows:

- 2,400 clean synthetic files
- 12,390 robustness files
- `unseen_combined`: 390 rows, all test-only

Current Saaras transcript cache:

- Expected rows: 14,790
- Successful transcripts: 2,860
- Failed or missing rows: 11,930
- All configured Saaras keys are currently quota-blocked with HTTP 402
- Existing successful transcripts must be preserved
- Final classifier metrics are currently blocked because coverage is incomplete

Do not claim the Saaras baseline is complete.

---

## Main technical approach

### Preferred approach: local Indic ASR replacement

Because Saaras quota is exhausted, investigate and implement a legitimate local alternative ASR backend.

Preferred first candidate:

- AI4Bharat IndicConformer-600M
- Hugging Face:
  https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
- Official repository:
  https://github.com/AI4Bharat/indicConformer-finetuning

Potential ready-made self-hosted wrapper:

- https://github.com/SASHI117/ai4bharat_stt

The wrapper exposes a local FastAPI endpoint and supports language-specific IndicConformer transcription.

The alternative must support the project languages:

- Bengali: bn
- English: en
- Gujarati: gu
- Hindi: hi
- Kannada: kn
- Marathi: mr
- Tamil: ta
- Telugu: te

IndicConformer requires an explicit language code and does not automatically identify the language. Use the manifest `language_code` field. Do not guess language codes.

Audio is 8 kHz telephone-band audio. Resampling to 16 kHz is acceptable, but document that this does not restore missing high-frequency content.

A second candidate for later comparison is:

- https://huggingface.co/bodhan-ai/indic-transcribe-core

Do not start with Indic-Transcribe-core unless IndicConformer is unusable; it is heavier and approximately 4.6 GB.

---

## Required implementation design

Make the ASR provider configurable rather than duplicating the entire pipeline.

Possible design:

- Keep Saaras support intact.
- Add a local/provider-neutral transcription backend.
- Add a provider argument such as:
  - `--provider saaras`
  - `--provider indicconformer`
- Keep separate caches:
  - `saaras_transcripts.csv`
  - `indicconformer_transcripts.csv`
- Add provider metadata to output reports.
- Preserve the original Saaras cache and error log.
- Do not silently label IndicConformer results as Saaras results.

If using a local HTTP wrapper, make the endpoint configurable through CLI arguments or environment variables. Do not hardcode private URLs or credentials.

Potential provider-neutral response fields:

- `record_key`
- `id`
- `audio_path`
- `split`
- `intent`
- `other_type`
- `language_code`
- `condition`
- `transcript`
- `status`
- `provider`
- `latency_ms`
- `error`
- `attempts`

The provider must:

- Resume safely
- Skip already successful cache rows
- Retry transient errors
- Preserve failed rows explicitly
- Never fabricate transcripts
- Record latency
- Record provider name
- Validate non-empty transcript output
- Stop or report clearly on model/server failure

---

## Required execution order

### Phase 1: Audit

Before changing code:

1. Check Git status.
2. Check Python environment and installed dependencies.
3. Run the existing dataset validator.
4. Verify audio paths from the ASR pool.
5. Verify current transcript-cache counts.
6. Verify duplicate cache keys.
7. Verify split and condition coverage.
8. Confirm all current Saaras failures are quota/path/configuration failures.
9. Inspect available GPU, CPU RAM, disk space, and FFmpeg availability.
10. Do not print `.env` contents or API keys.

### Phase 2: Smoke test local ASR

Before processing the full dataset:

1. Select a small deterministic smoke set:
   - one row from each available Indic language
   - at least one clean file
   - at least one noisy file
   - at least one `other` example
2. Run the local ASR.
3. Verify:
   - The audio loads successfully
   - The language code is accepted
   - A non-empty transcript is returned
   - The output is written to a separate cache
   - Latency is recorded
4. Save the smoke results.
5. If the smoke test fails, diagnose the exact dependency/model issue before bulk processing.

### Phase 3: Full local transcription

If the smoke test passes:

1. Process the frozen `synthetic_asr_pool.csv`.
2. Use bounded resumable batches.
3. Preserve successful cache rows.
4. Continue until all 14,790 rows are either successfully transcribed or explicitly failed.
5. Report:
   - Total rows
   - Successful rows
   - Failed rows
   - Coverage by split
   - Coverage by condition
   - Coverage by language
   - Average and percentile latency
   - Failure categories
6. Do not use test data for model training or calibration.

### Phase 4: Train the text classifier

Use only successful local-ASR transcripts from the permitted training split:

- Character TF-IDF
- N-gram range 3–5
- `min_df=2`
- `sublinear_tf=True`
- Balanced Logistic Regression
- Ten banking intents plus `other`

Use only calibration/dev rows to select the reject threshold.

The reject gate may use maximum predicted probability. Do not choose the threshold using final test rows.

### Phase 5: Evaluate

Evaluate only after training and threshold decisions are fixed.

Required evaluation groups:

- Clean final test
- Full robustness final test
- Per condition:
  - clean
  - gaussian 10 dB
  - gaussian 20 dB
  - traffic
  - office
  - telephone
  - unseen_combined
- `unseen_combined` must remain final-test-only.

Report separately:

- Overall accuracy
- In-scope accuracy
- Confusion matrix
- Coverage
- Abstention rate
- Overall `other` false-accept rate
- False-accept rate by `other_type`
- Recall at fixed operating points where estimable
- Confidence intervals
- ASR transcript coverage
- ASR latency
- Failure counts

Do not merge synthetic and public metrics.

Do not use untrusted IndicVoices candidates as supervised labels or trusted intent metrics.

### Phase 6: Documentation

Update only the directly relevant documentation after measured results exist:

- README.md
- CLAUDE.md
- dataset/README.md
- dataset/docs/DATA_READINESS_REPORT.md

Document:

- Saaras quota blocker
- Alternative provider name
- Model version
- License
- Hardware
- Resampling details
- Commands used
- Cache path
- Coverage
- Metrics
- Limitations
- The fact that alternative-ASR results are not Saaras results
- The fact that synthetic TTS does not establish real-caller performance

---

## Tests required

Add or update focused tests for:

- Cache-key uniqueness across conditions
- Resume behavior
- Provider-specific cache separation
- Empty transcript rejection
- Language-code mapping
- Audio-path resolution
- Split leakage
- Test-only `unseen_combined`
- Calibration threshold not using test data
- Metric calculations
- Failure logging

Run:

```powershell
conda run -n torch26 python scripts\validate_dataset.py
conda run -n torch26 python -m pytest tests -q
git diff --check
```

Use the actual configured Python environment; do not assume the global `python` command points to the correct environment.

---

## Safety and repository rules

- Never search for leaked API keys.
- Never print or commit credentials.
- Never modify `.env`.
- Never fabricate transcripts or metrics.
- Never substitute source utterance text for ASR output.
- Never claim a replacement ASR result is a Saaras result.
- Never use final test data for training, calibration, or model selection.
- Never delete existing Whisper work from the separate VoxIntel-R repository.
- Do not commit or push automatically.
- Do not modify unrelated projects or files.
- Keep generated audio/model binaries out of Git.
- Inspect `git diff` before finishing.

---

## Final response required

At the end, report:

1. Repository used
2. Files changed
3. ASR provider/model used
4. Exact command(s) executed
5. Smoke-test result
6. Total transcription coverage
7. Coverage by split, condition, and language
8. Classifier metrics, or the exact reason metrics are blocked
9. Tests and validators run
10. API/model/dependency blockers
11. Remaining work
12. Explicit statement that no secrets were exposed and no commit/push was performed

Do not report success unless the commands actually completed and the output was verified.
