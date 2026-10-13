# IndicVoices Banking-Relevant Subset

This directory contains the locally filtered subset of
[ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)
for the VoxIntel Indic assignment.

## Contents

- `audio/`: 1,915 downloaded FLAC recordings
- `manifest.jsonl`: complete source annotations plus assignment labels
- `manifest.csv`: tabular version of the manifest
- `summary.json`: filtering scope and counts

Every audio record has these assignment-compatible fields:

- `id`
- `intent`
- `language_code`
- `text`
- `other_type`
- `source`
- `script`
- `mix`

Additional fields retain IndicVoices provenance, speaker metadata, source split,
matched keyword categories, audio format, and SHA-256 checksum.

## Scope

The source CSV languages were mapped as follows:

| CSV code | IndicVoices config |
|---|---|
| `bn-IN` | Bengali |
| `gu-IN` | Gujarati |
| `hi-IN` | Hindi |
| `kn-IN` | Kannada |
| `mr-IN` | Marathi |
| `ta-IN` | Tamil |
| `te-IN` | Telugu |

`en-IN` was skipped because IndicVoices does not provide an English config.
`und` was skipped because it is unspecified. No unrelated language was
downloaded.

Only the IndicVoices `valid` split was downloaded in this pass. Records were
kept when their transcript, normalized transcript, verbatim transcript,
unsanitized transcript, or task name contained at least one assignment-related
banking, account, transaction, card, payment, bill, EMI, fraud, complaint,
customer-support, branch, or profile-update keyword.

## Intent assignment

The intent labels are conservative mappings into the frozen assignment
taxonomy. They are generated from annotation keywords, not claimed as
human-verified intent labels:

- `balance`
- `recent_transactions`
- `block_card`
- `emi_due`
- `report_fraud`
- `complaint_status`
- `update_details`
- `branch_timing`
- `pay_bill`
- `transfer_to_agent`
- `other`

Ambiguous matches are assigned to `other` with an `other_type` such as
`ambiguous_in_domain`, `unsupported_near_miss`, or a blank value when the
assignment is specific enough. Review these labels before using them as
supervised training data. The original IndicVoices annotation fields remain
available for audit.

## Reproduction

Run with the `torch26` environment and a Hugging Face token in `.env`:

```powershell
$env:HF_TOKEN = (Get-Content .env | `
  Where-Object { $_ -match '^HF_TOKEN=' }) -replace '^HF_TOKEN=', ''

& 'C:\Users\ACER\anaconda3\envs\torch26\python.exe' -X utf8 `
  scripts\filter_indicvoices.py `
  --utterances utterances_v0.2.csv `
  --output indicvoices_filtered `
  --splits valid
```

The original dataset is not modified or deleted. Only non-matching records
were excluded from this local filtered output.

IndicVoices is released under CC BY 4.0. Cite the IndicVoices ACL Findings
paper when distributing or reporting results based on this subset.
