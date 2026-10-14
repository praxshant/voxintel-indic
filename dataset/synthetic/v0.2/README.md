# VoxIntel Indic Dataset v0.2

This directory contains the generated speech dataset for the Indic speech-to-intent pilot.

## Contents

- `audio/`: mono WAV files at 8 kHz, 16-bit PCM
- `manifest.csv`: tabular labels and provenance
- `manifest.jsonl`: one JSON record per audio file
- `dataset_summary.json`: generation configuration and split counts

## Generation

- Source labels: `../../source/utterances_v0.2.csv`
- TTS: Sarvam Bulbul v3 REST API
- TTS voices:
  - `shubh`: training/dev voice
  - `simran`: held-out test voice
- Requested sample rate: 8,000 Hz
- Variants: two per TTS utterance
- `non_005`: locally generated two-second silence
- `non_006`: locally generated deterministic two-second low-level noise

The API key is not stored in this directory. The generator reads `SARVAM_API_KEY`
from the environment or the project `.env` file.

## Splits

The held-out `simran` voice is assigned to `test`. `shubh` files are split
deterministically into `train` and `dev`. This makes the test voice-disjoint, but
it does not make the text fully paraphrase-disjoint because v0.2 contains only
one text row per utterance ID. A larger release must split by paraphrase cluster,
speaker, and voice before training.

## Important limitations

- This is a TTS pilot, not evidence of natural telephone-speech performance.
- The dataset has only 98 TTS source rows and is too small for final model claims.
- Gujarati and Kannada have one TTS row each; language coverage is exploratory.
- Real speaker recordings and IndicVoices non-intent speech are not included yet.
- `non_003` and `non_004` remain labeled as `non_intent_audio` according to the
  source CSV; review these manually before using them in a final benchmark.

Use the SHA-256 values in the manifests to verify that audio files have not changed.
