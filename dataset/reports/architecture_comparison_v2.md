# Architecture Comparison: VoxIntel Indic

| Metric | ASR-cascade (char n-gram) | Direct audio (wav2vec2+MLP) |
|--------|-------|-------|
| **Architecture** | | |
| Encoder / features | char-ngram TF-IDF (3-5gram) + LogReg | facebook/wav2vec2-base-960h |
| Trainable params | N/A (sklearn) | 200203 |
| Train samples | 9720 | 8100 |
| Reject threshold | 0.1234 | 0.3500 |
| Latency / pipeline | Hybrid ASR (IndicConformer/Whisper) + TF-IDF + LogReg | GPU inference (frozen encoder + MLP head) |
| **Clean Final Test** | | |
| Accuracy | 100.00% | 64.80% |
| Coverage | 76.92% | 96.15% |
| In-scope accuracy | 100.00% | 62.24% |
| False-accept (other) | 0.00% | 26.97% |
| 95% CI | [1.0, 1.0] | [0.5973, 0.6933] |
| **Robustness Final Test** | | |
| Accuracy | 99.96% | 69.87% |
| Coverage | 76.92% | 96.15% |
| In-scope accuracy | 99.94% | 71.78% |
| False-accept (other) | 0.00% | 36.56% |
| **Unseen Combined (test-only)** | | |
| Accuracy | 99.74% | 73.02% |
| Coverage | 76.92% | 96.92% |
| **Risk-Coverage** | | |
| AURC (↓ better) | 0.000000 | 0.137886 |
| Sel. accuracy @80% cov | N/A | 76.23% |
| Sel. accuracy @90% cov | N/A | 72.79% |
| Sel. accuracy @95% cov | N/A | 70.31% |

### Per-condition accuracy (accepted predictions)
| Condition | ASR-cascade (char n-gram) | Direct audio (wav2vec2+MLP) |
|-----------|-------|-------|
| clean | 100.00% | N/A |
| gaussian_snr10 | 100.00% | 58.95% |
| gaussian_snr20 | 100.00% | 72.22% |
| office_snr15 | 100.00% | 75.80% |
| telephone_channel | 100.00% | 71.24% |
| traffic_snr15 | 100.00% | 67.55% |
| unseen_combined | 99.74% | 73.02% |

---

**Notes:**
- ASR-cascade: hybrid IndicConformer+Whisper ASR -> char 3-5gram TF-IDF -> Logistic Regression
- Direct audio: frozen facebook/wav2vec2-base-960h encoder -> mean pool -> 2-layer MLP
- Both systems use the same frozen splits (train/dev/test), identical threshold selection on dev only.
- Audio is synthetic TTS at 8 kHz, resampled to 16 kHz (does not restore missing high-frequency content).
- IndicVoices public labels excluded (unverified). Only synthetic ground truth used.
- unseen_combined (390 rows) is strictly test-only in both systems.
