# Architecture Comparison: VoxIntel Indic

| Metric | ASR-cascade (char n-gram) | Direct audio (wav2vec2+MLP) |
|--------|-------|-------|
| **Architecture** | | |
| Encoder / features | char-ngram TF-IDF (3-5gram) + LogReg | facebook/wav2vec2-base-960h |
| Trainable params | N/A (sklearn) | 200203 |
| Train samples | 9720 | 8100 |
| Reject threshold | 0.1234 | 0.2450 |
| Latency / pipeline | Hybrid ASR (IndicConformer/Whisper) + TF-IDF + LogReg | GPU inference (frozen encoder + MLP head) |
| **Clean Final Test** | | |
| Accuracy | 100.00% | 62.30% |
| Coverage | 76.92% | 95.90% |
| In-scope accuracy | 100.00% | 64.83% |
| False-accept (other) | 0.00% | 46.43% |
| 95% CI | [1.0, 1.0] | [0.5748, 0.6738] |
| **Robustness Final Test** | | |
| Accuracy | 99.96% | 64.34% |
| Coverage | 76.92% | 95.98% |
| In-scope accuracy | 99.94% | 71.73% |
| False-accept (other) | 0.00% | 60.90% |
| **Unseen Combined (test-only)** | | |
| Accuracy | 99.74% | 64.86% |
| Coverage | 76.92% | 94.87% |
| **Risk-Coverage** | | |
| AURC (↓ better) | 0.000000 | 0.183962 |
| Sel. accuracy @80% cov | N/A | 69.98% |
| Sel. accuracy @90% cov | N/A | 66.71% |
| Sel. accuracy @95% cov | N/A | 64.73% |

### Per-condition accuracy (accepted predictions)
| Condition | ASR-cascade (char n-gram) | Direct audio (wav2vec2+MLP) |
|-----------|-------|-------|
| clean | 100.00% | N/A |
| gaussian_snr10 | 100.00% | 55.49% |
| gaussian_snr20 | 100.00% | 67.56% |
| office_snr15 | 100.00% | 68.17% |
| telephone_channel | 100.00% | 64.66% |
| traffic_snr15 | 100.00% | 65.00% |
| unseen_combined | 99.74% | 64.86% |

---

**Notes:**
- ASR-cascade: hybrid IndicConformer+Whisper ASR -> char 3-5gram TF-IDF -> Logistic Regression
- Direct audio: frozen facebook/wav2vec2-base-960h encoder -> mean pool -> 2-layer MLP
- Both systems use the same frozen splits (train/dev/test), identical threshold selection on dev only.
- Audio is synthetic TTS at 8 kHz, resampled to 16 kHz (does not restore missing high-frequency content).
- IndicVoices public labels excluded (unverified). Only synthetic ground truth used.
- unseen_combined (390 rows) is strictly test-only in both systems.
