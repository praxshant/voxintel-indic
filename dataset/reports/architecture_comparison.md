# Architecture Comparison: VoxIntel Indic

| Metric | ASR-cascade (char n-gram) | Direct audio (wav2vec2+MLP) |
|--------|-------|-------|
| **Architecture** | | |
| Encoder / features | char-ngram TF-IDF (3-5gram) + LogReg | facebook/wav2vec2-base-960h |
| Trainable params | N/A (sklearn) | 200203 |
| Train samples | 9720 | 9720 |
| Reject threshold | 0.1234 | 0.3600 |
| Latency / pipeline | Hybrid ASR (IndicConformer/Whisper) + TF-IDF + LogReg | GPU inference (frozen encoder + MLP head) |
| **Clean Final Test** | | |
| Accuracy | 100.00% | 57.49% |
| Coverage | 76.92% | 94.10% |
| In-scope accuracy | 100.00% | 54.48% |
| False-accept (other) | 0.00% | 32.95% |
| 95% CI | [1.0, 1.0] | [0.5232, 0.6267] |
| **Robustness Final Test** | | |
| Accuracy | 99.96% | 55.92% |
| Coverage | 76.92% | 92.56% |
| In-scope accuracy | 99.94% | 51.66% |
| False-accept (other) | 0.00% | 31.15% |
| **Unseen Combined (test-only)** | | |
| Accuracy | 99.74% | 55.40% |
| Coverage | 76.92% | 92.56% |
| **Risk-Coverage** | | |
| AURC (↓ better) | 0.000000 | 0.259047 |
| Sel. accuracy @80% cov | N/A | 59.80% |
| Sel. accuracy @90% cov | N/A | 56.94% |
| Sel. accuracy @95% cov | N/A | 55.09% |

### Per-condition accuracy (accepted predictions)
| Condition | ASR-cascade (char n-gram) | Direct audio (wav2vec2+MLP) |
|-----------|-------|-------|
| clean | 100.00% | 57.49% |
| gaussian_snr10 | 100.00% | 55.56% |
| gaussian_snr20 | 100.00% | 55.56% |
| office_snr15 | 100.00% | 55.99% |
| telephone_channel | 100.00% | 55.40% |
| traffic_snr15 | 100.00% | 55.99% |
| unseen_combined | 99.74% | 55.40% |

---

**Notes:**
- ASR-cascade: hybrid IndicConformer+Whisper ASR -> char 3-5gram TF-IDF -> Logistic Regression
- Direct audio: frozen facebook/wav2vec2-base-960h encoder -> mean pool -> 2-layer MLP
- Both systems use the same frozen splits (train/dev/test), identical threshold selection on dev only.
- Audio is synthetic TTS at 8 kHz, resampled to 16 kHz (does not restore missing high-frequency content).
- IndicVoices public labels excluded (unverified). Only synthetic ground truth used.
- unseen_combined (390 rows) is strictly test-only in both systems.
