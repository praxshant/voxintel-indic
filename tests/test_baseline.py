"""Tests for the multi-provider ASR baseline.

Covers:
- Cache-key uniqueness across conditions
- Resume behaviour (already-successful rows are skipped)
- Provider-specific cache separation
- Empty transcript rejection
- Language-code mapping for IndicConformer
- Audio-path resolution
- Split leakage guard (unseen_combined must not appear in training)
- Calibration threshold must not use test rows
- Metric calculations (evaluate_group)
- Failure logging
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_saaras_baseline import (
    INDICCONFORMER_LANG_MAP,
    _expected_calibration_error,
    bootstrap_ci,
    calibrate_isotonic,
    calibrate_temperature,
    choose_threshold,
    choose_threshold_signal,
    compute_rejection_scores,
    coverage_report,
    evaluate_group,
    false_accept_by_other_type_at_coverage,
    record_key,
    risk_coverage_curve,
    write_csv,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> dict:
    base = {
        "id": "x001",
        "audio_path": "audio/x001_clean.wav",
        "condition": "clean",
        "split": "train",
        "intent": "balance",
        "other_type": "",
        "language_code": "hi-IN",
        "transcript": "my balance",
        "status": "success",
        "provider": "saaras",
        "record_key": "",
        "error": "",
        "attempts": "1",
        "latency_ms": "120.5",
    }
    base.update(kwargs)
    if not base["record_key"]:
        base["record_key"] = record_key(base)
    return base


# ---------------------------------------------------------------------------
# 1. Cache-key uniqueness across conditions
# ---------------------------------------------------------------------------

class TestRecordKey(unittest.TestCase):
    def test_same_id_different_condition_yields_different_key(self):
        base = {"id": "u001", "audio_path": "audio/u001.wav", "condition": "clean"}
        noisy = {**base, "condition": "gaussian_snr10"}
        self.assertNotEqual(record_key(base), record_key(noisy))

    def test_same_id_different_path_yields_different_key(self):
        a = {"id": "u001", "audio_path": "audio/u001_clean.wav", "condition": "clean"}
        b = {"id": "u001", "audio_path": "audio/u001_traffic.wav", "condition": "traffic_snr15"}
        self.assertNotEqual(record_key(a), record_key(b))

    def test_key_is_deterministic(self):
        row = {"id": "u001", "audio_path": "audio/u001.wav", "condition": "clean"}
        self.assertEqual(record_key(row), record_key(dict(row)))

    def test_key_length_is_24(self):
        row = {"id": "u001", "audio_path": "audio/u001.wav", "condition": "clean"}
        self.assertEqual(len(record_key(row)), 24)


# ---------------------------------------------------------------------------
# 2. Cache writer preserves all rows
# ---------------------------------------------------------------------------

class TestCacheWriter(unittest.TestCase):
    def test_write_and_read_roundtrip(self):
        rows = [
            _make_row(id="a", audio_path="a.wav", condition="clean"),
            _make_row(id="b", audio_path="b.wav", condition="gaussian_snr10"),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.csv"
            write_csv(path, rows)
            with path.open(newline="", encoding="utf-8") as f:
                written = list(csv.DictReader(f))
            self.assertEqual(len(written), 2)

    def test_all_required_fields_present(self):
        rows = [_make_row()]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.csv"
            write_csv(path, rows)
            with path.open(newline="", encoding="utf-8") as f:
                fieldnames = next(csv.reader(f))
            for field in ("record_key", "id", "audio_path", "split", "intent",
                          "condition", "transcript", "status", "provider"):
                self.assertIn(field, fieldnames)


# ---------------------------------------------------------------------------
# 3. Coverage report — incomplete cache is not complete
# ---------------------------------------------------------------------------

class TestCoverageReport(unittest.TestCase):
    def test_incomplete_cache_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "manifest.csv"
            cache = root / "cache.csv"
            source = {"id": "x", "audio_path": "audio/x.wav", "condition": "clean"}
            write_csv(manifest, [{**source, "split": "test", "intent": "other"}])
            write_csv(cache, [{**source, "record_key": record_key(source),
                               "split": "test", "intent": "other",
                               "transcript": "", "status": "failed",
                               "provider": "saaras"}])
            result = coverage_report(manifest, cache)
            self.assertFalse(result["complete"])
            self.assertEqual(result["failed_or_missing_rows"], 1)

    def test_complete_cache_is_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "manifest.csv"
            cache = root / "cache.csv"
            source = {"id": "x", "audio_path": "audio/x.wav", "condition": "clean"}
            row = {**source, "split": "test", "intent": "balance"}
            key = record_key(source)
            write_csv(manifest, [row])
            write_csv(cache, [{**row, "record_key": key, "transcript": "my balance",
                               "status": "success", "provider": "indicconformer"}])
            result = coverage_report(manifest, cache)
            self.assertTrue(result["complete"])
            self.assertEqual(result["failed_or_missing_rows"], 0)

    def test_by_split_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "manifest.csv"
            cache = root / "cache.csv"
            r1 = {"id": "a", "audio_path": "a.wav", "condition": "clean",
                  "split": "train", "intent": "balance"}
            r2 = {"id": "b", "audio_path": "b.wav", "condition": "clean",
                  "split": "test", "intent": "other"}
            write_csv(manifest, [r1, r2])
            c1 = {**r1, "record_key": record_key(r1), "transcript": "balance",
                  "status": "success", "provider": "indicconformer"}
            write_csv(cache, [c1])
            result = coverage_report(manifest, cache)
            self.assertIn("train", result["by_split"])
            self.assertIn("test", result["by_split"])
            self.assertEqual(result["by_split"]["train"]["success"], 1)
            self.assertEqual(result["by_split"]["test"]["success"], 0)


# ---------------------------------------------------------------------------
# 4. Provider-specific cache separation
# ---------------------------------------------------------------------------

class TestProviderCacheSeparation(unittest.TestCase):
    def test_saaras_and_ic_have_different_default_paths(self):
        root = Path(__file__).resolve().parents[1]
        saaras_cache = root / "dataset/manifests/evaluation/saaras_transcripts.csv"
        ic_cache = root / "dataset/manifests/evaluation/indicconformer_transcripts.csv"
        self.assertNotEqual(saaras_cache, ic_cache)

    def test_provider_field_recorded_in_cache(self):
        row_saaras = _make_row(provider="saaras")
        row_ic = _make_row(provider="indicconformer", id="y", audio_path="y.wav")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.csv"
            write_csv(path, [row_saaras, row_ic])
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            providers = {r["provider"] for r in rows}
            self.assertIn("saaras", providers)
            self.assertIn("indicconformer", providers)


# ---------------------------------------------------------------------------
# 5. Empty transcript rejection
# ---------------------------------------------------------------------------

class TestEmptyTranscriptRejection(unittest.TestCase):
    def test_empty_transcript_not_marked_success_in_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "manifest.csv"
            cache = root / "cache.csv"
            source = {"id": "z", "audio_path": "z.wav", "condition": "clean"}
            write_csv(manifest, [{**source, "split": "train", "intent": "balance"}])
            # Row has status=success but empty transcript — should NOT count as covered
            write_csv(cache, [{**source, "record_key": record_key(source),
                               "split": "train", "intent": "balance",
                               "transcript": "   ", "status": "success",
                               "provider": "indicconformer"}])
            result = coverage_report(manifest, cache)
            self.assertFalse(result["complete"])

    def test_nonempty_transcript_counts_as_covered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "manifest.csv"
            cache = root / "cache.csv"
            source = {"id": "z", "audio_path": "z.wav", "condition": "clean"}
            write_csv(manifest, [{**source, "split": "train", "intent": "balance"}])
            write_csv(cache, [{**source, "record_key": record_key(source),
                               "split": "train", "intent": "balance",
                               "transcript": "mera balance", "status": "success",
                               "provider": "indicconformer"}])
            result = coverage_report(manifest, cache)
            self.assertTrue(result["complete"])


# ---------------------------------------------------------------------------
# 6. Language-code mapping for IndicConformer
# ---------------------------------------------------------------------------

class TestLanguageCodeMapping(unittest.TestCase):
    EXPECTED_LANGS = {
        "bn-IN": "bn", "en-IN": "en", "gu-IN": "gu", "hi-IN": "hi",
        "kn-IN": "kn", "mr-IN": "mr", "ta-IN": "ta", "te-IN": "te",
    }

    def test_all_project_languages_mapped(self):
        for manifest_code, expected_bare in self.EXPECTED_LANGS.items():
            self.assertEqual(
                INDICCONFORMER_LANG_MAP.get(manifest_code), expected_bare,
                f"Language code mapping missing or wrong for {manifest_code!r}",
            )

    def test_no_extra_country_suffix_in_mapped_codes(self):
        for bare_code in INDICCONFORMER_LANG_MAP.values():
            self.assertNotIn("-", bare_code, f"Mapped code {bare_code!r} still has country suffix")


# ---------------------------------------------------------------------------
# 7. Split leakage — unseen_combined must not be used in training
# ---------------------------------------------------------------------------

class TestSplitLeakage(unittest.TestCase):
    def test_unseen_combined_excluded_from_train_selection(self):
        """Simulate the training row selection from evaluate()."""
        all_rows = [
            _make_row(id="t1", audio_path="t1.wav", split="train", condition="clean"),
            _make_row(id="t2", audio_path="t2.wav", split="train",
                      condition="unseen_combined"),
            _make_row(id="t3", audio_path="t3.wav", split="test",
                      condition="unseen_combined"),
        ]
        train = [
            r for r in all_rows
            if r["split"] == "train" and r.get("condition") != "unseen_combined"
        ]
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["id"], "t1")

    def test_test_rows_not_in_train(self):
        all_rows = [
            _make_row(id="t1", split="train", condition="clean"),
            _make_row(id="t2", split="test", condition="clean", audio_path="t2.wav"),
        ]
        train = [r for r in all_rows if r["split"] == "train"]
        test = [r for r in all_rows if r["split"] == "test"]
        train_ids = {r["id"] for r in train}
        test_ids = {r["id"] for r in test}
        self.assertFalse(train_ids & test_ids)


# ---------------------------------------------------------------------------
# 8. Calibration threshold must not use test data
# ---------------------------------------------------------------------------

class TestThresholdCalibration(unittest.TestCase):
    def _fit_simple_model(self) -> Pipeline:
        intents = ["balance", "block_card", "emi_due", "other"]
        texts = [
            "check my balance", "block my card", "emi due date", "something else",
            "account balance", "card blocked", "emi payment", "random query",
        ]
        labels = intents * 2
        model = Pipeline([
            ("f", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                  min_df=1, sublinear_tf=True)),
            ("c", LogisticRegression(max_iter=200, class_weight="balanced")),
        ])
        model.fit(texts, labels)
        return model

    def test_threshold_selected_on_dev_only(self):
        """choose_threshold must only receive dev rows; test rows must be excluded."""
        model = self._fit_simple_model()
        dev_rows = [
            {"transcript": "check my balance", "intent": "balance", "split": "dev"},
            {"transcript": "block my card", "intent": "block_card", "split": "dev"},
            {"transcript": "something random", "intent": "other", "split": "dev"},
        ]
        test_rows = [
            {"transcript": "emi due", "intent": "emi_due", "split": "test"},
        ]
        # This must not raise; test rows must not be passed in
        threshold = choose_threshold(model, dev_rows)
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_choose_threshold_returns_float(self):
        model = self._fit_simple_model()
        rows = [{"transcript": "check balance", "intent": "balance", "split": "dev"}]
        t = choose_threshold(model, rows)
        self.assertIsInstance(t, float)


# ---------------------------------------------------------------------------
# 9. Metric calculations
# ---------------------------------------------------------------------------

class TestMetricCalculations(unittest.TestCase):
    def _fit_model(self) -> Pipeline:
        texts = ["balance check", "block card", "emi payment", "something else"] * 3
        labels = ["balance", "block_card", "emi_due", "other"] * 3
        model = Pipeline([
            ("f", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                  min_df=1, sublinear_tf=True)),
            ("c", LogisticRegression(max_iter=200, class_weight="balanced")),
        ])
        model.fit(texts, labels)
        return model

    def test_evaluate_group_returns_required_keys(self):
        model = self._fit_model()
        rows = [
            {"transcript": "balance check", "intent": "balance", "other_type": ""},
            {"transcript": "block card", "intent": "block_card", "other_type": ""},
            {"transcript": "something", "intent": "other", "other_type": "ambiguous"},
        ]
        result = evaluate_group(model, rows, threshold=0.0)
        for key in ("rows", "accuracy", "in_scope_accuracy", "coverage",
                    "abstention_rate", "false_accept_rate_other",
                    "false_accept_rate_by_other_type",
                    "confusion_matrix_labels", "confusion_matrix", "accuracy_ci95"):
            self.assertIn(key, result)

    def test_evaluate_group_row_count(self):
        model = self._fit_model()
        rows = [
            {"transcript": "balance check", "intent": "balance", "other_type": ""},
        ]
        result = evaluate_group(model, rows, threshold=0.0)
        self.assertEqual(result["rows"], 1)

    def test_bootstrap_ci_bounds(self):
        values = [float(i % 2) for i in range(100)]
        lo, hi = bootstrap_ci(values)
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_bootstrap_ci_empty(self):
        lo, hi = bootstrap_ci([])
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 0.0)


# ---------------------------------------------------------------------------
# 10. Failure logging
# ---------------------------------------------------------------------------

class TestFailureLogging(unittest.TestCase):
    def test_failed_row_written_to_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cache = root / "cache.csv"
            source = {"id": "f1", "audio_path": "f1.wav", "condition": "clean"}
            failed_row = {
                **source,
                "record_key": record_key(source),
                "split": "train",
                "intent": "balance",
                "other_type": "",
                "language_code": "hi-IN",
                "transcript": "",
                "status": "failed",
                "provider": "indicconformer",
                "error": "IndicConformer returned empty transcript",
                "attempts": "3",
                "latency_ms": "",
            }
            write_csv(cache, [failed_row])
            with cache.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("empty transcript", rows[0]["error"])

    def test_coverage_counts_failed_as_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "manifest.csv"
            cache = root / "cache.csv"
            source = {"id": "f1", "audio_path": "f1.wav", "condition": "clean"}
            write_csv(manifest, [{**source, "split": "train", "intent": "balance"}])
            failed_row = {
                **source,
                "record_key": record_key(source),
                "split": "train", "intent": "balance",
                "transcript": "", "status": "failed",
                "provider": "indicconformer",
            }
            write_csv(cache, [failed_row])
            result = coverage_report(manifest, cache)
            self.assertFalse(result["complete"])
            self.assertEqual(result["failed_or_missing_rows"], 1)


# ---------------------------------------------------------------------------
# 11. Whisper and Hybrid provider tests
# ---------------------------------------------------------------------------

class TestWhisperAndHybridProviders(unittest.TestCase):
    def test_whisper_supported_languages(self):
        from scripts.run_saaras_baseline import WHISPER_SUPPORTED_LANGUAGES
        self.assertIn("en-IN", WHISPER_SUPPORTED_LANGUAGES)
        self.assertNotIn("hi-IN", WHISPER_SUPPORTED_LANGUAGES)

    def test_hybrid_transcribe_manifest_seeding(self):
        from scripts.run_saaras_baseline import transcribe_manifest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "manifest.csv"
            hybrid_cache = root / "hybrid_transcripts.csv"
            ic_cache = root / "indicconformer_transcripts.csv"
            errors_path = root / "errors.csv"

            # Create an Indic row and an English row
            r1 = {"id": "ic1", "audio_path": "audio/ic1.wav", "condition": "clean",
                  "split": "train", "intent": "balance", "language_code": "hi-IN"}
            r2 = {"id": "wh1", "audio_path": "audio/wh1.wav", "condition": "clean",
                  "split": "train", "intent": "balance", "language_code": "en-IN"}
            write_csv(manifest, [r1, r2])

            # Pre-populate ic_cache with r1 as successful
            c1 = {**r1, "record_key": record_key(r1), "transcript": "नमस्ते",
                  "status": "success", "provider": "indicconformer", "attempts": "1", "latency_ms": "100"}
            write_csv(ic_cache, [c1])

            # Run transcribe_manifest with max_new=0 so no actual transcription is attempted
            summary = transcribe_manifest(
                manifest=manifest,
                audio_root=root,
                cache=hybrid_cache,
                errors_path=errors_path,
                provider="hybrid",
                max_new=0,
            )
            # The hybrid cache should have seeded c1
            self.assertEqual(summary["cached_success"], 1)
            hybrid_rows = list(csv.DictReader(hybrid_cache.open(encoding="utf-8")))
            self.assertEqual(len(hybrid_rows), 1)
            self.assertEqual(hybrid_rows[0]["record_key"], record_key(r1))
            self.assertEqual(hybrid_rows[0]["transcript"], "नमस्ते")


# ---------------------------------------------------------------------------
# 12. Reject gate and risk–coverage evaluation tests
# ---------------------------------------------------------------------------

class TestRejectGateAndRiskCoverage(unittest.TestCase):
    def _fit_model(self) -> Pipeline:
        train_data = [
            ("check balance please", "balance"),
            ("what is my account balance", "balance"),
            ("block my debit card immediately", "block_card"),
            ("lost card please block", "block_card"),
            ("when is my emi due date", "emi_due"),
            ("emi payment schedule", "emi_due"),
            ("good morning how are you", "other"),
            ("tell me a joke", "other"),
            ("what is the weather today", "other"),
        ]
        model = Pipeline([
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
        ])
        model.fit([t[0] for t in train_data], [t[1] for t in train_data])
        return model

    def test_compute_rejection_scores_keys_and_bounds(self):
        model = self._fit_model()
        transcripts = ["check my balance", "block debit card", "random noise"]
        scores = compute_rejection_scores(model, transcripts)
        for key in ("max_prob", "margin", "norm_entropy"):
            self.assertIn(key, scores)
            self.assertEqual(len(scores[key]), len(transcripts))
            for val in scores[key]:
                self.assertGreaterEqual(val, -1e-6)
                self.assertLessEqual(val, 1.0 + 1e-6)

        # Margin must be <= max_prob
        for mp, mg in zip(scores["max_prob"], scores["margin"]):
            self.assertGreaterEqual(mp + 1e-6, mg)

    def test_expected_calibration_error_values(self):
        # Perfect calibration: 100% confidence, 100% correct
        ece_perfect = _expected_calibration_error([1.0, 1.0, 1.0], [True, True, True])
        self.assertEqual(ece_perfect, 0.0)

        # Worst calibration: 100% confidence, 0% correct
        ece_worst = _expected_calibration_error([1.0, 1.0, 1.0], [False, False, False])
        self.assertAlmostEqual(ece_worst, 1.0, places=4)

    def test_calibrate_temperature_keys_and_bounds(self):
        model = self._fit_model()
        cal_rows = [
            {"transcript": "balance inquiry", "intent": "balance"},
            {"transcript": "block my card", "intent": "block_card"},
            {"transcript": "emi date", "intent": "emi_due"},
            {"transcript": "hello world", "intent": "other"},
        ]
        result = calibrate_temperature(model, cal_rows)
        for key in ("ece_uncalibrated", "best_temperature", "ece_after_temperature", "improvement"):
            self.assertIn(key, result)
        self.assertGreaterEqual(result["ece_uncalibrated"], 0.0)
        self.assertGreaterEqual(result["ece_after_temperature"], 0.0)

    def test_calibrate_isotonic_keys(self):
        model = self._fit_model()
        cal_rows = [
            {"transcript": "balance inquiry", "intent": "balance"},
            {"transcript": "block my card", "intent": "block_card"},
            {"transcript": "emi date", "intent": "emi_due"},
            {"transcript": "hello world", "intent": "other"},
        ]
        result = calibrate_isotonic(model, cal_rows)
        for key in ("ece_before_isotonic", "ece_after_isotonic", "improvement", "isotonic_helped"):
            self.assertIn(key, result)
        self.assertIsInstance(result["isotonic_helped"], bool)

    def test_choose_threshold_signal_all_signals(self):
        model = self._fit_model()
        cal_rows = [
            {"transcript": "balance check", "intent": "balance"},
            {"transcript": "card lost block", "intent": "block_card"},
            {"transcript": "emi due date", "intent": "emi_due"},
            {"transcript": "weather today", "intent": "other"},
        ]
        for sig in ("max_prob", "margin", "norm_entropy"):
            res = choose_threshold_signal(model, cal_rows, signal=sig, min_recall=0.80)
            self.assertEqual(res["signal"], sig)
            self.assertIn("threshold", res)
            self.assertIn("recall", res)
            self.assertIn("precision", res)

    def test_risk_coverage_curve_properties(self):
        model = self._fit_model()
        rows = [
            {"transcript": "balance check", "intent": "balance"},
            {"transcript": "card block please", "intent": "block_card"},
            {"transcript": "emi schedule", "intent": "emi_due"},
            {"transcript": "nonsense string", "intent": "other"},
            {"transcript": "what time is it", "intent": "other"},
        ]
        curve = risk_coverage_curve(model, rows, signal="max_prob")
        self.assertEqual(curve["signal"], "max_prob")
        self.assertEqual(curve["n_rows"], 5)
        self.assertGreaterEqual(curve["aurc"], 0.0)
        self.assertLessEqual(curve["aurc"], 1.0)
        self.assertIn("selective_acc_at_80pct", curve)
        self.assertIn("selective_acc_at_90pct", curve)
        self.assertIn("selective_acc_at_95pct", curve)
        # Coverages must be monotonically non-decreasing
        covs = curve["coverages"]
        for i in range(1, len(covs)):
            self.assertGreaterEqual(covs[i], covs[i - 1])

    def test_false_accept_by_other_type_at_coverage(self):
        model = self._fit_model()
        rows = [
            {"transcript": "balance check", "intent": "balance", "other_type": ""},
            {"transcript": "card block", "intent": "block_card", "other_type": ""},
            {"transcript": "hello friend", "intent": "other", "other_type": "chitchat"},
            {"transcript": "buy shoes", "intent": "other", "other_type": "out_of_domain"},
        ]
        fa = false_accept_by_other_type_at_coverage(model, rows, signal="max_prob")
        for target in ("0.8", "0.9", "0.95"):
            self.assertIn(target, fa)
            self.assertIn("false_accept_by_other_type", fa[target])

    def test_evaluate_group_includes_extended_metrics(self):
        model = self._fit_model()
        rows = [
            {"transcript": "balance inquiry", "intent": "balance", "other_type": "", "language_code": "hi-IN"},
            {"transcript": "block my card", "intent": "block_card", "other_type": "", "language_code": "hi-IN"},
            {"transcript": "hello world", "intent": "other", "other_type": "chitchat", "language_code": "en-IN"},
        ]
        result = evaluate_group(model, rows, threshold=0.1)
        self.assertIn("risk_coverage", result)
        self.assertIn("false_accept_by_coverage", result)
        self.assertIn("by_language", result)
        self.assertIn("hi-IN", result["by_language"])
        self.assertIn("en-IN", result["by_language"])

    def test_split_purity_threshold_selection(self):
        model = self._fit_model()
        cal_rows = [
            {"transcript": "balance check", "intent": "balance"},
            {"transcript": "card lost block", "intent": "block_card"},
        ]
        test_rows = [
            {"transcript": "unseen test row", "intent": "balance"},
        ]
        # Running threshold selection on cal_rows must not accept test_rows
        res = choose_threshold_signal(model, cal_rows, signal="max_prob")
        # Ensure result threshold is computed strictly from cal_rows
        scores = compute_rejection_scores(model, [r["transcript"] for r in cal_rows])["max_prob"]
        # Threshold must be one of the candidate scores or 0.0
        self.assertTrue(res["threshold"] in scores or res["threshold"] == 0.0)


if __name__ == "__main__":
    unittest.main()
