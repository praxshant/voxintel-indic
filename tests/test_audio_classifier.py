"""Unit tests for the direct audio-to-intent classifier and comparison scripts.

Covers:
- Split integrity: no train/test or dev/test overlap
- unseen_combined strictly test-only
- MLPPClassifier forward pass & probability shapes
- Reject gate threshold selection on dev split only
- AURC and risk-coverage monotonic behavior
- compare_architectures metric extraction and table generation
"""

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train_audio_classifier import (
    INTENTS,
    LABEL2IDX,
    ENCODER_CANDIDATES,
    MLP_HIDDEN_DIM,
    NUM_CLASSES,
    MLPClassifier,
    encoder_cache_tag,
    compute_aurc,
    load_splits,
    rc_at_targets,
    select_threshold_on_dev,
)
from scripts.compare_architectures import extract_metrics, build_markdown_table
from scripts.analyze_audio_errors import class_metrics, top_confusions


class TestAudioClassifierArchitecture(unittest.TestCase):
    def test_encoder_cache_tag_is_model_specific(self):
        self.assertIn("facebook/wav2vec2-xls-r-300m", ENCODER_CANDIDATES)
        self.assertNotEqual(
            encoder_cache_tag("facebook/wav2vec2-base-960h"),
            encoder_cache_tag("facebook/wav2vec2-xls-r-300m"),
        )

    def test_num_classes(self):
        self.assertEqual(NUM_CLASSES, 11)
        self.assertIn("other", INTENTS)
        self.assertIn("balance", INTENTS)

    def test_mlp_classifier_forward(self):
        head = MLPClassifier(in_dim=768, hidden_dim=MLP_HIDDEN_DIM, num_classes=NUM_CLASSES)
        dummy_feats = torch.randn(4, 768)
        logits = head(dummy_feats)
        self.assertEqual(logits.shape, (4, NUM_CLASSES))

        probs = head.predict_proba(dummy_feats)
        self.assertEqual(probs.shape, (4, NUM_CLASSES))
        # Sum of probabilities across classes should be 1
        sums = probs.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)


class TestSplitIntegrity(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        manifest_dir = self.repo_root / "dataset/manifests/evaluation"
        self.train_manifest = manifest_dir / "synthetic_train.csv"
        self.calibration_manifest = manifest_dir / "synthetic_calibration.csv"
        self.clean_test_manifest = manifest_dir / "synthetic_clean_final_test.csv"
        self.robust_test_manifest = manifest_dir / "synthetic_final_test.csv"
        self.transcripts = self.repo_root / "dataset/manifests/evaluation/hybrid_transcripts.csv"

    def load_frozen_splits(self):
        return load_splits(
            self.train_manifest,
            self.calibration_manifest,
            self.clean_test_manifest,
            self.robust_test_manifest,
            self.transcripts,
        )

    def test_no_split_leakage(self):
        if not self.transcripts.exists():
            self.skipTest("Manifests not found")
        train_rows, dev_rows, clean_test_rows, robust_test_rows = self.load_frozen_splits()
        train_ids = {r["id"] for r in train_rows}
        dev_ids = {r["id"] for r in dev_rows}
        test_ids = {r["id"] for r in clean_test_rows + robust_test_rows}

        self.assertEqual(len(train_rows), 8100)
        self.assertEqual(len(dev_rows), 1950)
        self.assertEqual(len(clean_test_rows), 390)
        self.assertEqual(len(robust_test_rows), 2340)
        self.assertEqual(len(train_ids & test_ids), 0, "Train and test sets overlap!")
        self.assertEqual(len(dev_ids & test_ids), 0, "Dev and test sets overlap!")
        self.assertEqual(len(train_ids & dev_ids), 0, "Train and dev sets overlap!")

    def test_unseen_combined_strictly_test_only(self):
        if not self.transcripts.exists():
            self.skipTest("Manifests not found")
        train_rows, dev_rows, clean_test_rows, robust_test_rows = self.load_frozen_splits()

        train_unseen = [r for r in train_rows if r.get("condition") == "unseen_combined"]
        dev_unseen = [r for r in dev_rows if r.get("condition") == "unseen_combined"]
        clean_unseen = [r for r in clean_test_rows if r.get("condition") == "unseen_combined"]
        test_unseen = [r for r in robust_test_rows if r.get("condition") == "unseen_combined"]

        self.assertEqual(len(train_unseen), 0, "unseen_combined found in training split!")
        self.assertEqual(len(dev_unseen), 0, "unseen_combined found in dev split!")
        self.assertEqual(len(clean_unseen), 0, "unseen_combined found in clean test!")
        self.assertGreater(len(test_unseen), 0, "unseen_combined missing from test split!")


class TestCalibrationAndRiskCoverage(unittest.TestCase):
    def test_threshold_selection_on_dev(self):
        # 100 dev samples with dummy probabilities
        np.random.seed(42)
        dev_feats = torch.randn(100, 768)
        dev_labels = np.random.randint(0, NUM_CLASSES, size=100)
        head = MLPClassifier(in_dim=768, hidden_dim=MLP_HIDDEN_DIM, num_classes=NUM_CLASSES)

        threshold, cal = select_threshold_on_dev(
            head, dev_feats, dev_labels, device=torch.device("cpu"), target_recall=0.95
        )
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertEqual(cal["split"], "dev")
        self.assertIn("dev_coverage", cal)

    def test_aurc_bounds(self):
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ])
        # Perfect predictions: labels match max prob argmax
        labels_perfect = np.array([0, 0, 0, 0])
        aurc_perfect = compute_aurc(probs, labels_perfect)
        self.assertEqual(aurc_perfect, 0.0)

        # Inverted predictions
        labels_wrong = np.array([1, 1, 1, 1])
        aurc_wrong = compute_aurc(probs, labels_wrong)
        self.assertGreater(aurc_wrong, 0.0)
        self.assertLessEqual(aurc_wrong, 1.0)

    def test_rc_at_targets(self):
        probs = np.ones((10, 2)) * 0.5
        probs[:, 0] = np.linspace(0.55, 0.95, 10)
        labels = np.zeros(10, dtype=int)
        rc = rc_at_targets(probs, labels, targets=(0.8, 0.9))
        self.assertIn("selective_acc_at_80pct_coverage", rc)
        self.assertIn("selective_acc_at_90pct_coverage", rc)
        self.assertEqual(rc["selective_acc_at_80pct_coverage"], 1.0)


class TestComparisonScript(unittest.TestCase):
    def test_versioned_audio_schema_is_supported(self):
        report = {
            "schema_version": "audio_classifier_v3",
            "model": {"encoder": "frozen-encoder", "n_params_trainable": 1},
            "calibration": {"threshold": 0.4},
            "data": {"n_train": 10},
            "evaluation": {
                "clean_final_test": {"accuracy": 0.5},
                "full_robustness_test": {"accuracy": 0.4},
                "per_condition": {},
                "aurc_full_test": 0.2,
            },
        }
        metrics = extract_metrics(report, "Direct audio")
        self.assertEqual(metrics["clean_accuracy"], 0.5)
        self.assertEqual(metrics["robust_accuracy"], 0.4)

    def test_markdown_table_generation(self):
        m1 = {
            "system": "ASR-cascade",
            "encoder": "char-ngram",
            "n_trainable_params": "N/A",
            "n_train": 9720,
            "threshold": 0.1234,
            "latency_note": "hybrid",
            "clean_accuracy": 1.0,
            "clean_coverage": 0.7692,
            "clean_in_scope_acc": 1.0,
            "clean_false_accept_other": 0.0,
            "clean_ci": [1.0, 1.0],
            "robust_accuracy": 0.9996,
            "robust_coverage": 0.7692,
            "robust_in_scope_acc": 0.9994,
            "robust_false_accept_other": 0.0,
            "robust_ci": [0.9987, 1.0],
            "unseen_accuracy": 0.9974,
            "unseen_coverage": 0.7692,
            "aurc": 0.0,
            "selective_acc_80pct": 1.0,
            "selective_acc_90pct": 1.0,
            "selective_acc_95pct": 1.0,
            "per_condition": {"clean": 1.0, "gaussian_snr10": 1.0},
        }
        m2 = {
            "system": "Direct audio",
            "encoder": "wav2vec2-base",
            "n_trainable_params": 200203,
            "n_train": 9720,
            "threshold": 0.36,
            "latency_note": "gpu",
            "clean_accuracy": 0.5749,
            "clean_coverage": 0.941,
            "clean_in_scope_acc": 0.5448,
            "clean_false_accept_other": 0.3295,
            "clean_ci": [0.5232, 0.6267],
            "robust_accuracy": 0.5592,
            "robust_coverage": 0.9256,
            "robust_in_scope_acc": 0.5166,
            "robust_false_accept_other": 0.3115,
            "robust_ci": [0.53, 0.58],
            "unseen_accuracy": 0.5540,
            "unseen_coverage": 0.9256,
            "aurc": 0.259047,
            "selective_acc_80pct": 0.598,
            "selective_acc_90pct": 0.5694,
            "selective_acc_95pct": 0.5509,
            "per_condition": {"clean": 0.5749, "gaussian_snr10": 0.5556},
        }
        md = build_markdown_table(m1, m2)
        self.assertIn("Clean Final Test", md)
        self.assertIn("ASR-cascade", md)
        self.assertIn("Direct audio", md)
        self.assertIn("100.00%", md)
        self.assertIn("57.49%", md)


class TestAudioErrorAnalysis(unittest.TestCase):
    def test_confusions_and_class_metrics(self):
        matrix = [[4, 2], [1, 3]]
        metrics = {row["label"]: row for row in class_metrics(matrix, ["a", "b"])}
        self.assertEqual(metrics["a"]["support"], 6)
        self.assertEqual(metrics["b"]["recall"], 0.75)
        self.assertEqual(
            top_confusions(matrix, ["a", "b"], limit=1),
            [{"actual": "a", "predicted": "b", "count": 2}],
        )


if __name__ == "__main__":
    unittest.main()
