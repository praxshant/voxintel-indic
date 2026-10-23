"""Train a direct audio → intent classifier using frozen speech encoder features.

Architecture:
  Frozen speech encoder (facebook/wav2vec2-base-960h or whisper-base.en)
  → Mean-pooled frame representations (768-dim)
  → 2-layer MLP classification head (768 → 256 → 11)
  → 11 classes: 10 banking intents + explicit "other"

Key optimizations:
  - Features are extracted once with the frozen encoder in eval mode (torch.no_grad).
  - Pre-extracted features are cached to dataset/models/features_{split}.pt.
  - Head training runs on GPU in seconds with balanced class weights.
  - Reject gate threshold is selected exclusively on the dev (calibration) split.
  - Final evaluation reads test data only once, keeping unseen_combined strictly test-only.

Usage:
  conda run -n torch26 python scripts/train_audio_classifier.py --epochs 30 --batch-size 32 --lr 1e-3

Outputs:
  dataset/models/audio_classifier_v2.pt    — trained MLP head weights + metadata
  dataset/reports/audio_classifier_v2.json — complete evaluation report
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENTS = sorted([
    "balance", "block_card", "branch_timing", "complaint_status",
    "emi_due", "other", "pay_bill", "recent_transactions",
    "report_fraud", "transfer_to_agent", "update_details",
])
LABEL2IDX: dict[str, int] = {label: i for i, label in enumerate(INTENTS)}
IDX2LABEL: dict[int, str] = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(INTENTS)

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 15.0
MAX_SAMPLES = int(TARGET_SR * MAX_AUDIO_SECONDS)
MLP_HIDDEN_DIM = 256

ENCODER_CANDIDATES = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-xls-r-300m",
    "openai/whisper-base.en",
]


# ---------------------------------------------------------------------------
# Encoder backend abstraction
# ---------------------------------------------------------------------------

class EncoderBackend:
    def __init__(self, model_id: str, encoder_dim: int):
        self.model_id = model_id
        self.encoder_dim = encoder_dim

    def extract_features(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class Wav2Vec2Backend(EncoderBackend):
    def __init__(self, model_id: str, model):
        super().__init__(model_id, model.config.hidden_size)
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def extract_features(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(waveforms, attention_mask=attention_mask)
            hs = out.last_hidden_state  # (B, T_frames, D)

        B, T_input = attention_mask.shape
        T_frames = hs.shape[1]
        if T_frames == 0:
            return hs.mean(dim=1)

        stride = T_input / T_frames
        input_lens = attention_mask.sum(dim=1).float()
        frame_counts = (input_lens / stride).long().clamp(1, T_frames)

        pooled = torch.zeros(B, hs.shape[2], device=hs.device, dtype=hs.dtype)
        for i, count in enumerate(frame_counts):
            pooled[i] = hs[i, :count].mean(dim=0)
        return pooled

    def to(self, device):
        self.model = self.model.to(device)
        return self


class WhisperEncoderBackend(EncoderBackend):
    def __init__(self, model_id: str, model, feature_extractor):
        super().__init__(model_id, model.config.d_model)
        self.encoder = model.encoder
        self.encoder.eval()
        self.feature_extractor = feature_extractor
        for p in self.encoder.parameters():
            p.requires_grad = False

    def extract_features(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        B = waveforms.shape[0]
        device = waveforms.device

        mel_batch = []
        for i in range(B):
            wav_np = waveforms[i].cpu().float().numpy()
            target_len = 480_000
            if len(wav_np) < target_len:
                wav_np = np.pad(wav_np, (0, target_len - len(wav_np)))
            else:
                wav_np = wav_np[:target_len]
            inputs = self.feature_extractor(
                wav_np, sampling_rate=16000, return_tensors="pt"
            )
            mel_batch.append(inputs["input_features"].squeeze(0))

        mel_tensor = torch.stack(mel_batch, dim=0).to(device)
        with torch.no_grad():
            out = self.encoder(mel_tensor)
            hs = out.last_hidden_state

        T_frames = hs.shape[1]
        input_lens = attention_mask.sum(dim=1).float()
        stride_samples = 480_000 / T_frames
        frame_counts = (input_lens / stride_samples).long().clamp(1, T_frames)

        pooled = torch.zeros(B, hs.shape[2], device=device, dtype=hs.dtype)
        for i, count in enumerate(frame_counts):
            pooled[i] = hs[i, :count].mean(dim=0)
        return pooled

    def to(self, device):
        self.encoder = self.encoder.to(device)
        return self


import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


def load_encoder(model_id: str) -> EncoderBackend:
    """Load one explicitly selected locally cached speech encoder."""
    from transformers import Wav2Vec2Model, WhisperFeatureExtractor, WhisperModel

    if model_id not in ENCODER_CANDIDATES:
        raise ValueError(f"Unsupported encoder: {model_id}. Choose one of {ENCODER_CANDIDATES}")
    try:
        print(f"  Attempting local load for {model_id}...")
        if "whisper" in model_id.lower():
            m = WhisperModel.from_pretrained(model_id, local_files_only=True)
            fe = WhisperFeatureExtractor.from_pretrained(model_id, local_files_only=True)
            backend = WhisperEncoderBackend(model_id, m, fe)
            print(f"  [OK] Loaded Whisper encoder: d_model={backend.encoder_dim}")
            return backend
        m = Wav2Vec2Model.from_pretrained(model_id, local_files_only=True)
        backend = Wav2Vec2Backend(model_id, m)
        print(f"  [OK] Loaded wav2vec2 encoder: hidden_size={backend.encoder_dim}")
        return backend
    except Exception as e:
        raise RuntimeError(
            f"Could not load selected encoder {model_id} from the local cache: "
            f"{type(e).__name__}: {str(e)[:240]}"
        ) from e


def encoder_cache_tag(model_id: str) -> str:
    """Return a filesystem-safe tag that prevents cross-encoder cache reuse."""
    return model_id.replace("/", "__").replace("-", "_")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def resolve_audio(audio_path: str, audio_roots: list[Path]) -> Optional[Path]:
    for root in audio_roots:
        candidate = root / audio_path
        if candidate.exists():
            return candidate
    return None


def load_waveform(audio_path: str, audio_roots: list[Path]) -> Optional[torch.Tensor]:
    resolved = resolve_audio(audio_path, audio_roots)
    if resolved is None:
        return None
    try:
        waveform, sr = torchaudio.load(str(resolved))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.squeeze(0).float()
        if waveform.shape[0] > MAX_SAMPLES:
            waveform = waveform[:MAX_SAMPLES]
        return waveform
    except Exception as e:
        print(f"  [WARN] Failed to load {resolved}: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset & Collate
# ---------------------------------------------------------------------------

class RawAudioDataset(Dataset):
    def __init__(self, rows: list[dict], audio_roots: list[Path]):
        self.rows = rows
        self.audio_roots = audio_roots

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        waveform = load_waveform(row["audio_path"], self.audio_roots)
        return {
            "idx": idx,
            "waveform": waveform,
            "label": LABEL2IDX[row["intent"]],
            "audio_path": row["audio_path"],
        }


def raw_collate_fn(batch: list[dict]) -> Optional[dict]:
    valid = [b for b in batch if b["waveform"] is not None]
    if not valid:
        return None
    max_len = max(b["waveform"].shape[0] for b in valid)
    waveforms = torch.zeros(len(valid), max_len)
    attention_mask = torch.zeros(len(valid), max_len)
    for i, b in enumerate(valid):
        L = b["waveform"].shape[0]
        waveforms[i, :L] = b["waveform"]
        attention_mask[i, :L] = 1.0
    return {
        "indices": [b["idx"] for b in valid],
        "input_values": waveforms,
        "attention_mask": attention_mask,
        "labels": torch.tensor([b["label"] for b in valid], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Feature extraction with caching
# ---------------------------------------------------------------------------

def extract_features_for_rows(
    backend: EncoderBackend,
    rows: list[dict],
    audio_roots: list[Path],
    device: torch.device,
    cache_path: Optional[Path] = None,
    batch_size: int = 16,
    desc: str = "rows",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Extract or load cached features.

    Returns:
        features: (N, D) float32 tensor
        labels: (N,) int64 tensor
        valid_indices: indices from `rows` that had valid audio
    """
    if cache_path and cache_path.exists():
        print(f"  [Cache hit] Loading precomputed features from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        if data["encoder_model_id"] == backend.model_id and len(data["labels"]) == len(rows):
            return data["features"], data["labels"], data.get("valid_indices", list(range(len(rows))))
        print("  Cache mismatch; recomputing features...")

    ds = RawAudioDataset(rows, audio_roots)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=raw_collate_fn, num_workers=0)

    features_list = []
    labels_list = []
    indices_list = []
    t0 = time.time()
    n_processed = 0

    print(f"  Extracting features for {len(rows)} {desc} audio files...")
    for batch in loader:
        if batch is None:
            continue
        iv = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device)
        feat = backend.extract_features(iv, am)
        features_list.append(feat.cpu())
        labels_list.append(batch["labels"])
        indices_list.extend(batch["indices"])

        n_processed += len(batch["indices"])
        if n_processed % 400 == 0 or n_processed >= len(rows):
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 0.001)
            print(f"    [{desc}] {n_processed}/{len(rows)} files ({elapsed:.1f}s, {rate:.1f} files/s)")

    features = torch.cat(features_list, dim=0) if features_list else torch.empty(0, backend.encoder_dim)
    labels = torch.cat(labels_list, dim=0) if labels_list else torch.empty(0, dtype=torch.long)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "features": features,
            "labels": labels,
            "valid_indices": indices_list,
            "encoder_model_id": backend.model_id,
        }, cache_path)
        print(f"  Saved extracted features to {cache_path}")

    return features, labels, indices_list


# ---------------------------------------------------------------------------
# Classification Head Model
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = MLP_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(
    train_manifest_path: Path,
    calibration_manifest_path: Path,
    clean_test_manifest_path: Path,
    robust_test_manifest_path: Path,
    transcripts_path: Path,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Load only the frozen evaluation manifests and require complete ASR coverage.

    ``synthetic_asr_pool.csv`` exists solely to schedule ASR work. It deliberately
    contains rows that are not eligible for direct-model fitting or a particular
    evaluation role, so it must never be used to construct these splits.
    """
    key_columns = ["id", "audio_path", "split", "intent", "condition"]
    transcripts = pd.read_csv(transcripts_path)
    missing = set(key_columns + ["status"]) - set(transcripts.columns)
    if missing:
        raise KeyError(f"Transcript cache missing required columns: {sorted(missing)}")
    ok = transcripts.loc[transcripts["status"] == "success"].copy()
    if ok.duplicated(key_columns).any():
        raise RuntimeError("Transcript cache has duplicate successful record keys")

    def load_manifest(path: Path, expected_split: str, name: str) -> list[dict]:
        manifest = pd.read_csv(path)
        missing = set(key_columns + ["language_code", "other_type"]) - set(manifest.columns)
        if missing:
            raise KeyError(f"{path.name} missing required columns: {sorted(missing)}")
        if set(manifest["split"].astype(str)) != {expected_split}:
            raise RuntimeError(f"{path.name} is not exclusively the {expected_split!r} split")

        merged = manifest.merge(ok[key_columns], on=key_columns, how="left", indicator=True)
        absent = merged[merged["_merge"] != "both"]
        if not absent.empty:
            raise RuntimeError(
                f"{name} has {len(absent)} rows without successful hybrid transcripts"
            )
        return [
            {
                "id": str(r["id"]),
                "audio_path": str(r["audio_path"]),
                "intent": str(r["intent"]),
                "condition": str(r["condition"]),
                "language_code": str(r["language_code"]),
                "other_type": str(r["other_type"]) if pd.notna(r["other_type"]) else "",
            }
            for _, r in merged.iterrows()
        ]

    train = load_manifest(train_manifest_path, "train", "training manifest")
    dev = load_manifest(calibration_manifest_path, "dev", "calibration manifest")
    clean_test = load_manifest(clean_test_manifest_path, "test", "clean test manifest")
    robust_test = load_manifest(robust_test_manifest_path, "test", "robustness test manifest")
    return train, dev, clean_test, robust_test


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _bootstrap_ci(y_true, y_pred, n_boot=1000, conf=0.95):
    if len(y_true) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    scores = [
        accuracy_score(y_true[idx], y_pred[idx])
        for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))
    ]
    alpha = (1 - conf) / 2
    return float(np.percentile(scores, 100 * alpha)), float(np.percentile(scores, 100 * (1 - alpha)))


def evaluate_features(
    model: MLPClassifier,
    features: torch.Tensor,
    labels: np.ndarray,
    threshold: float,
    device: torch.device,
) -> dict:
    if len(labels) == 0:
        return {"error": "no_samples"}

    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(features.to(device)).cpu().numpy()

    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accept = max_probs >= threshold

    n_total = len(labels)
    n_acc = int(accept.sum())
    coverage = n_acc / n_total if n_total > 0 else 0.0
    abstention = 1.0 - coverage

    acc_labels = labels[accept]
    acc_preds  = preds[accept]
    accuracy   = float(accuracy_score(acc_labels, acc_preds)) if n_acc > 0 else 0.0
    raw_acc    = float(accuracy_score(labels, preds))

    other_idx = LABEL2IDX["other"]
    in_scope = acc_labels != other_idx
    in_scope_acc = float(accuracy_score(acc_labels[in_scope], acc_preds[in_scope])) if in_scope.sum() > 0 else 0.0

    other_total = (acc_labels == other_idx).sum()
    far = float(((acc_labels == other_idx) & (acc_preds != other_idx)).sum() / other_total) if other_total > 0 else 0.0

    ci_lo, ci_hi = _bootstrap_ci(acc_labels, acc_preds)

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist()

    return {
        "n_total": int(n_total),
        "n_accepted": n_acc,
        "n_abstained": int(n_total - n_acc),
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention, 4),
        "accuracy": round(accuracy, 4),
        "raw_accuracy": round(raw_acc, 4),
        "in_scope_accuracy": round(in_scope_acc, 4),
        "false_accept_rate_other": round(far, 4),
        "confidence_interval_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "confusion_matrix": cm,
        "label_order": [IDX2LABEL[i] for i in range(NUM_CLASSES)],
    }


def select_threshold_on_dev(
    model: MLPClassifier,
    dev_features: torch.Tensor,
    dev_labels: np.ndarray,
    device: torch.device,
    target_recall: float = 0.95,
) -> tuple[float, dict]:
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(dev_features.to(device)).cpu().numpy()

    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        mask = max_probs >= t
        if mask.mean() < target_recall:
            break
        acc = float(accuracy_score(dev_labels[mask], preds[mask])) if mask.sum() > 0 else 0.0
        if acc >= best_acc:
            best_acc = acc
            best_t = float(t)

    mask = max_probs >= best_t
    n_total = len(dev_labels)
    n_acc = int(mask.sum())
    cal = {
        "threshold": round(best_t, 4),
        "dev_n_total": n_total,
        "dev_n_accepted": n_acc,
        "dev_coverage": round(n_acc / n_total, 4) if n_total else 0.0,
        "dev_accuracy_at_threshold": round(
            float(accuracy_score(dev_labels[mask], preds[mask])) if n_acc > 0 else 0.0, 4
        ),
        "target_recall": target_recall,
        "split": "dev",
    }
    return best_t, cal


def compute_aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    order = np.argsort(-max_probs)
    correct_s = correct[order]
    n = len(correct_s)
    risks = [1 - correct_s[:k].mean() for k in range(1, n + 1)]
    coverages = [k / n for k in range(1, n + 1)]
    return round(float(np.trapz(risks, coverages)), 6)


def rc_at_targets(probs: np.ndarray, labels: np.ndarray, targets=(0.8, 0.9, 0.95)) -> dict:
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    result = {}
    for cov in targets:
        n_keep = max(1, int(math.ceil(cov * len(max_probs))))
        kept = np.argsort(-max_probs)[:n_keep]
        acc = float(accuracy_score(labels[kept], preds[kept]))
        result[f"selective_acc_at_{int(cov*100)}pct_coverage"] = round(acc, 4)
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(
    model: MLPClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    dev_features: Optional[torch.Tensor] = None,
    dev_labels: Optional[torch.Tensor] = None,
    status_file: Optional[Path] = None,
) -> tuple[list[float], dict]:
    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    losses = []
    best_state = None
    best_epoch = 0
    best_dev_acc = float("-inf")
    epochs_without_improvement = 0
    t_start = time.time()
    n_total_batches = len(loader)

    print("\n  " + "=" * 90, flush=True)
    print(
        f"  {'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^9} | {'Dev Loss':^10} | {'Dev Acc':^9} | {'LR':^9} | {'Time':^7} | {'ETA':^6}",
        flush=True,
    )
    print("  " + "-" * 90, flush=True)

    for epoch in range(epochs):
        t_ep_0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        correct_train, n_train_samples = 0, 0

        for b_idx, (feat_b, lbl_b) in enumerate(loader):
            feat_b = feat_b.to(device)
            lbl_b = lbl_b.to(device)
            opt.zero_grad()
            logits = model(feat_b)
            loss = crit(logits, lbl_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=-1)
            correct_train += (preds == lbl_b).sum().item()
            n_train_samples += lbl_b.size(0)

            # Intra-epoch progress every 50 batches
            if (b_idx + 1) % 50 == 0:
                running_loss = total_loss / n_batches
                running_acc = (correct_train / n_train_samples) * 100
                print(
                    f"    [Epoch {epoch+1:2d}/{epochs:2d} | Step {b_idx+1:3d}/{n_total_batches:3d}] "
                    f"step_loss={running_loss:.4f} | step_acc={running_acc:5.1f}%",
                    flush=True,
                )

        avg_loss = total_loss / n_batches if n_batches else 0.0
        train_acc = correct_train / n_train_samples if n_train_samples else 0.0
        losses.append(avg_loss)

        # Validation on dev split (calibration split only, zero test leakage)
        dev_loss_val = 0.0
        dev_acc_val = 0.0
        if dev_features is not None and dev_labels is not None:
            model.eval()
            with torch.no_grad():
                d_logits = model(dev_features.to(device))
                d_loss = crit(d_logits, dev_labels.to(device))
                dev_loss_val = float(d_loss.item())
                d_preds = d_logits.argmax(dim=-1)
                dev_acc_val = float((d_preds.cpu() == dev_labels.cpu()).float().mean().item())
            if dev_acc_val > best_dev_acc:
                best_dev_acc = dev_acc_val
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        lr_cur = opt.param_groups[0]["lr"]
        sched.step()

        t_ep = time.time() - t_ep_0
        elapsed = time.time() - t_start
        avg_ep_time = elapsed / (epoch + 1)
        remaining = avg_ep_time * (epochs - (epoch + 1))
        eta_str = f"{int(remaining):2d}s" if remaining >= 1 else "0s"

        print(
            f"  {epoch+1:3d}/{epochs:2d}  | {avg_loss:^10.4f} | {train_acc*100:^8.2f}% | "
            f"{dev_loss_val:^10.4f} | {dev_acc_val*100:^8.2f}% | {lr_cur:^9.2e} | {t_ep:^6.2f}s | {eta_str:^6}",
            flush=True,
        )

        if status_file:
            try:
                status_file.parent.mkdir(parents=True, exist_ok=True)
                status_dict = {
                    "status": "training",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "progress_pct": round(((epoch + 1) / epochs) * 100, 1),
                    "train_loss": round(avg_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "dev_loss": round(dev_loss_val, 4),
                    "dev_acc": round(dev_acc_val, 4),
                    "lr": round(lr_cur, 6),
                    "epoch_time_seconds": round(t_ep, 2),
                    "elapsed_seconds": round(elapsed, 1),
                    "remaining_seconds": round(remaining, 1),
                    "eta_str": eta_str,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                status_file.write_text(json.dumps(status_dict, indent=2), encoding="utf-8")
            except Exception:
                pass

        if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
            print(
                f"  Early stopping at epoch {epoch + 1}; best dev accuracy "
                f"was {best_dev_acc * 100:.2f}% at epoch {best_epoch}.",
                flush=True,
            )
            break

    print("  " + "=" * 90 + "\n", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    training_summary = {
        "best_epoch": best_epoch if best_state is not None else len(losses),
        "best_dev_accuracy": round(best_dev_acc, 4) if best_state is not None else None,
        "epochs_completed": len(losses),
    }

    if status_file:
        try:
            status_dict = {
                "status": "completed",
                "total_epochs": epochs,
                **training_summary,
                "final_train_loss": round(losses[-1], 4),
                "final_train_acc": round(train_acc, 4),
                "final_dev_loss": round(dev_loss_val, 4),
                "final_dev_acc": round(dev_acc_val, 4),
                "total_training_time_seconds": round(time.time() - t_start, 2),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            status_file.write_text(json.dumps(status_dict, indent=2), encoding="utf-8")
        except Exception:
            pass

    return losses, training_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]

    parser.add_argument("--audio-roots", nargs="+", type=Path, default=[
        repo_root / "dataset/synthetic/v0.3_robustness",
        repo_root / "dataset/synthetic/v0.3",
    ])
    parser.add_argument("--transcripts", type=Path,
                        default=repo_root / "dataset/manifests/evaluation/hybrid_transcripts.csv")
    parser.add_argument("--train-manifest", type=Path,
                        default=repo_root / "dataset/manifests/evaluation/synthetic_train.csv")
    parser.add_argument("--calibration-manifest", type=Path,
                        default=repo_root / "dataset/manifests/evaluation/synthetic_calibration.csv")
    parser.add_argument("--clean-test-manifest", type=Path,
                        default=repo_root / "dataset/manifests/evaluation/synthetic_clean_final_test.csv")
    parser.add_argument("--robust-test-manifest", type=Path,
                        default=repo_root / "dataset/manifests/evaluation/synthetic_final_test.csv")
    parser.add_argument("--output", type=Path,
                        default=repo_root / "dataset/reports/audio_classifier_v2.json")
    parser.add_argument("--model-dir", type=Path, default=repo_root / "dataset/models")
    parser.add_argument("--model-output", type=Path, default=None,
                        help="Checkpoint path. Defaults to <model-dir>/audio_classifier_v2.pt.")
    parser.add_argument("--schema-version", default="audio_classifier_v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encoder", choices=ENCODER_CANDIDATES,
        default="facebook/wav2vec2-base-960h",
        help="Locally cached frozen encoder to probe.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=MLP_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="Stop after this many non-improving dev epochs and restore the best dev checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-batch-size", type=int, default=16,
        help="Batch size for frozen feature extraction; lower it for larger encoders.",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test on 100 rows per split.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    # 1. Load splits
    print("\n[1/6] Loading splits...")
    train_rows, dev_rows, clean_test_rows, robust_test_rows = load_splits(
        args.train_manifest,
        args.calibration_manifest,
        args.clean_test_manifest,
        args.robust_test_manifest,
        args.transcripts,
    )

    # Integrity check: no split leakage
    test_ids  = {r["id"] for r in clean_test_rows + robust_test_rows}
    train_ids = {r["id"] for r in train_rows}
    dev_ids   = {r["id"] for r in dev_rows}
    if train_ids & test_ids or dev_ids & test_ids or train_ids & dev_ids:
        raise RuntimeError(
            "Split leakage: "
            f"train∩dev={len(train_ids & dev_ids)}, "
            f"train∩test={len(train_ids & test_ids)}, "
            f"dev∩test={len(dev_ids & test_ids)}"
        )
    if any(r["condition"] == "unseen_combined" for r in train_rows + dev_rows):
        raise RuntimeError("unseen_combined must remain strictly test-only")
    if any(r["condition"] != "clean" for r in clean_test_rows):
        raise RuntimeError("The clean test manifest contains a non-clean condition")
    if any(r["condition"] == "clean" for r in robust_test_rows):
        raise RuntimeError("The robustness test manifest must not contain clean audio")

    if args.smoke_test:
        train_rows = train_rows[:100]
        dev_rows = dev_rows[:100]
        clean_test_rows = clean_test_rows[:100]
        robust_test_rows = robust_test_rows[:100]
        args.epochs = 2
        print("  [SMOKE TEST] 100 rows/role, 2 epochs")

    print(
        f"  Train={len(train_rows)}, Dev={len(dev_rows)}, "
        f"Clean test={len(clean_test_rows)}, Robust test={len(robust_test_rows)}"
    )

    # Class weights for training split
    train_labels_int = [LABEL2IDX[r["intent"]] for r in train_rows]
    unique_cls = np.unique(train_labels_int)
    weights = compute_class_weight("balanced", classes=unique_cls, y=train_labels_int)
    class_weights = torch.ones(NUM_CLASSES)
    for cls, w in zip(unique_cls, weights):
        class_weights[int(cls)] = float(w)

    # 2. Load encoder
    print("\n[2/6] Loading frozen speech encoder...")
    backend = load_encoder(args.encoder)
    backend = backend.to(device)

    # 3. Extract features (with disk cache)
    print("\n[3/6] Extracting / loading audio features...")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke_test else ""

    cache_tag = encoder_cache_tag(backend.model_id)
    cache_train = args.model_dir / f"features_frozen_train_{cache_tag}{suffix}.pt"
    cache_dev = args.model_dir / f"features_frozen_dev_{cache_tag}{suffix}.pt"
    cache_clean = args.model_dir / f"features_frozen_clean_test_{cache_tag}{suffix}.pt"
    cache_robust = args.model_dir / f"features_frozen_robust_test_{cache_tag}{suffix}.pt"

    t_feat_0 = time.time()
    train_feats, train_lbls, train_valid_idx = extract_features_for_rows(
        backend, train_rows, args.audio_roots, device, cache_train,
        batch_size=args.feature_batch_size, desc="train"
    )
    dev_feats, dev_lbls, dev_valid_idx = extract_features_for_rows(
        backend, dev_rows, args.audio_roots, device, cache_dev,
        batch_size=args.feature_batch_size, desc="dev"
    )
    clean_test_feats, clean_test_lbls, clean_test_valid_idx = extract_features_for_rows(
        backend, clean_test_rows, args.audio_roots, device, cache_clean,
        batch_size=args.feature_batch_size, desc="clean test"
    )
    robust_test_feats, robust_test_lbls, robust_test_valid_idx = extract_features_for_rows(
        backend, robust_test_rows, args.audio_roots, device, cache_robust,
        batch_size=args.feature_batch_size, desc="robustness test"
    )
    feat_elapsed = time.time() - t_feat_0
    print(f"  All features ready in {feat_elapsed:.1f}s.")

    # 4. Train MLP head
    print(f"\n[4/6] Training MLP head ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})...")
    head = MLPClassifier(
        in_dim=backend.encoder_dim,
        hidden_dim=args.hidden_dim,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    )
    head = head.to(device)

    t_train_0 = time.time()
    status_file = repo_root / "dataset/reports/live_training_status.json"
    losses, training_selection = train_mlp(
        head, train_feats, train_lbls, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        class_weights=class_weights,
        dev_features=dev_feats, dev_labels=dev_lbls,
        status_file=status_file,
    )
    train_elapsed = time.time() - t_train_0
    print(f"  Head trained in {train_elapsed:.2f}s, final loss={losses[-1]:.4f}")

    # Save model weights
    model_save_path = args.model_output or (args.model_dir / "audio_classifier_v2.pt")
    torch.save({
        "mlp_state_dict": head.state_dict(),
        "backend_model_id": backend.model_id,
        "encoder_dim": backend.encoder_dim,
        "mlp_hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "label2idx": LABEL2IDX,
        "idx2label": IDX2LABEL,
        "train_losses": losses,
    }, model_save_path)
    print(f"  Model saved to {model_save_path}")

    # 5. Threshold selection (dev split only)
    print("\n[5/6] Selecting reject threshold on dev split...")
    t_dev_0 = time.time()
    dev_lbls_np = dev_lbls.numpy()
    threshold, cal_report = select_threshold_on_dev(head, dev_feats, dev_lbls_np, device)
    print(f"  Threshold={threshold:.4f}, dev_coverage={cal_report['dev_coverage']}, "
          f"dev_accuracy={cal_report['dev_accuracy_at_threshold']} ({time.time()-t_dev_0:.2f}s)")

    # 6. Final evaluation on the separately frozen clean and robustness tests
    print("\n[6/6] Final evaluation on test splits...")
    clean_test_lbls_np = clean_test_lbls.numpy()
    robust_test_lbls_np = robust_test_lbls.numpy()
    robust_test_valid_rows = [robust_test_rows[i] for i in robust_test_valid_idx]
    clean_result = evaluate_features(
        head, clean_test_feats, clean_test_lbls_np, threshold, device
    )
    robust_result = evaluate_features(
        head, robust_test_feats, robust_test_lbls_np, threshold, device
    )

    # Per condition
    conditions = sorted({r.get("condition", "") for r in robust_test_valid_rows})
    per_condition = {}
    for cond in conditions:
        cond_mask = np.array([r.get("condition") == cond for r in robust_test_valid_rows])
        if cond_mask.sum() > 0:
            per_condition[cond] = evaluate_features(
                head, robust_test_feats[cond_mask], robust_test_lbls_np[cond_mask], threshold, device
            )
            print(f"  {cond:25s}: acc={per_condition[cond].get('accuracy','?')}, cov={per_condition[cond].get('coverage','?')}")

    # AURC & selective accuracy
    with torch.no_grad():
        robust_test_probs = head.predict_proba(robust_test_feats.to(device)).cpu().numpy()
    aurc = compute_aurc(robust_test_probs, robust_test_lbls_np)
    rc_targets = rc_at_targets(robust_test_probs, robust_test_lbls_np)

    # Build report
    report = {
        "schema_version": args.schema_version,
        "model": {
            "encoder": backend.model_id,
            "encoder_type": type(backend).__name__,
            "encoder_dim": backend.encoder_dim,
            "encoder_frozen": True,
            "head_architecture": f"Linear({backend.encoder_dim},{args.hidden_dim})->LayerNorm->ReLU->Dropout({args.dropout})->Linear({args.hidden_dim},{NUM_CLASSES})",
            "n_params_encoder": sum(p.numel() for p in (backend.model.parameters() if hasattr(backend, "model") else backend.encoder.parameters())),
            "n_params_head": sum(p.numel() for p in head.parameters()),
            "n_params_trainable": sum(p.numel() for p in head.parameters()),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
            "early_stop_patience": args.early_stop_patience,
            **training_selection,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "class_weighting": "balanced",
            "n_train": len(train_lbls),
            "feature_extraction_seconds": round(feat_elapsed, 1),
            "head_train_seconds": round(train_elapsed, 2),
            "final_train_loss": round(losses[-1], 4) if losses else None,
            "train_losses": [round(l, 4) for l in losses],
        },
        "calibration": {
            "method": "max_probability_threshold_on_dev",
            **cal_report,
        },
        "evaluation": {
            "clean_final_test": clean_result,
            "full_robustness_test": robust_result,
            "per_condition": per_condition,
            "aurc_full_test": aurc,
            **rc_targets,
        },
        "data": {
            "train_manifest": str(args.train_manifest),
            "calibration_manifest": str(args.calibration_manifest),
            "clean_test_manifest": str(args.clean_test_manifest),
            "robust_test_manifest": str(args.robust_test_manifest),
            "transcripts": str(args.transcripts),
            "n_train": len(train_lbls),
            "n_dev": len(dev_lbls),
            "n_clean_test": len(clean_test_lbls),
            "n_robust_test": len(robust_test_lbls),
            "leakage_check": {
                "train_test_overlap": len(train_ids & test_ids),
                "dev_test_overlap": len(dev_ids & test_ids),
                "train_dev_overlap": len(train_ids & dev_ids),
                "passed": not (train_ids & test_ids or dev_ids & test_ids or train_ids & dev_ids),
            },
        },
        "audio": {
            "roots_tried": [str(r) for r in args.audio_roots],
            "source_sample_rate_hz": 8000,
            "target_sample_rate_hz": TARGET_SR,
            "resampling_note": (
                "8 kHz -> 16 kHz upsampling does NOT restore high-frequency content "
                "that was absent from the original telephone-band recordings."
            ),
            "max_audio_seconds": MAX_AUDIO_SECONDS,
        },
        "limitations": [
            "Encoder completely frozen (no end-to-end backprop into speech backbone).",
            "Audio is synthetic TTS, not real caller speech.",
            "8 kHz → 16 kHz resampling does not restore missing high-frequency content.",
            "Results not comparable to real telephone deployment.",
            "IndicVoices candidates excluded (unverified intent labels).",
            f"Encoder {backend.model_id} was pre-trained on English; Indic acoustic generalization is limited.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.output}")

    print("\n=== SUMMARY ===")
    print(f"Encoder:       {backend.model_id}")
    print(f"Clean test:    acc={clean_result.get('accuracy')}, coverage={clean_result.get('coverage')}")
    print(f"Robust test:   acc={robust_result.get('accuracy')}, coverage={robust_result.get('coverage')}")
    print(f"AURC:          {aurc}")
    print(f"Threshold:     {threshold:.4f}")


if __name__ == "__main__":
    main()
