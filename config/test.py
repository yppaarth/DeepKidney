#!/usr/bin/env python3
"""
evaluate_face_voice.py

Evaluate a pretrained face-voice verification model and produce a MAV-Celeb-style score file.

Usage:
    python evaluate_face_voice.py \
        --model_weights model.pth \
        --test_pairs test_pairs.txt \
        --data_root ./MAV-Celeb-v1/test \
        --output_file sub_score_English_heard.txt \
        --batch_size 32

Dependencies:
    pip install torch torchvision torchaudio facenet-pytorch modelscope pillow scikit-learn
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T
import torchaudio
import numpy as np

# Optional for EER
from sklearn.metrics import roc_curve

# Try imports for pretrained bases
try:
    from modelscope.models.audio.sv import ECAPA_TDNN  # type: ignore
except Exception as e:
    ECAPA_TDNN = None
    ECAPA_ERROR = e

try:
    from facenet_pytorch import InceptionResnetV1
except Exception as e:
    InceptionResnetV1 = None
    FACENET_ERROR = e


# ---------------------------
# Model definitions (same architecture used during training)
# ---------------------------
class AudioEncoder(nn.Module):
    """
    Frozen RDINO ECAPA-TDNN base + trainable adapter head (192 -> 512 -> ReLU -> 512)
    """

    def __init__(self, device: torch.device):
        super().__init__()
        if ECAPA_TDNN is None:
            raise ImportError(
                "Failed to import ECAPA_TDNN from modelscope. "
                f"Original error: {ECAPA_ERROR}"
            )
        model_id = "damo/speech_rdino_ecapa-tdnn_sv_en_voxceleb_16k"
        # load pretrained
        try:
            base = ECAPA_TDNN.from_pretrained(model_id)
        except Exception:
            try:
                base = ECAPA_TDNN(model_id)
            except Exception as e:
                raise RuntimeError(f"Unable to load ECAPA_TDNN pretrained model '{model_id}': {e}")
        self.base = base
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T) or (B, 1, T)
        returns: (B, 512) (raw adapter outputs)
        """
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)

        with torch.no_grad():
            out = None
            try:
                out = self.base(waveform)
            except Exception:
                # try alternative calling signature
                try:
                    out = self.base(waveform.unsqueeze(1))
                except Exception as e:
                    raise RuntimeError(f"ECAPA_TDNN forward invocation failed: {e}")

            if isinstance(out, (list, tuple)):
                emb = next((x for x in out if isinstance(x, torch.Tensor)), None)
                if emb is None:
                    raise RuntimeError("ECAPA_TDNN returned no tensor embeddings.")
            elif isinstance(out, torch.Tensor):
                emb = out
            else:
                raise RuntimeError(f"Unexpected ECAPA_TDNN output type: {type(out)}")

            emb = emb.squeeze()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)

        emb = emb.to(next(self.adapter.parameters()).device)
        return self.adapter(emb)


class FaceEncoder(nn.Module):
    """
    Frozen FaceNet InceptionResnetV1 (vggface2) base + trainable adapter (512 -> 512 -> ReLU -> 512)
    """

    def __init__(self, device: torch.device):
        super().__init__()
        if InceptionResnetV1 is None:
            raise ImportError(
                "Failed to import InceptionResnetV1 from facenet_pytorch. "
                f"Original error: {FACENET_ERROR}"
            )
        base = InceptionResnetV1(pretrained="vggface2")
        base.eval()
        for p in base.parameters():
            p.requires_grad = False
        self.base = base

        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W), normalized to [-1,1]
        returns: (B, 512)
        """
        with torch.no_grad():
            emb = self.base(images)
        emb = emb.to(next(self.adapter.parameters()).device)
        return self.adapter(emb)


class FaceVoiceVerificationModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.audio = AudioEncoder(device)
        self.face = FaceEncoder(device)

    def forward(self, audio_input: torch.Tensor, face_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        audio_input: (B, T)
        face_input: (B, 3, H, W)
        returns: E1 (B,512), E2 (B,512) both L2-normalized
        """
        # Move inputs to adapters' devices
        audio_input = audio_input.to(next(self.audio.adapter.parameters()).device)
        face_input = face_input.to(next(self.face.adapter.parameters()).device)

        a = self.audio(audio_input)
        f = self.face(face_input)

        E1 = F.normalize(a, p=2, dim=1)
        E2 = F.normalize(f, p=2, dim=1)
        return E1, E2


# ---------------------------
# TestPairsDataset for evaluation
# ---------------------------
class TestPairsDataset(Dataset):
    """
    Parses a pairs file where each line is: unique_id voice_path face_path
    and loads/preprocesses the audio and face exactly as in training.
    """

    def __init__(self, pairs_file_path: str, data_root: str, audio_len: int = 32000, target_sr: int = 16000):
        """
        Args:
            pairs_file_path: path to text file containing lines: uid voice_path face_path
            data_root: root directory to prepend to voice_path and face_path
            audio_len: samples to crop/pad audio to (e.g., 32000)
            target_sr: required sample rate (16000)
        """
        super().__init__()
        self.pairs_file = Path(pairs_file_path)
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_file}")

        self.data_root = Path(data_root)
        self.entries = []  # list of tuples (uid, full_voice_path, full_face_path)
        with self.pairs_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid line in pairs file (expected 3 columns): {line}")
                uid, voice_rel, face_rel = parts[0], parts[1], parts[2]
                voice_path = (self.data_root / voice_rel).resolve()
                face_path = (self.data_root / face_rel).resolve()
                if not voice_path.exists():
                    raise FileNotFoundError(f"Voice path not found for uid {uid}: {voice_path}")
                if not face_path.exists():
                    raise FileNotFoundError(f"Face path not found for uid {uid}: {face_path}")
                self.entries.append((uid, voice_path, face_path))

        if len(self.entries) == 0:
            raise RuntimeError(f"No entries parsed from {self.pairs_file}")

        # transforms for faces (same as training): Resize->ToTensor->Normalize to [-1,1]
        self.face_transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.audio_len = int(audio_len)
        self.target_sr = int(target_sr)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        uid, voice_path, face_path = self.entries[idx]

        # load face
        with Image.open(face_path).convert("RGB") as img:
            face_tensor = self.face_transform(img)  # (3,160,160)

        # load audio
        waveform, sr = torchaudio.load(str(voice_path))  # [channels, samples], sr
        # convert to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # resample to target_sr if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        waveform = waveform.float()
        cur_len = waveform.size(0)
        if cur_len == self.audio_len:
            audio_tensor = waveform
        elif cur_len > self.audio_len:
            # center crop deterministic in evaluation (to be consistent)
            start = (cur_len - self.audio_len) // 2
            audio_tensor = waveform[start:start + self.audio_len]
        else:
            pad_len = self.audio_len - cur_len
            audio_tensor = torch.cat([waveform, torch.zeros(pad_len, dtype=waveform.dtype)], dim=0)

        return uid, face_tensor, audio_tensor


# ---------------------------
# EER utility
# ---------------------------
def calculate_eer(y_true: List[int], y_scores: List[float]) -> float:
    """
    Compute EER given ground truth labels (1: match, 0: non-match) and scores.
    Returns EER (in same [0,1] scale).
    """
    if len(y_true) == 0:
        raise ValueError("Empty y_true provided to calculate_eer.")
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # find point where abs(fpr - fnr) is minimal
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer


# ---------------------------
# Evaluation routine
# ---------------------------
def collate_eval(batch):
    """
    batch: list of (uid, face_tensor, audio_tensor)
    returns: list uids, faces stacked (B,3,160,160), audios stacked (B, audio_len)
    """
    uids = [b[0] for b in batch]
    faces = torch.stack([b[1] for b in batch], dim=0)
    audios = torch.stack([b[2] for b in batch], dim=0)
    return uids, faces, audios


def load_model_weights_into(model: FaceVoiceVerificationModel, weights_path: str, device: torch.device):
    """
    Robustly load weights. Accepts either state_dict directly or a dict with 'model_state_dict' or full checkpoint.
    """
    ckpt = torch.load(weights_path, map_location=device)
    # ckpt might be dict with keys 'model_state_dict' or raw state_dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # Possibly the whole saved model state dict
            # Heuristic: keys are like 'audio.adapter.0.weight' etc.
            state = ckpt
    else:
        # Unexpected type
        raise RuntimeError("Loaded checkpoint is not a mapping-type state dict.")

    # Attempt to load; allow missing keys if adapter head naming differs a bit by prefix
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # Try non-strict load and report missing/unexpected
        print("Strict load failed, attempting non-strict load. Error:", e)
        res = model.load_state_dict(state, strict=False)
        print("Non-strict load result:", res)
    return model


def evaluate_and_write_scores(
    model_weights: str,
    pairs_file: str,
    data_root: str,
    output_file: str,
    batch_size: int = 32,
    audio_len: int = 32000,
    device: torch.device = torch.device("cpu"),
    calc_eer_local: bool = False,
):
    # Instantiate dataset and loader
    ds = TestPairsDataset(pairs_file, data_root, audio_len=audio_len, target_sr=16000)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_eval)

    # Build model and load weights
    model = FaceVoiceVerificationModel(device)
    # Move adapters/bases to device
    model.to(device)
    model.audio.adapter = model.audio.adapter.to(device)
    model.face.adapter = model.face.adapter.to(device)
    model.audio.base = model.audio.base  # base kept in eval mode and might already be on CPU; ensure no param grads
    model.face.base = model.face.base

    # Load weights
    print(f"Loading model weights from: {model_weights}")
    model = load_model_weights_into(model, model_weights, device)
    model.eval()

    uids_all = []
    scores_all = []

    # Optional for EER if ground truth labels are embedded in pairs file; user must supply if desired
    y_true = []  # if pairs file provides true labels as extra column, we could populate; currently left empty

    with torch.no_grad():
        for uids, faces, audios in loader:
            faces = faces.to(device)
            audios = audios.to(device)
            # Compute embeddings
            E1, E2 = model(audios, faces)  # E1 audio, E2 face, both (B,512) normalized
            # Compute cosine similarity: dot product per row (since both L2-normalized)
            # score = torch.sum(E1 * E2, dim=1)
            scores = torch.einsum("bd,bd->b", E1, E2)  # (B,)
            scores = scores.detach().cpu().numpy().tolist()
            uids_all.extend(uids)
            scores_all.extend(scores)

    # Write output file in exact format: unique_id score (one per line)
    out_path = Path(output_file)
    with out_path.open("w", encoding="utf-8") as fh:
        for uid, score in zip(uids_all, scores_all):
            # Format score with 6 decimal places (challenge example used 4, 6 is safer); change if needed
            fh.write(f"{uid} {score:.6f}\n")

    print(f"Saved {len(uids_all)} scores to {out_path.resolve()}")

    # Optionally compute EER if ground truth labels were available
    if calc_eer_local and len(y_true) == len(scores_all) and len(y_true) > 0:
        eer = calculate_eer(y_true, scores_all)
        print(f"Local EER: {eer:.6f}")
    elif calc_eer_local:
        print("EER requested but no ground-truth labels were provided / parsed; skipping EER computation.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face-Voice verification model and produce MAV-Celeb score file.")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to model weights (.pth) saved from training.")
    parser.add_argument("--test_pairs", type=str, required=True, help="Path to test pairs .txt file (uid voice_path face_path per line).")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory to prepend to paths in the pairs file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write output scores (unique_id score per line).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--audio_len", type=int, default=32000, help="Audio crop/pad length in samples (default 32000).")
    parser.add_argument("--use_eer", action="store_true", help="If provided and ground-truth labels are available in pairs file, compute EER locally.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    evaluate_and_write_scores(
        model_weights=args.model_weights,
        pairs_file=args.test_pairs,
        data_root=args.data_root,
        output_file=args.output_file,
        batch_size=args.batch_size,
        audio_len=args.audio_len,
        device=device,
        calc_eer_local=args.use_eer,
    )