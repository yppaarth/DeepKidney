#!/usr/bin/env python3
"""
train_face_voice_with_ckpt.py

Complete script to train a face-voice verification model with:
 - dataset scanning for face<->voice pairs
 - facenet and RDINO ECAPA_TDNN as frozen bases
 - adapter heads for both streams
 - contrastive loss
 - training + validation loop
 - checkpoint saving (latest & best) and resume loading

Usage:
    pip install torch torchvision torchaudio facenet-pytorch modelscope pillow
    python train_face_voice_with_ckpt.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import torchvision.transforms as T
import torchaudio

# Try imports for required pretrained bases
try:
    from modelscope.models.audio.sv import ECAPA_TDNN  # type: ignore
except Exception as e:
    ECAPA_TDNN = None
    ECAPA_IMPORT_ERROR = e

try:
    from facenet_pytorch import InceptionResnetV1
except Exception as e:
    InceptionResnetV1 = None
    FACENET_IMPORT_ERROR = e


# ---------------------------
# Dataset
# ---------------------------
class MAVCelebDataset(Dataset):
    """
    Disk-based dataset mapping faces/.../idXXXX/.../name.jpg -> voices/.../idXXXX/.../name.wav
    """

    def __init__(self, root_dir: str, audio_crop_len: int = 32000):
        super().__init__()
        self.root = Path(root_dir)
        if not self.root.exists():
            raise ValueError(f"Root dir does not exist: {self.root}")
        self.faces_dir = self.root / "faces"
        self.voices_dir = self.root / "voices"
        if not self.faces_dir.exists() or not self.voices_dir.exists():
            raise ValueError(f"Expected faces/ and voices/ under {self.root}; found faces: {self.faces_dir.exists()}, voices: {self.voices_dir.exists()}")

        self.face_paths: List[Path] = list(self.faces_dir.rglob("*.jpg"))
        self.pairs: List[Tuple[Path, Path]] = []
        self.labels_str: List[str] = []

        for fpath in self.face_paths:
            try:
                rel = fpath.relative_to(self.root)
            except Exception:
                # fallback
                parts = list(fpath.parts)
                try:
                    idx = parts.index("faces")
                    rel = Path(*parts[idx:])
                except ValueError:
                    continue
            if rel.parts[0] != "faces":
                continue
            rel_voice = Path("voices", *rel.parts[1:]).with_suffix(".wav")
            voice_path = self.root / rel_voice
            if voice_path.exists():
                self.pairs.append((fpath, voice_path))
                # speaker id assumed to be rel.parts[1] (faces/idXXXX/...)
                spk = rel.parts[1] if len(rel.parts) > 1 else "unknown"
                self.labels_str.append(spk)

        if len(self.pairs) == 0:
            raise RuntimeError(f"No face-voice pairs found under {self.root} (checked {len(self.face_paths)} face files).")

        unique_spks = sorted(set(self.labels_str))
        self.spk2int: Dict[str, int] = {s: i for i, s in enumerate(unique_spks)}
        self.labels_int = [self.spk2int[s] for s in self.labels_str]

        self.face_transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.audio_len = audio_crop_len
        self.target_sr = 16000

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError("Index out of range")
        fpath, vpath = self.pairs[idx]
        label = self.labels_int[idx]

        with Image.open(fpath).convert("RGB") as im:
            face = self.face_transform(im)

        waveform, sr = torchaudio.load(str(vpath))  # [channels, samples], sr
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != self.target_sr:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resample(waveform)
        waveform = waveform.float()

        cur_len = waveform.size(0)
        if cur_len == self.audio_len:
            audio = waveform
        elif cur_len > self.audio_len:
            max_start = cur_len - self.audio_len
            start = torch.randint(0, max_start + 1, (1,)).item()
            audio = waveform[start:start + self.audio_len]
        else:
            pad_len = self.audio_len - cur_len
            audio = torch.cat([waveform, torch.zeros(pad_len, dtype=waveform.dtype)], dim=0)

        return face, audio, int(label)


# ---------------------------
# Model components
# ---------------------------
class AudioEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        if ECAPA_TDNN is None:
            raise ImportError(f"Cannot import ECAPA_TDNN from modelscope. Error: {ECAPA_IMPORT_ERROR}")
        model_id = "damo/speech_rdino_ecapa-tdnn_sv_en_voxceleb_16k"
        try:
            base = ECAPA_TDNN.from_pretrained(model_id)
        except Exception:
            try:
                base = ECAPA_TDNN(model_id)
            except Exception as e:
                raise RuntimeError(f"Failed to load ECAPA_TDNN '{model_id}'. Error: {e}")
        self.base = base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T) or (B,1,T)
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        with torch.no_grad():
            out = None
            try:
                out = self.base(waveform)
            except Exception:
                try:
                    out = self.base(waveform.unsqueeze(1))
                except Exception as e:
                    raise RuntimeError(f"ECAPA_TDNN forward failed: {e}")
            if isinstance(out, (list, tuple)):
                emb = next((x for x in out if isinstance(x, torch.Tensor)), None)
                if emb is None:
                    raise RuntimeError("ECAPA_TDNN returned non-tensor outputs.")
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
    def __init__(self, device: torch.device):
        super().__init__()
        if InceptionResnetV1 is None:
            raise ImportError(f"Cannot import InceptionResnetV1. Error: {FACENET_IMPORT_ERROR}")
        base = InceptionResnetV1(pretrained="vggface2").eval()
        for p in base.parameters():
            p.requires_grad = False
        self.base = base
        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
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
        audio_input = audio_input.to(next(self.audio.adapter.parameters()).device)
        face_input = face_input.to(next(self.face.adapter.parameters()).device)
        a = self.audio(audio_input)
        f = self.face(face_input)
        E1 = F.normalize(a, p=2, dim=1)
        E2 = F.normalize(f, p=2, dim=1)
        return E1, E2


# ---------------------------
# Loss
# ---------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, E1: torch.Tensor, E2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = E1.device
        labels = labels.to(device)
        sim = torch.matmul(E1, E2.t())
        lr = labels.unsqueeze(1)
        lc = labels.unsqueeze(0)
        target = (lr == lc).float().to(device)
        pos_mask = target == 1.0
        neg_mask = target == 0.0
        pos_term = (1.0 - sim)[pos_mask]
        neg_term = F.relu(sim - self.margin)[neg_mask]
        pos_mean = pos_term.mean() if pos_term.numel() > 0 else torch.tensor(0.0, device=device)
        neg_mean = neg_term.mean() if neg_term.numel() > 0 else torch.tensor(0.0, device=device)
        loss = 0.5 * (pos_mean + neg_mean)
        return loss


# ---------------------------
# Utilities
# ---------------------------
def collate_fn(batch):
    faces = torch.stack([b[0] for b in batch], dim=0)
    audios = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return faces, audios, labels


def get_adapter_params(model: FaceVoiceVerificationModel):
    params = list(model.audio.adapter.parameters()) + list(model.face.adapter.parameters())
    if len(params) == 0:
        raise RuntimeError("No adapter params found for training.")
    return params


# ---------------------------
# Training & Validation with checkpointing
# ---------------------------
if __name__ == "__main__":
    # Hyperparameters
    DATA_ROOT = "./v1"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 10
    AUDIO_LEN = 32000
    VAL_SPLIT = 0.1
    CHECKPOINT_LATEST = "checkpoint_latest.pth"
    CHECKPOINT_BEST = "checkpoint_best.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")

    # Prepare dataset
    print("Preparing dataset...")
    dataset = MAVCelebDataset(DATA_ROOT, audio_crop_len=AUDIO_LEN)
    num_samples = len(dataset)
    val_size = max(1, int(num_samples * VAL_SPLIT))
    train_size = num_samples - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"Total samples: {num_samples} | Train: {train_size} | Val: {val_size} | Speakers: {len(dataset.spk2int)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)

    # Model
    print("Initializing model (this may download pretrained weights)...")
    model = FaceVoiceVerificationModel(DEVICE)
    # Move adapters and bases to DEVICE
    model.to(DEVICE)
    # Ensure adapter modules are on same device
    model.audio.adapter = model.audio.adapter.to(DEVICE)
    model.face.adapter = model.face.adapter.to(DEVICE)

    # Freeze base networks explicitly
    model.audio.base.eval()
    model.face.base.eval()
    for p in model.audio.base.parameters():
        p.requires_grad = False
    for p in model.face.base.parameters():
        p.requires_grad = False

    criterion = ContrastiveLoss(margin=0.2)
    optimizer = torch.optim.Adam(get_adapter_params(model), lr=LEARNING_RATE)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume if checkpoint exists
    if os.path.exists(CHECKPOINT_LATEST):
        print(f"Found checkpoint {CHECKPOINT_LATEST}, attempting to resume...")
        ckpt = torch.load(CHECKPOINT_LATEST, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            print(f"Resumed from epoch {ckpt['epoch']}. Best val loss so far: {best_val_loss:.6f}")
        except Exception as e:
            print(f"Warning: failed to fully load checkpoint: {e}. Continuing from scratch.")

    # Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, (faces, audios, labels) in enumerate(train_loader, start=1):
            faces = faces.to(DEVICE)
            audios = audios.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            E1, E2 = model(audios, faces)
            loss = criterion(E1, E2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 10 == 0 or step == 1:
                avg = running_loss / step
                print(f"[Train] Epoch {epoch}/{EPOCHS} Step {step}/{len(train_loader)} Loss: {loss.item():.6f} Avg: {avg:.6f}")

        epoch_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch} training complete. Avg training loss: {epoch_train_loss:.6f}")

        # Validation
        model.eval()
        val_running = 0.0
        val_steps = 0
        with torch.no_grad():
            for faces, audios, labels in val_loader:
                faces = faces.to(DEVICE)
                audios = audios.to(DEVICE)
                labels = labels.to(DEVICE)
                E1, E2 = model(audios, faces)
                vloss = criterion(E1, E2, labels)
                val_running += vloss.item()
                val_steps += 1
        val_loss = val_running / val_steps if val_steps > 0 else 0.0
        print(f"[Val] Epoch {epoch} validation loss: {val_loss:.6f}")

        # Checkpoint: save latest
        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "spk2int": dataset.spk2int,
        }
        torch.save(ckpt_dict, CHECKPOINT_LATEST)
        print(f"Saved checkpoint: {CHECKPOINT_LATEST}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dict["best_val_loss"] = best_val_loss
            torch.save(ckpt_dict, CHECKPOINT_BEST)
            print(f"New best validation loss {best_val_loss:.6f}. Saved best checkpoint: {CHECKPOINT_BEST}")

    print("Training finished. Final best val loss:", best_val_loss)