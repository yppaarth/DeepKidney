#!/usr/bin/env python3
"""
train_face_voice.py

Complete, self-contained script to train a face-voice verification model
as specified by the user.

Requirements (install before running):
    pip install torch torchvision facenet-pytorch modelscope

Notes:
- This script expects ModelScope's ECAPA_TDNN implementation to be importable
  from modelscope.models.audio.sv import ECAPA_TDNN and that it provides
  either a .from_pretrained(...) loader or can be instantiated and loaded.
- If the exact API differs, please adapt the ECAPA_TDNN loading section.
"""

import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Attempt to import ModelScope ECAPA_TDNN and facenet-pytorch InceptionResnetV1
try:
    # ModelScope audio speaker-verification models
    from modelscope.models.audio.sv import ECAPA_TDNN  # type: ignore
except Exception as e:
    ECAPA_TDNN = None
    _ms_error = e

try:
    from facenet_pytorch import InceptionResnetV1
except Exception as e:
    InceptionResnetV1 = None
    _fn_error = e


# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Audio Encoder
# ---------------------------
class AudioEncoder(nn.Module):
    """
    Audio stream:
      - Loads pretrained RDINO ECAPA_TDNN from ModelScope (frozen)
      - Uses it to extract 192-d embeddings (assumed)
      - Trains a small projection head to map 192 -> 512 -> 512
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        if ECAPA_TDNN is None:
            raise ImportError(
                "Could not import ECAPA_TDNN from modelscope.models.audio.sv. "
                "Original import error:\n" + str(_ms_error)
            )

        # Load the pretrained RDINO ECAPA-TDNN model identifier specified
        model_id = "damo/speech_rdino_ecapa-tdnn_sv_en_voxceleb_16k"

        # Try common loading patterns for ModelScope models:
        base = None
        load_errors = []
        try:
            # Try classmethod from_pretrained
            base = ECAPA_TDNN.from_pretrained(model_id)
        except Exception as ex1:
            load_errors.append(("from_pretrained", ex1))
            try:
                # Try direct constructor (some versions allow passing model_id)
                base = ECAPA_TDNN(model_id)
            except Exception as ex2:
                load_errors.append(("constructor_with_id", ex2))
                try:
                    # Try parameterless constructor and then load_state_dict if available
                    base = ECAPA_TDNN()
                except Exception as ex3:
                    load_errors.append(("bare_constructor", ex3))
                    # If still not possible, re-raise with collected info
                    msg = "Failed to load ECAPA_TDNN with multiple strategies. Errors:\n"
                    for tag, err in load_errors:
                        msg += f"- {tag}: {repr(err)}\n"
                    raise RuntimeError(msg)

        self.base = base.to(self.device)
        self.base.eval()
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        # In many ECAPA implementations the embedding dim is 192. We'll assume 192 as specified.
        base_embedding_dim = 192

        # Trainable projection head: 192 -> 512 -> ReLU -> 512
        self.proj = nn.Sequential(
            nn.Linear(base_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        ).to(self.device)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        waveforms: Tensor shape (B, T) or (B, 1, T)
        Returns: Tensor (B, 512) - raw (not normalized) embeddings (projection head output)
        """
        # Ensure shape (B, T)
        if waveforms.dim() == 3 and waveforms.size(1) == 1:
            waveforms = waveforms.squeeze(1)

        # Move to device
        waveforms = waveforms.to(self.device)

        # Use base model to extract embedding.
        # Different ModelScope versions might expect different input shapes/keyword args.
        # We'll try common invocation patterns and return the first tensor-like output interpreted as embedding.

        with torch.no_grad():
            base_out = None
            invocation_errors = []
            # Try calling base directly
            try:
                base_out = self.base(waveforms)
            except Exception as e:
                invocation_errors.append(("base(waveforms)", e))
                try:
                    # try passing as (waveforms, sample_rate) if required
                    base_out = self.base(waveforms, 16000)
                except Exception as e2:
                    invocation_errors.append(("base(waveforms,16000)", e2))
                    try:
                        # some implementations want (waveforms.unsqueeze(1))
                        base_out = self.base(waveforms.unsqueeze(1))
                    except Exception as e3:
                        invocation_errors.append(("base(waveforms.unsqueeze(1))", e3))
                        # As a last resort, try calling an attribute like .extract_embedding
                        if hasattr(self.base, "extract_embedding"):
                            try:
                                base_out = self.base.extract_embedding(waveforms)
                            except Exception as e4:
                                invocation_errors.append(("extract_embedding", e4))

            if base_out is None:
                # Provide helpful debugging output
                msg = "Failed to get embeddings from the ECAPA_TDNN base model using common invocation patterns. "
                msg += "Invocation attempts and errors:\n"
                for tag, err in invocation_errors:
                    msg += f"- {tag}: {repr(err)}\n"
                raise RuntimeError(msg)

            # base_out might be a tensor or tuple/list. Extract the embedding tensor.
            if isinstance(base_out, (tuple, list)):
                # often returns (embedding, ...). Take first tensor-like entry.
                emb = None
                for x in base_out:
                    if isinstance(x, torch.Tensor):
                        emb = x
                        break
                if emb is None:
                    raise RuntimeError("ECAPA_TDNN returned a sequence but no tensor was found.")
            elif isinstance(base_out, torch.Tensor):
                emb = base_out
            else:
                raise RuntimeError(f"Unexpected ECAPA_TDNN output type: {type(base_out)}")

            # At this point emb should be (B, embedding_dim) or (B, C, 1). Squeeze and ensure shape.
            emb = emb.squeeze()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)

            # If emb has extra dims (B, C, 1), flatten last dims
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)

        # Now pass embedding through projection head (trainable)
        proj_out = self.proj(emb.to(self.proj[0].weight.device))
        return proj_out


# ---------------------------
# Face Encoder
# ---------------------------
class FaceEncoder(nn.Module):
    """
    Face stream:
      - Loads pretrained InceptionResnetV1 from facenet-pytorch (vggface2)
      - Freezes it
      - Adds a trainable projection head 512 -> 512 -> ReLU -> 512
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        if InceptionResnetV1 is None:
            raise ImportError(
                "Could not import InceptionResnetV1 from facenet_pytorch. "
                "Original import error:\n" + str(_fn_error)
            )

        # Load pretrained FaceNet (vggface2)
        face_base = InceptionResnetV1(pretrained="vggface2").to(self.device)
        face_base.eval()
        for p in face_base.parameters():
            p.requires_grad = False

        self.base = face_base

        # FaceNet embedding dim is 512
        base_emb_dim = 512
        self.proj = nn.Sequential(
            nn.Linear(base_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        ).to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: Tensor (B, 3, H, W) already preprocessed as facenet expects: normalized in [-1, 1]
        Return: Tensor (B, 512)
        """
        images = images.to(self.device)
        with torch.no_grad():
            base_emb = self.base(images)  # (B, 512)

        proj_out = self.proj(base_emb.to(self.proj[0].weight.device))
        return proj_out


# ---------------------------
# Main Model
# ---------------------------
class FaceVoiceVerificationModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.audio_enc = AudioEncoder(device)
        self.face_enc = FaceEncoder(device)

    def forward(self, waveforms: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        waveforms: (B, T) or (B,1,T)
        images: (B, 3, H, W)
        Returns:
          E1: audio embeddings L2-normalized (B, 512)
          E2: face embeddings L2-normalized (B, 512)
        """
        # Get projected embeddings
        a_emb = self.audio_enc(waveforms)  # (B, 512)
        f_emb = self.face_enc(images)      # (B, 512)

        # L2 normalize along dim=1
        E1 = F.normalize(a_emb, p=2, dim=1)
        E2 = F.normalize(f_emb, p=2, dim=1)
        return E1, E2


# ---------------------------
# Contrastive Loss Module
# ---------------------------
class ContrastiveLoss(nn.Module):
    """
    Given audio embeddings E1 and face embeddings E2 (both L2-normalized)
    and integer speaker labels, compute pairwise cosine similarities and
    apply the described contrastive loss with margin.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, E1: torch.Tensor, E2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        E1: (B, D) - audio embeddings (L2-normed)
        E2: (B, D) - face embeddings (L2-normed)
        labels: (B,) integer labels
        """
        # Ensure float, device alignment
        device = E1.device
        labels = labels.to(device)

        # Cosine similarity matrix: since embeddings are normalized, matrix = E1 @ E2^T
        sim_matrix = torch.matmul(E1, E2.t())  # (B, B), values in [-1, 1]

        # Build target matrix: 1 if same speaker, else 0
        labels_row = labels.unsqueeze(1)  # (B,1)
        labels_col = labels.unsqueeze(0)  # (1,B)
        target = (labels_row == labels_col).float().to(device)  # (B,B)

        # Positive part: (1 - similarity) for target == 1
        pos_mask = target == 1.0
        neg_mask = target == 0.0

        pos_losses = (1.0 - sim_matrix)[pos_mask]  # shape: num_pos
        neg_losses = F.relu(sim_matrix - self.margin)[neg_mask]  # shape: num_neg

        # If no positive or negative pairs exist in batch (very unlikely), handle gracefully
        losses = []
        if pos_losses.numel() > 0:
            losses.append(pos_losses.mean())
        else:
            losses.append(torch.tensor(0.0, device=device))

        if neg_losses.numel() > 0:
            losses.append(neg_losses.mean())
        else:
            losses.append(torch.tensor(0.0, device=device))

        loss = torch.stack(losses).mean()
        return loss


# ---------------------------
# Dummy MAVCelebDataset (simulated)
# ---------------------------
class MAVCelebDataset(Dataset):
    """
    Produces synthetic random data resembling:
      - face image tensor: (3, 160, 160) values in [-1, 1] (facenet input)
      - audio waveform: (16000,) raw waveform in [-1, 1]
      - label: integer speaker id from 0..(num_speakers-1)
    """

    def __init__(self, num_samples: int = 500, num_speakers: int = 10, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.num_speakers = num_speakers
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Face image: random floats in [-1,1]
        face = (torch.rand(3, 160, 160) * 2.0 - 1.0).float()

        # Audio waveform: 1 second at 16kHz -> shape (16000,)
        audio = (torch.rand(16000) * 2.0 - 1.0).float()

        # Label: choose random speaker id
        label = self.rng.randrange(0, self.num_speakers)
        return face, audio, label


# ---------------------------
# Training routine
# ---------------------------
def train(
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
):
    print(f"Using device: {device}")

    # Dataset and dataloader
    dataset = MAVCelebDataset(num_samples=200, num_speakers=10, seed=123)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = FaceVoiceVerificationModel(device).to(device)

    # Only parameters of projection heads should be trainable (base models frozen)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            # print which params are trainable
    print("Trainable parameter names:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("  ", n)

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = ContrastiveLoss(margin=0.2)

    model.train()
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        running = 0
        for batch_idx, (faces, audios, labels) in enumerate(dataloader, start=1):
            # faces: (B,3,160,160), audios: (B,16000), labels: list or tensor
            # Convert labels to tensor
            labels = torch.tensor(labels, dtype=torch.long)

            # Move to device and forward
            faces = faces.to(device)
            audios = audios.to(device)

            optimizer.zero_grad()
            try:
                E1, E2 = model(audios, faces)  # E1 audio, E2 face
            except Exception as e:
                # If model invocation fails, raise helpful message
                raise RuntimeError(
                    "Model forward failed. This could be due to ModelScope ECAPA_TDNN invocation mismatch. "
                    f"Error: {e}"
                )

            loss = criterion(E1, E2, labels.to(device))
            loss.backward()
            optimizer.step()

            running += 1
            if batch_idx % 1 == 0:
                print(f"  Batch {batch_idx:03d} Loss: {loss.item():.6f}")

    print("\nTraining complete.")


# ---------------------------
# Main entrypoint
# ---------------------------
if __name__ == "__main__":
    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device=device, epochs=5, batch_size=4, lr=1e-4)