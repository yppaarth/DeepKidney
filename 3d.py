import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
import argparse
import hashlib

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_wav_mono_16k(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
        wav = wav_t.squeeze(0).numpy()
    return wav

def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-10)

def deterministic_projection(in_dim=192, out_dim=512, seed_text="3dspeaker_projection"):
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((out_dim, in_dim))
    Q, _ = np.linalg.qr(A.T)
    return Q.T[:out_dim, :in_dim].astype(np.float32)

# ---------------------------------------------------------
# Embedding extraction logic
# ---------------------------------------------------------

def extract_embedding(model_path, wav_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Load TorchScript model
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    # Load audio
    wav = load_wav_mono_16k(wav_path)
    x = torch.from_numpy(wav).float().unsqueeze(0)

    with torch.no_grad():
        emb = model(x)
    emb = emb.squeeze(0).cpu().numpy()
    emb = l2norm(emb)

    # Save native (192D)
    np.save(os.path.join(out_dir, "native_192d.npy"), emb)

    # Project to 512D
    W = deterministic_projection(in_dim=emb.shape[-1], out_dim=512)
    emb512 = l2norm(W @ emb)
    np.save(os.path.join(out_dir, "embed_512d.npy"), emb512)

    print(f"✅ Saved embeddings in {out_dir}")
    print(f"Native shape: {emb.shape}, Projected shape: {emb512.shape}")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to local model.pt file")
    parser.add_argument("--wav_path", required=True, help="Path to input WAV file (≤1s)")
    parser.add_argument("--out_dir", default="./embeddings", help="Output directory")
    args = parser.parse_args()

    extract_embedding(args.model_path, args.wav_path, args.out_dir)


import argparse, os, json, hashlib
import numpy as np
import soundfile as sf
import torch
import torchaudio

from modelscope.hub.snapshot_download import snapshot_download
from pathlib import Path

# Minimal wrapper to use the 3D-Speaker ModelScope checkpoint via their infer script logic
# without depending on full training stack. We call their shipped "infer_sv.py" model loader.

# --- Utilities ---
def load_wav_mono_16k(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=16000)
        wav = wav_t.squeeze(0).numpy()
        sr = 16000
    return wav, sr

def l2norm(x, eps=1e-10):
    n = np.linalg.norm(x) + eps
    return x / n

def deterministic_ortho_proj(in_dim=192, out_dim=512, seed_text="3d_speaker_proj_v1"):
    # Build a deterministic orthonormal projection (no training needed)
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % (2**31-1)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((out_dim, in_dim))
    # Orthonormalize rows via QR
    Q, _ = np.linalg.qr(A.T)
    W = Q.T[:out_dim, :in_dim]
    return W.astype(np.float32)

# --- Model loader via ModelScope ---
def load_model(model_id: str):
    # Download model files locally
    model_dir = snapshot_download(model_id)
    # Try to locate a TorchScript/pt model and a config used by 3D-Speaker
    # Common names include 'model.pt', 'pytorch_model.bin', or 'model.pth'
    # We’ll try standard names and let torch.load resolve weights.
    # The repo’s inference scripts wrap the backbone; here we use torch.jit if available,
    # otherwise we import their speakerlab if present.
    pt_candidates = [
        "model.pt", "pytorch_model.bin", "model.pth", "weights.pt"
    ]
    found = None
    for c in pt_candidates:
        p = Path(model_dir) / c
        if p.exists():
            found = str(p)
            break
    if found is None:
        raise FileNotFoundError(f"Cannot find a model .pt in {model_dir}.")
    # Load TorchScript if scripted
    try:
        model = torch.jit.load(found, map_location="cpu")
        model.eval()
        return model, "jit"
    except Exception:
        # Fallback – try regular state_dict with a simple wrapper (if the checkpoint is plain)
        # Many ModelScope SV models export TorchScript. If not, you can import speakerlab here.
        raise RuntimeError("This checkpoint is not TorchScript. Please use the infer script shipped by 3D-Speaker.")

def extract_embed_with_jit(model, wav_16k: np.ndarray):
    # Most JIT speaker encoders expect (1, T) float32 @16k, return (1, D)
    x = torch.from_numpy(wav_16k).float().unsqueeze(0)
    with torch.no_grad():
        emb = model(x)  # (1, D)
    return emb.squeeze(0).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Path to single WAV (≤1s ok).")
    ap.add_argument("--model_id", default="iic/speech_eres2netv2_sv_zh-cn_16k-common")
    ap.add_argument("--out_dir", default="emb_out")
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wav, _ = load_wav_mono_16k(args.wav)
    model, kind = load_model(args.model_id)

    # Native embedding (likely 192-D for ERes2NetV2)
    emb = extract_embed_with_jit(model, wav)
    emb = l2norm(emb)
    np.save(os.path.join(args.out_dir, Path(args.wav).stem + "_native.npy"), emb)

    # Project to 512-D deterministically
    W = deterministic_ortho_proj(in_dim=emb.shape[-1], out_dim=512)
    emb512 = l2norm(W @ emb)
    np.save(os.path.join(args.out_dir, Path(args.wav).stem + "_512d.npy"), emb512)

    if args.save_json:
        meta = {
            "wav": args.wav,
            "model_id": args.model_id,
            "native_dim": int(emb.shape[-1]),
            "projected_dim": 512,
            "note": "Native from 3D-Speaker model via ModelScope; projected to 512-D with deterministic orthonormal map."
        }
        with open(os.path.join(args.out_dir, Path(args.wav).stem + "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print("Saved:",
          os.path.join(args.out_dir, Path(args.wav).stem + "_native.npy"),
          os.path.join(args.out_dir, Path(args.wav).stem + "_512d.npy"))

if __name__ == "__main__":
    main()    