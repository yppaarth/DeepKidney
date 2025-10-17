# setup.py

import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1

def get_device():
    """
    Check for GPU availability and return the appropriate device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    return device

def load_facenet_model(device):
    """
    Load the pre-trained FaceNet model (vggface2), move it to the device,
    set it to evaluation mode, and return it.
    """
    print("[INFO] Loading FaceNet model (vggface2)...")
    model = InceptionResnetV1(pretrained='vggface2').to(device)
    model.eval()
    print("[INFO] Model loaded and set to evaluation mode.")
    return model

if __name__ == '__main__':
    try:
        device = get_device()
        model = load_facenet_model(device)
        print("[SUCCESS] FaceNet model setup completed successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to set up FaceNet model: {e}")


# image_utils.py

import cv2
import numpy as np
from typing import Optional, Tuple

def _load_haar_cascade() -> cv2.CascadeClassifier:
    """
    Load OpenCV's default frontal face Haar cascade.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError(f"Failed to load Haar cascade from: {cascade_path}")
    return cascade

def _largest_face(detections: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Given an array of detections (x, y, w, h), return the largest one by area.
    """
    if detections is None or len(detections) == 0:
        return None
    areas = detections[:, 2] * detections[:, 3]
    idx = int(np.argmax(areas))
    return tuple(int(v) for v in detections[idx])

def preprocess_face_for_pca(
    image_path: str,
    target_size: Tuple[int, int] = (200, 200)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reads an image, detects the main face, crops it, and pads to target_size while
    preserving aspect ratio for PCA/Eigenfaces.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    target_size : Tuple[int, int], optional
        Desired (width, height) of the output standardized image. Default is (200, 200).

    Returns
    -------
    standardized : Optional[np.ndarray]
        Grayscale image of shape (target_h, target_w) with black padding (uint8).
        Returns None if no face is detected or image can't be read.
    original_crop : Optional[np.ndarray]
        Grayscale cropped face (unpadded) as detected. Returns None if no face is detected.

    Notes
    -----
    - Uses OpenCV Haar cascade ('haarcascade_frontalface_default.xml') for detection.
    - If multiple faces are detected, the largest one is used as the "main" face.
    - Both returned arrays are grayscale (uint8).
    """
    target_w, target_h = target_size

    # Read image (as BGR), validate
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Image path invalid or unreadable
        return None, None

    # Convert to grayscale for detection and PCA workflow
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = _load_haar_cascade()
    detections = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30)
    )

    # Choose the largest face
    best = _largest_face(detections)
    if best is None:
        return None, None

    x, y, w, h = best

    # Safety clamp to image bounds
    H, W = img_gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None, None

    # Crop original face (grayscale)
    face_crop = img_gray[y0:y1, x0:x1]

    # Resize with aspect ratio to fit within target_size
    crop_h, crop_w = face_crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return None, None

    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding to reach target_size (centered; extra pixel goes to bottom/right)
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    standardized = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # black padding
    )

    # Ensure exact target size and dtype
    standardized = standardized.astype(np.uint8)
    if standardized.shape != (target_h, target_w):
        standardized = cv2.resize(standardized, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return standardized, face_crop


if __name__ == "__main__":
    # Simple manual test; replace with a real file path
    test_path = "test_face.jpg"
    std_img, orig_crop = preprocess_face_for_pca(test_path, target_size=(200, 200))
    if std_img is None:
        print("[INFO] No face detected or image could not be read.")
    else:
        print("[INFO] Preprocessing successful.")
        print(f"Standardized shape: {std_img.shape}, dtype: {std_img.dtype}")
        print(f"Original crop shape: {orig_crop.shape}, dtype: {orig_crop.dtype}")



# eigenface_extractor.py

import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from image_utils import preprocess_face_for_pca


class EigenfaceExtractor:
    """
    Train a PCA (Eigenfaces) model on standardized 200x200 face images and
    transform new standardized faces into fixed-length embeddings.

    Parameters
    ----------
    n_components : int
        Number of PCA components (embedding size). Default: 512
    """

    def __init__(self, n_components: int = 512):
        self.n_components = int(n_components)
        self.pca: Optional[PCA] = None
        self.fitted_components_: Optional[int] = None
        self.image_shape: Tuple[int, int] = (200, 200)

    def _iter_image_paths(self, image_folder: str) -> List[str]:
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        paths = []
        for root, _, files in os.walk(image_folder):
            for f in files:
                if os.path.splitext(f.lower())[1] in valid_ext:
                    paths.append(os.path.join(root, f))
        return sorted(paths)

    def fit(self, image_folder: str) -> "EigenfaceExtractor":
        """
        Iterate images in `image_folder`, preprocess faces to 200x200,
        flatten to 1D vectors, and fit PCA with n_components=512.

        Images without a detectable face are skipped.

        Raises
        ------
        ValueError
            If no faces are found, or if the dataset contains fewer samples
            than requested n_components.
        """
        image_paths = self._iter_image_paths(image_folder)
        if not image_paths:
            raise ValueError(f"No images found in folder: {image_folder}")

        vectors: List[np.ndarray] = []

        for p in image_paths:
            standardized, _ = preprocess_face_for_pca(p, target_size=self.image_shape)
            if standardized is None:
                continue  # skip images with no detectable face
            if standardized.shape != self.image_shape:
                # safety: enforce shape if upstream changed
                standardized = np.asarray(standardized, dtype=np.uint8)
                standardized = standardized.reshape(self.image_shape)

            # Flatten to 1D vector (length 40000)
            vec = standardized.astype(np.float32).reshape(-1)  # PCA will center
            vectors.append(vec)

        if not vectors:
            raise ValueError("No faces detected in the dataset after preprocessing.")

        X = np.stack(vectors, axis=0)  # shape: (n_samples, 40000)
        n_samples, n_features = X.shape

        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} exceeds min(n_samples, n_features) "
                f"= {min(n_samples, n_features)}. Provide more images or lower n_components."
            )

        # Use randomized SVD for speed on high-dimensional data
        self.pca = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            whiten=False,
            random_state=42,
        )
        self.pca.fit(X)
        self.fitted_components_ = self.n_components
        return self

    def transform(self, standardized_face_image: np.ndarray) -> np.ndarray:
        """
        Transform a single standardized (200x200) face image into a 512-D vector.

        Parameters
        ----------
        standardized_face_image : np.ndarray
            Grayscale 200x200 face (uint8 or float), already standardized as in preprocess_face_for_pca.

        Returns
        -------
        np.ndarray
            1D numpy array of shape (n_components,), the Eigenface embedding.

        Raises
        ------
        RuntimeError
            If the PCA model has not been fitted.
        ValueError
            If input image does not match the expected shape.
        """
        if self.pca is None:
            raise RuntimeError("PCA model is not fitted. Call fit(image_folder) first.")

        if standardized_face_image is None or standardized_face_image.shape != self.image_shape:
            raise ValueError(
                f"Expected image of shape {self.image_shape}, got {None if standardized_face_image is None else standardized_face_image.shape}"
            )

        vec = standardized_face_image.astype(np.float32).reshape(1, -1)  # shape: (1, 40000)
        emb = self.pca.transform(vec)  # shape: (1, n_components)
        return emb.ravel()


if __name__ == "__main__":
    # Minimal sanity check (optional usage example).
    # Provide a folder path via environment variable EIGENFACE_DATASET or edit below.
    folder = os.environ.get("EIGENFACE_DATASET", "")
    if folder and os.path.isdir(folder):
        extractor = EigenfaceExtractor(n_components=512)
        try:
            extractor.fit(folder)
            print("[SUCCESS] PCA model fitted.")
            print(f"Components: {extractor.fitted_components_}")
        except Exception as e:
            print(f"[ERROR] {e}")
    else:
        print("Set EIGENFACE_DATASET to a folder path with face images to test fitting.")







# main.py

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

from facenet_pytorch import MTCNN

from setup import get_device, load_facenet_model
from image_utils import preprocess_face_for_pca
from eigenface_extractor import EigenfaceExtractor

# Optional: import cvlib for gender prediction
try:
    import cvlib as cv
    _HAS_CVLIB = True
except Exception:
    _HAS_CVLIB = False
    print("[WARN] cvlib not available. Gender will default to 0 (male).")


def get_combined_features(image_path, facenet_model, eigenface_model, device):
    """
    Build a 1025-D feature vector from:
      - 512-D Eigenface vector (PCA on 200x200 standardized face)
      - 512-D FaceNet embedding
      - 1-D Gender (0 = male, 1 = female)

    Steps
    -----
    1) preprocess_face_for_pca -> (standardized_200x200_gray, original_face_gray)
    2) Eigenface path: eigenface_model.transform(standardized) -> (512,)
    3) FaceNet path: convert original crop to PIL RGB, use MTCNN -> tensor, then facenet_model -> (512,)
    4) Gender path: cvlib.detect_gender on original crop (BGR or RGB), map to {male:0, female:1}
    5) Concatenate -> (1025,)
    """
    # --- Preprocess for PCA / crop extraction
    standardized, original_crop = preprocess_face_for_pca(image_path, target_size=(200, 200))
    if standardized is None or original_crop is None:
        return None  # No face detected or image not readable

    # --- Eigenface path (expects standardized 200x200 grayscale)
    eigen_vec = eigenface_model.transform(standardized)  # shape: (512,)

    # --- Prepare original crop for FaceNet
    # original_crop from preprocess_face_for_pca is grayscale; convert to BGR for compatibility with downstream
    if len(original_crop.shape) == 2:  # (H, W)
        crop_bgr = cv2.cvtColor(original_crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = original_crop  # already (H, W, 3) BGR

    # Convert BGR -> RGB and then to PIL
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    # --- FaceNet path with MTCNN preprocessing
    mtcnn = MTCNN(image_size=160, margin=0, post_process=True, device=device)
    with torch.no_grad():
        face_tensor = mtcnn(crop_pil)  # shape: (3,160,160) or None
        if face_tensor is None:
            # Fallback: try resizing the whole crop if alignment failed
            face_tensor = torch.tensor(crop_rgb).permute(2, 0, 1).float() / 255.0
            face_tensor = torch.nn.functional.interpolate(
                face_tensor.unsqueeze(0), size=(160, 160), mode="bilinear", align_corners=False
            ).squeeze(0)

        face_tensor = face_tensor.unsqueeze(0).to(device)  # (1,3,160,160)
        facenet_emb = facenet_model(face_tensor)          # (1,512)
        facenet_vec = facenet_emb.squeeze(0).cpu().numpy()  # (512,)

    # --- Gender path
    gender_val = 0  # default male
    if _HAS_CVLIB:
        try:
            # cvlib expects RGB (uint8). It returns (labels, confidences)
            labels, confidences = cv.detect_gender(crop_rgb)
            # labels like ['male', 'female'], confidences e.g. [0.73, 0.27]
            if labels and confidences and len(labels) == len(confidences):
                idx = int(np.argmax(confidences))
                gender_val = 0 if labels[idx].lower().startswith("m") else 1
        except Exception:
            # keep default
            pass
    gender_vec = np.array([gender_val], dtype=np.float32)  # (1,)

    # --- Concatenate: 512 (Eigen) + 512 (FaceNet) + 1 (Gender) = 1025
    combined = np.concatenate([eigen_vec.astype(np.float32), facenet_vec.astype(np.float32), gender_vec], axis=0)
    return combined  # (1025,)


if __name__ == "__main__":
    """
    Example usage:
      python main.py /path/to/image.jpg /path/to/trained_pca.npz
    The PCA model is assumed already trained & available through your EigenfaceExtractor instance.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)

    # Device + FaceNet model
    device = get_device()
    facenet = load_facenet_model(device)

    # Assume EigenfaceExtractor is trained & ready to use
    # If you need to load a persisted PCA, you can store/reload via joblib or numpy elsewhere in your codebase.
    # For demonstration, we create a placeholder and assume it has been fitted.
    extractor = EigenfaceExtractor(n_components=512)
    if extractor.pca is None:
        print("[WARN] EigenfaceExtractor PCA is not fitted in this demo. "
              "Fit it beforehand or load a persisted PCA model.")

    # Attempt feature extraction
    feats = get_combined_features(img_path, facenet, extractor, device)
    if feats is None:
        print("[INFO] No face detected or preprocessing failed.")
        sys.exit(0)

    print(f"[SUCCESS] Combined feature vector shape: {feats.shape} (should be 1025)")



# network.py

import torch
import torch.nn as nn


class FeatureReducer(nn.Module):
    """
    A small feed-forward network that reduces a 1025-D feature vector
    to a 512-D embedding via a hidden layer.
    """
    def __init__(self, input_dim: int = 1025, hidden_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 1025)
        returns: (batch_size, 512)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def build_reducer_network(device: torch.device) -> FeatureReducer:
    """
    Instantiate FeatureReducer and move it to the specified device.
    """
    model = FeatureReducer()
    model.to(device)
    return model


if __name__ == "__main__":
    # Quick sanity check
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_reducer_network(dev)
    dummy = torch.randn(4, 1025, device=dev)  # batch of 4
    out = net(dummy)
    print("Output shape:", out.shape)  # should be (4, 512)



# main.py

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

from facenet_pytorch import MTCNN
from setup import get_device, load_facenet_model
from image_utils import preprocess_face_for_pca
from eigenface_extractor import EigenfaceExtractor
from network import build_reducer_network

# Optional: cvlib for gender prediction
try:
    import cvlib as cv
    _HAS_CVLIB = True
except Exception:
    _HAS_CVLIB = False
    print("[WARN] cvlib not available. Gender will default to 0 (male).")

IMAGE_DIR = "./dataset/"  # Directory containing input images


def get_combined_features(image_path, facenet_model, eigenface_model, device):
    """
    Build a 1025-D feature vector from:
      - 512-D Eigenface vector (PCA on 200x200 standardized face)
      - 512-D FaceNet embedding
      - 1-D Gender (0 = male, 1 = female)
    Returns None if face not found / preprocessing fails.
    """
    # Preprocess: standardized 200x200 (gray) and original face crop (gray)
    standardized, original_crop = preprocess_face_for_pca(image_path, target_size=(200, 200))
    if standardized is None or original_crop is None:
        return None

    # --- Eigenface (expects standardized 200x200 grayscale)
    try:
        eigen_vec = eigenface_model.transform(standardized)  # (512,)
    except Exception as e:
        print(f"[ERROR] Eigenface transform failed for {image_path}: {e}")
        return None

    # --- Prepare original crop for FaceNet
    if len(original_crop.shape) == 2:  # gray -> BGR
        crop_bgr = cv2.cvtColor(original_crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = original_crop
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    # --- FaceNet path (MTCNN alignment/resize)
    mtcnn = MTCNN(image_size=160, margin=0, post_process=True, device=device)
    with torch.no_grad():
        face_tensor = mtcnn(crop_pil)  # (3,160,160) or None
        if face_tensor is None:
            # Fallback: resize the whole crop if alignment fails
            face_tensor = torch.tensor(crop_rgb).permute(2, 0, 1).float() / 255.0
            face_tensor = torch.nn.functional.interpolate(
                face_tensor.unsqueeze(0), size=(160, 160), mode="bilinear", align_corners=False
            ).squeeze(0)

        face_tensor = face_tensor.unsqueeze(0).to(device)  # (1,3,160,160)
        facenet_emb = facenet_model(face_tensor)          # (1,512)
        facenet_vec = facenet_emb.squeeze(0).cpu().numpy()

    # --- Gender
    gender_val = 0  # default male
    if _HAS_CVLIB:
        try:
            labels, confidences = cv.detect_gender(crop_rgb)
            if labels and confidences and len(labels) == len(confidences):
                idx = int(np.argmax(confidences))
                gender_val = 0 if labels[idx].lower().startswith("m") else 1
        except Exception:
            pass
    gender_vec = np.array([gender_val], dtype=np.float32)

    # --- Concatenate: 512 + 512 + 1 = 1025
    combined = np.concatenate(
        [eigen_vec.astype(np.float32), facenet_vec.astype(np.float32), gender_vec], axis=0
    )
    return combined


def _iter_image_files(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(files)


if __name__ == "__main__":
    # --- Init device and models
    device = get_device()
    facenet_model = load_facenet_model(device)
    facenet_model.eval()

    # --- Load/assume trained Eigenface model
    # Expect that EigenfaceExtractor().pca is already trained and assigned.
    # If you persist with joblib, you can load here and assign to extractor.pca.
    extractor = EigenfaceExtractor(n_components=512)
    if extractor.pca is None:
        # Try to load a persisted PCA if present.
        # Comment out this block if you load your model elsewhere.
        try:
            from joblib import load as joblib_load
            if os.path.isfile("eigenface_pca.joblib"):
                extractor.pca = joblib_load("eigenface_pca.joblib")
                extractor.fitted_components_ = extractor.pca.n_components_
                print("[INFO] Loaded trained Eigenface PCA from eigenface_pca.joblib")
            else:
                print("[ERROR] Trained Eigenface PCA not found. Place 'eigenface_pca.joblib' next to this script,")
                print("        or set extractor.pca to your trained PCA before running.")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to load Eigenface PCA: {e}")
            sys.exit(1)

    # --- Build reducer network and set to eval
    reducer = build_reducer_network(device)
    reducer.eval()

    # --- Iterate images and build outputs
    image_files = _iter_image_files(IMAGE_DIR)
    if not image_files:
        print(f"[ERROR] No images found under {IMAGE_DIR}")
        sys.exit(1)

    final_output_vectors = []  # list of (512,) numpy arrays
    kept_paths = []            # optional: track which images produced outputs

    for img_path in image_files:
        feats_1025 = get_combined_features(img_path, facenet_model, extractor, device)
        if feats_1025 is None or feats_1025.shape[0] != 1025:
            print(f"[INFO] Skipping (no face or error): {img_path}")
            continue

        # To tensor on device with batch dim
        x = torch.from_numpy(feats_1025).float().unsqueeze(0).to(device)  # (1,1025)

        with torch.no_grad():
            out_512 = reducer(x)  # (1,512)

        vec_512 = out_512.squeeze(0).cpu().numpy()  # (512,)
        final_output_vectors.append(vec_512)
        kept_paths.append(img_path)

    if not final_output_vectors:
        print("[ERROR] No valid feature vectors produced.")
        sys.exit(1)

    # --- Save to CSV
    df = pd.DataFrame(final_output_vectors)
    df.to_csv("facial_features_output.csv", index=False)
    print(f"[SUCCESS] Saved {len(final_output_vectors)} embeddings to 'facial_features_output.csv'")

    # Optional: also save a mapping of row -> image path for traceability
    mapping_path = "facial_features_output_mapping.csv"
    pd.DataFrame({"image_path": kept_paths}).to_csv(mapping_path, index=False)
    print(f"[INFO] Saved image-path mapping to '{mapping_path}'")