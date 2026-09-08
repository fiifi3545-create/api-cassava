"""
CustomCNN classifier for cassava leaf images — ONNX Runtime inference.

The network was trained in `nn-final-project (3).ipynb` (PyTorch) and exported
with export_onnx.py. The API runs the int8-quantized graph, so torch is NOT a
runtime dependency; preprocessing below reproduces the notebook's
`valid_transforms` exactly.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from PIL import Image

IMG_SIZE = 224
NUM_CLASSES = 5

# ImageNet stats — same values as `valid_transforms` in the notebook.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_HERE, "cassava_customcnn.int8.onnx")
# Full-precision graph, used when CASSAVA_USE_FP32=1.
FP32_MODEL_PATH = os.path.join(_HERE, "cassava_customcnn.onnx")


@dataclass(frozen=True)
class CassavaClassInfo:
    """One class as trained (integer label) and human-readable label."""

    index: int
    folder_name: str
    label: str


# Kaggle `label_num_to_disease_map.json` order — the integer labels in train.csv
# that the notebook trained against. Order matters: index == output neuron.
CASSAVA_DATASET_CLASSES: tuple[CassavaClassInfo, ...] = (
    CassavaClassInfo(0, "Cassava___bacterial_blight", "Cassava bacterial blight"),
    CassavaClassInfo(1, "Cassava___brown_streak_disease", "Cassava brown streak disease"),
    CassavaClassInfo(2, "Cassava___green_mottle", "Cassava green mottle"),
    CassavaClassInfo(3, "cassava _mosaic_disease", "Cassava mosaic disease"),
    CassavaClassInfo(4, "Cassava___healthy", "Healthy"),
)

UNCLEAR_LABEL = "Unclear / not a cassava leaf"


class ModelNotAvailable(RuntimeError):
    """Model file missing or unreadable."""


def _resolve_path(path: str | None) -> str:
    if path:
        return path
    env = os.getenv("CASSAVA_MODEL_PATH")
    if env:
        return env
    if os.getenv("CASSAVA_USE_FP32", "").strip() in {"1", "true", "yes"}:
        return FP32_MODEL_PATH
    return DEFAULT_MODEL_PATH


def _download_model(url: str, dest: str) -> None:
    """Fetch the graph at boot — for hosts that don't ship it in the image."""
    import urllib.request

    print(f"[cassava_guard] downloading model from {url}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = f"{dest}.part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        raise ModelNotAvailable(f"Could not download model from {url}: {e}") from e


@lru_cache(maxsize=1)
def load_session(path: str | None = None) -> ort.InferenceSession:
    """Build the ONNX Runtime session once per process."""
    model_path = _resolve_path(path)

    if not os.path.isfile(model_path):
        url = os.getenv("CASSAVA_MODEL_URL")
        if url:
            _download_model(url, model_path)
    if not os.path.isfile(model_path):
        raise ModelNotAvailable(
            f"Model not found at {model_path}. Run export_onnx.py, set "
            "CASSAVA_MODEL_PATH, or set CASSAVA_MODEL_URL to download it at startup."
        )

    opts = ort.SessionOptions()
    # One host thread per request keeps memory flat on small instances.
    opts.intra_op_num_threads = int(os.getenv("CASSAVA_ORT_THREADS", "1"))
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        return ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    except Exception as e:  # noqa: BLE001
        raise ModelNotAvailable(f"Could not load {model_path}: {e}") from e


# Backwards-compatible alias — main.py warms the model through this.
load_model = load_session


def model_info(path: str | None = None) -> dict:
    resolved = _resolve_path(path)
    return {
        "path": resolved,
        "precision": "int8" if resolved.endswith(".int8.onnx") else "fp32",
        "size_mb": (
            round(os.path.getsize(resolved) / 1e6, 1) if os.path.isfile(resolved) else None
        ),
    }


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Bytes -> (1, 3, 224, 224) float32, matching the notebook's valid_transforms."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001 — surfaced to the caller as a 400
        raise ValueError(f"Could not decode image: {e}") from e

    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def predict(image_bytes: bytes, model_path: str | None = None) -> dict:
    """Run the CNN and return the label, confidence and full probability vector."""
    session = load_session(model_path)
    tensor = preprocess(image_bytes)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0][0]
    probs = _softmax(np.asarray(logits, dtype=np.float32))

    top_idx = int(np.argmax(probs))
    info = CASSAVA_DATASET_CLASSES[top_idx]
    return {
        "index": top_idx,
        "label": info.label,
        "folder_name": info.folder_name,
        "confidence": float(probs[top_idx]),
        "probabilities": {
            c.label: round(float(probs[c.index]), 4) for c in CASSAVA_DATASET_CLASSES
        },
    }
