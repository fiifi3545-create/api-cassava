"""
Convert best_CustomCNN.pth -> ONNX, then int8-quantize it.

Run once on a machine that has torch installed; the API itself only needs
onnxruntime. Produces:
  cassava_customcnn.onnx      (~104 MB, fp32 — intermediate)
  cassava_customcnn.int8.onnx (~26 MB  — this is what the API serves)

Usage:
  pip install torch onnx onnxruntime
  python export_onnx.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

CKPT = os.getenv("CASSAVA_MODEL_PATH", "best_CustomCNN.pth")
FP32_OUT = "cassava_customcnn.onnx"
INT8_OUT = "cassava_customcnn.int8.onnx"


def export_fp32() -> None:
    import torch

    from model_torch import CustomCNN, NUM_CLASSES

    if not os.path.isfile(CKPT):
        sys.exit(f"Checkpoint not found: {CKPT}")

    model = CustomCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    # dynamo=False: torch 2.12's dynamo exporter emits value_info that conflicts
    # with ONNX shape inference on the flatten -> Gemm boundary, which then breaks
    # quantization. The TorchScript exporter produces a clean graph here.
    torch.onnx.export(
        model,
        dummy,
        FP32_OUT,
        input_names=["input"],
        output_names=["logits"],
        # Batch stays dynamic so the same graph can score several images at once.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {FP32_OUT} ({os.path.getsize(FP32_OUT) / 1e6:.1f} MB)")


def quantize() -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # torch writes the weights beside the graph as external data; fold them back
    # into one self-contained file so the API only has to ship a single artifact.
    merged = "_merged.onnx"
    model = onnx.load(FP32_OUT, load_external_data=True)
    onnx.save_model(model, merged, save_as_external_data=False)

    # Dynamic quantization: weights -> int8, activations stay float. No
    # calibration dataset needed, and the big Linear(50176, 512) is where
    # almost all of the size lives. Symbolic shape pre-processing is skipped —
    # it can't resolve the dynamic batch axis and this graph doesn't need it.
    quantize_dynamic(merged, INT8_OUT, weight_type=QuantType.QInt8)
    os.remove(merged)
    print(f"wrote {INT8_OUT} ({os.path.getsize(INT8_OUT) / 1e6:.1f} MB)")


def verify() -> None:
    """Compare torch vs fp32 ONNX vs int8 ONNX on the same random inputs."""
    import onnxruntime as ort
    import torch

    from model_torch import CustomCNN, NUM_CLASSES

    model = CustomCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(0)
    batch = rng.standard_normal((8, 3, 224, 224)).astype(np.float32)

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(batch)).numpy()

    for path in (FP32_OUT, INT8_OUT):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        onnx_logits = sess.run(None, {"input": batch})[0]
        max_diff = float(np.abs(torch_logits - onnx_logits).max())
        agree = int((torch_logits.argmax(1) == onnx_logits.argmax(1)).sum())
        print(f"{path}: max logit diff {max_diff:.5f} | argmax agreement {agree}/8")


if __name__ == "__main__":
    export_fp32()
    quantize()
    verify()
