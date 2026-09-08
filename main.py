"""
Cassava Guard — vision API backed by the CustomCNN trained in
`nn-final-project (3).ipynb` (checkpoint: best_CustomCNN.pth).

POST /analyze-cassava with an image; returns disease class, confidence,
analysis and suggestions.

Run:
  pip install -r requirements_api.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Env (.env):
  CASSAVA_MODEL_PATH=./best_CustomCNN.pth   # checkpoint location
  CASSAVA_MIN_CONFIDENCE=0.40               # below this -> "Unclear / not a cassava leaf"
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from advice import analysis_for, suggestions_for
from model import (
    CASSAVA_DATASET_CLASSES,
    UNCLEAR_LABEL,
    ModelNotAvailable,
    load_model,
    model_info,
    predict,
)

load_dotenv()

MODEL_PATH = os.getenv("CASSAVA_MODEL_PATH")
MODEL_NAME = "CustomCNN (ONNX)"

# The network only knows five classes, so it will still answer confidently on a
# photo of something else. A softmax floor is a crude but useful guard.
try:
    MIN_CONFIDENCE = float(os.getenv("CASSAVA_MIN_CONFIDENCE", "0.40"))
except ValueError:
    MIN_CONFIDENCE = 0.40

app = FastAPI(title="Cassava Guard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_origins=["*"],
)


@app.on_event("startup")
def _warm_model() -> None:
    """Load weights at boot so the first scan is not the slow one."""
    try:
        load_model(MODEL_PATH)
    except ModelNotAvailable as e:
        # Don't crash the process — /health reports the problem instead.
        print(f"[cassava_guard] WARNING: {e}")


class CassavaAnalysisResponse(BaseModel):
    disease_class: str = Field(..., description="Predicted class, or Unclear below threshold")
    analysis: str = Field(..., description="What the prediction means; includes confidence")
    suggestions: str = Field(..., description="Farmer-facing management advice")
    confidence: float = Field(..., description="Softmax probability of the predicted class, 0-1")
    probabilities: dict[str, float] = Field(..., description="Probability per class")
    pest: str = Field(..., description="Deprecated alias for disease_class (same value)")


@app.get("/")
def index():
    """Root: what this service is and where to go."""
    return {
        "app": "cassava_guard",
        "model": MODEL_NAME,
        "docs": "/docs",
        "endpoints": {
            "POST /analyze-cassava": "multipart form field 'file' = leaf image",
            "POST /analyze-pest": "legacy alias of /analyze-cassava",
            "GET /health": "service and model status",
            "GET /classes": "the five classes the model predicts",
        },
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    try:
        load_model(MODEL_PATH)
        model_state = "loaded"
    except ModelNotAvailable as e:
        model_state = f"unavailable: {e}"
    return {
        "status": "ok",
        "app": "cassava_guard",
        "model": MODEL_NAME,
        "model_state": model_state,
        "model_file": model_info(MODEL_PATH),
        "min_confidence": MIN_CONFIDENCE,
    }


@app.get("/classes")
def list_classes():
    """Training labels and display labels (for app / training alignment)."""
    return {
        "model": MODEL_NAME,
        "dataset_classes": [
            {"index": c.index, "folder_name": c.folder_name, "label": c.label}
            for c in CASSAVA_DATASET_CLASSES
        ],
        "extra_outcomes": [UNCLEAR_LABEL],
        "note": (
            f"Predictions below {MIN_CONFIDENCE:.2f} softmax confidence are reported as "
            f"'{UNCLEAR_LABEL}'."
        ),
    }


@app.post("/analyze-cassava", response_model=CassavaAnalysisResponse)
async def analyze_cassava(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file.")

    image_data = await file.read()
    if len(image_data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

    try:
        result = predict(image_data, MODEL_PATH)
    except ModelNotAvailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Inference error: {e!s}") from e

    label = result["label"]
    confidence = result["confidence"]
    if confidence < MIN_CONFIDENCE:
        label = UNCLEAR_LABEL

    return CassavaAnalysisResponse(
        disease_class=label,
        pest=label,
        analysis=analysis_for(label, confidence),
        suggestions=suggestions_for(label),
        confidence=round(confidence, 4),
        probabilities=result["probabilities"],
    )


@app.post("/analyze-pest", response_model=CassavaAnalysisResponse)
async def analyze_pest_legacy(file: UploadFile = File(...)):
    """Legacy path name; same behavior as /analyze-cassava."""
    return await analyze_cassava(file)
