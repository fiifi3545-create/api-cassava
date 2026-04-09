"""
Cassava Guard — vision API (OpenRouter-compatible) for cassava leaf images.
POST /analyze-cassava with an image; returns disease class, analysis, suggestions.

Dataset folder names (training labels) are listed in CASSAVA_DATASET_CLASSES.
The model may also return other outcomes when the leaf does not fit those classes.

Run:
  pip install -r requirements_api.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

.env:
  OPENROUTER_API_KEY=sk-or-...
  # optional: CASSAVA_VISION_MODEL=openai/gpt-4o-mini
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv("CASSAVA_VISION_MODEL") or os.getenv("PEST_VISION_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


@dataclass(frozen=True)
class CassavaClassInfo:
    """One class as stored on disk (folder name) and human-readable label."""

    folder_name: str
    label: str


# Order matches common PlantVillage-style cassava naming; folder_name must match your dataset dirs.
CASSAVA_DATASET_CLASSES: tuple[CassavaClassInfo, ...] = (
    CassavaClassInfo("cassava _mosaic_disease", "Cassava mosaic disease"),
    CassavaClassInfo("Cassava___bacterial_blight", "Cassava bacterial blight"),
    CassavaClassInfo("Cassava___brown_streak_disease", "Cassava brown streak disease"),
    CassavaClassInfo("Cassava___green_mottle", "Cassava green mottle"),
    CassavaClassInfo("Cassava___healthy", "Healthy"),
)

# Section 1 lines the LLM is allowed to use (canonical outputs).
_CANONICAL_LABELS: tuple[str, ...] = tuple(c.label for c in CASSAVA_DATASET_CLASSES) + (
    "Other cassava problem (specify briefly)",
    "Unclear / not a cassava leaf",
)


def _class_prompt_block() -> str:
    lines = "\n".join(f"- {lab}" for lab in _CANONICAL_LABELS)
    folders = "\n".join(
        f"- {c.folder_name} → report as “{c.label}”" for c in CASSAVA_DATASET_CLASSES
    )
    return f"""Training/dataset classes (folder names on disk → label to use in Section 1):
{folders}

Section 1 must be EXACTLY one of these lines (copy spelling):
{lines}

If the leaf shows a different cassava disorder not in the list, use “Other cassava problem (specify briefly)” and name it in Section 2.
If the image is not a cassava leaf, too blurry, or unrelated, use “Unclear / not a cassava leaf”."""


def _get_client() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("CHRISKEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Set OPENROUTER_API_KEY (or CHRISKEY) in .env.",
        )
    return OpenAI(base_url=OPENROUTER_BASE, api_key=key)


app = FastAPI(title="Cassava Guard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_origins=["*"],
)


class CassavaAnalysisResponse(BaseModel):
    disease_class: str = Field(..., description="Canonical class or other/unclear outcome")
    analysis: str = Field(..., description="What is visible; uncertainty")
    suggestions: str = Field(..., description="Farmer-facing management advice")
    pest: str = Field(
        ...,
        description="Deprecated alias for disease_class (same value)",
    )


def _encode_image(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _analyze_cassava_leaf_image(image_base64: str) -> str:
    system = f"""You are an agricultural extension assistant for smallholder farmers in Ghana and similar climates.
The user sends a photo that should be a cassava (Manihot esculenta) leaf for disease/health assessment.

{_class_prompt_block()}

Respond in exactly three sections separated by the line ---SECTION---
Section 1: exactly one of the allowed lines above (single line).
Section 2 (2–5 sentences): visible signs; note uncertainty; if “Other”, name the suspected issue.
Section 3: practical suggestions — cultural/mechanical first; involve extension when needed; avoid specific pesticide brands unless essential.

Crops only — not human medical advice."""

    user_text = (
        "Classify this cassava leaf image using the allowed Section 1 labels. "
        "Prefer the dataset class names when symptoms fit; otherwise use Other or Unclear as defined."
    )

    cli = _get_client()
    rsp = cli.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return rsp.choices[0].message.content or ""


def _split_sections(raw: str) -> tuple[str, str, str]:
    parts = re.split(r"\s*---SECTION---\s*", raw.strip(), maxsplit=2)
    if len(parts) >= 3:
        return parts[0].strip(), parts[1].strip(), parts[2].strip()
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs[0], paragraphs[1], paragraphs[2]
    if len(paragraphs) == 2:
        return paragraphs[0], paragraphs[1], "Consult your agriculture extension agent for confirmation."
    if len(paragraphs) == 1:
        return "Unclear / not a cassava leaf", paragraphs[0], "Verify in the field with extension support."
    return "Unclear / not a cassava leaf", raw.strip() or "No content", "Retake a closer, well-lit photo of the leaf."


def _snap_disease_line(line: str) -> str:
    """Map close matches to canonical labels; otherwise return stripped line."""
    s = line.strip()
    if s in _CANONICAL_LABELS:
        return s
    low = s.lower()
    for lab in _CANONICAL_LABELS:
        if lab.lower() == low:
            return lab
    for c in CASSAVA_DATASET_CLASSES:
        if c.folder_name.lower() in low or c.label.lower() in low:
            return c.label
    return s


def _normalize_unclear(
    disease_class: str, analysis: str, suggestions: str
) -> tuple[str, str, str]:
    text = f"{disease_class} {analysis}".lower()
    unclear_markers = [
        "unclear",
        "not a cassava",
        "non-crop",
        "cannot identify",
        "insufficient detail",
        "blurry",
        "out of focus",
        "not cassava",
    ]
    if any(marker in text for marker in unclear_markers):
        return (
            "Unclear / not a cassava leaf",
            analysis,
            "Retake the photo: single cassava leaf, fill the frame, in good daylight. "
            "If symptoms persist, consult a local extension officer.",
        )
    return disease_class, analysis, suggestions


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "app": "cassava_guard"}


@app.get("/classes")
def list_classes():
    """Dataset folder names and display labels (for app / training alignment)."""
    return {
        "dataset_classes": [
            {"folder_name": c.folder_name, "label": c.label} for c in CASSAVA_DATASET_CLASSES
        ],
        "section1_options": list(_CANONICAL_LABELS),
        "note": "API may return Other/Unclear when the image does not match the five dataset classes.",
    }


def _analysis_result(disease: str, analysis: str, suggestions: str) -> CassavaAnalysisResponse:
    return CassavaAnalysisResponse(
        disease_class=disease,
        pest=disease,
        analysis=analysis,
        suggestions=suggestions,
    )


@app.post("/analyze-cassava", response_model=CassavaAnalysisResponse)
async def analyze_cassava(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file.")

    image_data = await file.read()
    if len(image_data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

    try:
        full = _analyze_cassava_leaf_image(_encode_image(image_data))
        disease, analysis, suggestions = _split_sections(full)
        disease = _snap_disease_line(disease)
        disease, analysis, suggestions = _normalize_unclear(disease, analysis, suggestions)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vision model error: {e!s}") from e

    return _analysis_result(disease, analysis, suggestions)


@app.post("/analyze-pest", response_model=CassavaAnalysisResponse)
async def analyze_pest_legacy(file: UploadFile = File(...)):
    """Legacy path name; same behavior as /analyze-cassava."""
    return await analyze_cassava(file)
