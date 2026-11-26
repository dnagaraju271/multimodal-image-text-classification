from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.config import CLASSES, IMAGE_SIZE, MODELS, RESULTS

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
MODEL_PATHS = {
    "image_only": MODELS / "image_only.keras",
    "text_only": MODELS / "text_only.keras",
    "multimodal": MODELS / "multimodal.keras",
}
VOCABULARY_PATH = MODELS / "text_vocabulary.json"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ARTIFACT_PATTERNS = [
    "image_only.keras",
    "text_only.keras",
    "multimodal.keras",
    "text_vocabulary.json",
]
_ARTIFACT_DOWNLOAD_ATTEMPTED = False
_ARTIFACT_DOWNLOAD_ERROR: str | None = None

app = FastAPI(
    title="Multimodal Image + Text Classification",
    description="Research demo comparing image-only, caption-only and fused multi-label classifiers.",
    version="1.1.0",
)

# GitHub Pages hosts the browser interface; inference runs on a separate HTTPS API.
_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://dnagaraju271.github.io,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _ensure_model_artifacts() -> None:
    """Download private/public model artifacts from the Hugging Face Hub when configured."""
    global _ARTIFACT_DOWNLOAD_ATTEMPTED, _ARTIFACT_DOWNLOAD_ERROR

    if all(path.is_file() for path in MODEL_PATHS.values()) and VOCABULARY_PATH.is_file():
        return

    repo_id = os.getenv("HF_MODEL_REPO", "").strip()
    if not repo_id or _ARTIFACT_DOWNLOAD_ATTEMPTED:
        return

    _ARTIFACT_DOWNLOAD_ATTEMPTED = True
    try:
        from huggingface_hub import snapshot_download

        MODELS.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=os.getenv("HF_TOKEN") or None,
            revision=os.getenv("HF_MODEL_REVISION", "main"),
            allow_patterns=ARTIFACT_PATTERNS,
            local_dir=str(MODELS),
        )
        _ARTIFACT_DOWNLOAD_ERROR = None
    except Exception as exc:  # surfaced by the status endpoint without crashing app startup
        _ARTIFACT_DOWNLOAD_ERROR = f"{type(exc).__name__}: {exc}"[:500]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "multimodal-classifier-api"}


@app.get("/api/models/status")
def model_status():
    _ensure_model_artifacts()
    states = {name: path.is_file() for name, path in MODEL_PATHS.items()}
    vocabulary_ready = VOCABULARY_PATH.is_file()

    if all(states.values()) and vocabulary_ready:
        try:
            models, _ = _load_artifacts()
            output_classes = {
                name: int(model.output_shape[-1])
                for name, model in models.items()
            }
            expected = len(CLASSES)
            compatible = all(value == expected for value in output_classes.values())
            message = (
                "All three seven-class models and the text vocabulary are ready."
                if compatible
                else (
                    "Model artifacts are from an older class configuration. "
                    f"Expected {expected} outputs per model; found {output_classes}. "
                    "Retrain all three models with the current seven-class dataset."
                )
            )
            return {
                "models": states,
                "vocabulary": vocabulary_ready,
                "ready": compatible,
                "expected_classes": CLASSES,
                "expected_class_count": expected,
                "model_output_classes": output_classes,
                "message": message,
            }
        except HTTPException as exc:
            return {
                "models": states,
                "vocabulary": vocabulary_ready,
                "ready": False,
                "expected_classes": CLASSES,
                "expected_class_count": len(CLASSES),
                "message": str(exc.detail),
            }

    if _ARTIFACT_DOWNLOAD_ERROR:
        message = f"Model artifact download failed: {_ARTIFACT_DOWNLOAD_ERROR}"
    elif not os.getenv("HF_MODEL_REPO"):
        message = "Model artifacts are missing. Configure HF_MODEL_REPO or place the trained files in models/."
    else:
        message = "Waiting for all three model files and the text vocabulary."
    return {
        "models": states,
        "vocabulary": vocabulary_ready,
        "ready": False,
        "expected_classes": CLASSES,
        "expected_class_count": len(CLASSES),
        "message": message,
    }


@app.get("/api/metrics")
def metrics():
    path = RESULTS / "model_comparison.json"
    if not path.exists():
        return {"available": False, "message": "No evaluation report. Train models first."}
    return {"available": True, "report": json.loads(path.read_text(encoding="utf-8"))}


@lru_cache(maxsize=1)
def _load_artifacts():
    _ensure_model_artifacts()
    missing = [str(path.name) for path in MODEL_PATHS.values() if not path.is_file()]
    if missing or not VOCABULARY_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="The inference service is not ready. Required model files and text vocabulary are not all available.",
        )

    models = {
        name: tf.keras.models.load_model(path, compile=False)
        for name, path in MODEL_PATHS.items()
    }
    output_classes = {
        name: int(model.output_shape[-1])
        for name, model in models.items()
    }
    expected = len(CLASSES)
    incompatible = {
        name: count for name, count in output_classes.items()
        if count != expected
    }
    if incompatible:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artifacts have incompatible output dimensions: {incompatible}. "
                f"The current dataset requires {expected} output classes. "
                "Retrain all three models before running inference."
            ),
        )
    vocabulary = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        output_mode="int",
        output_sequence_length=60,
        standardize="lower_and_strip_punctuation",
        vocabulary=vocabulary,
    )
    return models, vectorizer


def _image_tensor(data: bytes):
    try:
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
    return np.expand_dims(array, axis=0)


def _format_scores(model, inputs):
    logits = np.asarray(model(inputs, training=False))[0]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    ranked = sorted(
        [{"class": name, "probability": float(probabilities[index])}
         for index, name in enumerate(CLASSES)],
        key=lambda item: item["probability"], reverse=True,
    )
    return {
        "predicted_labels": [item["class"] for item in ranked if item["probability"] >= 0.5],
        "ranked_classes": ranked,
        "threshold": 0.5,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), caption: str = Form("")):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Select an image file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum upload size is 12 MB.")

    image = _image_tensor(content)
    models, vectorizer = _load_artifacts()

    # The image-only branch is the primary visual prediction and never receives the caption.
    results = {
        "image_only": _format_scores(models["image_only"], image),
    }

    # Run text and fusion branches only when the user supplies a caption for comparison.
    caption_text = caption.strip() if caption else ""
    if caption_text:
        tokens = vectorizer(tf.constant([caption_text])).numpy()
        results["text_only"] = _format_scores(models["text_only"], tokens)
        results["multimodal"] = _format_scores(
            models["multimodal"], {"image": image, "text": tokens}
        )

    return {
        "results": results,
        "classes": CLASSES,
        "prediction_mode": "image_only" if not caption_text else "image_plus_caption",
    }
