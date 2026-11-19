from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

app = FastAPI(
    title="Multimodal Image + Text Classification",
    description="Research demo comparing image-only, caption-only and fused multi-label classifiers.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models/status")
def model_status():
    states = {name: path.is_file() for name, path in MODEL_PATHS.items()}
    return {
        "models": states,
        "vocabulary": (MODELS / "text_vocabulary.json").is_file(),
        "ready": any(states.values()),
        "message": "Run python -m src.train before making predictions."
        if not any(states.values()) else "Available model files are listed.",
    }


@app.get("/api/metrics")
def metrics():
    path = RESULTS / "model_comparison.json"
    if not path.exists():
        return {"available": False, "message": "No evaluation report. Train models first."}
    return {"available": True, "report": json.loads(path.read_text(encoding="utf-8"))}


@lru_cache(maxsize=1)
def _load_artifacts():
    models = {
        name: tf.keras.models.load_model(path, compile=False)
        for name, path in MODEL_PATHS.items() if path.exists()
    }
    vectorizer = None
    vocabulary_path = MODELS / "text_vocabulary.json"
    if vocabulary_path.exists():
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=None, output_mode="int", output_sequence_length=60,
            standardize="lower_and_strip_punctuation", vocabulary=vocabulary,
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
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum upload size is 12 MB.")
    image = _image_tensor(content)
    models, vectorizer = _load_artifacts()
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Trained model files are missing. Run python -m src.train first.",
        )

    results = {}
    for name, model in models.items():
        if name == "image_only":
            results[name] = _format_scores(model, image)
        elif vectorizer is not None:
            tokens = vectorizer(tf.constant([caption])).numpy()
            if name == "text_only":
                results[name] = _format_scores(model, tokens)
            else:
                results[name] = _format_scores(model, {"image": image, "text": tokens})
    return {"results": results, "classes": CLASSES}
