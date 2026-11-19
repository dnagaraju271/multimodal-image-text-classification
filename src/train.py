from __future__ import annotations

import argparse
import json

import pandas as pd
import tensorflow as tf

from src.config import CLASSES, MODELS, RESULTS
from src.data_pipeline import fit_vectorizer, label_matrix, load_manifest
from src.metrics import calculate_metrics
from src.training import MODEL_KINDS, fit_model, load_vectorizer, predict_probabilities, save_vocabulary


def evaluate_saved(frame, vectorizer):
    validation = frame[frame.split == "official_val"].reset_index(drop=True)
    if validation.empty:
        raise ValueError("official_val partition is empty.")
    report = {"task": "multi-label classification", "classes": CLASSES,
              "validation_split": "official_val", "models": {}}
    for kind in MODEL_KINDS:
        path = MODELS / f"{kind}.keras"
        if not path.exists():
            print(f"Skipping {kind}; model artifact is missing.")
            continue
        model = tf.keras.models.load_model(path, compile=False)
        probabilities = predict_probabilities(model, validation, vectorizer, kind)
        report["models"][kind] = calculate_metrics(label_matrix(validation), probabilities)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["models"]:
        rows = [{
            "model": name,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "mean_average_precision": metrics["mean_average_precision"],
            "exact_match_accuracy": metrics["exact_match_accuracy"],
        } for name, metrics in report["models"].items()]
        pd.DataFrame(rows).to_csv(RESULTS / "model_comparison.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Train and compare all three modalities.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    frame = load_manifest()
    train = frame[frame.split == "official_train"].reset_index(drop=True)
    validation = frame[frame.split == "official_val"].reset_index(drop=True)
    print(f"Train images: {len(train)} | held-out validation images: {len(validation)}")
    print("Image counts per class in train:", dict(zip(CLASSES, label_matrix(train).sum(axis=0).astype(int))))
    if args.evaluate_only:
        evaluate_saved(frame, load_vectorizer())
        return

    # Fit text vocabulary only on train2017.
    vectorizer = fit_vectorizer(train.caption.fillna("").astype(str).to_numpy())
    save_vocabulary(vectorizer)
    MODELS.mkdir(parents=True, exist_ok=True)
    for kind in MODEL_KINDS:
        print(f"\\n===== {kind} =====")
        model, history = fit_model(
            kind, train, validation, vectorizer,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, dropout=args.dropout,
            fine_tune=args.fine_tune,
        )
        model.save(MODELS / f"{kind}.keras")
        print(f"Saved {kind}; epochs completed: {len(history.history.get('loss', []))}")
    evaluate_saved(frame, vectorizer)


if __name__ == "__main__":
    main()
