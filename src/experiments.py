from __future__ import annotations

import argparse
import itertools
import json
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from src.config import RESULTS
from src.data_pipeline import fit_vectorizer, label_matrix, load_manifest
from src.metrics import calculate_metrics
from src.training import fit_model, predict_probabilities


def _training_partition(max_images=0):
    frame = load_manifest()
    train = frame[frame.split == "official_train"].reset_index(drop=True)
    if max_images and len(train) > max_images:
        train = train.sample(max_images, random_state=42).reset_index(drop=True)
    return train


def run_cv(folds=5, epochs=3, batch_size=8, models=("multimodal",), max_images=0):
    train = _training_partition(max_images)
    if len(train) < folds:
        raise ValueError("Not enough training images for requested folds.")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    results = []
    for kind in models:
        for fold, (train_idx, val_idx) in enumerate(splitter.split(train), start=1):
            tf.keras.backend.clear_session()
            fit = train.iloc[train_idx].reset_index(drop=True)
            holdout = train.iloc[val_idx].reset_index(drop=True)
            vectorizer = fit_vectorizer(fit.caption.astype(str).to_numpy())
            model, _ = fit_model(
                kind, fit, holdout, vectorizer, epochs=epochs,
                batch_size=batch_size, fine_tune=False, verbose=0,
            )
            probs = predict_probabilities(model, holdout, vectorizer, kind, batch_size)
            scores = calculate_metrics(label_matrix(holdout), probs)
            results.append({"model": kind, "fold": fold, **scores})
            print(f"{kind} fold {fold}/{folds}: macro-F1 {scores['macro_f1']:.4f}")
            del model
    summary = {}
    for kind in models:
        group = [r for r in results if r["model"] == kind]
        values = [r["macro_f1"] for r in group]
        summary[kind] = {
            "folds": len(group),
            "macro_f1_mean": float(np.mean(values)),
            "macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "micro_f1_mean": float(np.mean([r["micro_f1"] for r in group])),
            "map_mean": float(np.mean([r["mean_average_precision"] for r in group])),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "KFold on official_train image IDs only; no official_val usage",
              "summary": summary, "fold_results": results}
    (RESULTS / "cross_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return report


def run_search(trials=4, folds=3, epochs=2, max_images=1800):
    train = _training_partition(max_images)
    configs = list(itertools.product([5e-5, 1e-4, 3e-4], [0.3, 0.4, 0.5], [8, 16]))
    random.Random(42).shuffle(configs)
    configs = configs[:trials]
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    reports = []
    for number, (lr, dropout, batch) in enumerate(configs, start=1):
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(train), start=1):
            tf.keras.backend.clear_session()
            fit = train.iloc[train_idx].reset_index(drop=True)
            holdout = train.iloc[val_idx].reset_index(drop=True)
            vectorizer = fit_vectorizer(fit.caption.astype(str).to_numpy())
            model, _ = fit_model(
                "multimodal", fit, holdout, vectorizer, epochs=epochs,
                batch_size=batch, learning_rate=lr, dropout=dropout,
                fine_tune=False, verbose=0,
            )
            probs = predict_probabilities(model, holdout, vectorizer, "multimodal", batch)
            score = calculate_metrics(label_matrix(holdout), probs)["macro_f1"]
            fold_scores.append(score)
            print(f"trial {number} fold {fold}: macro-F1 {score:.4f}")
            del model
        reports.append({
            "trial": number, "learning_rate": lr, "dropout": dropout, "batch_size": batch,
            "macro_f1_mean": float(np.mean(fold_scores)),
            "macro_f1_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            "fold_scores": fold_scores,
        })
    best = max(reports, key=lambda item: item["macro_f1_mean"])
    output = {"search": "bounded randomized search on multimodal model",
              "selection_metric": "mean K-fold macro-F1",
              "best_configuration": best, "trials": reports}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "random_search.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="Cross-validation or randomized hyperparameter search.")
    parser.add_argument("--mode", choices=("cv", "search"), default="cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["multimodal"],
                        choices=("image_only", "text_only", "multimodal"))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "cv":
        run_cv(args.folds, args.epochs, args.batch_size, args.models, args.max_images)
    else:
        run_search(args.trials, args.folds, args.epochs, args.max_images or 1800)


if __name__ == "__main__":
    main()
