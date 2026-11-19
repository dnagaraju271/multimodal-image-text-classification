from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support

from src.config import CLASSES, THRESHOLD


def calculate_metrics(y_true, probabilities, threshold=THRESHOLD):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    p, r, f, support = precision_recall_fscore_support(
        y_true, predicted, average=None, zero_division=0
    )
    per_class, aps = {}, []
    for i, name in enumerate(CLASSES):
        ap = float(average_precision_score(y_true[:, i], probabilities[:, i])) if y_true[:, i].sum() else None
        if ap is not None:
            aps.append(ap)
        per_class[name] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "support": int(support[i]), "average_precision": ap,
        }
    micro_p, micro_r, _, _ = precision_recall_fscore_support(
        y_true, predicted, average="micro", zero_division=0
    )
    macro_p, macro_r, _, _ = precision_recall_fscore_support(
        y_true, predicted, average="macro", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(f1_score(y_true, predicted, average="micro", zero_division=0)),
        "mean_average_precision": float(np.mean(aps)) if aps else 0.0,
        "exact_match_accuracy": float(accuracy_score(y_true, predicted)),
        "num_images": int(len(y_true)),
        "per_class": per_class,
    }
