from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    CATEGORY_TO_CLASS, CLASSES, IMAGE_SIZE, MANIFEST, MAX_TOKENS,
    MULTI_LABEL_FRACTION, PROCESSED, RAW, SEQUENCE_LENGTH, TARGET_CATEGORIES,
)

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"


def _download(url: str, target: Path, timeout: int = 90) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "ISEP-Multimodal-Project/1.0"})
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_annotations() -> Path:
    folder = RAW / "annotations"
    required = [
        folder / "instances_train2017.json", folder / "captions_train2017.json",
        folder / "instances_val2017.json", folder / "captions_val2017.json",
    ]
    if all(p.exists() for p in required):
        return folder
    archive = RAW / "annotations_trainval2017.zip"
    # Re-download a stale or incomplete archive instead of trusting any non-empty file.
    if archive.exists():
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != ANNOTATIONS_SHA256:
            print("Existing annotation archive failed SHA-256 verification; downloading again.")
            archive.unlink()
    print("Downloading COCO 2017 annotations archive from the official HTTP endpoint...")
    _download(ANNOTATIONS_URL, archive, timeout=240)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ANNOTATIONS_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            "COCO annotation archive SHA-256 mismatch. The download was removed; "
            "check the network or use the official COCO download page."
        )
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing annotation files: " + ", ".join(missing))
    return folder


def _candidates(split: str, folder: Path) -> list[dict]:
    with (folder / f"instances_{split}2017.json").open(encoding="utf-8") as f:
        instance_data = json.load(f)
    with (folder / f"captions_{split}2017.json").open(encoding="utf-8") as f:
        caption_data = json.load(f)
    image_labels: dict[int, set[str]] = {}
    for annotation in instance_data["annotations"]:
        category = int(annotation["category_id"])
        if category in CATEGORY_TO_CLASS:
            image_labels.setdefault(int(annotation["image_id"]), set()).add(CATEGORY_TO_CLASS[category])
    image_captions: dict[int, list[str]] = {}
    for annotation in caption_data["annotations"]:
        caption = " ".join(str(annotation.get("caption", "")).split())
        if caption:
            image_captions.setdefault(int(annotation["image_id"]), []).append(caption)
    result = []
    for image in instance_data["images"]:
        image_id = int(image["id"])
        if image_id not in image_labels or image_id not in image_captions:
            continue
        result.append({
            "image_id": image_id,
            "file_name": image["file_name"],
            "url": image.get("coco_url") or "",
            "labels": sorted(image_labels[image_id]),
            "caption": " . ".join(image_captions[image_id]),
        })
    return result


def _stratified_subset(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Sample single-label examples equally by class plus a fixed multi-label share.

    The default 13.2% multi-label share yields 372 single-label examples per
    class plus 396 multi-label examples for a 3,000-image training subset,
    and 93 per class plus 99 multi-label examples for a 750-image validation
    subset. Candidate rows are sampled only within their official COCO split.
    """
    if limit < len(CLASSES):
        raise ValueError(
            f"Image limit must be at least {len(CLASSES)} to include every class."
        )

    image_ids = [int(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Candidate rows contain duplicate image IDs.")

    multi_limit = int(round(limit * MULTI_LABEL_FRACTION))
    single_total = limit - multi_limit
    base_quota, remainder = divmod(single_total, len(CLASSES))
    if base_quota < 1:
        raise ValueError("Image limit is too small for the configured class strata.")

    single_quotas = {
        name: base_quota + int(index < remainder)
        for index, name in enumerate(CLASSES)
    }
    single_pools = {
        name: [
            row for row in rows
            if set(row["labels"]) == {name}
        ]
        for name in CLASSES
    }
    multi_pool = [
        row for row in rows if len(set(row["labels"])) > 1
    ]

    for name, quota in single_quotas.items():
        available = len(single_pools[name])
        if available < quota:
            raise ValueError(
                f"Not enough {name}-only candidates: need {quota}, have {available}."
            )
    if len(multi_pool) < multi_limit:
        raise ValueError(
            f"Not enough multi-label candidates: need {multi_limit}, "
            f"have {len(multi_pool)}."
        )

    rng = random.Random(seed)
    selected = []
    for name in CLASSES:
        selected.extend(rng.sample(single_pools[name], single_quotas[name]))
    if multi_limit:
        selected.extend(rng.sample(multi_pool, multi_limit))

    selected.sort(key=lambda row: int(row["image_id"]))
    selected_ids = [int(row["image_id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Stratified sampling produced duplicate image IDs.")
    if len(selected) != limit:
        raise RuntimeError(
            f"Stratified sampling produced {len(selected)} rows; expected {limit}."
        )
    return selected


def _download_image(row: dict, split: str) -> tuple[dict, str | None]:
    target = RAW / "images" / f"{split}2017" / row["file_name"]
    # COCO publishes these image endpoints as HTTP. Do not rewrite them to HTTPS:
    # on some networks images.cocodataset.org presents a certificate for another hostname.
    url = row["url"] or f"http://images.cocodataset.org/{split}2017/{row['file_name']}"
    try:
        _download(url, target, timeout=60)
        result = row.copy()
        result["image_path"] = str(target.relative_to(MANIFEST.parents[2]))
        result["split"] = "official_" + split
        return result, None
    except Exception as exc:
        return row, str(exc)


def prepare(max_train: int = 3000, max_val: int = 750, workers: int = 8) -> pd.DataFrame:
    folder = _ensure_annotations()
    rows = []
    for split, limit in (("train", max_train), ("val", max_val)):
        candidates = _candidates(split, folder)
        print(f"{split}2017 candidate images with labels and captions: {len(candidates)}")
        selected = _stratified_subset(candidates, limit, 42 if split == "train" else 43)
        print(
            f"  stratified selection: "
            f"{sum(len(set(row['labels'])) == 1 for row in selected)} single-label, "
            f"{sum(len(set(row['labels'])) > 1 for row in selected)} multi-label"
        )
        completed, failures = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_image, row, split) for row in selected]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                item, error = future.result()
                if error:
                    failures.append(error)
                else:
                    completed.append(item)
                if index % 100 == 0 or index == len(futures):
                    print(f"  {split}: {index}/{len(futures)} processed")
        if failures:
            print(f"Warning: {len(failures)} image downloads failed; first error: {failures[0]}")
        rows.extend(completed)
    if not rows:
        raise RuntimeError("No paired COCO images were downloaded.")
    frame = pd.DataFrame(rows)[["image_id", "file_name", "image_path", "caption", "labels", "split"]]
    frame = frame.sort_values(["split", "image_id"]).drop_duplicates("image_id").reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.assign(labels=frame.labels.map(json.dumps)).to_csv(MANIFEST, index=False)
    coverage = {}
    sampling_summary = {}
    for split, group in frame.groupby("split"):
        coverage[split] = class_image_counts(group, labels_column="labels")
        label_sets = [set(_labels_as_list(value)) for value in group["labels"]]
        sampling_summary[split] = {
            "single_label_images": int(sum(len(labels) == 1 for labels in label_sets)),
            "multi_label_images": int(sum(len(labels) > 1 for labels in label_sets)),
            "single_label_images_by_class": {
                name: int(sum(labels == {name} for labels in label_sets))
                for name in CLASSES
            },
        }
    report = {
        "task": "seven-class multi-label classification",
        "classes": CLASSES,
        "coco_category_ids": TARGET_CATEGORIES,
        "images": int(len(frame)),
        "split_counts": {k: int(v) for k, v in frame.split.value_counts().items()},
        "class_image_counts": coverage,
        "sampling": {
            "strategy": "equal single-class strata plus a sampled multi-label stratum",
            "target_multi_label_fraction": MULTI_LABEL_FRACTION,
            "actual_by_split": sampling_summary,
        },
        "note": "Class counts overlap because an image can contain several target classes.",
    }
    (PROCESSED / "coverage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return frame


def load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        raise FileNotFoundError("Dataset not prepared. Run python -m src.data_pipeline first.")
    frame = pd.read_csv(MANIFEST)
    required = {"image_id", "image_path", "caption", "labels", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest must contain columns {sorted(required)}")
    if frame.image_id.duplicated().any():
        raise ValueError("Expected one manifest row per image_id.")
    frame.image_id = frame.image_id.astype(int)
    frame["labels_list"] = frame.labels.map(json.loads)
    for labels in frame.labels_list:
        unknown = set(labels).difference(CLASSES)
        if unknown:
            raise ValueError(f"Unknown labels: {sorted(unknown)}")
        if not labels:
            raise ValueError("Empty target label list found.")
    root = MANIFEST.parents[2]
    missing = [p for p in frame.image_path if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")
    train_ids = set(frame.loc[frame.split == "official_train", "image_id"])
    val_ids = set(frame.loc[frame.split == "official_val", "image_id"])
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Image leakage across official splits: {sorted(overlap)[:10]}")
    if not train_ids or not val_ids:
        raise ValueError("Both official_train and official_val partitions are required.")
    return frame


def _labels_as_list(value) -> list[str]:
    """Normalize labels held as Python lists or JSON-serialized CSV strings."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple, set)):
        raise TypeError(f"Expected labels as a list or JSON string, got {type(value).__name__}.")
    return list(value)


def class_image_counts(frame: pd.DataFrame, labels_column: str = "labels") -> dict[str, int]:
    """Count images containing each class; one image may count for multiple classes."""
    counts = {name: 0 for name in CLASSES}
    for value in frame[labels_column]:
        present = set(_labels_as_list(value))
        for name in CLASSES:
            counts[name] += int(name in present)
    return counts


def label_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [[int(label in row) for label in CLASSES] for row in frame.labels_list],
        dtype="float32",
    )


def fit_vectorizer(captions):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(np.asarray(captions, dtype=str)).batch(256))
    return vectorizer


def make_dataset(frame, vectorizer, modality: str, batch_size: int = 8, shuffle: bool = False):
    root = MANIFEST.parents[2]
    paths = np.asarray([str(root / path) for path in frame.image_path], dtype=str)
    captions = frame.caption.fillna("").astype(str).to_numpy()
    labels = label_matrix(frame)
    ds = tf.data.Dataset.from_tensor_slices((paths, captions, labels))
    if shuffle:
        ds = ds.shuffle(len(frame), seed=42, reshuffle_each_iteration=True)

    def transform(path, caption, target):
        inputs = {}
        if modality in ("image_only", "multimodal"):
            image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
            inputs["image"] = tf.cast(image, tf.float32)
        if modality in ("text_only", "multimodal"):
            inputs["text"] = vectorizer(caption)

        # A single-input Keras Functional model expects a tensor, not a
        # one-key dictionary. Fusion accepts its named input dictionary.
        if modality == "image_only":
            return inputs["image"], target
        if modality == "text_only":
            return inputs["text"], target
        return inputs, target

    return ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description="Prepare a bounded MS COCO 2017 image-caption dataset.")
    parser.add_argument("--max-train-images", type=int, default=3000)
    parser.add_argument("--max-val-images", type=int, default=750)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    prepare(args.max_train_images, args.max_val_images, args.workers)


if __name__ == "__main__":
    main()
