from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw" / "coco2017"
PROCESSED = ROOT / "dataset" / "processed"
MANIFEST = PROCESSED / "manifest.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Seven COCO target categories used for multi-label image classification.
TARGET_CATEGORIES = {
    "bicycle": 2,
    "bird": 16,
    "bus": 6,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "motorcycle": 4,
}

# The proposed 3,000/750 subsets allocate 13.2% of images to multi-label
# examples and distribute the remaining images equally across single-label
# strata. For 3,000 training and 750 validation images this gives:
# 372/93 single-label images per class and 396/99 multi-label images.
MULTI_LABEL_FRACTION = 0.132
CLASSES = list(TARGET_CATEGORIES)
CATEGORY_TO_CLASS = {v: k for k, v in TARGET_CATEGORIES.items()}
IMAGE_SIZE = (224, 224)
MAX_TOKENS = 20000
SEQUENCE_LENGTH = 60
SEED = 42
THRESHOLD = 0.5
