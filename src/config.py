from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw" / "coco2017"
PROCESSED = ROOT / "dataset" / "processed"
MANIFEST = PROCESSED / "manifest.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Initial candidate classes. Confirm image counts in coverage_report.json.
TARGET_CATEGORIES = {
    "bicycle": 2,
    "bird": 16,
    "bus": 6,
    "cat": 17,
    "dog": 18,
    "horse": 19,
}
CLASSES = list(TARGET_CATEGORIES)
CATEGORY_TO_CLASS = {v: k for k, v in TARGET_CATEGORIES.items()}
IMAGE_SIZE = (224, 224)
MAX_TOKENS = 20000
SEQUENCE_LENGTH = 60
SEED = 42
THRESHOLD = 0.5
