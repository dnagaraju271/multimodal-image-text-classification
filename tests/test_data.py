import pytest

from src.config import CATEGORY_TO_CLASS, CLASSES
from src.data_pipeline import class_image_counts, label_matrix
from src.data_pipeline import load_manifest, _stratified_subset


def test_candidate_category_map_is_unique():
    assert len(CATEGORY_TO_CLASS) == len(CLASSES)
    assert CLASSES == [
        "bicycle", "bird", "bus", "cat", "dog", "horse", "motorcycle"
    ]


def test_label_matrix_encodes_multiple_labels():
    import pandas as pd
    frame = pd.DataFrame({"labels_list": [["cat", "bicycle"], ["dog"]]})
    matrix = label_matrix(frame)
    assert matrix.tolist() == [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]


def test_manifest_requires_prepared_data():
    from src.config import MANIFEST
    if not MANIFEST.exists():
        with pytest.raises(FileNotFoundError):
            load_manifest()



def test_class_coverage_counts_in_memory_label_lists():
    import pandas as pd

    frame = pd.DataFrame({
        "labels": [["cat", "bicycle"], ["dog"], ["cat", "dog", "horse"]]
    })
    counts = class_image_counts(frame)
    assert counts["bicycle"] == 1
    assert counts["cat"] == 2
    assert counts["dog"] == 2
    assert counts["horse"] == 1


def test_class_coverage_counts_json_serialized_labels():
    import json
    import pandas as pd

    frame = pd.DataFrame({
        "labels": [json.dumps(["bird", "cat"]), json.dumps(["bird", "bus"])]
    })
    counts = class_image_counts(frame)
    assert counts["bird"] == 2
    assert counts["cat"] == 1
    assert counts["bus"] == 1



def test_stratified_subset_is_reproducible_and_covers_seven_classes():
    from collections import Counter
    from src.config import CLASSES

    rows = []
    image_id = 1

    # Supply enough examples in each single-label stratum.
    for name in CLASSES:
        for _ in range(12):
            rows.append({"image_id": image_id, "labels": [name]})
            image_id += 1

    # Supply enough multi-label examples for the configured quota.
    for index in range(12):
        rows.append({
            "image_id": image_id,
            "labels": [CLASSES[index % len(CLASSES)],
                       CLASSES[(index + 1) % len(CLASSES)]],
        })
        image_id += 1

    first = _stratified_subset(rows, limit=75, seed=123)
    second = _stratified_subset(rows, limit=75, seed=123)

    first_ids = [row["image_id"] for row in first]
    second_ids = [row["image_id"] for row in second]

    assert len(first) == 75
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))

    single_counts = Counter(
        row["labels"][0]
        for row in first
        if len(row["labels"]) == 1
    )
    assert set(single_counts) == set(CLASSES)
    assert min(single_counts.values()) >= 9
    assert sum(len(row["labels"]) > 1 for row in first) == 10


def test_stratified_subset_rejects_insufficient_class_candidates():
    rows = []
    image_id = 1

    # The first six single-label strata have enough examples; motorcycle does not.
    for name in CLASSES[:-1]:
        for _ in range(12):
            rows.append({"image_id": image_id, "labels": [name]})
            image_id += 1

    # Include enough multi-label candidates so the missing stratum is the failure.
    for index in range(12):
        rows.append({
            "image_id": image_id,
            "labels": [CLASSES[index % 6], CLASSES[(index + 1) % 6]],
        })
        image_id += 1

    with pytest.raises(ValueError, match="Not enough motorcycle-only candidates"):
        _stratified_subset(rows, limit=75, seed=42)
