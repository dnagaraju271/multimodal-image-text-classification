import pytest

from src.config import CATEGORY_TO_CLASS, CLASSES
from src.data_pipeline import class_image_counts, label_matrix
from src.data_pipeline import load_manifest


def test_candidate_category_map_is_unique():
    assert len(CATEGORY_TO_CLASS) == len(CLASSES)
    assert CLASSES == ["bicycle", "bird", "bus", "cat", "dog", "horse"]


def test_label_matrix_encodes_multiple_labels():
    import pandas as pd
    frame = pd.DataFrame({"labels_list": [["cat", "bicycle"], ["dog"]]})
    matrix = label_matrix(frame)
    assert matrix.tolist() == [[1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]


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
