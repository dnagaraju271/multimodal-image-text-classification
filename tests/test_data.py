import pytest

from src.config import CATEGORY_TO_CLASS, CLASSES
from src.data_pipeline import label_matrix
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
