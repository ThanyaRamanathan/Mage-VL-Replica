import zipfile

import pytest

from surgical_tokens.cataract101 import _find_named, _split_map, _safe_extract_zip


def test_cataract_split_is_deterministic_and_procedure_disjoint():
    first = _split_map(list(range(1, 102)), seed=17)
    second = _split_map(list(range(1, 102)), seed=17)
    assert first == second
    assert sum(value == "train" for value in first.values()) == 71
    assert sum(value == "val" for value in first.values()) == 15
    assert sum(value == "test" for value in first.values()) == 15


def test_cataract_zip_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="unsafe"):
        _safe_extract_zip(archive, tmp_path / "out")


def test_named_files_ignore_macos_metadata(tmp_path):
    real = tmp_path / "dataset" / "annotations.csv"
    sidecar = tmp_path / "__MACOSX" / "dataset" / "annotations.csv"
    real.parent.mkdir(parents=True)
    sidecar.parent.mkdir(parents=True)
    real.write_text("real")
    sidecar.write_text("metadata")
    assert _find_named(tmp_path, "annotations.csv") == real
