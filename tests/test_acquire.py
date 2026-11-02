import io
import tarfile

import pytest

from surgical_tokens.acquire import safe_extract
from surgical_tokens.tfrecords import split_for_video


def _archive(path, name):
    with tarfile.open(path, "w:gz") as bundle:
        info = tarfile.TarInfo(name)
        value = b"content"
        info.size = len(value)
        bundle.addfile(info, io.BytesIO(value))


def test_safe_extract(tmp_path):
    archive = tmp_path / "ok.tar.gz"
    _archive(archive, "nested/value.txt")
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out/nested/value.txt").read_text() == "content"


def test_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    _archive(archive, "../escape.txt")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(archive, tmp_path / "out")


def test_procedure_split():
    assert split_for_video(1) == "train"
    assert split_for_video(5) == "val"
    assert split_for_video(41) == "test"
