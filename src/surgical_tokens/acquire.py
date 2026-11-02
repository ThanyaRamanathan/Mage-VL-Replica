"""Safe acquisition helpers for the official CAMMA TF-Cholec80 release."""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

OFFICIAL_URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz"


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, destination: Path) -> None:
    """Download atomically, retaining a partial file if the network fails."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "surgical-codec-tokens/0.1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    partial.replace(destination)


def _safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"unsafe archive member: {member_name}") from error
    return target


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files/directories only; reject traversal and links."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            target = _safe_target(destination, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"links/devices are not allowed: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read archive member: {member.name}")
                with source, target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
            else:
                raise ValueError(f"unsupported archive member: {member.name}")


def acquire(root: Path, expected_digest: str | None = None, keep_archive: bool = False) -> Path:
    archive = root / "cholec80.tar.gz"
    output = root / "cholec80"
    if not archive.exists():
        download(OFFICIAL_URL, archive)
    if expected_digest:
        algorithm = "md5" if len(expected_digest) == 32 else "sha256"
        actual = digest(archive, algorithm)
        if actual.lower() != expected_digest.strip().lower():
            raise ValueError(f"{algorithm} mismatch: expected {expected_digest}, got {actual}")
    safe_extract(archive, output)
    if not keep_archive:
        archive.unlink()
    return output
