"""Acquisition and manifest conversion for the Cataract-101 video dataset."""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path

CATARACT101_URL = (
    "https://ftp.itec.aau.at/datasets/ovid/cat-101/downloads/cataract-101.zip"
)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError(f"unsafe ZIP member: {info.filename}") from error
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)


def download_cataract101(root: Path, keep_archive: bool = True) -> Path:
    """Download the public archive with HTTP range resume, then safely extract it."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cataract-101.zip"
    partial = root / "cataract-101.zip.part"
    output = root / "cataract-101"
    real_videos = ([path for path in output.rglob("*.mp4")
                    if "__MACOSX" not in path.parts and not path.name.startswith("._")]
                   if output.exists() else [])
    if len(real_videos) == 101 and all(list(output.rglob(name)) for name in (
            "annotations.csv", "phases.csv", "videos.csv")):
        return output
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            bad = bundle.testzip()
            if bad:
                raise ValueError(f"corrupt ZIP member: {bad}")
        _safe_extract_zip(archive, output)
        if not keep_archive:
            archive.unlink()
        return output
    headers = {"User-Agent": "surgical-codec-tokens/0.2"}
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(CATARACT101_URL, headers=headers)
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as output:
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
    partial.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError(f"corrupt ZIP member: {bad}")
    _safe_extract_zip(archive, output)
    if not keep_archive:
        archive.unlink()
    return output


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _read_semicolon(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _field(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    by_name = {_norm(key): value for key, value in row.items()}
    for alias in aliases:
        if _norm(alias) in by_name:
            return by_name[_norm(alias)].strip()
    raise ValueError(f"none of fields {aliases} found in {list(row)}")


def _integer(value: str) -> int:
    return int(float(value.strip().replace(",", ".")))


def _video_key(value: str) -> int:
    match = re.search(r"\d+", Path(value).stem)
    if not match:
        raise ValueError(f"could not identify video ID from {value!r}")
    return int(match.group())


def _find_named(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name)
               if "__MACOSX" not in path.parts and not path.name.startswith("._")]
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, found {len(matches)} under {root}")
    return matches[0]


def _split_map(video_ids: list[int], seed: int = 17) -> dict[int, str]:
    ids = sorted(video_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    train_end, val_end = round(n * 0.70), round(n * 0.85)
    return {video_id: ("train" if i < train_end else "val" if i < val_end else "test")
            for i, video_id in enumerate(ids)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_cataract101(source: Path, frames_root: Path, manifest: Path,
                        sample_fps: float = 2.0, split_seed: int = 17) -> int:
    """Decode a sampled index while retaining references to source H.264 frames."""
    import av
    from PIL import Image

    annotation_rows = _read_semicolon(_find_named(source, "annotations.csv"))
    phase_rows = _read_semicolon(_find_named(source, "phases.csv"))
    video_rows = _read_semicolon(_find_named(source, "videos.csv"))
    phase_names = {}
    for index, row in enumerate(phase_rows, start=1):
        try:
            phase_id = _integer(_field(row, ("phase_id", "phaseid", "id")))
        except ValueError:
            phase_id = index
        phase_names[phase_id] = _field(row, ("phase_name", "phasename", "name", "phase"))
    metadata = {
        _video_key(_field(row, ("video_id", "videoid", "id", "video"))): row
        for row in video_rows
    }
    videos = {}
    for path in source.rglob("*.mp4"):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        candidates = {int(value) for value in re.findall(r"\d+", path.stem)} & set(metadata)
        if len(candidates) != 1:
            raise ValueError(f"could not uniquely match MP4 to metadata ID: {path} ({candidates})")
        key = candidates.pop()
        if key in videos:
            raise ValueError(f"duplicate MP4 for video ID {key}: {path} and {videos[key]}")
        videos[key] = path
    if set(metadata) != set(videos):
        raise ValueError(f"video metadata/MP4 ID mismatch: metadata-only={set(metadata)-set(videos)}, "
                         f"mp4-only={set(videos)-set(metadata)}")

    transitions: dict[int, list[tuple[int, int]]] = {key: [] for key in videos}
    for row in annotation_rows:
        video_id = _video_key(_field(row, ("video_id", "videoid", "video")))
        start = _integer(_field(row, ("frame", "frame_no", "frameno", "startframe")))
        phase_id = _integer(_field(row, ("phase_id", "phaseid", "phase")))
        transitions[video_id].append((start, phase_id))
    for values in transitions.values():
        values.sort()
    splits = _split_map(list(videos), split_seed)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "video_id", "video_path", "frame_path", "frame_idx", "sample_idx",
              "time_seconds", "phase", "phase_id", "near_transition", "split"]
    count = 0
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for video_id, video_path in sorted(videos.items()):
            starts = [item[0] for item in transitions[video_id]]
            labels = [item[1] for item in transitions[video_id]]
            if not starts:
                raise ValueError(f"video {video_id} has no phase annotations")
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            interval = fps / sample_fps
            next_source = 0.0
            output_dir = frames_root / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            sample_idx = 0
            for source_idx, frame in enumerate(container.decode(stream)):
                if source_idx < round(next_source):
                    continue
                next_source += interval
                position = max(0, bisect.bisect_right(starts, source_idx) - 1)
                phase_id = labels[position]
                distance = min(abs(source_idx - boundary) for boundary in starts[1:]) if len(starts) > 1 else 10**9
                image_path = output_dir / f"{source_idx:08d}.jpg"
                if not image_path.exists():
                    Image.fromarray(frame.to_ndarray(format="rgb24")).save(image_path, quality=95)
                writer.writerow({
                    "dataset": "cataract101", "video_id": video_path.stem,
                    "video_path": str(video_path.resolve()), "frame_path": str(image_path.resolve()),
                    "frame_idx": source_idx, "sample_idx": sample_idx,
                    "time_seconds": source_idx / fps, "phase": phase_names[phase_id],
                    "phase_id": phase_id - min(phase_names),
                    "near_transition": int(distance <= round(5 * fps)), "split": splits[video_id],
                })
                sample_idx += 1
                count += 1
            container.close()
    split_file = manifest.with_suffix(".splits.json")
    split_file.write_text(json.dumps({
        "seed": split_seed,
        "splits": {name: [key for key, value in splits.items() if value == name]
                   for name in ("train", "val", "test")},
        "source_sha256": {str(key): _sha256(path) for key, path in sorted(videos.items())},
    }, indent=2))
    return count
