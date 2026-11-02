"""Deterministic controlled HEVC reconstruction from official 1-fps PNG frames."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def encode_hevc(frame_dir: Path, output: Path, fps: int = 1, crf: int = 23, gop: int = 32) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise ValueError(f"no PNG frames in {frame_dir}")
    try:
        start_number = int(frames[0].stem)
    except ValueError as error:
        raise ValueError("PNG filenames must be numeric, for example 000001.png") from error
    expected = [start_number + i for i in range(len(frames))]
    actual = [int(path.stem) for path in frames]
    if actual != expected:
        raise ValueError("PNG frame numbers must be contiguous")
    output.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "%06d.png")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
        "-start_number", str(start_number), "-i", pattern, "-c:v", "libx265",
        "-preset", "medium", "-crf", str(crf),
        "-x265-params", f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0",
        "-pix_fmt", "yuv420p", str(output),
    ]
    subprocess.run(command, check=True)
    provenance = {
        "source": "TF-Cholec80 1-fps PNG reconstruction",
        "not_original_acquisition_codec": True,
        "fps": fps, "crf": crf, "gop": gop, "frames": len(frames), "command": command,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(provenance, indent=2))
    return provenance
