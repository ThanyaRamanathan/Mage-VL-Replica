from __future__ import annotations

import argparse
import json
from pathlib import Path


def acquire_data(args):
    from .acquire import acquire

    if not args.accept_noncommercial_terms:
        raise SystemExit("pass --accept-noncommercial-terms after reviewing CAMMA's license")
    output = acquire(Path(args.root), args.checksum, args.keep_archive)
    print(json.dumps({"dataset": str(output.resolve())}, indent=2))


def convert_data(args):
    from .tfrecords import convert_tfrecords

    count = convert_tfrecords(Path(args.source), Path(args.frames), Path(args.manifest))
    print(json.dumps({"frames": count, "manifest": str(Path(args.manifest).resolve())}, indent=2))


def reencode(args):
    from .reencode import encode_hevc

    result = encode_hevc(Path(args.frames), Path(args.output), args.fps, args.crf, args.gop)
    print(json.dumps(result, indent=2))


def acquire_cataract(args):
    from .cataract101 import download_cataract101

    if not args.accept_noncommercial_terms:
        raise SystemExit("pass --accept-noncommercial-terms after reviewing Cataract-101 terms")
    output = download_cataract101(Path(args.root), args.keep_archive)
    print(json.dumps({"dataset": str(output.resolve())}, indent=2))


def convert_cataract(args):
    from .cataract101 import convert_cataract101

    count = convert_cataract101(Path(args.source), Path(args.frames), Path(args.manifest),
                                args.sample_fps, args.split_seed)
    print(json.dumps({"frames": count, "manifest": str(Path(args.manifest).resolve())}, indent=2))


def inspect_codec(args):
    from .codec import iter_codec
    from .selectors import codec_mask

    frames = [frame for index, frame in iter_codec(args.video, args.limit)
              if index % args.sample_step == 0]
    mask, scores = codec_mask(
        frames, args.budget, args.patch, args.motion_weight, args.anchor_fraction)
    payload = {
        "video": str(Path(args.video).resolve()),
        "frames": len(frames),
        "keyframes": sum(f.keyframe for f in frames),
        "grid": list(scores.shape[1:]),
        "requested_tokens": args.budget,
        "selected_tokens": int(mask.sum()),
        "residual_signal": "motion_compensated_l1_proxy",
        "tokens_per_frame": mask.sum((1, 2)).tolist(),
        "sample_step": args.sample_step,
        "anchor_fraction": args.anchor_fraction,
    }
    print(json.dumps(payload, indent=2))


def compare(args):
    rows = []
    for directory in args.runs:
        path = Path(directory) / "results.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        item = json.loads(path.read_text())
        rows.append(item)
    required = {"selector", "visual_tokens", "accuracy", "macro_f1", "latency_ms"}
    if any(not required <= row.keys() for row in rows):
        raise SystemExit(f"every results.json must contain: {sorted(required)}")
    rows.sort(key=lambda x: (x["visual_tokens"], x["selector"]))
    print("selector\ttokens\taccuracy\tmacro_f1\tlatency_ms")
    for x in rows:
        print(f'{x["selector"]}\t{x["visual_tokens"]}\t{x["accuracy"]:.4f}\t'
              f'{x["macro_f1"]:.4f}\t{x["latency_ms"]:.2f}')


def validate_config(args):
    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    selector = config.get("selector")
    if selector not in {"uniform", "codec", "clinical"}:
        raise SystemExit("selector must be uniform, codec, or clinical")
    manifest = Path(args.manifest or config["data"]["manifest"])
    errors = []
    if not manifest.exists():
        errors.append(f"manifest does not exist: {manifest}")
    budgets = config.get("budgets", [])
    if not budgets or any(not isinstance(x, int) or x <= 0 for x in budgets):
        errors.append("budgets must be non-empty positive integers")
    print(json.dumps({"config": str(args.config), "selector": selector, "budgets": budgets,
                      "ready": not errors, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Matched-budget surgical patch selection")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("acquire-tf-cholec80", help="download and safely extract official CAMMA data")
    p.add_argument("--root", required=True)
    p.add_argument("--checksum", help="expected MD5 (32 chars) or SHA-256")
    p.add_argument("--keep-archive", action="store_true")
    p.add_argument("--accept-noncommercial-terms", action="store_true")
    p.set_defaults(func=acquire_data)
    p = sub.add_parser("convert-tf-cholec80", help="convert per-video TFRecords to PNG + CSV")
    p.add_argument("--source", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=convert_data)
    p = sub.add_parser("acquire-cataract101", help="download and safely extract Cataract-101")
    p.add_argument("--root", required=True)
    p.add_argument("--keep-archive", action="store_true")
    p.add_argument("--accept-noncommercial-terms", action="store_true")
    p.set_defaults(func=acquire_cataract)
    p = sub.add_parser("convert-cataract101", help="create 1-fps frames and manifest")
    p.add_argument("--source", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--split-seed", type=int, default=17)
    p.set_defaults(func=convert_cataract)
    p = sub.add_parser("reencode-hevc", help="make a controlled HEVC stream from 1-fps PNGs")
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--gop", type=int, default=32)
    p.set_defaults(func=reencode)
    p = sub.add_parser("inspect-codec", help="extract and summarize codec-selected patches")
    p.add_argument("video")
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--motion-weight", type=float, default=0.5)
    p.add_argument("--limit", type=int)
    p.add_argument("--sample-step", type=int, default=1)
    p.add_argument("--anchor-fraction", type=float, default=0.5)
    p.set_defaults(func=inspect_codec)
    p = sub.add_parser("validate-config", help="validate an experiment before GPU submission")
    p.add_argument("config")
    p.add_argument("--manifest", help="override data.manifest from the config")
    p.set_defaults(func=validate_config)
    p = sub.add_parser("compare", help="tabulate generated results.json artifacts")
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
