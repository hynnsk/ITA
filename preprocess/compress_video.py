"""Prepare 3-fps videos with the shorter side resized to 224 pixels.

Preprocessing adapted from https://github.com/ArrowLuo/CLIP4Clip.
"""

import argparse
import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm


def compress(paths):
    source, target = paths
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-filter:v",
        "scale='if(gt(a,1),trunc(oh*a/2)*2,224)':'if(gt(a,1),224,trunc(ow*a/2)*2)'",
        "-map",
        "0:v:0",
        "-an",
        "-r",
        "3",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed for {source}:\n{result.stderr}")


def prepare_pairs(input_root, output_root):
    extensions = {".mp4", ".mkv", ".avi", ".webm", ".mov"}
    pairs, targets = [], {}
    for source in sorted(input_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in extensions:
            continue
        target = output_root / (source.stem + ".mp4")
        if target in targets:
            raise ValueError(f"Duplicate video ID: {targets[target]} and {source}")
        targets[target] = source
        if not target.exists():
            pairs.append((source, target))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    source, target = args.input_root.resolve(), args.output_root.resolve()
    if not source.is_dir():
        parser.error("--input_root must be an existing video directory")
    if source == target or source in target.parents or target in source.parents:
        parser.error(
            "Input and output directories must be separate, non-nested directories"
        )
    if args.num_workers < 1:
        parser.error("--num_workers must be positive")
    target.mkdir(parents=True, exist_ok=True)
    pairs = prepare_pairs(source, target)
    with Pool(args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(compress, pairs), total=len(pairs)):
            pass


if __name__ == "__main__":
    main()
