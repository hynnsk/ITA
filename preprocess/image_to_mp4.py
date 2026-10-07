"""Convert TVQA/TVR frame folders to videos before compression."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from natsort import natsorted
from tqdm import tqdm


def process_single_video(folder, target, fps):
    images = natsorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Cannot read {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {target}")
        for image in images:
            frame = cv2.imread(str(image))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Invalid frame: {image}")
            writer.write(frame)
    except Exception:
        writer.release()
        target.unlink(missing_ok=True)
        raise
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame_dir", type=Path, required=True)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()
    if not args.frame_dir.is_dir():
        parser.error("--frame_dir must be an existing directory")
    if args.fps < 1 or args.max_workers < 1:
        parser.error("FPS and worker count must be positive")
    folders = sorted(
        {
            p.parent
            for p in args.frame_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
    )
    names = [folder.name for folder in folders]
    if len(set(names)) != len(names):
        parser.error("Frame folders must have unique video IDs")
    args.video_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                process_single_video,
                folder,
                args.video_dir / (folder.name + ".mp4"),
                args.fps,
            )
            for folder in folders
            if not (args.video_dir / (folder.name + ".mp4")).exists()
        ]
        for future in tqdm(futures):
            future.result()


if __name__ == "__main__":
    main()
