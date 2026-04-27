#!/usr/bin/env python3
"""Convert a video-backed local LeRobot dataset to an image-backed one.

This creates a brand-new dataset directory without modifying the source.
Camera feature keys are preserved as-is; only their storage changes from
video streams to image structs with relative file paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import imageio
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task", type=str, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_tasks_table(tasks_rows: list[dict[str, Any]], task_override: str | None) -> list[dict[str, Any]]:
    normalized = []
    for row in tasks_rows:
        task = row.get("task")
        if task is None:
            task = row.get("__index_level_0__")
        if task_override is not None:
            task = task_override
        normalized.append(
            {
                "task_index": int(row["task_index"]),
                "task": str(task),
            }
        )
    return normalized


def extract_video_to_images(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    camera_keys: list[str],
    jpeg_quality: int,
) -> None:
    video_template = info["video_path"]

    for camera_key in camera_keys:
        file_refs = sorted(
            {
                (
                    int(episode[f"videos/{camera_key}/chunk_index"]),
                    int(episode[f"videos/{camera_key}/file_index"]),
                )
                for episode in episodes
            }
        )
        for chunk_index, file_index in file_refs:
            video_rel_path = video_template.format(
                video_key=camera_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            video_path = src_root / video_rel_path
            image_dir = (
                dst_root
                / "images"
                / camera_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}"
            )
            ensure_dir(image_dir)

            reader = imageio.get_reader(os.fspath(video_path), format="ffmpeg")
            try:
                for frame_index, frame in enumerate(reader):
                    image_path = image_dir / f"frame_{frame_index:06d}.jpg"
                    if image_path.exists():
                        continue
                    Image.fromarray(frame).save(
                        image_path,
                        format="JPEG",
                        quality=jpeg_quality,
                    )
            finally:
                reader.close()


def build_image_backed_rows(
    src_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    camera_keys: list[str],
) -> list[dict[str, Any]]:
    data_path = src_root / info["data_path"].format(chunk_index=0, file_index=0)
    rows = pq.read_table(data_path).to_pylist()
    rows.sort(key=lambda row: int(row["index"]))
    episodes_by_index = {int(episode["episode_index"]): episode for episode in episodes}
    fps = float(info["fps"])

    new_rows: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        episode = episodes_by_index[int(row["episode_index"])]
        frame_index = int(row["frame_index"])
        for camera_key in camera_keys:
            chunk_index = int(episode[f"videos/{camera_key}/chunk_index"])
            file_index = int(episode[f"videos/{camera_key}/file_index"])
            start_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
            global_frame_index = int(round(start_timestamp * fps)) + frame_index
            row[camera_key] = {
                "bytes": None,
                "path": (
                    f"images/{camera_key}/chunk-{chunk_index:03d}/"
                    f"file-{file_index:03d}/frame_{global_frame_index:06d}.jpg"
                ),
            }
        new_rows.append(row)
    return new_rows


def build_info_json(src_info: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    dst_info = dict(src_info)
    dst_info["video_files_size_in_mb"] = 0
    features = dict(src_info["features"])
    for camera_key in camera_keys:
        feature = dict(features[camera_key])
        feature["dtype"] = "image"
        feature.pop("info", None)
        feature["fps"] = src_info["fps"]
        features[camera_key] = feature
    for key, feature in list(features.items()):
        if key not in camera_keys and isinstance(feature, dict):
            feature = dict(feature)
            feature.setdefault("fps", src_info["fps"])
            features[key] = feature
    dst_info["features"] = features
    return dst_info


def main() -> None:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dst_root}")
        shutil.rmtree(dst_root)

    ensure_dir(dst_root)
    ensure_dir(dst_root / "data" / "chunk-000")
    ensure_dir(dst_root / "meta")
    ensure_dir(dst_root / "meta" / "episodes" / "chunk-000")

    info = json.loads((src_root / "meta" / "info.json").read_text())
    camera_keys = sorted(
        key
        for key, feature in info["features"].items()
        if key.startswith("observation.images.") and feature.get("dtype") == "video"
    )
    episodes = pq.read_table(src_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    tasks_rows = pq.read_table(src_root / "meta" / "tasks.parquet").to_pylist()

    print(f"camera_keys={camera_keys}")
    print(f"episodes={len(episodes)}")

    extract_video_to_images(
        src_root=src_root,
        dst_root=dst_root,
        info=info,
        episodes=episodes,
        camera_keys=camera_keys,
        jpeg_quality=args.jpeg_quality,
    )

    new_rows = build_image_backed_rows(
        src_root=src_root,
        info=info,
        episodes=episodes,
        camera_keys=camera_keys,
    )
    pq.write_table(
        pa.Table.from_pylist(new_rows),
        dst_root / "data" / "chunk-000" / "file-000.parquet",
    )

    pq.write_table(
        pa.Table.from_pylist(episodes),
        dst_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(normalize_tasks_table(tasks_rows, args.task)),
        dst_root / "meta" / "tasks.parquet",
    )
    (dst_root / "meta" / "info.json").write_text(
        json.dumps(build_info_json(info, camera_keys), indent=4)
    )

    print(f"wrote dataset to {dst_root}")


if __name__ == "__main__":
    main()
