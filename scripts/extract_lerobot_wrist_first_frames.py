#!/usr/bin/env python3
import argparse
import json
import pathlib
import re

import pyarrow.compute as pc
import pyarrow.parquet as pq


def slugify_task(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
    return slug or "unknown_task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract first-frame wrist images from a LeRobot dataset grouped by task."
    )
    parser.add_argument("--lerobot-dir", required=True, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where grouped wrist images and manifests will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lerobot_dir = pathlib.Path(args.lerobot_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_table = pq.read_table(lerobot_dir / "meta" / "tasks.parquet")
    task_rows = tasks_table.to_pylist()
    task_index_to_name = {row["task_index"]: row["task"] for row in task_rows}

    data_files = sorted((lerobot_dir / "data").glob("chunk-*/*.parquet"))
    manifest_records = []
    grouped_summary: dict[str, dict] = {}

    for data_file in data_files:
        table = pq.read_table(
            data_file,
            columns=[
                "observation.images.wrist_image",
                "episode_index",
                "frame_index",
                "task_index",
            ],
        )
        first_frame_mask = pc.equal(table["frame_index"], 0)
        first_frame_table = table.filter(first_frame_mask)

        for row in first_frame_table.to_pylist():
            task_index = row["task_index"]
            task_name = task_index_to_name[task_index]
            task_slug = slugify_task(task_name)
            task_dir = output_dir / task_slug
            task_dir.mkdir(parents=True, exist_ok=True)

            episode_index = int(row["episode_index"])
            image_bytes = row["observation.images.wrist_image"]["bytes"]
            image_name = f"episode_{episode_index:04d}_wrist_first.png"
            image_path = task_dir / image_name
            image_path.write_bytes(image_bytes)

            relative_path = image_path.relative_to(output_dir)
            manifest_records.append(
                {
                    "task": task_name,
                    "task_slug": task_slug,
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "relative_path": str(relative_path),
                    "source_data_file": str(data_file.relative_to(lerobot_dir)),
                    "frame_index": 0,
                }
            )

            grouped_summary.setdefault(
                task_name,
                {"task_slug": task_slug, "task_index": task_index, "count": 0, "files": []},
            )
            grouped_summary[task_name]["count"] += 1
            grouped_summary[task_name]["files"].append(str(relative_path))

    manifest_records.sort(key=lambda record: (record["task"], record["episode_index"]))
    for entry in grouped_summary.values():
        entry["files"].sort()

    summary = {
        "lerobot_dir": str(lerobot_dir),
        "output_dir": str(output_dir),
        "num_tasks": len(grouped_summary),
        "num_images": len(manifest_records),
        "tasks": grouped_summary,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_records, indent=2, ensure_ascii=True) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )


if __name__ == "__main__":
    main()
