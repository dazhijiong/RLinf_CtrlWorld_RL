#!/usr/bin/env python3
"""Create a balanced local LeRobot subset by selecting fixed episodes per task."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _episode_task_name(episode_row: dict) -> str:
    tasks = episode_row.get("tasks", [])
    if not tasks:
        raise ValueError(f"Episode row does not include tasks: {episode_row}")
    return str(tasks[0])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def select_episodes(
    episodes: list[dict], episodes_per_task: int
) -> tuple[list[dict], dict[int, list[int]]]:
    by_task: dict[int, list[dict]] = defaultdict(list)
    for episode in episodes:
        task_index = int(round(float(episode["stats/task_index/mean"][0])))
        by_task[task_index].append(episode)

    selected: list[dict] = []
    manifest_map: dict[int, list[int]] = {}
    for task_index in sorted(by_task):
        task_episodes = sorted(
            by_task[task_index], key=lambda row: int(row["episode_index"])
        )
        chosen = task_episodes[:episodes_per_task]
        if len(chosen) < episodes_per_task:
            raise ValueError(
                f"Task {task_index} only has {len(chosen)} episodes, expected "
                f"{episodes_per_task}."
            )
        selected.extend(chosen)
        manifest_map[task_index] = [int(row["episode_index"]) for row in chosen]

    return selected, manifest_map


def build_subset(src_root: Path, dst_root: Path, episodes_per_task: int) -> None:
    episodes_path = src_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = src_root / "meta" / "tasks.parquet"
    info_path = src_root / "meta" / "info.json"

    info = _load_json(info_path)
    tasks_table = pq.read_table(tasks_path)
    episodes = pq.read_table(episodes_path).to_pylist()

    selected_episodes, manifest_map = select_episodes(episodes, episodes_per_task)
    selected_old_ids = {int(row["episode_index"]) for row in selected_episodes}
    remap = {
        int(old_episode["episode_index"]): new_episode_index
        for new_episode_index, old_episode in enumerate(
            sorted(selected_episodes, key=lambda row: int(row["episode_index"]))
        )
    }

    selected_by_file: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for episode in selected_episodes:
        selected_by_file[
            (int(episode["data/chunk_index"]), int(episode["data/file_index"]))
        ].append(episode)

    data_rows: list[dict] = []
    manifest_rows: list[dict] = []
    next_global_index = 0
    rewritten_episode_rows: list[dict] = []

    for chunk_file in sorted(selected_by_file):
        chunk_index, file_index = chunk_file
        data_file = src_root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        table = pq.read_table(data_file)
        rows = table.to_pylist()
        rows_by_episode: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            episode_index = int(row["episode_index"])
            if episode_index in selected_old_ids:
                rows_by_episode[episode_index].append(row)

        for original_episode in sorted(
            selected_by_file[chunk_file], key=lambda row: int(row["episode_index"])
        ):
            old_episode_index = int(original_episode["episode_index"])
            new_episode_index = remap[old_episode_index]
            episode_rows = sorted(
                rows_by_episode[old_episode_index], key=lambda row: int(row["frame_index"])
            )
            if not episode_rows:
                raise ValueError(f"No data rows found for episode {old_episode_index}")

            dataset_from_index = next_global_index
            for row in episode_rows:
                rewritten = dict(row)
                rewritten["episode_index"] = new_episode_index
                rewritten["index"] = next_global_index
                data_rows.append(rewritten)
                next_global_index += 1
            dataset_to_index = next_global_index

            length = len(episode_rows)
            index_values = list(range(dataset_from_index, dataset_to_index))

            updated_episode = dict(original_episode)
            updated_episode["episode_index"] = new_episode_index
            updated_episode["data/chunk_index"] = 0
            updated_episode["data/file_index"] = 0
            updated_episode["meta/episodes/chunk_index"] = 0
            updated_episode["meta/episodes/file_index"] = 0
            updated_episode["dataset_from_index"] = dataset_from_index
            updated_episode["dataset_to_index"] = dataset_to_index
            updated_episode["length"] = length
            updated_episode["stats/episode_index/min"] = [new_episode_index]
            updated_episode["stats/episode_index/max"] = [new_episode_index]
            updated_episode["stats/episode_index/mean"] = [float(new_episode_index)]
            updated_episode["stats/episode_index/std"] = [0.0]
            updated_episode["stats/episode_index/count"] = [length]
            updated_episode["stats/index/min"] = [dataset_from_index]
            updated_episode["stats/index/max"] = [dataset_to_index - 1]
            updated_episode["stats/index/mean"] = [_mean(index_values)]
            updated_episode["stats/index/std"] = [_std(index_values)]
            updated_episode["stats/index/count"] = [length]
            rewritten_episode_rows.append(updated_episode)

            manifest_rows.append(
                {
                    "task_index": int(round(float(original_episode["stats/task_index/mean"][0]))),
                    "task": _episode_task_name(original_episode),
                    "old_episode_index": old_episode_index,
                    "new_episode_index": new_episode_index,
                    "length": length,
                }
            )

    dst_meta = dst_root / "meta" / "episodes" / "chunk-000"
    dst_data = dst_root / "data" / "chunk-000"
    dst_meta.mkdir(parents=True, exist_ok=True)
    dst_data.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.Table.from_pylist(data_rows), dst_data / "file-000.parquet")
    pq.write_table(
        pa.Table.from_pylist(rewritten_episode_rows), dst_meta / "file-000.parquet"
    )
    pq.write_table(tasks_table, dst_root / "meta" / "tasks.parquet")

    subset_info = dict(info)
    subset_info["total_episodes"] = len(rewritten_episode_rows)
    subset_info["total_frames"] = len(data_rows)
    subset_info["total_tasks"] = len(manifest_map)
    subset_info["splits"] = {"train": f"0:{len(rewritten_episode_rows)}"}
    _dump_json(dst_root / "meta" / "info.json", subset_info)

    _dump_json(
        dst_root / "meta" / "subset_manifest.json",
        {
            "source_root": str(src_root),
            "subset_root": str(dst_root),
            "episodes_per_task": episodes_per_task,
            "total_tasks": len(manifest_map),
            "total_episodes": len(rewritten_episode_rows),
            "total_frames": len(data_rows),
            "selected_episode_indices_by_task": manifest_map,
            "selected_episodes": manifest_rows,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_subset(args.src_root.resolve(), args.dst_root.resolve(), args.episodes_per_task)


if __name__ == "__main__":
    main()
