#!/usr/bin/env python3
"""Filter redundant low-motion frames from a local image-backed LeRobot book dataset.

This script creates a brand-new dataset directory. For each episode, it keeps the
first frame, then drops later frames whose robot motion amplitude relative to the
previous kept frame is too small. Kept frames are re-indexed contiguously, the
parquet rows are rewritten, and the corresponding images are copied over.

The source dataset is expected to follow the local ``lerobot_book`` layout:

- ``meta/info.json``
- ``meta/tasks.parquet``
- ``meta/episodes/chunk-000/file-000.parquet``
- ``data/chunk-000/file-000.parquet``
- ``images/<camera_key>/episode-XXXXXX/frame-XXXXXX.png``
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BOOK_IMAGE_KEYS = (
    "observation.images.d405_rgb",
    "observation.images.d405_1_rgb",
    "observation.images.d435_rgb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument(
        "--state-threshold",
        type=float,
        default=1e-3,
        help="Drop a frame when ||state_t - state_prev_kept||_2 < threshold.",
    )
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=None,
        help=(
            "Optional extra threshold on ||action_t - action_prev_kept||_2. "
            "If unset, only the state threshold is used."
        ),
    )
    parser.add_argument(
        "--logical-op",
        choices=("and", "or"),
        default="and",
        help=(
            "How to combine state/action low-motion checks when action-threshold is set. "
            "'and' means both must be small to drop; 'or' means either one is small."
        ),
    )
    parser.add_argument(
        "--min-frames-per-episode",
        type=int,
        default=2,
        help="Always keep at least this many frames per episode when possible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove dst-root first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many frames would be removed.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _l2_delta(prev_value: Any, curr_value: Any) -> float:
    prev_arr = np.asarray(prev_value, dtype=np.float32)
    curr_arr = np.asarray(curr_value, dtype=np.float32)
    return float(np.linalg.norm(curr_arr - prev_arr))


def _quantiles(values: list[float]) -> tuple[float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    return tuple(float(x) for x in np.quantile(arr, [0.01, 0.10, 0.50, 0.90, 0.99]))


def _array_stat_lists(rows: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    arr = np.asarray([row[key] for row in rows], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]

    out: dict[str, list[float]] = {
        "min": np.min(arr, axis=0).astype(np.float64).tolist(),
        "max": np.max(arr, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(arr, axis=0).astype(np.float64).tolist(),
        "std": np.std(arr, axis=0).astype(np.float64).tolist(),
        "count": [int(arr.shape[0])],
    }

    q01_list: list[float] = []
    q10_list: list[float] = []
    q50_list: list[float] = []
    q90_list: list[float] = []
    q99_list: list[float] = []
    for dim in range(arr.shape[1]):
        q01, q10, q50, q90, q99 = _quantiles(arr[:, dim].tolist())
        q01_list.append(q01)
        q10_list.append(q10)
        q50_list.append(q50)
        q90_list.append(q90)
        q99_list.append(q99)
    out["q01"] = q01_list
    out["q10"] = q10_list
    out["q50"] = q50_list
    out["q90"] = q90_list
    out["q99"] = q99_list
    return out


def _scalar_stat_lists(rows: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {
            "min": [0.0],
            "max": [0.0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0],
            "q01": [0.0],
            "q10": [0.0],
            "q50": [0.0],
            "q90": [0.0],
            "q99": [0.0],
        }
    q01, q10, q50, q90, q99 = _quantiles(values)
    return {
        "min": [float(min(values))],
        "max": [float(max(values))],
        "mean": [float(sum(values) / len(values))],
        "std": [float(np.std(np.asarray(values, dtype=np.float64)))],
        "count": [len(values)],
        "q01": [q01],
        "q10": [q10],
        "q50": [q50],
        "q90": [q90],
        "q99": [q99],
    }


def _done_stat_lists(rows: list[dict[str, Any]]) -> dict[str, list[float | bool]]:
    values = [bool(row["next.done"]) for row in rows]
    numeric = [1.0 if value else 0.0 for value in values]
    q01, q10, q50, q90, q99 = _quantiles(numeric)
    return {
        "min": [bool(min(values))],
        "max": [bool(max(values))],
        "mean": [float(sum(numeric) / len(numeric)) if numeric else 0.0],
        "std": [float(np.std(np.asarray(numeric, dtype=np.float64))) if numeric else 0.0],
        "count": [len(numeric)],
        "q01": [q01],
        "q10": [q10],
        "q50": [q50],
        "q90": [q90],
        "q99": [q99],
    }


def _update_episode_stats(episode_row: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    updated = dict(episode_row)
    length = len(rows)
    updated["length"] = length

    state_stats = _array_stat_lists(rows, "observation.state")
    action_stats = _array_stat_lists(rows, "action")
    timestamp_stats = _scalar_stat_lists(rows, "timestamp")
    frame_stats = _scalar_stat_lists(rows, "frame_index")
    index_stats = _scalar_stat_lists(rows, "index")
    task_stats = _scalar_stat_lists(rows, "task_index")
    episode_stats = _scalar_stat_lists(rows, "episode_index")
    done_stats = _done_stat_lists(rows)

    for name, stats in (
        ("observation.state", state_stats),
        ("action", action_stats),
        ("timestamp", timestamp_stats),
        ("frame_index", frame_stats),
        ("index", index_stats),
        ("task_index", task_stats),
        ("episode_index", episode_stats),
        ("next.done", done_stats),
    ):
        for stat_name, stat_value in stats.items():
            updated[f"stats/{name}/{stat_name}"] = stat_value

    return updated


def _copy_frame_set(
    src_root: Path,
    dst_root: Path,
    episode_index: int,
    old_to_new_frame: dict[int, int],
) -> None:
    for image_key in BOOK_IMAGE_KEYS:
        dst_episode_dir = (
            dst_root / "images" / image_key / f"episode-{episode_index:06d}"
        )
        _ensure_dir(dst_episode_dir)
        for old_frame_index, new_frame_index in old_to_new_frame.items():
            src = (
                src_root
                / "images"
                / image_key
                / f"episode-{episode_index:06d}"
                / f"frame-{old_frame_index:06d}.png"
            )
            dst = dst_episode_dir / f"frame-{new_frame_index:06d}.png"
            shutil.copy2(src, dst)


def _should_drop(
    prev_kept_row: dict[str, Any],
    curr_row: dict[str, Any],
    state_threshold: float,
    action_threshold: float | None,
    logical_op: str,
) -> bool:
    state_small = _l2_delta(
        prev_kept_row["observation.state"], curr_row["observation.state"]
    ) < state_threshold
    if action_threshold is None:
        return state_small

    action_small = _l2_delta(prev_kept_row["action"], curr_row["action"]) < action_threshold
    if logical_op == "and":
        return state_small and action_small
    return state_small or action_small


def filter_episode_rows(
    rows: list[dict[str, Any]],
    *,
    state_threshold: float,
    action_threshold: float | None,
    logical_op: str,
    min_frames_per_episode: int,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    if not rows:
        return [], {}

    kept_rows: list[dict[str, Any]] = [dict(rows[0])]
    kept_original_frame_indices = [int(rows[0]["frame_index"])]
    prev_kept_row = rows[0]

    for row in rows[1:]:
        remaining_rows = len(rows) - len(kept_original_frame_indices)
        must_keep_to_hit_min = len(kept_rows) + remaining_rows <= min_frames_per_episode
        if must_keep_to_hit_min or not _should_drop(
            prev_kept_row,
            row,
            state_threshold=state_threshold,
            action_threshold=action_threshold,
            logical_op=logical_op,
        ):
            kept_rows.append(dict(row))
            kept_original_frame_indices.append(int(row["frame_index"]))
            prev_kept_row = row

    old_to_new_frame = {
        old_frame_index: new_frame_index
        for new_frame_index, old_frame_index in enumerate(kept_original_frame_indices)
    }
    return kept_rows, old_to_new_frame


def build_filtered_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    state_threshold: float,
    action_threshold: float | None,
    logical_op: str,
    min_frames_per_episode: int,
    dry_run: bool,
) -> None:
    info = _load_json(src_root / "meta" / "info.json")
    tasks_table = pq.read_table(src_root / "meta" / "tasks.parquet")
    episodes_table = pq.read_table(src_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    data_table = pq.read_table(src_root / "data" / "chunk-000" / "file-000.parquet")

    episodes = sorted(
        episodes_table.to_pylist(), key=lambda row: int(row["episode_index"])
    )
    rows = sorted(data_table.to_pylist(), key=lambda row: int(row["index"]))

    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)

    rewritten_data_rows: list[dict[str, Any]] = []
    rewritten_episode_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    next_global_index = 0
    total_removed = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_rows = rows_by_episode.get(episode_index, [])
        kept_rows, old_to_new_frame = filter_episode_rows(
            episode_rows,
            state_threshold=state_threshold,
            action_threshold=action_threshold,
            logical_op=logical_op,
            min_frames_per_episode=min_frames_per_episode,
        )

        removed = len(episode_rows) - len(kept_rows)
        total_removed += removed
        manifest_rows.append(
            {
                "episode_index": episode_index,
                "original_length": len(episode_rows),
                "filtered_length": len(kept_rows),
                "removed_frames": removed,
            }
        )

        if dry_run:
            continue

        dataset_from_index = next_global_index
        for new_frame_index, row in enumerate(kept_rows):
            rewritten = dict(row)
            rewritten["frame_index"] = new_frame_index
            rewritten["index"] = next_global_index
            rewritten["next.done"] = bool(new_frame_index == len(kept_rows) - 1)
            rewritten_data_rows.append(rewritten)
            next_global_index += 1

        dataset_to_index = next_global_index
        updated_episode = dict(episode)
        updated_episode["dataset_from_index"] = dataset_from_index
        updated_episode["dataset_to_index"] = dataset_to_index
        updated_episode = _update_episode_stats(updated_episode, rewritten_data_rows[dataset_from_index:dataset_to_index])
        rewritten_episode_rows.append(updated_episode)

        _copy_frame_set(src_root, dst_root, episode_index, old_to_new_frame)

    if dry_run:
        print(
            f"Would remove {total_removed} / {len(rows)} frames "
            f"({(100.0 * total_removed / max(1, len(rows))):.2f}%)."
        )
        for entry in manifest_rows[:10]:
            print(entry)
        return

    _ensure_dir(dst_root / "meta" / "episodes" / "chunk-000")
    _ensure_dir(dst_root / "data" / "chunk-000")

    pq.write_table(
        pa.Table.from_pylist(rewritten_data_rows),
        dst_root / "data" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(rewritten_episode_rows),
        dst_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(tasks_table, dst_root / "meta" / "tasks.parquet")

    updated_info = dict(info)
    updated_info["total_episodes"] = len(rewritten_episode_rows)
    updated_info["total_frames"] = len(rewritten_data_rows)
    _dump_json(dst_root / "meta" / "info.json", updated_info)

    stats_src = src_root / "meta" / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, dst_root / "meta" / "stats.json")

    _dump_json(
        dst_root / "meta" / "filtered_manifest.json",
        {
            "source_root": str(src_root),
            "filtered_root": str(dst_root),
            "state_threshold": state_threshold,
            "action_threshold": action_threshold,
            "logical_op": logical_op,
            "min_frames_per_episode": min_frames_per_episode,
            "total_original_frames": len(rows),
            "total_filtered_frames": len(rewritten_data_rows),
            "total_removed_frames": total_removed,
            "episodes": manifest_rows,
        },
    )

    print(
        f"Filtered dataset written to {dst_root}. "
        f"Removed {total_removed} / {len(rows)} frames "
        f"({(100.0 * total_removed / max(1, len(rows))):.2f}%)."
    )


def main() -> None:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dst_root}")
        shutil.rmtree(dst_root)

    if not args.dry_run:
        _ensure_dir(dst_root)

    build_filtered_dataset(
        src_root,
        dst_root,
        state_threshold=args.state_threshold,
        action_threshold=args.action_threshold,
        logical_op=args.logical_op,
        min_frames_per_episode=args.min_frames_per_episode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
