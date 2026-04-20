#!/usr/bin/env python3
"""Manage Hugging Face model repositories for local training artifacts.

This helper is intended for workflows where we train locally, upload the
resulting Hugging Face model directory to a private repo, remove local weights,
and later download the model back only when needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, create_repo, snapshot_download


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _existing_dir(value: str) -> Path:
    path = _path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def _optional_patterns(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    patterns = [value for value in values if value]
    return patterns or None


def _print_summary(**items: object) -> None:
    for key, value in items.items():
        print(f"{key}: {value}")


def _create_repo(args: argparse.Namespace) -> str:
    repo_url = create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=args.exist_ok,
        token=args.token,
    )
    _print_summary(action="create", repo_id=args.repo_id, repo_url=repo_url)
    return str(repo_url)


def _upload_folder(args: argparse.Namespace) -> None:
    local_dir = args.local_dir.resolve()
    api = HfApi(token=args.token)

    if args.create_repo:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=args.token,
        )

    _print_summary(
        action="upload",
        repo_id=args.repo_id,
        local_dir=local_dir,
        path_in_repo=args.path_in_repo,
        large_folder=args.large_folder,
    )

    if args.large_folder:
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(local_dir),
        )
        return

    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        revision=args.revision,
        allow_patterns=_optional_patterns(args.allow_pattern),
        ignore_patterns=_optional_patterns(args.ignore_pattern),
        delete_patterns=_optional_patterns(args.delete_pattern),
    )
    _print_summary(commit_url=getattr(commit_info, "commit_url", None))


def _download_snapshot(args: argparse.Namespace) -> None:
    output_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
        local_dir=str(args.local_dir.resolve()) if args.local_dir else None,
        allow_patterns=_optional_patterns(args.allow_pattern),
        ignore_patterns=_optional_patterns(args.ignore_pattern),
        token=args.token,
        local_files_only=args.local_files_only,
    )
    _print_summary(action="download", repo_id=args.repo_id, output_path=output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, upload, and download Hugging Face model repos.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses local Hugging Face login state.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create",
        help="Create a Hugging Face model repo.",
    )
    create_parser.add_argument("repo_id", help="HF repo id, e.g. username/pi05-book-sft")
    create_parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repo as private. Default: true.",
    )
    create_parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not fail if the repo already exists. Default: true.",
    )
    create_parser.set_defaults(func=_create_repo)

    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload a local Hugging Face model directory to a repo.",
    )
    upload_parser.add_argument("repo_id", help="HF repo id, e.g. username/pi05-book-sft")
    upload_parser.add_argument("local_dir", type=_existing_dir, help="Local model directory")
    upload_parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Destination path inside the repo. Default: repo root.",
    )
    upload_parser.add_argument(
        "--create-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repo automatically before uploading. Default: true.",
    )
    upload_parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When creating the repo automatically, make it private. Default: true.",
    )
    upload_parser.add_argument(
        "--large-folder",
        action="store_true",
        help="Use upload_large_folder for resumable large uploads.",
    )
    upload_parser.add_argument(
        "--revision",
        default=None,
        help="Target branch, tag, or commit for the upload.",
    )
    upload_parser.add_argument(
        "--commit-message",
        default="Upload model artifacts",
        help="Commit message for upload_folder uploads.",
    )
    upload_parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Glob of files to include. Repeatable.",
    )
    upload_parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Glob of files to exclude. Repeatable.",
    )
    upload_parser.add_argument(
        "--delete-pattern",
        action="append",
        default=[],
        help="Glob of remote files to delete before uploading. Repeatable.",
    )
    upload_parser.set_defaults(func=_upload_folder)

    download_parser = subparsers.add_parser(
        "download",
        help="Download a model snapshot from a repo.",
    )
    download_parser.add_argument("repo_id", help="HF repo id, e.g. username/pi05-book-sft")
    download_parser.add_argument(
        "--local-dir",
        type=_path,
        default=None,
        help="Directory to place downloaded files in.",
    )
    download_parser.add_argument(
        "--cache-dir",
        type=_path,
        default=None,
        help="Custom Hugging Face cache directory.",
    )
    download_parser.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit to download.",
    )
    download_parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Glob of files to include. Repeatable.",
    )
    download_parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Glob of files to exclude. Repeatable.",
    )
    download_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use already cached local files; do not access the network.",
    )
    download_parser.set_defaults(func=_download_snapshot)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
