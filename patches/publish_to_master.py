#!/usr/bin/env python3
"""Generate and publish the TensorFlow patch release into a master worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PATCH_MANAGER = SCRIPT_DIR / "patch_manager.py"
PREPARE_SOURCE = SCRIPT_DIR / "prepare_source.py"
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"
RELEASE_TEMPLATE_DIR = SCRIPT_DIR / "release"
PUBLISHED_FEATURE_DIR = "feature"
PUBLISHED_FROZEN_DIR = "frozen_feature"

OLD_ROOT_PATCHES = (
    "0001-tensorflow_2.15.0-optimize.patch",
    "0002-tensorflow_2.15.0-annc-optimize.patch",
    "0003-tensorflow_2.15.0-kembedding.patch",
    "0004-tensorflow_2.15.0-sparse-matmul.patch",
)

STALE_DOCUMENT_NAMES = OLD_ROOT_PATCHES


class PublishError(RuntimeError):
    pass


def run(
    args: list[str],
    *,
    cwd: Path = REPO_ROOT,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def git_output(args: list[str], cwd: Path) -> str:
    return run(["git", *args], cwd=cwd, capture_output=True).stdout.strip()


def validate_master_worktree(master_worktree: Path) -> None:
    if not master_worktree.is_dir():
        raise PublishError(f"master worktree does not exist: {master_worktree}")
    top_level = Path(
        git_output(["rev-parse", "--show-toplevel"], master_worktree)
    ).resolve()
    if top_level != master_worktree:
        raise PublishError(
            f"target is not a worktree root: {master_worktree} (root: {top_level})"
        )
    branch = git_output(["branch", "--show-current"], master_worktree)
    if branch != "master":
        raise PublishError(
            f"target worktree must be on master, found {branch or 'detached HEAD'}"
        )
    if master_worktree == REPO_ROOT:
        raise PublishError("refusing to publish into the development worktree")


def load_outputs() -> tuple[str, list[str]]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    base_ref = data["base_ref"]
    outputs = [group["output"] for group in data["groups"]]
    if len(outputs) != len(set(outputs)):
        raise PublishError("manifest contains duplicate patch output names")
    return base_ref, outputs


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sync_patch_directory(
    source_dir: Path, destination_dir: Path, expected_names: set[str]
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for existing in destination_dir.glob("*.patch"):
        if existing.name not in expected_names:
            existing.unlink()
    for name in sorted(expected_names):
        source = source_dir / name
        if not source.is_file():
            raise PublishError(f"missing patch artifact: {source}")
        copy_file(source, destination_dir / name)


def write_checksums(release_dir: Path) -> None:
    patch_paths = sorted(
        [
            *release_dir.joinpath(PUBLISHED_FEATURE_DIR).glob("*.patch"),
            *release_dir.joinpath(PUBLISHED_FROZEN_DIR).glob("*.patch"),
        ],
        key=lambda path: path.relative_to(release_dir).as_posix(),
    )
    lines = []
    for path in patch_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(release_dir).as_posix()}")
    release_dir.joinpath("SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def verify_document_references(master_worktree: Path) -> None:
    document_paths = [
        master_worktree / "README.md",
        master_worktree / "README_en.md",
        *master_worktree.joinpath("docs").rglob("*.md"),
    ]
    stale = []
    for path in document_paths:
        content = path.read_text(encoding="utf-8")
        for name in STALE_DOCUMENT_NAMES:
            if name in content or name.replace("_", "\\_") in content:
                stale.append(f"{path.relative_to(master_worktree)}: {name}")
    if stale:
        raise PublishError(
            "stale patch names remain in release documents:\n  "
            + "\n  ".join(stale)
        )


def publish(source_ref: str, master_worktree: Path, check_docs: bool) -> None:
    validate_master_worktree(master_worktree)
    base_ref, outputs = load_outputs()
    source_commit = git_output(["rev-parse", f"{source_ref}^{{commit}}"], REPO_ROOT)

    with tempfile.TemporaryDirectory(prefix="tensorflow-patch-release-") as temp:
        generated_dir = Path(temp) / PUBLISHED_FEATURE_DIR
        run(
            [
                sys.executable,
                str(PATCH_MANAGER),
                "--source",
                source_ref,
                "audit",
            ]
        )
        run(
            [
                sys.executable,
                str(PATCH_MANAGER),
                "--source",
                source_ref,
                "generate",
                "--output-dir",
                str(generated_dir),
            ]
        )
        run(
            [
                sys.executable,
                str(PATCH_MANAGER),
                "--source",
                source_ref,
                "verify",
                "--patch-dir",
                str(generated_dir),
            ]
        )

        release_dir = master_worktree / "patches"
        sync_patch_directory(
            generated_dir, release_dir / PUBLISHED_FEATURE_DIR, set(outputs)
        )

        frozen_source = SCRIPT_DIR / PUBLISHED_FROZEN_DIR
        frozen_names = {path.name for path in frozen_source.glob("*.patch")}
        if not frozen_names:
            raise PublishError("no frozen legacy patch found")
        sync_patch_directory(
            frozen_source, release_dir / PUBLISHED_FROZEN_DIR, frozen_names
        )

        copy_file(MANIFEST_PATH, release_dir / "manifest.json")
        copy_file(PATCH_MANAGER, release_dir / "patch_manager.py")
        copy_file(PREPARE_SOURCE, release_dir / "prepare_source.py")
        copy_file(
            RELEASE_TEMPLATE_DIR / "README.md", release_dir / "README.md"
        )
        copy_file(
            RELEASE_TEMPLATE_DIR / "README_en.md",
            release_dir / "README_en.md",
        )

        release_dir.joinpath("SOURCE_COMMIT").write_text(
            f"source_commit={source_commit}\nbase_commit={base_ref}\n",
            encoding="ascii",
        )
        write_checksums(release_dir)

    for old_name in OLD_ROOT_PATCHES:
        old_path = master_worktree / old_name
        if old_path.exists():
            old_path.unlink()

    if check_docs:
        verify_document_references(master_worktree)

    print(f"published source {source_commit}")
    print(f"master worktree: {master_worktree}")
    print(f"release directory: {master_worktree / 'patches'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="HEAD",
        help="committed development source to publish (default: HEAD)",
    )
    parser.add_argument(
        "--master-worktree",
        type=Path,
        required=True,
        help="cleanly checked out master worktree to update",
    )
    parser.add_argument(
        "--skip-doc-check",
        action="store_true",
        help="allow legacy patch filenames to remain in release documents",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        publish(
            args.source,
            args.master_worktree.expanduser().resolve(),
            not args.skip_doc_check,
        )
    except (OSError, PublishError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
