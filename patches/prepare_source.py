#!/usr/bin/env python3
"""Create a complete TensorFlow source tree from a published feature set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from patch_manager import (
    SCRIPT_DIR,
    PatchError,
    load_manifest,
    materialize_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set",
        required=True,
        metavar="NAME",
        help="features to include (for example: full-default)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="PATH",
        type=Path,
        help="directory where the complete TensorFlow source tree is created",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=SCRIPT_DIR / "dist",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_manifest(SCRIPT_DIR / "manifest.json")
        materialize_profile(
            manifest,
            args.feature_set,
            args.patch_dir.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
            force=args.force,
        )
    except (OSError, PatchError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
