#!/usr/bin/env python3
"""Configuration loading and validation for coverage testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Configuration validation error."""


@dataclass
class TestTarget:
    """Single test target configuration."""
    id: str
    target: str
    kind: str  # "gtest" or "benchmark"
    gtest_filter: str = "*"
    env: dict[str, str] | None = None
    enabled: bool = True
    bazel_build_flags: list[str] | None = None  # Target-specific build flags

    def __post_init__(self):
        if self.env is None:
            self.env = {}
        if self.bazel_build_flags is None:
            self.bazel_build_flags = []


@dataclass
class FeatureConfig:
    """Feature-specific test configuration."""
    name: str
    description: str
    test_targets: list[TestTarget]
    bazel_args: list[str]
    bazel_build_flags: list[str]  # Feature-level build flags (e.g., -fexceptions)
    platform: str = "linux-*"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureConfig:
        """Load configuration from dictionary."""
        targets = [
            TestTarget(
                id=t["id"],
                target=t["target"],
                kind=t["kind"],
                gtest_filter=t.get("gtest_filter", "*"),
                env=t.get("env"),
                enabled=t.get("enabled", True),
                bazel_build_flags=t.get("bazel_build_flags", []),
            )
            for t in data.get("test_targets", [])
        ]

        return cls(
            name=data["name"],
            description=data["description"],
            test_targets=targets,
            bazel_args=data.get("bazel_args", []),
            bazel_build_flags=data.get("bazel_build_flags", []),
        )


def load_feature_config(feature_dir: Path) -> FeatureConfig:
    """Load feature configuration from config.json."""
    config_path = feature_dir / "config.json"
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return FeatureConfig.from_dict(data)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {config_path}: {e}")
    except KeyError as e:
        raise ConfigError(f"Missing required field in {config_path}: {e}")
