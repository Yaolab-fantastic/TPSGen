from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def installed_resource_root() -> Path:
    override = os.environ.get("TOMATO_PROMOTER_DESIGNER_RESOURCE_DIR")
    if override:
        return Path(override)
    return Path(sysconfig.get_path("data")) / "share" / "tomato-promoter-designer"


def find_example(name: str = "demo_input.fasta") -> Path:
    candidates = (
        repository_root() / "examples" / name,
        installed_resource_root() / "examples" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Bundled example not found: {name}")


def find_model(relative_path: str) -> Path:
    override = os.environ.get("TOMATO_PROMOTER_DESIGNER_MODELS_DIR")
    candidates = (
        *((Path(override) / relative_path,) if override else ()),
        repository_root() / "models" / relative_path,
        installed_resource_root() / "models" / relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Bundled model not found: models/{relative_path}")
