#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the IDE-oriented EUR-LexRD pipeline scripts."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

RANDOM_SEED = 123


def find_project_root(
    override: Path | None = None,
    *,
    script_file: str | Path | None = None,
) -> Path:
    """Find the repository root from an override, script location, or cwd."""
    if override is not None:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"PROJECT_ROOT_OVERRIDE is not a directory: {root}"
            )
        return root

    candidates: list[Path] = []

    if script_file:
        script_path = Path(script_file).expanduser().resolve()
        candidates.extend([script_path.parent, *script_path.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        has_data = (candidate / "data").is_dir()
        has_scripts = (candidate / "scripts").is_dir()
        has_code = (candidate / "code").is_dir()

        if has_data and (has_scripts or has_code):
            return candidate

        if has_data and candidate == cwd:
            return candidate

    raise FileNotFoundError(
        "Could not locate the EUR-LexRD project root. Set "
        "PROJECT_ROOT_OVERRIDE near the top of the script to the repository's "
        "absolute path."
    )


def resolve_project_path(project_root: Path, path: Path | str) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Seed standard random generators used by the pipeline."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require_environment_variables(names: Iterable[str]) -> dict[str, str]:
    """Return required environment variables or raise a clear error."""
    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise EnvironmentError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
        )
    return values


def require_any_environment_variable(names: Iterable[str]) -> tuple[str, str]:
    """Return the first available environment variable from ``names``."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return name, value
    raise EnvironmentError(
        "Set one of the following environment variables before running: "
        + ", ".join(names)
    )
