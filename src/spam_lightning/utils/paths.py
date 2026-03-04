"""Path helpers shared by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the repo root by walking upward until a `.git` directory is found."""

    origin = (start or Path.cwd()).resolve()
    search_roots = [origin] + list(origin.parents)
    for candidate in search_roots:
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root containing a .git directory.")


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the resolved path."""

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
