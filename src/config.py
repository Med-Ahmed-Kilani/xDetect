"""Config loading utility — reads YAML files relative to the project root."""
import os
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent


def project_root() -> Path:
    return _PROJECT_ROOT


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by filename stem (e.g. 'datasets' or 'models')."""
    path = _PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a project-relative path string to an absolute Path."""
    return _PROJECT_ROOT / relative
