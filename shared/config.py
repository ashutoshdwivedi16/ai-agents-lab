"""Config loading: YAML files -> Pydantic models."""

import logging
from pathlib import Path
from typing import Any

import yaml

from shared.models import AgentConfig, AppConfig

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Returns an empty dict on parse errors (logs a warning).
    """
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.warning("Malformed YAML in %s: %s — using defaults", path, e)
        return {}


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load the main application config.

    Falls back to defaults if the file doesn't exist or is malformed.
    """
    config_path = path or _CONFIG_DIR / "default.yaml"
    if not config_path.exists():
        return AppConfig()
    return AppConfig(**load_yaml(config_path))


def load_agent_config(agent_name: str, path: Path | None = None) -> AgentConfig:
    """Load an agent-specific config.

    Falls back to defaults if the YAML file doesn't exist or is malformed.
    """
    config_path = path or _CONFIG_DIR / "agents" / f"{agent_name}.yaml"
    if not config_path.exists():
        return AgentConfig()
    return AgentConfig(**load_yaml(config_path))
