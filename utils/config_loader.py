"""
Configuration loader for centralized settings management.

Loads configuration from config/config.yaml and provides easy access
to settings throughout the application.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container with easy attribute access."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.

        Example:
            config.get("retry.max_retries")  # Returns 3
            config.get("tokens.context_management.hard_prompt_cap")
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config

    @property
    def raw(self) -> Dict[str, Any]:
        """Get raw config dict."""
        return self._config


# Global config instance
_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml

    Returns:
        Config instance
    """
    global _config

    if config_path is None:
        # Try to find config relative to this file
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        # If provided path doesn't exist, try relative to project root
        if not config_path.exists():
            utils_dir = Path(__file__).parent
            project_root = utils_dir.parent
            config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    _config = Config(config_dict)
    return _config


def get_config() -> Config:
    """
    Get the global config instance.

    Loads config on first call.

    Returns:
        Config instance
    """
    global _config

    if _config is None:
        _config = load_config()

    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Force reload configuration from file.

    Useful for picking up config changes without restarting.

    Args:
        config_path: Path to config file

    Returns:
        New Config instance
    """
    global _config
    _config = None
    return load_config(config_path)


# Convenience functions for common config values

def get_default_provider() -> str:
    """Get default provider."""
    return get_config().get("providers.default", "openai")


def get_enabled_providers() -> list[str]:
    """Get list of enabled providers."""
    return get_config().get("providers.enabled", ["openai"])


def get_max_retries() -> int:
    """Get max retry attempts."""
    return get_config().get("retry.max_retries", 3)


def get_backoff_base() -> float:
    """Get base backoff time in seconds."""
    return get_config().get("retry.backoff.base_seconds", 0.5)


def get_backoff_jitter() -> float:
    """Get backoff jitter factor."""
    return get_config().get("retry.backoff.jitter_factor", 0.25)


def get_default_temperature(task_type: Optional[str] = None) -> float:
    """
    Get default temperature for task type.

    Args:
        task_type: Optional task type (extraction, classification, etc.)

    Returns:
        Temperature value
    """
    if task_type:
        temp = get_config().get(f"defaults.by_task.{task_type}.temperature")
        if temp is not None:
            return temp

    return get_config().get("defaults.temperature", 0.2)


def get_default_max_tokens(task_type: Optional[str] = None) -> int:
    """
    Get default max_tokens for task type.

    Args:
        task_type: Optional task type

    Returns:
        Max tokens value
    """
    if task_type:
        max_tok = get_config().get(f"defaults.by_task.{task_type}.max_tokens")
        if max_tok is not None:
            return max_tok

    return get_config().get("defaults.max_tokens", 1000)


def is_logging_enabled() -> bool:
    """Check if logging is enabled."""
    return get_config().get("logging.enabled", True)


def get_log_path() -> Path:
    """Get full path to log file."""
    log_dir = get_config().get("logging.output_dir", "logs")
    log_file = get_config().get("logging.output_file", "runs.csv")
    return Path(log_dir) / log_file


def should_auto_route_reasoning() -> bool:
    """Check if automatic reasoning model routing is enabled."""
    return get_config().get("models.auto_routing", True)


def get_reasoning_techniques() -> list[str]:
    """Get list of techniques that trigger reasoning model routing."""
    return get_config().get("models.reasoning_techniques", ["cot", "tot"])
