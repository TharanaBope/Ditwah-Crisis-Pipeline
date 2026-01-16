"""
Model routing utilities.

Automatically selects appropriate model tier based on prompt technique:
- Reasoning techniques (cot, tot) → reasoning models
- General techniques → general models
"""

import yaml
from pathlib import Path
from typing import Literal, Optional


def pick_model(
    provider: Literal["openai", "google", "groq"],
    technique: str,
    tier: Optional[Literal["general", "strong", "reason"]] = None,
    config_path: str = "config/models.yaml",
) -> str:
    """
    Select appropriate model based on provider and technique.

    Routing logic:
    - technique in {"cot", "tot", "cot_reasoning", "tot_reasoning"} → reason tier
    - technique in {"strong", "complex"} → strong tier
    - Otherwise → general tier
    - Explicit tier parameter overrides automatic routing

    Args:
        provider: API provider (openai, google, groq)
        technique: Prompt technique identifier
        tier: Optional explicit tier selection
        config_path: Path to models.yaml config file

    Returns:
        Model identifier string

    Raises:
        FileNotFoundError: If config file not found
        KeyError: If provider/tier not in config
    """
    config_file = Path(config_path)

    # If config file doesn't exist, try relative to this file's location
    if not config_file.exists():
        # Try relative to utils directory (project root)
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        config_file = project_root / "config" / "models.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Model config not found. Tried:\n"
            f"  - {config_path}\n"
            f"  - {config_file}\n"
            f"Current working directory: {Path.cwd()}"
        )

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if provider not in config:
        raise KeyError(f"Provider '{provider}' not found in {config_path}")

    # Determine tier based on technique if not explicitly provided
    if tier is None:
        technique_lower = technique.lower()

        # Reasoning techniques require reasoning models
        if any(x in technique_lower for x in ["cot", "tot", "reason", "think"]):
            tier = "reason"
        # Strong/complex techniques benefit from stronger models
        elif any(x in technique_lower for x in ["strong", "complex", "advanced"]):
            tier = "strong"
        # Default to general tier
        else:
            tier = "general"

    if tier not in config[provider]:
        # Fallback to general if requested tier not available
        tier = "general"

    return config[provider][tier]


def list_available_models(config_path: str = "config/models.yaml") -> dict[str, dict]:
    """
    List all available models from config.

    Args:
        config_path: Path to models.yaml config file

    Returns:
        Dict mapping provider -> tier -> model
    """
    config_file = Path(config_path)

    # If config file doesn't exist, try relative to this file's location
    if not config_file.exists():
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        config_file = project_root / "config" / "models.yaml"

    if not config_file.exists():
        return {}

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_context_window(model: str) -> int:
    """
    Get approximate context window size for a model.

    Note: These are approximate values. Check provider docs for exact limits.

    Args:
        model: Model identifier

    Returns:
        Context window size in tokens
    """
    # OpenAI models
    if "gpt-4o" in model or "o3" in model or "o1" in model:
        return 128_000
    if "gpt-4" in model:
        return 128_000
    if "gpt-3.5" in model:
        return 16_385

    # Google Gemini models
    if "gemini-2.0" in model:
        return 1_000_000
    if "gemini-1.5" in model:
        return 1_000_000

    # Groq models
    if "llama-3.1" in model:
        return 131_072
    if "deepseek-r1" in model:
        return 65_536
    if "llama-3.2" in model:
        return 131_072

    # Default conservative estimate
    return 8_000


def should_use_reasoning_model(technique: str) -> bool:
    """
    Check if technique requires reasoning model.

    Args:
        technique: Prompt technique identifier

    Returns:
        True if reasoning model recommended
    """
    from .config_loader import get_config, should_auto_route_reasoning, get_reasoning_techniques

    if not should_auto_route_reasoning():
        return False

    technique_lower = technique.lower()
    reasoning_techniques = get_reasoning_techniques()

    # Check exact matches first
    if technique_lower in reasoning_techniques:
        return True

    # Check substring matches
    return any(keyword in technique_lower for keyword in reasoning_techniques)
