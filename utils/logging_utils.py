"""
CSV logging utilities with token metrics and cost estimation.

Logs each LLM call to logs/runs.csv with:
- Basic metadata (timestamp, provider, model, technique)
- Token metrics (estimated + actual)
- Performance (latency, retry count)
- Cost estimates (with disclaimer)
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# Approximate cost per 1M tokens (as of late 2024, subject to change)
# Always check provider pricing pages for current rates
COST_PER_1M_TOKENS = {
    "openai": {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "o3-mini": {"input": 1.00, "output": 4.00},
        "o3": {"input": 10.00, "output": 40.00},
    },
    "google": {
        "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00},  # Free tier
        "gemini-2.0-flash-thinking-exp": {"input": 0.00, "output": 0.00},  # Free tier
    },
    "groq": {
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "deepseek-r1-distill-llama-70b": {"input": 0.40, "output": 1.00},
    },
}


def _get_log_path() -> Path:
    """Get path to logs/runs.csv, creating directory if needed."""
    # Try current directory first
    log_dir = Path("logs")

    # If logs doesn't exist in current dir, try relative to project root
    if not log_dir.exists():
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        log_dir = project_root / "logs"

    log_dir.mkdir(exist_ok=True)
    return log_dir / "runs.csv"


def _init_csv_if_needed(log_path: Path) -> None:
    """Initialize CSV with headers if file doesn't exist."""
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "provider",
                "model",
                "technique",
                "latency_ms",
                "input_tokens_est",
                "context_tokens_est",
                "total_est",
                "prompt_tokens_actual",
                "completion_tokens_actual",
                "total_tokens_actual",
                "retry_count",
                "backoff_ms_total",
                "overflow_handled",
                "cost_estimate_usd",
                "notes",
            ])


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> Optional[float]:
    """
    Estimate cost in USD based on token usage.

    Note: Prices change frequently. This is for educational purposes only.
    Always check provider pricing pages for accurate costs.

    Args:
        provider: API provider
        model: Model identifier
        prompt_tokens: Input token count
        completion_tokens: Output token count

    Returns:
        Estimated cost in USD or None if pricing unavailable
    """
    if not prompt_tokens or not completion_tokens:
        return None

    provider_pricing = COST_PER_1M_TOKENS.get(provider, {})
    model_pricing = None

    # Try exact match first
    if model in provider_pricing:
        model_pricing = provider_pricing[model]
    else:
        # Try partial match (e.g., "gpt-4o-mini-2024-07-18" matches "gpt-4o-mini")
        for key in provider_pricing:
            if key in model:
                model_pricing = provider_pricing[key]
                break

    if not model_pricing:
        return None

    input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]

    return round(input_cost + output_cost, 6)


def log_llm_call(
    provider: str,
    model: str,
    technique: str,
    latency_ms: int,
    usage: dict[str, Any],
    retry_count: int = 0,
    backoff_ms_total: int = 0,
    overflow_handled: bool = False,
    notes: str = "",
) -> None:
    """
    Log an LLM call to logs/runs.csv.

    Args:
        provider: API provider (openai, google, groq)
        model: Model identifier
        technique: Prompt technique used (zero_shot, few_shot, cot, etc.)
        latency_ms: Call latency in milliseconds
        usage: Token usage dict with estimated + actual fields
        retry_count: Number of retries performed
        backoff_ms_total: Total backoff time in milliseconds
        overflow_handled: Whether context overflow was handled
        notes: Optional notes or error messages
    """
    log_path = _get_log_path()
    _init_csv_if_needed(log_path)

    # Extract token counts
    input_tokens_est = usage.get("input_tokens_est", 0)
    context_tokens_est = usage.get("context_tokens_est", 0)
    total_est = usage.get("total_est", 0)
    prompt_tokens_actual = usage.get("prompt_tokens_actual")
    completion_tokens_actual = usage.get("completion_tokens_actual")
    total_tokens_actual = usage.get("total_tokens_actual")

    # Estimate cost using actual tokens when available
    cost_estimate = estimate_cost(
        provider,
        model,
        prompt_tokens_actual or input_tokens_est + context_tokens_est,
        completion_tokens_actual or 0,
    )

    # Format cost with disclaimer
    cost_str = f"~${cost_estimate:.6f}" if cost_estimate else "N/A"

    # Append to CSV
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            provider,
            model,
            technique,
            latency_ms,
            input_tokens_est,
            context_tokens_est,
            total_est,
            prompt_tokens_actual or "",
            completion_tokens_actual or "",
            total_tokens_actual or "",
            retry_count,
            backoff_ms_total,
            overflow_handled,
            cost_str,
            notes,
        ])


def get_log_summary() -> dict[str, Any]:
    """
    Get summary statistics from logs/runs.csv.

    Returns:
        Dict with total calls, avg latency, total tokens, etc.
    """
    log_path = _get_log_path()

    if not log_path.exists():
        return {"total_calls": 0, "message": "No logs found"}

    import pandas as pd

    try:
        df = pd.read_csv(log_path)

        # Handle empty CSV
        if len(df) == 0:
            return {"total_calls": 0, "message": "Log file is empty"}

        summary = {
            "total_calls": len(df),
            "avg_latency_ms": df["latency_ms"].mean(),
            "total_retries": df["retry_count"].sum(),
            "techniques_used": df["technique"].value_counts().to_dict(),
            "providers_used": df["provider"].value_counts().to_dict(),
        }

        # Token statistics (using actual when available, estimated otherwise)
        if "prompt_tokens_actual" in df.columns:
            actual_prompts = df["prompt_tokens_actual"].dropna()
            if len(actual_prompts) > 0:
                summary["total_prompt_tokens"] = int(actual_prompts.sum())

        if "completion_tokens_actual" in df.columns:
            actual_completions = df["completion_tokens_actual"].dropna()
            if len(actual_completions) > 0:
                summary["total_completion_tokens"] = int(actual_completions.sum())

        return summary

    except Exception as e:
        return {"error": f"Failed to read logs: {e}"}


def clear_logs() -> None:
    """Delete logs/runs.csv (use with caution)."""
    log_path = _get_log_path()
    if log_path.exists():
        os.remove(log_path)
