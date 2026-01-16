"""
Central prompt template catalog.

This module serves as the single source of truth for all prompt templates
in the Crisis Intelligence Pipeline. Each prompt is defined as a PromptSpec
with metadata and can be rendered with variables.

Usage:
    from utils.prompts import render

    text, spec = render("zero_shot.v1",
                       role="helpful assistant",
                       instruction="Summarize the following text",
                       constraints="Maximum 3 sentences",
                       format="Plain text paragraph")
"""

from dataclasses import dataclass
from string import Template
from typing import Optional, List, Dict, Tuple


@dataclass
class PromptSpec:
    """
    Specification for a prompt template.

    Attributes:
        id: Unique identifier (e.g., "zero_shot.v1")
        purpose: Human-readable description of use case
        template: Template string with ${variable} placeholders
        stop: Optional stop sequences
        max_tokens: Suggested max output tokens
        temperature: Suggested temperature (0.0-1.0)
    """

    id: str
    purpose: str
    template: str
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# Central prompt registry
PROMPTS: Dict[str, PromptSpec] = {
    "skeleton.v1": PromptSpec(
        id="skeleton.v1",
        purpose="Structured writing with fixed sections",
        template=(
            "Role: ${role}\n"
            "Task: ${task}\n\n"
            "Context:\n${context}\n\n"
            "Constraints:\n${constraints}\n\n"
            "Output format:\n${format}\n\n"
            "Checks:\n${checks}\n"
        ),
        stop=["</end>"],
        max_tokens=800,
        temperature=0.2,
    ),
    "zero_shot.v1": PromptSpec(
        id="zero_shot.v1",
        purpose="Direct instruction-first prompt",
        template=(
            "You are ${role}. Follow the instructions precisely.\n"
            "Instruction: ${instruction}\n"
            "Constraints: ${constraints}\n"
            "Output format: ${format}\n"
        ),
        temperature=0.2,
    ),
    "few_shot.v1": PromptSpec(
        id="few_shot.v1",
        purpose="Few-shot with explicit examples block",
        template=(
            "You are ${role}. Learn from the examples, then answer the query.\n\n"
            "Examples:\n${examples}\n\n"
            "Query: ${query}\n"
            "Constraints: ${constraints}\n"
            "Output format: ${format}\n"
        ),
        temperature=0.2,
    ),
    "cot_reasoning.v1": PromptSpec(
        id="cot_reasoning.v1",
        purpose="Chain-of-thought style reasoning (use with reasoning models only)",
        template=(
            "You are ${role}. Solve the problem carefully.\n"
            "Problem: ${problem}\n\n"
            "First, outline your reasoning steps briefly.\n"
            "Then provide the final answer clearly marked under 'Answer:'.\n"
            "Keep reasoning concise; avoid unnecessary prose.\n"
        ),
        temperature=0.3,
        max_tokens=4096,  # High limit to accommodate reasoning model "thinking" tokens
    ),
    "tot_reasoning.v1": PromptSpec(
        id="tot_reasoning.v1",
        purpose="Tree-of-thought branching with self-consistency",
        template=(
            "You are ${role}. Explore ${branches} distinct solution paths to the problem below.\n"
            "Problem: ${problem}\n\n"
            "For each path, provide: Hypothesis → Steps → Intermediate check.\n"
            "After exploring, select the best path and provide the final 'Answer:'.\n"
        ),
        temperature=0.5,
        max_tokens=4096,  # High limit to accommodate reasoning model "thinking" tokens
    ),
    "json_extract.v1": PromptSpec(
        id="json_extract.v1",
        purpose="Strict schema-first JSON extraction",
        template=(
            "Extract the requested fields and return ONLY valid JSON matching this schema:\n"
            "${schema}\n\n"
            "Text:\n${text}\n\n"
            "Return ONLY JSON, no extra text."
        ),
        temperature=0.0,
        max_tokens=400,
        stop=None,
    ),
    "tool_call.v1": PromptSpec(
        id="tool_call.v1",
        purpose="Instruct model to choose & call a tool when needed",
        template=(
            "You can decide to call a tool if it helps.\n"
            "Available tools:\n${tools}\n\n"
            "User request: ${request}\n\n"
            "If a tool is needed, respond with JSON:\n"
            '{"tool": "tool_name", "arguments": {...}}\n\n'
            "Otherwise, answer directly and concisely."
        ),
        temperature=0.2,
    ),
    "overflow_summarize.v1": PromptSpec(
        id="overflow_summarize.v1",
        purpose="Token-window safeguard: summarize context, then solve",
        template=(
            "Step 1 — Summarize the following context into ≤ ${max_tokens_context} tokens, "
            "preserving key facts, numbers, and names needed for the task:\n"
            "${context}\n\n"
            "Step 2 — Using ONLY that summary, complete the task:\n"
            "${task}\n"
            "Output format: ${format}\n"
        ),
        temperature=0.2,
    ),
    "rate_limit_retry.v1": PromptSpec(
        id="rate_limit_retry.v1",
        purpose="Idempotent retry phrasing to avoid redoing stochastic steps",
        template=(
            "If a previous attempt failed due to a temporary error, do not redo stochastic planning. "
            "Reuse the prior plan and produce only the final result below.\n"
            "Task: ${task}\n"
            "Keep the answer concise (≤ ${max_tokens_answer} tokens)."
        ),
        temperature=0.0,
    ),
    "style_persona.v1": PromptSpec(
        id="style_persona.v1",
        purpose="Lock tone/persona/reading level",
        template=(
            "Persona: ${persona}\n"
            "Tone: ${tone}\n"
            "Reading level: ${reading_level}\n\n"
            "Task: ${task}\n"
            "Constraints: ${constraints}\n"
            "Output format: ${format}\n"
        ),
        temperature=0.2,
    ),
    "router_classify.v1": PromptSpec(
        id="router_classify.v1",
        purpose="Light intent classification into categories",
        template=(
            "Classify the user query into one of the categories:\n"
            "${labels}\n\n"
            "Query: ${query}\n\n"
            "Return ONLY the label, nothing else."
        ),
        temperature=0.0,
        max_tokens=20,
    ),
}


def render(prompt_id: str, **vars) -> Tuple[str, PromptSpec]:
    """
    Render a prompt template with variables.

    Args:
        prompt_id: Prompt identifier from PROMPTS registry
        **vars: Variables to substitute in template

    Returns:
        Tuple of (rendered_text, prompt_spec)

    Raises:
        KeyError: If prompt_id not found in registry

    Example:
        >>> text, spec = render("zero_shot.v1",
        ...                     role="helpful assistant",
        ...                     instruction="Summarize this",
        ...                     constraints="Max 50 words",
        ...                     format="Plain text")
    """
    if prompt_id not in PROMPTS:
        raise KeyError(
            f"Prompt '{prompt_id}' not found. "
            f"Available: {', '.join(PROMPTS.keys())}"
        )

    spec = PROMPTS[prompt_id]
    text = Template(spec.template).safe_substitute(**vars)
    return text, spec


def list_prompts() -> List[str]:
    """
    List all available prompt IDs.

    Returns:
        List of prompt identifiers
    """
    return list(PROMPTS.keys())


def get_prompt_info(prompt_id: str) -> PromptSpec:
    """
    Get metadata for a prompt without rendering.

    Args:
        prompt_id: Prompt identifier

    Returns:
        PromptSpec instance

    Raises:
        KeyError: If prompt_id not found
    """
    if prompt_id not in PROMPTS:
        raise KeyError(f"Prompt '{prompt_id}' not found")
    return PROMPTS[prompt_id]
