"""
Unified LLM client with multi-provider support, retry logic, and token guards.

Supports:
- OpenAI, Google Gemini, Groq via single abstraction
- Automatic retry with exponential backoff for 429/5xx/timeouts
- Token estimation pre-call with context overflow handling
- Usage reconciliation (estimated vs actual tokens)
- Comprehensive error handling
"""

import time
import random
from typing import Literal, Optional, Any, Dict, List
from openai import OpenAI, OpenAIError
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv
import os

from .token_utils import (
    count_messages_tokens,
    reconcile_usage,
    fit_within_context,
)
from .router import get_context_window

# Load environment variables
load_dotenv()


class LLMClient:
    """
    Unified client for multiple LLM providers with robust error handling.

    Features:
    - Automatic token estimation and context overflow handling
    - Retry logic with exponential backoff + jitter
    - Usage tracking (estimated vs actual)
    - Consistent return format across providers
    """

    def __init__(
        self,
        provider: Literal["openai", "google", "groq"],
        model: str,
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
        backoff_jitter: Optional[float] = None,
        hard_prompt_cap: Optional[int] = None,
    ):
        """
        Initialize LLM client.

        Args:
            provider: API provider (openai, google, groq)
            model: Model identifier
            max_retries: Maximum retry attempts (None = use config default)
            backoff_base: Base backoff time in seconds (None = use config default)
            backoff_jitter: Random jitter factor (None = use config default)
            hard_prompt_cap: Optional hard limit on prompt tokens (triggers summarization)
        """
        from .config_loader import get_max_retries, get_backoff_base, get_backoff_jitter

        self.provider = provider
        self.model = model
        self.max_retries = max_retries if max_retries is not None else get_max_retries()
        self.backoff_base = backoff_base if backoff_base is not None else get_backoff_base()
        self.backoff_jitter = backoff_jitter if backoff_jitter is not None else get_backoff_jitter()
        self.hard_prompt_cap = hard_prompt_cap

        # Initialize provider client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize provider-specific client."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(
                                api_key=api_key
                                )

        elif self.provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            self.client = genai.Client(api_key=api_key)

        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            self.client = Groq(
                               api_key=api_key
                               )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_wait = self.backoff_base * (2 ** attempt)
        jitter = random.uniform(0, self.backoff_jitter * base_wait)
        return base_wait + jitter

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is transient and should be retried."""
        error_str = str(error).lower()

        # Rate limits (429)
        if "429" in error_str or "rate limit" in error_str:
            return True

        # Server errors (5xx)
        if any(x in error_str for x in ["500", "502", "503", "504", "server error"]):
            return True

        # Timeouts
        if "timeout" in error_str or "timed out" in error_str:
            return True

        # Context overflow (may be handled differently)
        if "context" in error_str and ("length" in error_str or "too long" in error_str):
            return True

        return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        context_strs: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send chat completion request with automatic retry and token management.

        Args:
            messages: OpenAI-style messages array
            context_strs: Optional context strings (counted separately)
            temperature: Sampling temperature
            max_tokens: Max completion tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with text, usage (estimated + actual), latency_ms, meta
        """
        # Pre-call token estimation
        token_counts = count_messages_tokens(
            messages, self.provider, self.model, context_strs
        )

        # Check hard prompt cap
        overflow_handled = False
        if self.hard_prompt_cap and token_counts["estimated_total"] > self.hard_prompt_cap:
            # Apply context-fit strategy
            messages, context_strs, fit_meta = fit_within_context(
                messages,
                self.provider,
                self.model,
                self.hard_prompt_cap,
                strategy="truncate",
                context_strs=context_strs,
            )
            overflow_handled = fit_meta.get("overflow", False)
            # Recalculate after fitting
            token_counts = count_messages_tokens(
                messages, self.provider, self.model, context_strs
            )

        # Retry loop
        retry_count = 0
        total_backoff_ms = 0
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                # Call provider-specific implementation
                if self.provider == "openai":
                    response = self._call_openai(messages, temperature, max_tokens, **kwargs)
                elif self.provider == "google":
                    response = self._call_google(messages, temperature, max_tokens, **kwargs)
                elif self.provider == "groq":
                    response = self._call_groq(messages, temperature, max_tokens, **kwargs)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")

                latency_ms = int((time.time() - start_time) * 1000)

                # Extract text and usage
                text = response["text"]
                provider_usage = response.get("usage")

                # Reconcile token usage
                usage = reconcile_usage(token_counts, provider_usage)

                return {
                    "text": text,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "raw": response.get("raw"),
                    "meta": {
                        "retry_count": retry_count,
                        "backoff_ms_total": total_backoff_ms,
                        "overflow_handled": overflow_handled,
                    },
                }

            except Exception as e:
                last_error = e

                # Check if we should retry
                if attempt < self.max_retries and self._is_retryable_error(e):
                    retry_count += 1
                    backoff_sec = self._calculate_backoff(attempt)
                    backoff_ms = int(backoff_sec * 1000)
                    total_backoff_ms += backoff_ms

                    time.sleep(backoff_sec)
                    continue

                # Context overflow error - try summarization
                error_str = str(e).lower()
                if (
                    "context" in error_str
                    and ("length" in error_str or "too long" in error_str)
                    and not overflow_handled
                ):
                    # This should be handled by caller using overflow_summarize prompt
                    raise ValueError(
                        "Context window exceeded. Use overflow_summarize.v1 prompt."
                    ) from e

                # Non-retryable error or max retries exceeded
                raise

        # Should not reach here, but just in case
        raise last_error or Exception("Unknown error in LLM call")

    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        params = {
            "model": self.model,
            "messages": messages,
        }

        # Detect reasoning models (o1, o3 series)
        is_reasoning_model = any(
            self.model.startswith(prefix) for prefix in ["o1-", "o3-"]
        )

        # Reasoning models don't support temperature parameter
        if temperature is not None and not is_reasoning_model:
            params["temperature"] = temperature

        if max_tokens is not None:
            # Reasoning models use max_completion_tokens instead of max_tokens
            if is_reasoning_model:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

        params.update(kwargs)

        response = self.client.chat.completions.create(**params)

        return {
            "text": response.choices[0].message.content or "",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "raw": response,
        }

    def _call_google(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Call Google Gemini API using new google-genai SDK."""
        # Convert OpenAI format to Gemini format
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=content)])
                )
            elif role == "assistant":
                gemini_contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=content)])
                )

        # Build generation config
        config_params = {}
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        generation_config = types.GenerateContentConfig(**config_params) if config_params else None

        # Generate content using new client API
        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=generation_config,
        )

        # Extract usage metadata
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "promptTokenCount": response.usage_metadata.prompt_token_count,
                "candidatesTokenCount": response.usage_metadata.candidates_token_count,
            }

        return {
            "text": response.text,
            "usage": usage,
            "raw": response,
        }

    def _call_groq(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Call Groq API (OpenAI-compatible)."""
        params = {
            "model": self.model,
            "messages": messages,
        }

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        params.update(kwargs)

        response = self.client.chat.completions.create(**params)

        return {
            "text": response.choices[0].message.content or "",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "raw": response,
        }

    def json_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat with JSON mode enabled (where supported).

        Returns same format as chat() but attempts to enforce JSON output.
        """
        if self.provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}

        # Note: Gemini and Groq may not support JSON mode directly
        # Fallback to prompt engineering (handled by caller using json_extract prompt)

        return self.chat(messages, temperature=temperature, **kwargs)

    def tool_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat with function calling / tools.

        Args:
            messages: Messages array
            tools: Tool definitions in OpenAI format
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Same format as chat() with potential tool_calls in raw response
        """
        if self.provider == "openai":
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        elif self.provider == "groq":
            # Groq supports OpenAI-compatible function calling
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        else:
            # Fallback: embed tools in prompt (handled by caller using tool_call prompt)
            pass

        return self.chat(messages, temperature=temperature, **kwargs)
