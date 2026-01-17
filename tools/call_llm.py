"""
LLM call utilities for making API calls to language models.
Supports OpenAI and can be extended to other providers.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    **kwargs,
) -> str:
    """
    Call an LLM API with the given prompt.

    Args:
        prompt: The prompt text to send to the LLM
        model: Model name to use (defaults based on provider)
        temperature: Sampling temperature (0.0 for deterministic,
            higher for creative)
        max_tokens: Maximum tokens in response
            (None for default/provider max)
        provider: LLM provider to use ("openai", "anthropic", etc.)
        **kwargs: Additional provider-specific parameters

    Returns:
        The LLM response text

    Raises:
        ValueError: If provider is not supported or API key is missing
        Exception: If the API call fails
    """
    if provider == "openai":
        return _call_openai(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    elif provider == "anthropic":
        return _call_anthropic(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    else:
        supported = "'openai', 'anthropic'"
        raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")


def _call_openai(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Call OpenAI API.

    Args:
        prompt: The prompt text
        model: Model name (defaults to gpt-4o-mini or gpt-3.5-turbo)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional OpenAI API parameters

    Returns:
        The response text from OpenAI

    Raises:
        ValueError: If API key is missing
        Exception: If the API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI library not installed. Install it with: pip install openai"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it to your OpenAI API key."
        )

    client = OpenAI(api_key=api_key)

    # Default model if not specified
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Default max_tokens if not specified
    if max_tokens is None:
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    logger.info(f"Calling OpenAI API with model: {model}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        result = response.choices[0].message.content
        if result is None:
            logger.error("OpenAI API returned None content")
            raise ValueError(
                "OpenAI API returned None content. The model may have been filtered or returned no response."
            )
        tokens_used = response.usage.total_tokens if response.usage else "unknown"
        logger.info(f"OpenAI API call successful. Tokens used: {tokens_used}")
        return result

    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise


def _call_anthropic(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Call Anthropic (Claude) API.

    Args:
        prompt: The prompt text
        model: Model name (defaults to claude-3-5-sonnet-20241022)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional Anthropic API parameters

    Returns:
        The response text from Anthropic

    Raises:
        ValueError: If API key is missing
        Exception: If the API call fails
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        msg = (
            "Anthropic library not installed. " "Install it with: pip install anthropic"
        )
        raise ImportError(msg)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it to your Anthropic API key."
        )

    client = Anthropic(api_key=api_key)

    # Default model if not specified
    if model is None:
        default_model = "claude-3-5-sonnet-20241022"
        model = os.getenv("ANTHROPIC_MODEL", default_model)

    # Default max_tokens if not specified
    if max_tokens is None:
        max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))

    logger.info(f"Calling Anthropic API with model: {model}")

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        if not message.content or len(message.content) == 0:
            logger.error("Anthropic API returned empty content")
            raise ValueError(
                "Anthropic API returned empty content. The model may have been filtered or returned no response."
            )
        result = message.content[0].text
        if result is None:
            logger.error("Anthropic API returned None text")
            raise ValueError(
                "Anthropic API returned None text. The model may have been filtered or returned no response."
            )
        if message.usage:
            tokens_used = message.usage.input_tokens + message.usage.output_tokens
        else:
            tokens_used = "unknown"
        logger.info(f"Anthropic API call successful. Tokens used: {tokens_used}")
        return result

    except Exception as e:
        logger.error(f"Anthropic API call failed: {str(e)}")
        raise


def call_llm_json(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    **kwargs,
) -> Dict[str, Any]:
    """
    Call an LLM and parse the response as JSON.

    Args:
        prompt: The prompt text to send to the LLM
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        provider: LLM provider to use
        **kwargs: Additional provider-specific parameters

    Returns:
        Parsed JSON response as a dictionary

    Raises:
        json.JSONDecodeError: If the response is not valid JSON
        Exception: If the API call fails
    """
    response = call_llm(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
        **kwargs,
    )

    # Check if response is None or empty
    if response is None:
        logger.error("LLM returned None response")
        raise ValueError(
            "LLM response is None. The API call may have failed or returned no content."
        )

    if not isinstance(response, str):
        logger.error(f"LLM returned non-string response: {type(response)}")
        raise ValueError(f"LLM response is not a string: {type(response)}")

    # Try to extract JSON from markdown code blocks if present
    response = response.strip()

    # Check if response is empty after stripping
    if not response:
        logger.error("LLM returned empty response")
        raise ValueError(
            "LLM response is empty. The API call may have returned no content."
        )

    if "```json" in response:
        # Extract JSON from markdown code block
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end != -1:
            response = response[start:end].strip()
    elif "```" in response:
        # Extract from generic code block
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            response = response[start:end].strip()

    # Check again after extracting from code blocks
    if not response:
        logger.error("LLM response is empty after extracting from code blocks")
        raise ValueError("LLM response is empty after extracting from code blocks.")

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse JSON response. Response preview: {response[:200]}..."
        )
        logger.error(f"Full response length: {len(response)} characters")
        raise json.JSONDecodeError(
            f"LLM response is not valid JSON: {str(e)}. Response preview: {response[:200]}",
            response,
            e.pos if hasattr(e, "pos") else 0,
        )


async def call_llm_async(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    **kwargs,
) -> str:
    """
    Async version of call_llm for use in FastAPI endpoints.

    Args:
        prompt: The prompt text to send to the LLM
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        provider: LLM provider to use
        **kwargs: Additional provider-specific parameters

    Returns:
        The LLM response text
    """
    # For now, we'll use the sync version in a thread pool
    # In the future, this can be updated to use async clients
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            **kwargs,
        ),
    )


async def call_llm_json_async(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    **kwargs,
) -> Dict[str, Any]:
    """
    Async version of call_llm_json for use in FastAPI endpoints.

    Args:
        prompt: The prompt text to send to the LLM
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        provider: LLM provider to use
        **kwargs: Additional provider-specific parameters

    Returns:
        Parsed JSON response as a dictionary
    """
    response = await call_llm_async(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
        **kwargs,
    )

    # Check if response is None or empty
    if response is None:
        logger.error("LLM returned None response")
        raise ValueError(
            "LLM response is None. The API call may have failed or returned no content."
        )

    if not isinstance(response, str):
        logger.error(f"LLM returned non-string response: {type(response)}")
        raise ValueError(f"LLM response is not a string: {type(response)}")

    # Try to extract JSON from markdown code blocks if present
    response = response.strip()

    # Check if response is empty after stripping
    if not response:
        logger.error("LLM returned empty response")
        raise ValueError(
            "LLM response is empty. The API call may have returned no content."
        )

    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end != -1:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            response = response[start:end].strip()

    # Check again after extracting from code blocks
    if not response:
        logger.error("LLM response is empty after extracting from code blocks")
        raise ValueError("LLM response is empty after extracting from code blocks.")

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse JSON response. Response preview: {response[:200]}..."
        )
        logger.error(f"Full response length: {len(response)} characters")
        raise json.JSONDecodeError(
            f"LLM response is not valid JSON: {str(e)}. Response preview: {response[:200]}",
            response,
            e.pos if hasattr(e, "pos") else 0,
        )
