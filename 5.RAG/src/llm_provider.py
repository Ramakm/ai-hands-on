"""Cloud LLM provider for RAG answer generation.

Supports Atlas Cloud, MiniMax, OpenAI, and any OpenAI-compatible API as an
alternative to local BART generation. Provider is auto-detected from
environment variables or can be explicitly configured.
"""

import os
from dataclasses import dataclass

# Provider presets: name -> (base_url, default_model, env_key)
PROVIDER_PRESETS = {
    "atlas_cloud": {
        "base_url": "https://api.atlascloud.ai/v1",
        "default_model": "deepseek-ai/DeepSeek-V3-0324",
        "env_key": "ATLAS_CLOUD_API_KEY",
        "model_env_key": "ATLAS_CLOUD_MODEL",
        "display_name": "Atlas Cloud",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M3",
        "env_key": "MINIMAX_API_KEY",
        "display_name": "MiniMax",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "display_name": "OpenAI",
    },
}

# Auto-detection order: prefer Atlas Cloud when configured locally.
_DETECTION_ORDER = ["atlas_cloud", "minimax", "openai"]


@dataclass
class LLMConfig:
    """Configuration for a cloud LLM provider."""

    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024


def detect_provider():
    """Auto-detect available LLM provider from environment variables.

    Returns the provider name (e.g. ``"minimax"``) or ``None`` if no API key
    is found.
    """
    for name in _DETECTION_ORDER:
        preset = PROVIDER_PRESETS[name]
        if os.environ.get(preset["env_key"]):
            return name
    return None


def get_llm_config(
    provider=None,
    model=None,
    temperature=0.7,
    max_tokens=1024,
):
    """Build an :class:`LLMConfig` for the given or auto-detected provider.

    Parameters
    ----------
    provider : str, optional
        Provider name (for example ``"atlas_cloud"``, ``"minimax"``, or
        ``"openai"``). When *None*, the provider is auto-detected from
        environment variables.
    model : str, optional
        Model identifier.  Defaults to the provider preset.
    temperature : float
        Sampling temperature.  Clamped to ``(0, 1]`` for MiniMax.
    max_tokens : int
        Maximum tokens in the completion.

    Returns
    -------
    LLMConfig or None
        ``None`` when no provider could be resolved.
    """
    if provider is None:
        provider = detect_provider()
    if provider is None:
        return None

    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return None

    api_key = os.environ.get(preset["env_key"], "")
    if not api_key:
        return None

    env_model = preset.get("model_env_key")
    env_model_value = os.environ.get(env_model, "") if env_model else ""
    resolved_model = model or env_model_value or preset["default_model"]

    # MiniMax temperature must be in (0, 1]
    if provider == "minimax":
        temperature = max(0.01, min(temperature, 1.0))

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=preset["base_url"],
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _strip_think_tags(text):
    """Remove ``<think>...</think>`` blocks that some models emit."""
    import re

    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def chat_completion(config, messages):
    """Send a chat completion request and return the assistant message text.

    Uses the ``openai`` Python SDK so any OpenAI-compatible endpoint works.

    Parameters
    ----------
    config : LLMConfig
        Provider configuration.
    messages : list[dict]
        Chat messages in the ``[{"role": ..., "content": ...}]`` format.

    Returns
    -------
    str
        The assistant's reply.
    """
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    text = response.choices[0].message.content or ""
    return _strip_think_tags(text)


def chat_completion_stream(config, messages):
    """Send a streaming chat completion request and collect streamed text."""
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    stream = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )

    chunks = []
    for event in stream:
        if not getattr(event, "choices", None):
            continue

        delta = getattr(event.choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            chunks.append(content)

    return _strip_think_tags("".join(chunks))
