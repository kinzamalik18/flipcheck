"""Provider factory for FlipCheck.

Returns the correct provider instance for a given Version config.
"""

from __future__ import annotations

from flipcheck.interfaces import Provider
from flipcheck.types import Version


def get_provider(version: Version) -> Provider:
    """Return a provider instance for the given version.

    Args:
        version: The Version whose config.type selects the provider.

    Returns:
        A provider implementing the Provider protocol.

    Raises:
        ValueError: When config.type is not a known provider type.
    """
    provider_type = version.config.type

    if provider_type == "mock":
        from flipcheck.providers.mock import MockProvider

        return MockProvider(version.config.mock or {})
    elif provider_type == "llm":
        from flipcheck.providers.llm import LLMProvider

        return LLMProvider(version.config)
    elif provider_type == "http":
        from flipcheck.providers.http import HTTPProvider

        return HTTPProvider(version.config)
    elif provider_type == "python":
        from flipcheck.providers.python_target import PythonProvider

        return PythonProvider(version.config)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
