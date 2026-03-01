from __future__ import annotations

import logging
import os
import sys

from google import genai

log = logging.getLogger("chat.client")


class GeminiClient:
    """Wraps Google Gemini API client creation and API key management."""

    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def get_client(self) -> genai.Client:
        """Load API key from environment and create genai.Client. Exits on missing key."""
        if self._client is not None:
            return self._client

        log.debug("Loading Google API key from environment")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            log.critical("GOOGLE_API_KEY environment variable is not set")
            log.critical("Export it with: export GOOGLE_API_KEY='your-key-here'")
            sys.exit(1)
        log.debug("Creating genai.Client")
        self._client = genai.Client(api_key=api_key)
        return self._client
