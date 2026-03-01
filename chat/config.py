from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai.errors import ServerError

log = logging.getLogger("chat.config")

# --- Model ---
MODEL = "gemini-3-flash-preview"
MAX_TOOL_CALLS = 10

# --- Audio ---
SAMPLE_RATE = 16000
CHANNELS = 1

# --- Workout ---
HISTORY_KEEP_STEPS = 3
MAX_REVIEW_RETRIES = 2

# --- Paths ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_DIR = _PROJECT_ROOT / "MCPs"
MCP_REAL_CONFIG = MCP_DIR / "Go2_MCP.json"
MCP_SIM_CONFIG = MCP_DIR / "Go2_MCP_simulator.json"
WORKOUT_PATH = _PROJECT_ROOT / "WORKOUT_PROCEDURE_DEFAULT.md"
WORKOUT_GENERATED_PATH = _PROJECT_ROOT / "WORKOUT_PROCEDURE.md"
RULES_PATH = _PROJECT_ROOT / "RULES.md"

# --- Error handling ---
SERVER_UNAVAILABLE_MSG = (
    "The Gemini API is currently unavailable due to high demand. Please try again later."
)


class ServiceUnavailableError(Exception):
    """Raised when the Gemini API returns 503 UNAVAILABLE."""


def check_503(err: ServerError) -> None:
    """Re-raise as ServiceUnavailableError if the server returned 503."""
    if err.code == 503:
        log.error("Gemini API 503 UNAVAILABLE: %s", err)
        raise ServiceUnavailableError(SERVER_UNAVAILABLE_MSG) from err
