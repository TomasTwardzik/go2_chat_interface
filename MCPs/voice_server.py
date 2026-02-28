"""MCP server for voice output — Say tool.

Speaks the exact text passed as argument through the speakers.

Usage:
    python MCPs/voice_server.py
"""

from __future__ import annotations

import logging
import sys
import tempfile

import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("voice_mcp")

mcp = FastMCP("voice")


def _speak(text: str) -> None:
    """Convert text to speech and play through speakers."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tts = gTTS(text=text)
        tts.write_to_fp(tmp)
        tmp.flush()
        tmp.seek(0)
        data, samplerate = sf.read(tmp.name)
        sd.play(data, samplerate)
        sd.wait()


@mcp.tool()
def say(text: str) -> str:
    """Speak the exact text out loud through the speakers.

    Use this for direct speech output — the text is spoken exactly as provided.

    Args:
        text: The exact text to speak aloud.
    """
    log.info("say: %s", text)
    _speak(text)
    return f"OK: spoke '{text}'"


if __name__ == "__main__":
    log.info("Voice MCP server starting")
    mcp.run(transport="stdio")
