# Chat Interface

A CLI chatbot agent powered by Google Gemini (gemini-3-flash-preview) with MCP tool support.

## Tech Stack

- **Python 3.10+**
- **uv** — package manager and runner
- **google-genai** — Google Gemini API client
- **python-dotenv** — loads `.env` for API keys
- **sounddevice** — microphone audio capture
- **soundfile** — WAV encoding / MP3 decoding
- **numpy** — audio buffer handling
- **gTTS** — Google Text-to-Speech for voice output
- **mcp** — Model Context Protocol client for external tool integration

## Dev Tools

- **ruff** — linter and formatter (line-length 100)
- **ty** — type checker

## Commands

```bash
# Install dependencies
uv sync

# Run the chatbot
uv run python main.py

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check .
```

## Project Structure

- `main.py` — entry point, async chatbot CLI with text input, voice input, and MCP tool use
- `Go2_MCP.json` — MCP server configuration (define external tools here)
- `.env` — holds `GOOGLE_API_KEY` (gitignored, never commit)
- `pyproject.toml` — project config, dependencies, ruff/ty settings

## Chat Commands

- Type text and press Enter — send a text message
- `voice` — push-to-talk: records mic audio, transcribes it, sends to Gemini, and speaks the response aloud
- `clear` — reset conversation history
- `quit` / `exit` / Ctrl+C — stop the chatbot

## Conventions

- Use `from __future__ import annotations` in all Python files
- Type-annotate all function signatures
- All code must pass `ruff check`, `ruff format --check`, and `ty check` before committing
- Keep `.env` out of version control — secrets go in `.env`, example keys in `.env.example`

## MCP Configuration

Define MCP servers in `Go2_MCP.json`:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "uvx",
      "args": ["mcp-server-package"],
      "env": {}
    }
  }
}
```

On startup, the chatbot connects to all configured MCP servers, lists their tools, and exposes them to Gemini via function calling. When Gemini invokes a tool, the chatbot routes the call to the correct MCP server and feeds the result back in an agentic loop.

## Claude Workflow

- Use thinking mode (extended thinking) for all non-trivial actions
