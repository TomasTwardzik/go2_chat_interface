# Chat Interface

A CLI chatbot agent powered by Google Gemini (gemini-2.5-flash).

## Tech Stack

- **Python 3.10+**
- **uv** — package manager and runner
- **google-genai** — Google Gemini API client
- **python-dotenv** — loads `.env` for API keys
- **sounddevice** — microphone audio capture
- **soundfile** — WAV encoding
- **numpy** — audio buffer handling

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

- `main.py` — entry point, chatbot CLI loop with text and voice input
- `.env` — holds `GOOGLE_API_KEY` (gitignored, never commit)
- `pyproject.toml` — project config, dependencies, ruff/ty settings

## Chat Commands

- Type text and press Enter — send a text message
- `voice` — push-to-talk: records mic audio, transcribes it, then sends to Gemini
- `clear` — reset conversation history
- `quit` / `exit` / Ctrl+C — stop the chatbot

## Conventions

- Use `from __future__ import annotations` in all Python files
- Type-annotate all function signatures
- All code must pass `ruff check`, `ruff format --check`, and `ty check` before committing
- Keep `.env` out of version control — secrets go in `.env`, example keys in `.env.example`

## Claude Workflow

- Use thinking mode (extended thinking) for all non-trivial actions
