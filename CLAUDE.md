# Chat Interface

A CLI chatbot agent powered by Google Gemini (gemini-3-flash-preview) with MCP tool support, voice I/O, and scripted workout runner.

## Tech Stack

- **Python 3.10+**
- **uv** — package manager and runner
- **google-genai** — Google Gemini API client
- **python-dotenv** — loads `.env` for API keys
- **sounddevice** — microphone audio capture and playback
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

# Run the chatbot (real MCP server)
uv run python main.py

# Run with simulator MCP server
uv run python main.py -s

# Run with debug logging
uv run python main.py -d

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check .
```

## Project Structure

- `main.py` — thin entry point (arg parsing, logging config, MCP setup, launches ChatLoop)
- `chat/` — main application package
  - `__init__.py` — re-exports: ChatLoop, GeminiClient, MCPManager, config constants
  - `config.py` — constants, paths, `ServiceUnavailableError`, `check_503()`
  - `client.py` — `GeminiClient` class (wraps `genai.Client` creation)
  - `mcp_manager.py` — `MCPManager` class (connect servers, list/cache tools)
  - `audio.py` — `AudioIO` class (record, transcribe, speak)
  - `chat_loop.py` — `ChatLoop` class (main REPL, command dispatch)
  - `workout_runner.py` — `WorkoutRunner` class (parse procedure, execute steps)
  - `workout_planner.py` — `WorkoutPlanner` class (multi-turn generation, review, save)
- `MCPs/` — MCP server configs and custom servers
  - `Go2_MCP.json` — real robot MCP config
  - `Go2_MCP_simulator.json` — simulator MCP config
  - `voice_server.py` — voice output MCP server (`say` tool for TTS)
- `WORKOUT_PROCEDURE_DEFAULT.md` — default scripted workout with Action/Say/Voice/Wait keywords
- `WORKOUT_PROCEDURE.md` — generated workout plan (created by `generate_workout`)
- `RULES.md` — LLM instructions for generating valid workout plans
- `.env` — holds `GOOGLE_API_KEY` (gitignored, never commit)
- `pyproject.toml` — project config, dependencies, ruff/ty settings

## Chat Commands

- Type text and press Enter — send a text message
- `voice` — push-to-talk: records mic audio, transcribes, sends to Gemini, speaks response
- `start_workout` — run the scripted workout from `WORKOUT_PROCEDURE_DEFAULT.md`
- `generate_workout` — interactively create a custom workout plan, saved to `WORKOUT_PROCEDURE.md`
- `clear` — reset conversation history
- `quit` / `exit` / Ctrl+C — stop the chatbot

## CLI Flags

- `-s` / `--sim` — use simulator MCP server instead of real robot
- `-d` / `--debug` — enable DEBUG-level logging (shows context sizes, AFC history, API calls)

## MCP Configuration

MCP server configs live in `MCPs/`. Both configs include `go2` (robot) and `voice` (TTS) servers.

```json
{
  "mcpServers": {
    "server-name": {
      "command": "/path/to/python",
      "args": ["server_script.py"],
      "cwd": "/optional/working/dir",
      "env": {}
    }
  }
}
```

On startup, the app connects to all configured MCP servers, lists their tools (cached to avoid repeated ListToolsRequest RPCs), and exposes them to Gemini via automatic function calling (AFC). The SDK handles tool discovery, schema conversion, and routing.

## Voice MCP Server

`MCPs/voice_server.py` provides the `say` tool — speaks exact text via gTTS + sounddevice. Used by the workout runner for Say() and Voice() commands (Voice steps are rephrased by Gemini then passed to `say`).

## Workout Runner

`WORKOUT_PROCEDURE_DEFAULT.md` defines scripted workouts with keyword-driven steps:
- `Action(<name>)` — call Go2 MCP tool
- `Say(<text>)` — call `say` MCP tool with exact text
- `Voice(<text>)` — Gemini rephrases text, then calls `say`
- `Wait(<time>)` — model outputs `[WAIT:<seconds>]`, runner sleeps

The runner parses the procedure into prepare phase and workout stages, executes each step via Gemini with AFC, and uses a **sliding context window** (last 3 step exchanges + system prompt) to bound latency. Older exchanges are collapsed to user+model pairs.

## Workout Plan Generator

The `generate_workout` command launches a multi-turn conversation where Gemini asks the user about their workout preferences (fitness level, target muscles, time, reps, injuries) one question at a time. Once confirmed, the model generates a workout plan following the rules in `RULES.md`.

`RULES.md` defines the required plan structure, valid keywords, exercise breakdowns (push-ups, squats, burpees, stretches, etc.), and a validation checklist. After generation, a separate review model call checks the plan for compliance. If issues are found, the generator retries (up to 2 times). The full Q&A history is kept during the conversation; context is compacted after the plan is extracted before entering the review loop.

The final plan is saved to `WORKOUT_PROCEDURE.md`.

## Logging

Uses Python `logging` library with hierarchical loggers under `chat` (`chat.loop`, `chat.mcp`, `chat.client`, `chat.audio`, `chat.workout.runner`, `chat.workout.planner`, `chat.config`). INFO level by default, DEBUG with `-d` flag. Debug output includes context sizes, AFC history, API call traces, and audio stats. All output goes through logging; `input()` prompts remain for stdin reading.

## Conventions

- Use `from __future__ import annotations` in all Python files
- Type-annotate all function signatures
- All code must pass `ruff check`, `ruff format --check`, and `ty check` before committing
- Keep `.env` out of version control — secrets go in `.env`

## Claude Workflow

- Use thinking mode (extended thinking) for all non-trivial actions
