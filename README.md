# Chat Interface

A CLI chatbot agent powered by Google Gemini with MCP tool support, voice I/O, and a scripted workout runner.

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- A **Google Gemini API key** — get one at https://aistudio.google.com/apikey

### System dependencies (for voice features)

The voice input/output functionality requires PortAudio and libsndfile:

```bash
# Debian / Ubuntu
sudo apt install portaudio19-dev libsndfile1

# macOS
brew install portaudio libsndfile
```

These are only needed if you plan to use the `voice` command or the workout runner's speech output.

## Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd test_chat_interface
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Create a `.env` file with your API key:

   ```bash
   echo 'GOOGLE_API_KEY=your-key-here' > .env
   ```

## Usage

### Run the chatbot

```bash
uv run python main.py
```

### CLI flags

| Flag | Description |
|------|-------------|
| `-s` / `--sim` | Use the simulator MCP server instead of the real robot |
| `-d` / `--debug` | Enable DEBUG-level logging |

```bash
# Simulator mode
uv run python main.py -s

# Debug logging
uv run python main.py -d

# Both
uv run python main.py -s -d
```

### Chat commands

Once the chatbot is running, you can use these commands at the prompt:

| Command | Description |
|---------|-------------|
| *(any text)* | Send a text message to Gemini |
| `voice` | Push-to-talk: record from mic, transcribe, get spoken response |
| `start_workout` | Run the scripted workout from `WORKOUT_PROCEDURE_DEFAULT.md` |
| `generate_workout` | Interactively create a custom workout plan |
| `clear` | Reset conversation history |
| `quit` / `exit` | Stop the chatbot (or press Ctrl+C) |

## Project Structure

```
main.py                  Entry point — async chatbot CLI
MCPs/
  Go2_MCP.json           Real robot MCP server config
  Go2_MCP_simulator.json Simulator MCP server config
  voice_server.py        Voice output MCP server (say tool)
WORKOUT_PROCEDURE_DEFAULT.md     Default scripted workout procedure
WORKOUT_PROCEDURE.md             Generated workout plan (from generate_workout)
RULES.md                         LLM rules for generating valid plans
.env                             API key (gitignored)
pyproject.toml                   Project config and dependencies
```

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check .
```

## Future Development Ideas

### ~~1. Dynamic workout plan generation~~ (Implemented)

Implemented as the `generate_workout` command. Uses multi-turn Gemini conversation to gather user preferences, generates a plan following `RULES.md`, and reviews it for compliance before saving to `WORKOUT_PROCEDURE.md`.

### 2. Unified voice and text input

Refactor input handling so voice and text work interchangeably at any point in the conversation. Instead of the current mode-switching (`voice` command), the system should accept either input method seamlessly — for example, listening for a push-to-talk key while the text prompt is active, or automatically falling back to text when no microphone is available. This applies during normal chat as well as during workout sessions.

### 3. Visual form assessment via camera

Use a pose estimation model (OpenPose or YOLOv8-Pose) to visually assess the user's exercise form during workouts via camera input. Compare detected joint angles and body positions against reference data for each exercise and assign a quality score.

**Advanced extensions:**

- **Robot repositioning** — the robot autonomously moves to a better viewing angle for the current exercise, then returns to its demonstration position in time to show the next exercise.
- **Verbal coaching** — when form quality drops below a threshold, the robot uses the `say` tool to give real-time corrective hints (e.g. "keep your back straight", "lower your hips more").
