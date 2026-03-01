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

### Go2 MCP server (robot control)

The chatbot connects to a Go2 robot via an MCP server provided by the upstream [go2_main](https://github.com/zhaw-physical-ai/go2_main) repository. You need to clone it and set up its environment separately:

```bash
git clone https://github.com/zhaw-physical-ai/go2_main.git
cd go2_main
```

Follow the repo's setup instructions to install dependencies and source its Python environment. To run the simulator MCP server (no real robot needed):

```bash
python go2_mcp/dummy_simulator.py
```

The MCP server configs in `MCPs/` reference the Go2 server's Python path and script location. You may need to update `Go2_MCP.json` and `Go2_MCP_simulator.json` to match your local paths.

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
| `start_workout` | Run a workout (prefers generated plan, falls back to default) |
| `generate_workout` | Interactively create a custom workout plan |
| `clear` | Reset conversation history |
| `quit` / `exit` | Stop the chatbot (or press Ctrl+C) |

## Project Structure

```
main.py                          Thin entry point (arg parsing, logging, MCP setup)
chat/                            Main application package
  __init__.py                    Re-exports for clean imports
  config.py                      Constants, paths, error classes
  client.py                      GeminiClient (wraps genai.Client)
  mcp_manager.py                 MCPManager (server connections, tool caching)
  audio.py                       AudioIO (record, transcribe, speak)
  chat_loop.py                   ChatLoop (main REPL, command dispatch)
  workout_runner.py              WorkoutRunner (parse & execute workout procedures)
  workout_planner.py             WorkoutPlanner (multi-turn generation, review)
MCPs/
  Go2_MCP.json                   Real robot MCP server config
  Go2_MCP_simulator.json         Simulator MCP server config
  voice_server.py                Voice output MCP server (say tool)
WORKOUT_PROCEDURE_DEFAULT.md     Default scripted workout procedure
WORKOUT_PROCEDURE.md             Generated workout plan (from generate_workout)
RULES.md                         LLM rules for generating valid plans
.env                             API key (gitignored)
pyproject.toml                   Project config and dependencies
```

## Workout Procedure Format

Workout procedures use a structured exercise block format. Each exercise is a `### <Name>` block with three phases:

1. **Announcement** — `Say()` or `Voice()` stating exercise name and reps/duration
2. **Prepare** — `Action()` steps to position the robot in starting position
3. **Movement** — `reps: N` (repeated N times) or `period: Ns` (executed once, then hold)

Example:

```markdown
### Push-ups
- Announcement: Say(Let's do some push-ups, 5 repetitions)
- Prepare:
  1. Action(stand_up)
- Movement (reps: 5):
  1. Action(stand_down)
  2. Wait(0.5 sec)
  3. Action(stand_up)
  4. Wait(0.5 sec)
```

`WORKOUT_PROCEDURE_DEFAULT.md` provides a built-in workout. `generate_workout` creates custom plans saved to `WORKOUT_PROCEDURE.md`. The `start_workout` command prefers the generated plan if it exists, otherwise falls back to the default.

## Workout Plan Generator

The `generate_workout` command uses a multi-turn conversation where Gemini asks about your preferences (fitness level, target muscles, time, reps, injuries), then generates a plan following `RULES.md`. The plan is saved immediately, then reviewed for compliance — only major structural issues trigger a rework; minor issues are logged and the plan is kept as-is.

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

Implemented as the `generate_workout` command. Uses multi-turn Gemini conversation to gather user preferences, generates a structured exercise plan following `RULES.md`, saves immediately, then reviews with severity-aware classification (PASS/MINOR/MAJOR). The runner executes each exercise through its three phases (announcement, prepare, movement loop).

### 2. Unified voice and text input

Refactor input handling so voice and text work interchangeably at any point in the conversation. Instead of the current mode-switching (`voice` command), the system should accept either input method seamlessly — for example, listening for a push-to-talk key while the text prompt is active, or automatically falling back to text when no microphone is available. This applies during normal chat as well as during workout sessions.

### 3. Visual form assessment via camera

Use a pose estimation model (OpenPose or YOLOv8-Pose) to visually assess the user's exercise form during workouts via camera input. Compare detected joint angles and body positions against reference data for each exercise and assign a quality score.

**Advanced extensions:**

- **Robot repositioning** — the robot autonomously moves to a better viewing angle for the current exercise, then returns to its demonstration position in time to show the next exercise.
- **Verbal coaching** — when form quality drops below a threshold, the robot uses the `say` tool to give real-time corrective hints (e.g. "keep your back straight", "lower your hips more").
