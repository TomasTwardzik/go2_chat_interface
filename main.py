from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from gtts import gTTS
from mcp import ClientSession, StdioServerParameters, stdio_client

load_dotenv()

log = logging.getLogger("chat")

MODEL = "gemini-3-flash-preview"
MAX_TOOL_CALLS = 10
HISTORY_KEEP_STEPS = 3
SAMPLE_RATE = 16000
CHANNELS = 1
MCP_DIR = Path(__file__).parent / "MCPs"
MCP_REAL_CONFIG = MCP_DIR / "Go2_MCP.json"
MCP_SIM_CONFIG = MCP_DIR / "Go2_MCP_simulator.json"
WORKOUT_PATH = Path(__file__).parent / "WORKOUT_PROCEDURE_DEFAULT.md"
WORKOUT_GENERATED_PATH = Path(__file__).parent / "WORKOUT_PROCEDURE.md"
RULES_PATH = Path(__file__).parent / "RULES.md"
MAX_REVIEW_RETRIES = 2


def get_client() -> genai.Client:
    log.debug("Loading Google API key from environment")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY environment variable is not set")
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Export it with: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)
    log.debug("Creating genai.Client")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


def load_mcp_config(config_path: Path) -> dict[str, Any]:
    """Load MCP server configuration from the given JSON file."""
    log.debug("Loading MCP config from %s", config_path)
    if not config_path.exists():
        log.warning("MCP config file not found: %s", config_path)
        return {}
    with open(config_path) as f:
        data = json.load(f)
    log.debug("MCP config loaded: %d server(s)", len(data.get("mcpServers", {})))
    return data


async def connect_mcp_servers(
    config: dict[str, Any],
    exit_stack: AsyncExitStack,
) -> dict[str, ClientSession]:
    """Connect to all MCP servers defined in config. Returns {name: session}."""
    servers: dict[str, ClientSession] = {}
    mcp_servers = config.get("mcpServers", {})
    log.debug("Connecting to %d MCP server(s)", len(mcp_servers))

    for name, server_cfg in mcp_servers.items():
        command = server_cfg["command"]
        args = server_cfg.get("args", [])
        cwd = server_cfg.get("cwd")
        log.info("[%s] Launching: %s %s", name, command, " ".join(args))
        if cwd:
            log.info("[%s] Working dir: %s", name, cwd)

        params = StdioServerParameters(
            command=command,
            args=args,
            env={**os.environ, **server_cfg.get("env", {})},
            cwd=cwd,
        )
        try:
            log.debug("[%s] Opening stdio transport", name)
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(params))
            log.debug("[%s] Creating ClientSession", name)
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            log.debug("[%s] Initializing session", name)
            result = await session.initialize()
            server_info = result.serverInfo if result else None
            if server_info:
                version = f" v{server_info.version}" if server_info.version else ""
                log.info("[%s] Connected: %s%s", name, server_info.name, version)
            else:
                log.info("[%s] Connected (no server info)", name)

            capabilities = session.get_server_capabilities()
            log.debug("[%s] Capabilities: %s", name, capabilities)
            if capabilities and capabilities.tools:
                log.info("[%s] Capabilities: tools supported", name)
            servers[name] = session
        except Exception as e:
            log.error("[%s] FAILED to connect: %s", name, e)

    log.debug("Connected to %d/%d server(s)", len(servers), len(mcp_servers))
    return servers


async def list_mcp_tools(sessions: dict[str, ClientSession]) -> int:
    """Print discovered tools and cache results on each session.

    The google-genai SDK calls session.list_tools() on every generate_content()
    call. By caching the result here we avoid repeated ListToolsRequest RPCs.
    """
    total = 0
    for server_name, session in sessions.items():
        log.debug("[%s] Listing tools", server_name)
        result = await session.list_tools()
        count = len(result.tools)
        total += count
        log.info("[%s] Discovered %d tool(s)", server_name, count)
        for tool in result.tools:
            desc = f" - {tool.description}" if tool.description else ""
            log.info("  - %s%s", tool.name, desc)

        # Patch session so the SDK reuses the cached tool list
        # instead of issuing a new ListToolsRequest per generate_content() call.
        cached = result
        log.debug("[%s] Caching list_tools result (%d tools)", server_name, count)

        async def _cached_list_tools(_cached: Any = cached) -> Any:
            return _cached

        session.list_tools = _cached_list_tools  # type: ignore[assignment]
    log.debug("Total tools discovered: %d", total)
    return total


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def record_audio() -> bytes:
    """Record audio from the microphone until Enter is pressed. Returns WAV bytes."""
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        frames.append(indata.copy())

    log.debug("Starting audio recording (rate=%d, channels=%d)", SAMPLE_RATE, CHANNELS)
    print("\n🎙  Recording... press [Enter] to stop and send.")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    )
    stream.start()

    wait_thread = threading.Thread(target=lambda: (input(), stop_event.set()))
    wait_thread.start()
    stop_event.wait()
    print("⏹  Recording stopped. Processing...")

    stream.stop()
    stream.close()

    audio_data = np.concatenate(frames, axis=0)
    log.debug("Recorded %d frames, %.1f seconds", len(audio_data), len(audio_data) / SAMPLE_RATE)
    buf = io.BytesIO()
    sf.write(buf, audio_data, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    log.debug("Encoded WAV: %d bytes", len(wav_bytes))
    return wav_bytes


def transcribe_audio(client: genai.Client, audio_bytes: bytes) -> str:
    """Send audio to Gemini for transcription. Returns the transcript text."""
    log.debug("Transcribing %d bytes of audio via %s", len(audio_bytes), MODEL)
    audio_part = genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    text_part = genai.types.Part(
        text="Transcribe this audio exactly as spoken. Return only the transcription, nothing else."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=[genai.types.Content(role="user", parts=[audio_part, text_part])],
    )
    transcript = (response.text or "").strip()
    log.debug("Transcription result: %r", transcript)
    return transcript


def speak_text(text: str) -> None:
    """Convert text to speech and play it through speakers."""
    log.debug("TTS: generating speech for %d chars", len(text))
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tts = gTTS(text=text)
        tts.write_to_fp(tmp)
        tmp.flush()
        tmp.seek(0)
        log.debug("TTS: reading audio from %s", tmp.name)
        data, samplerate = sf.read(tmp.name)
        log.debug("TTS: playing audio (samplerate=%d, samples=%d)", samplerate, len(data))
        sd.play(data, samplerate)
        sd.wait()
    log.debug("TTS: playback complete")


# ---------------------------------------------------------------------------
# Workout runner
# ---------------------------------------------------------------------------


def parse_procedure(text: str) -> dict[str, Any]:
    """Parse WORKOUT_PROCEDURE_DEFAULT.md into header, prepare steps, and workout steps."""
    log.debug("Parsing workout procedure (%d chars)", len(text))
    header_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            sections[current_section] = []
            log.debug("Found section: %s", current_section)
            continue
        if current_section is None:
            # Skip the # title line
            if stripped and not stripped.startswith("#"):
                header_lines.append(stripped)
        elif stripped and stripped[0].isdigit():
            step_text = stripped.split(".", 1)[1].strip() if "." in stripped else stripped
            sections[current_section].append(step_text)

    result = {
        "header": "\n".join(header_lines),
        "prepare": sections.get("Prepare phase", []),
        "stages": sections.get("Workout Stages", []),
    }
    log.debug(
        "Parsed: %d header lines, %d prepare steps, %d workout stages",
        len(header_lines),
        len(result["prepare"]),
        len(result["stages"]),
    )
    return result


async def execute_steps(
    label: str,
    steps: list[str],
    client: genai.Client,
    system_prompt: list[genai.types.Content],
    config: genai.types.GenerateContentConfig | None,
) -> None:
    """Execute a list of workout steps with a sliding context window.

    Keeps the system prompt pair plus only the last HISTORY_KEEP_STEPS step
    exchanges to bound latency and token usage as the workout progresses.
    """
    wait_pattern = re.compile(r"\[WAIT:([\d.]+)]")
    log.debug("[%s] Executing %d step(s) (window=%d)", label, len(steps), HISTORY_KEEP_STEPS)

    # Each "step exchange" is a list of Content entries (user + AFC + model)
    # produced by one step. We keep at most HISTORY_KEEP_STEPS of these.
    step_exchanges: list[list[genai.types.Content]] = []

    for i, step in enumerate(steps, 1):
        log.info("[%s] Step %d/%d: %s", label, i, len(steps), step)

        user_msg = genai.types.Content(
            role="user",
            parts=[genai.types.Part(text=f"Execute step {i}: {step}")],
        )

        # Build context: system prompt + recent step exchanges + current user msg
        recent: list[genai.types.Content] = []
        for exchange in step_exchanges:
            recent.extend(exchange)
        context = system_prompt + recent + [user_msg]
        log.debug(
            "[%s] Context: %d system + %d recent + 1 new = %d entries",
            label,
            len(system_prompt),
            len(recent),
            len(context),
        )

        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=context,
                config=config,
            )
            result_text = response.text or ""
            log.debug("[%s] Response text: %r", label, result_text[:200])

            # Collect all entries produced by this step
            exchange: list[genai.types.Content] = [user_msg]

            afc_history = response.automatic_function_calling_history
            if afc_history:
                log.debug("[%s] AFC history: %d entries", label, len(afc_history))
                for entry in afc_history:
                    if entry.parts:
                        for part in entry.parts:
                            if part.function_call:
                                fc = part.function_call
                                log.info(
                                    "[%s]   action: %s(%s)",
                                    label,
                                    fc.name,
                                    dict(fc.args or {}),
                                )
                            if part.function_response:
                                fr = part.function_response
                                log.info("[%s]   result: %s -> %s", label, fr.name, fr.response)
                exchange.extend(afc_history)
            else:
                log.debug("[%s] No AFC history for this step", label)

            exchange.append(
                genai.types.Content(role="model", parts=[genai.types.Part(text=result_text)])
            )

            # Append and trim to sliding window.
            # For older exchanges, collapse to just user + model summary
            # to prevent AFC history from bloating context.
            step_exchanges.append(exchange)
            if len(step_exchanges) > HISTORY_KEEP_STEPS:
                trimmed = len(step_exchanges) - HISTORY_KEEP_STEPS
                step_exchanges = step_exchanges[-HISTORY_KEEP_STEPS:]
                log.debug(
                    "[%s] Trimmed %d old exchange(s), keeping %d",
                    label,
                    trimmed,
                    HISTORY_KEEP_STEPS,
                )
            # Collapse all but the most recent exchange: keep only user + model
            for idx in range(len(step_exchanges) - 1):
                ex = step_exchanges[idx]
                if len(ex) > 2:
                    old_len = len(ex)
                    step_exchanges[idx] = [ex[0], ex[-1]]
                    log.debug(
                        "[%s] Collapsed exchange %d: %d -> 2 entries",
                        label,
                        idx,
                        old_len,
                    )
            total_entries = sum(len(e) for e in step_exchanges)
            log.debug(
                "[%s] Step exchanges: %d, total entries in window: %d",
                label,
                len(step_exchanges),
                total_entries,
            )

            for line in result_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                wait_match = wait_pattern.search(line)
                if wait_match:
                    delay = float(wait_match.group(1))
                    log.info("[%s]   wait: %.1fs", label, delay)
                    await asyncio.sleep(delay)
                else:
                    log.info("[%s]   text: %s", label, line)

        except Exception as e:
            log.error("[%s] Step %d failed: %s", label, i, e, exc_info=True)

    log.debug("[%s] All %d step(s) complete", label, len(steps))


async def run_workout(
    client: genai.Client,
    sessions: list[ClientSession],
) -> None:
    """Load WORKOUT_PROCEDURE_DEFAULT.md, run prepare phase, then workout stages."""
    log.info("Starting workout procedure")
    if not WORKOUT_PATH.exists():
        log.error("WORKOUT_PROCEDURE_DEFAULT.md not found at %s", WORKOUT_PATH)
        return

    log.debug("Reading %s", WORKOUT_PATH)
    procedure = parse_procedure(WORKOUT_PATH.read_text())
    header = procedure["header"]
    prepare_steps: list[str] = procedure["prepare"]
    workout_steps: list[str] = procedure["stages"]

    log.info("Instructions:\n%s", header)
    log.info("%d prepare step(s), %d workout step(s)", len(prepare_steps), len(workout_steps))

    # Build config with MCP sessions
    tools: list[Any] = list(sessions)
    log.debug("Building GenerateContentConfig with %d MCP session(s)", len(tools))
    config = (
        genai.types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_TOOL_CALLS,
            ),
        )
        if tools
        else None
    )

    # System prompt derived from the procedure header
    system_prompt = (
        "You are a workout coach controlling a Go2 robot via MCP tools.\n\n"
        "Instructions from the workout procedure:\n"
        f"{header}\n\n"
        "Additional rules:\n"
        '- For "Say(<text>)" steps, call the `say` MCP tool with the exact text.\n'
        '- For "Voice(<text>)" steps, rephrase the text into short, friendly, '
        "motivating speech and then call the `say` MCP tool with your rephrased version.\n"
        '- For "Wait(<time>)" output exactly [WAIT:<seconds>] on its own line '
        "(e.g. Wait(0.5s) becomes [WAIT:0.5]). The system will handle the pause.\n"
        '- For "Action(<action>)" steps, use the appropriate Go2 MCP tool.\n'
        "- Execute ONLY the current step. Be concise.\n"
    )
    log.debug("System prompt: %s", system_prompt)

    system_prefix: list[genai.types.Content] = [
        genai.types.Content(role="user", parts=[genai.types.Part(text=system_prompt)]),
        genai.types.Content(
            role="model",
            parts=[genai.types.Part(text="Understood. Ready to execute steps.")],
        ),
    ]

    # --- Prepare phase ---
    if prepare_steps:
        log.info("--- Prepare phase (%d steps) ---", len(prepare_steps))
        await execute_steps("prepare", prepare_steps, client, system_prefix, config)
        log.info("Prepare phase complete")

    # --- Workout stages ---
    if workout_steps:
        log.info("--- Workout stages (%d steps) ---", len(workout_steps))
        await execute_steps("workout", workout_steps, client, system_prefix, config)

    log.info("Workout complete!")


# ---------------------------------------------------------------------------
# Workout plan generation
# ---------------------------------------------------------------------------


def _extract_markdown_block(text: str) -> str | None:
    """Extract content from a ```markdown fenced code block, if present."""
    pattern = re.compile(r"```markdown\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


async def _review_plan(client: genai.Client, plan: str, rules: str) -> str | None:
    """Review a generated workout plan against RULES.md.

    Returns None if the plan passes, or a string describing the issues found.
    """
    review_prompt = (
        "You are a strict reviewer. Check the following workout plan against the "
        "validation checklist and all rules below. If the plan fully complies, respond "
        'with exactly "PASS". Otherwise, respond with "FAIL" on the first line, '
        "followed by a numbered list of specific issues that must be fixed.\n\n"
        f"## Rules\n\n{rules}\n\n"
        f"## Workout Plan to Review\n\n{plan}"
    )
    log.debug("Sending plan for review (%d chars)", len(plan))
    response = await client.aio.models.generate_content(model=MODEL, contents=review_prompt)
    result = (response.text or "").strip()
    log.debug("Review result: %s", result[:200])
    if result.upper().startswith("PASS"):
        return None
    return result


async def generate_workout_plan(client: genai.Client) -> None:
    """Interactive workout plan generator: gather user prefs, generate, review, save."""
    log.info("Starting workout plan generation")

    if not RULES_PATH.exists():
        log.error("RULES.md not found at %s", RULES_PATH)
        print("Error: RULES.md not found.")
        return

    rules = RULES_PATH.read_text()

    system_prompt = (
        "You are a workout plan designer for a Go2 robot workout coach. "
        "You must follow these rules exactly:\n\n"
        f"{rules}\n\n"
        "Your task:\n"
        "1. Ask the user about their workout preferences ONE question at a time. "
        "Cover: fitness level, target muscle groups, available time, number of exercises, "
        "rep/duration preference, and any injuries or limitations.\n"
        "2. After gathering enough information, summarize the specification and ask "
        "the user to confirm.\n"
        "3. Once confirmed, generate the complete workout plan and wrap it in a "
        "```markdown code block.\n\n"
        "Start by asking your first question."
    )

    history: list[genai.types.Content] = [
        genai.types.Content(role="user", parts=[genai.types.Part(text=system_prompt)]),
    ]

    print("\n--- Workout Plan Generator ---")
    print("Answer the questions below to create a custom workout plan.")
    print("Type 'cancel' to abort.\n")

    # Step 1: Get initial model response (first question)
    response = await client.aio.models.generate_content(model=MODEL, contents=history)
    assistant_text = response.text or ""
    history.append(genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)]))
    print(f"Coach: {assistant_text}")

    # Step 2: Multi-turn conversation until plan is generated
    plan_content: str | None = None
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nYou: ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print("\nPlan generation cancelled.")
            return

        if not user_input:
            continue
        if user_input.lower() == "cancel":
            print("Plan generation cancelled.")
            return

        history.append(genai.types.Content(role="user", parts=[genai.types.Part(text=user_input)]))

        response = await client.aio.models.generate_content(model=MODEL, contents=history)
        assistant_text = response.text or ""
        history.append(
            genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
        )

        # Check if the response contains a generated plan
        plan_content = _extract_markdown_block(assistant_text)
        if plan_content:
            log.info("Plan generated (%d chars)", len(plan_content))
            print(f"\nCoach: {assistant_text}")
            break

        print(f"\nCoach: {assistant_text}")

    # Compact context: drop full Q&A history, keep only the system prompt + generated plan.
    log.debug("Compacting context: %d entries -> 2 (system + plan)", len(history))
    history = [
        history[0],  # original system prompt
        genai.types.Content(
            role="model",
            parts=[genai.types.Part(text=f"```markdown\n{plan_content}\n```")],
        ),
    ]

    # Step 3: Review and retry loop
    for attempt in range(1, MAX_REVIEW_RETRIES + 2):
        log.info("Review attempt %d", attempt)
        print(f"\nReviewing plan (attempt {attempt})...")
        issues = await _review_plan(client, plan_content, rules)

        if issues is None:
            log.info("Plan passed review")
            print("Plan passed review!")
            break

        log.warning("Plan review failed: %s", issues[:200])
        if attempt > MAX_REVIEW_RETRIES:
            print(f"Review found issues but max retries reached. Saving anyway.\n{issues}")
            break

        print(f"Review found issues, regenerating...\n{issues}")

        # Ask the generator to fix the issues
        fix_prompt = (
            f"The plan you generated has the following issues:\n\n{issues}\n\n"
            "Please fix all issues and output the corrected plan in a ```markdown code block."
        )
        history.append(genai.types.Content(role="user", parts=[genai.types.Part(text=fix_prompt)]))
        response = await client.aio.models.generate_content(model=MODEL, contents=history)
        assistant_text = response.text or ""
        history.append(
            genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
        )

        fixed_plan = _extract_markdown_block(assistant_text)
        if fixed_plan:
            plan_content = fixed_plan
            log.info("Regenerated plan (%d chars)", len(plan_content))
        else:
            log.warning("Regeneration did not produce a markdown block, keeping previous plan")

    # Step 4: Save to file
    WORKOUT_GENERATED_PATH.write_text(plan_content + "\n")
    log.info("Saved workout plan to %s", WORKOUT_GENERATED_PATH)
    print(f"\nWorkout plan saved to {WORKOUT_GENERATED_PATH.name}")
    print("Use 'start_workout' to run it (after updating the workout path if needed).")


# ---------------------------------------------------------------------------
# Chat loop (uses SDK automatic function calling for MCP tools)
# ---------------------------------------------------------------------------


async def chat_loop(
    client: genai.Client,
    sessions: list[ClientSession],
    tool_count: int,
) -> None:
    history: list[genai.types.Content] = []

    # Build config: pass MCP sessions directly — the SDK handles
    # tool discovery, schema conversion, routing, and the agentic loop.
    tools: list[Any] = list(sessions)
    log.debug("Building chat config with %d MCP session(s)", len(tools))
    config = (
        genai.types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_TOOL_CALLS,
            ),
        )
        if tools
        else None
    )

    print(f"\nChat with {MODEL} ({tool_count} MCP tools available)")
    print("Commands: 'quit'/'exit', 'clear', 'voice', 'start_workout', 'generate_workout'")
    print("-" * 60)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nYou: ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            log.debug("Received interrupt/EOF, exiting chat loop")
            print("\nGoodbye!")
            break

        if not user_input:
            log.debug("Empty input, skipping")
            continue
        if user_input.lower() in ("quit", "exit"):
            log.debug("Quit command received")
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            log.debug("Conversation history cleared")
            print("Conversation cleared.")
            continue
        if user_input.lower() == "start_workout":
            log.debug("start_workout command received")
            await run_workout(client, sessions)
            continue
        if user_input.lower() == "generate_workout":
            log.debug("generate_workout command received")
            await generate_workout_plan(client)
            continue

        voice_mode = user_input.lower() == "voice"
        if voice_mode:
            log.debug("Voice mode activated")
            try:
                audio_bytes = await asyncio.to_thread(record_audio)
                transcript = transcribe_audio(client, audio_bytes)
                if not transcript:
                    log.warning("Transcription returned empty result")
                    print("\n(Could not transcribe audio)")
                    continue
                print(f"\nYou (voice): {transcript}")
                user_input = transcript

                # Check if voice command triggers workout
                if "start workout" in transcript.lower():
                    log.debug("Voice command triggered start_workout")
                    await run_workout(client, sessions)
                    continue
            except Exception as e:
                log.error("Voice input failed: %s", e, exc_info=True)
                print(f"\nError: {e}")
                continue

        log.debug("Sending user message: %r", user_input[:100])
        history.append(genai.types.Content(role="user", parts=[genai.types.Part(text=user_input)]))
        try:
            log.debug("Calling generate_content (history=%d)", len(history))
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=history,
                config=config,
            )
            assistant_text = response.text or "(empty response)"
            log.debug("Response: %r", assistant_text[:200])

            # Preserve AFC tool-call history in conversation context
            afc_history = response.automatic_function_calling_history
            if afc_history:
                log.debug("AFC history: %d entries", len(afc_history))
                history.extend(afc_history)

            history.append(
                genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
            )
            log.debug("History size: %d entries", len(history))
        except Exception as e:
            log.error("generate_content failed: %s", e, exc_info=True)
            print(f"\nError: {e}")
            history.pop()
            continue

        print(f"\nAssistant: {assistant_text}")

        if voice_mode:
            log.debug("Speaking response in voice mode")
            await asyncio.to_thread(speak_text, assistant_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat Interface - Gemini + MCP")
    parser.add_argument(
        "-s",
        "--sim",
        action="store_true",
        help="Use the simulator MCP server instead of the real one",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = MCP_SIM_CONFIG if args.sim else MCP_REAL_CONFIG
    mode = "SIMULATION" if args.sim else "REAL"

    log.info("Chat Interface starting up [%s]", mode)
    log.info("Model: %s", MODEL)
    log.debug("Config path: %s", config_path)

    client = get_client()
    log.info("Google API key loaded")

    mcp_config = load_mcp_config(config_path)
    mcp_servers = mcp_config.get("mcpServers", {})

    if not mcp_servers:
        log.warning("No MCP servers configured in %s", config_path.name)
    else:
        log.info("Loading %d server(s) from %s", len(mcp_servers), config_path.name)

    async with AsyncExitStack() as stack:
        sessions = await connect_mcp_servers(mcp_config, stack)

        connected = len(sessions)
        failed = len(mcp_servers) - connected
        if mcp_servers:
            log.info("MCP: %d connected, %d failed", connected, failed)

        tool_count = await list_mcp_tools(sessions)

        log.info("%d MCP tool(s) available to %s", tool_count, MODEL)
        log.info("Automatic function calling enabled (max %d calls/turn)", MAX_TOOL_CALLS)

        await chat_loop(client, list(sessions.values()), tool_count)


if __name__ == "__main__":
    asyncio.run(main())
