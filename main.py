from __future__ import annotations

import argparse
import asyncio
import io
import json
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

MODEL = "gemini-3-flash-preview"
MAX_TOOL_CALLS = 10
STEP_PAUSE = 1.0
SAMPLE_RATE = 16000
CHANNELS = 1
MCP_DIR = Path(__file__).parent / "MCPs"
MCP_REAL_CONFIG = MCP_DIR / "Go2_MCP.json"
MCP_SIM_CONFIG = MCP_DIR / "Go2_MCP_simulator.json"
WORKOUT_PATH = Path(__file__).parent / "Workout_procedure.md"


def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Export it with: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


def load_mcp_config(config_path: Path) -> dict[str, Any]:
    """Load MCP server configuration from the given JSON file."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


async def connect_mcp_servers(
    config: dict[str, Any],
    exit_stack: AsyncExitStack,
) -> dict[str, ClientSession]:
    """Connect to all MCP servers defined in config. Returns {name: session}."""
    servers: dict[str, ClientSession] = {}
    mcp_servers = config.get("mcpServers", {})

    for name, server_cfg in mcp_servers.items():
        command = server_cfg["command"]
        args = server_cfg.get("args", [])
        cwd = server_cfg.get("cwd")
        print(f"  [{name}] Launching: {command} {' '.join(args)}")
        if cwd:
            print(f"  [{name}] Working dir: {cwd}")

        params = StdioServerParameters(
            command=command,
            args=args,
            env={**os.environ, **server_cfg.get("env", {})},
            cwd=cwd,
        )
        try:
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(params))
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            result = await session.initialize()
            server_info = result.serverInfo if result else None
            if server_info:
                version = f" v{server_info.version}" if server_info.version else ""
                print(f"  [{name}] Connected: {server_info.name}{version}")
            else:
                print(f"  [{name}] Connected (no server info)")

            capabilities = session.get_server_capabilities()
            if capabilities and capabilities.tools:
                print(f"  [{name}] Capabilities: tools supported")
            servers[name] = session
        except Exception as e:
            print(f"  [{name}] FAILED to connect: {e}")

    return servers


async def list_mcp_tools(sessions: dict[str, ClientSession]) -> int:
    """Print discovered tools from all sessions. Returns total tool count."""
    total = 0
    for server_name, session in sessions.items():
        result = await session.list_tools()
        count = len(result.tools)
        total += count
        print(f"  [{server_name}] Discovered {count} tool(s):")
        for tool in result.tools:
            desc = f" - {tool.description}" if tool.description else ""
            print(f"    - {tool.name}{desc}")
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
    buf = io.BytesIO()
    sf.write(buf, audio_data, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe_audio(client: genai.Client, audio_bytes: bytes) -> str:
    """Send audio to Gemini for transcription. Returns the transcript text."""
    audio_part = genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    text_part = genai.types.Part(
        text="Transcribe this audio exactly as spoken. Return only the transcription, nothing else."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=[genai.types.Content(role="user", parts=[audio_part, text_part])],
    )
    return (response.text or "").strip()


def speak_text(text: str) -> None:
    """Convert text to speech and play it through speakers."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tts = gTTS(text=text)
        tts.write_to_fp(tmp)
        tmp.flush()
        tmp.seek(0)
        data, samplerate = sf.read(tmp.name)
        sd.play(data, samplerate)
        sd.wait()


# ---------------------------------------------------------------------------
# Workout runner
# ---------------------------------------------------------------------------


async def get_tool_list_summary(sessions: list[ClientSession]) -> str:
    """Return a text summary of all available MCP tools."""
    lines: list[str] = []
    for session in sessions:
        result = await session.list_tools()
        for tool in result.tools:
            desc = f": {tool.description}" if tool.description else ""
            lines.append(f"  - {tool.name}{desc}")
    return "\n".join(lines) if lines else "  (none)"


async def run_workout(
    client: genai.Client,
    sessions: list[ClientSession],
) -> None:
    """Load Workout_procedure.md and execute it step by step."""
    if not WORKOUT_PATH.exists():
        print("[workout] Workout_procedure.md not found.")
        return

    procedure = WORKOUT_PATH.read_text()
    tool_summary = await get_tool_list_summary(sessions)

    print("\n" + "=" * 60)
    print("[workout] Starting workout procedure")
    print("=" * 60)
    print(f"[workout] Available MCP tools:\n{tool_summary}")

    # Extract numbered steps from the ## Stages section
    steps: list[str] = []
    in_stages = False
    for line in procedure.splitlines():
        if line.strip().startswith("## Stages"):
            in_stages = True
            continue
        if in_stages and line.strip() and line.strip()[0].isdigit():
            # Strip leading number + dot
            step_text = line.strip().split(".", 1)[1].strip() if "." in line else line.strip()
            steps.append(step_text)

    if not steps:
        print("[workout] No steps found in procedure.")
        return

    print(f"[workout] {len(steps)} step(s) loaded\n")

    tools: list[Any] = list(sessions)
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

    system_prompt = (
        "You are a workout coach controlling a Go2 robot via MCP tools.\n"
        f"Available tools:\n{tool_summary}\n\n"
        "Keyword reference from the procedure:\n"
        '- "Action(<name>)" means call the MCP tool with that name.\n'
        '- "Say(<text>)" means output exactly that text (it will be spoken aloud).\n'
        '- "Voice: <instruction>" means generate short, motivating speech text.\n'
        '- "Wait(<time>)" means output exactly [WAIT:<seconds>] on its own line '
        "(e.g. Wait(0.5s) becomes [WAIT:0.5]). The system will pause.\n"
        '- "Do N times (actions)" means repeat those actions N times, '
        "including any Wait directives in each iteration.\n"
        "- Don't use speakers for debug output, only for Say and Voice.\n"
        "- Execute ONLY the current step. Be concise.\n"
    )

    history: list[genai.types.Content] = [
        genai.types.Content(role="user", parts=[genai.types.Part(text=system_prompt)]),
        genai.types.Content(
            role="model",
            parts=[genai.types.Part(text="Ready. Send me each step and I will execute it.")],
        ),
    ]

    for i, step in enumerate(steps, 1):
        print(f"[workout] Step {i}/{len(steps)}: {step}")

        history.append(
            genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=f"Execute step {i}: {step}")],
            )
        )

        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=history,
                config=config,
            )
            result_text = response.text or ""

            afc_history = response.automatic_function_calling_history
            if afc_history:
                for entry in afc_history:
                    if entry.parts:
                        for part in entry.parts:
                            if part.function_call:
                                fc = part.function_call
                                print(f"  [action] {fc.name}({dict(fc.args or {})})")
                            if part.function_response:
                                fr = part.function_response
                                print(f"  [result] {fr.name} -> {fr.response}")
                history.extend(afc_history)

            history.append(
                genai.types.Content(role="model", parts=[genai.types.Part(text=result_text)])
            )

            # Process response: handle [WAIT:X] markers and speak the rest
            wait_pattern = re.compile(r"\[WAIT:([\d.]+)]")
            for line in result_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                wait_match = wait_pattern.search(line)
                if wait_match:
                    delay = float(wait_match.group(1))
                    print(f"  [wait]   {delay}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"  [say]    {line}")
                    await asyncio.to_thread(speak_text, line)

        except Exception as e:
            print(f"  [error]  {e}")

        if i < len(steps):
            print(f"  [pause]  {STEP_PAUSE}s")
            await asyncio.sleep(STEP_PAUSE)

    print("\n[workout] Workout complete!")
    print("=" * 60)


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
    print("Commands: 'quit'/'exit', 'clear', 'voice', 'start_workout'")
    print("-" * 60)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nYou: ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if user_input.lower() == "start_workout":
            await run_workout(client, sessions)
            continue

        voice_mode = user_input.lower() == "voice"
        if voice_mode:
            try:
                audio_bytes = await asyncio.to_thread(record_audio)
                transcript = transcribe_audio(client, audio_bytes)
                if not transcript:
                    print("\n(Could not transcribe audio)")
                    continue
                print(f"\nYou (voice): {transcript}")
                user_input = transcript

                # Check if voice command triggers workout
                if "start workout" in transcript.lower():
                    await run_workout(client, sessions)
                    continue
            except Exception as e:
                print(f"\nError: {e}")
                continue

        history.append(genai.types.Content(role="user", parts=[genai.types.Part(text=user_input)]))
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=history,
                config=config,
            )
            assistant_text = response.text or "(empty response)"

            # Preserve AFC tool-call history in conversation context
            afc_history = response.automatic_function_calling_history
            if afc_history:
                history.extend(afc_history)

            history.append(
                genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
            )
        except Exception as e:
            print(f"\nError: {e}")
            history.pop()
            continue

        print(f"\nAssistant: {assistant_text}")

        if voice_mode:
            await asyncio.to_thread(speak_text, assistant_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat Interface - Gemini + MCP")
    parser.add_argument(
        "-s",
        "--sim",
        action="store_true",
        help="Use the simulator MCP server instead of the real one",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config_path = MCP_SIM_CONFIG if args.sim else MCP_REAL_CONFIG
    mode = "SIMULATION" if args.sim else "REAL"

    print("=" * 60)
    print(f"Chat Interface - Starting up [{mode}]")
    print("=" * 60)

    print(f"\n[model] {MODEL}")
    print(f"[mode]  {mode}")

    client = get_client()
    print("[api]   Google API key loaded")

    mcp_config = load_mcp_config(config_path)
    mcp_servers = mcp_config.get("mcpServers", {})

    if not mcp_servers:
        print(f"\n[mcp]   No MCP servers configured in {config_path.name}")
    else:
        print(f"\n[mcp]   Loading {len(mcp_servers)} server(s) from {config_path.name}")

    async with AsyncExitStack() as stack:
        sessions = await connect_mcp_servers(mcp_config, stack)

        connected = len(sessions)
        failed = len(mcp_servers) - connected
        if mcp_servers:
            print(f"\n[mcp]   {connected} connected, {failed} failed")

        tool_count = await list_mcp_tools(sessions)

        print(f"\n[ready] {tool_count} MCP tool(s) available to {MODEL}")
        print(f"[afc]   Automatic function calling enabled (max {MAX_TOOL_CALLS} calls/turn)")
        print("=" * 60)

        await chat_loop(client, list(sessions.values()), tool_count)


if __name__ == "__main__":
    asyncio.run(main())
