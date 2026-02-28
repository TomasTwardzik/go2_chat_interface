from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
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
SAMPLE_RATE = 16000
CHANNELS = 1
MCP_DIR = Path(__file__).parent / "MCPs"
MCP_REAL_CONFIG = MCP_DIR / "Go2_MCP.json"
MCP_SIM_CONFIG = MCP_DIR / "Go2_MCP_simulator.json"


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
    print("Commands: 'quit'/'exit', 'clear', 'voice' (push-to-talk)")
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
