from __future__ import annotations

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
SAMPLE_RATE = 16000
CHANNELS = 1
MCP_CONFIG_PATH = Path(__file__).parent / "Go2_MCP.json"


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


def load_mcp_config() -> dict[str, Any]:
    """Load MCP server configuration from Go2_MCP.json."""
    if not MCP_CONFIG_PATH.exists():
        return {}
    with open(MCP_CONFIG_PATH) as f:
        return json.load(f)


async def connect_mcp_servers(
    config: dict[str, Any],
    exit_stack: AsyncExitStack,
) -> dict[str, ClientSession]:
    """Connect to all MCP servers defined in config. Returns {name: session}."""
    servers: dict[str, ClientSession] = {}
    mcp_servers = config.get("mcpServers", {})

    for name, server_cfg in mcp_servers.items():
        params = StdioServerParameters(
            command=server_cfg["command"],
            args=server_cfg.get("args", []),
            env={**os.environ, **server_cfg.get("env", {})},
        )
        read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(params))
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        servers[name] = session
        print(f"  Connected to MCP server: {name}")

    return servers


async def gather_mcp_tools(
    sessions: dict[str, ClientSession],
) -> tuple[list[genai.types.Tool], dict[str, ClientSession]]:
    """List tools from all MCP sessions. Returns Gemini tools and a name->session map."""
    declarations: list[genai.types.FunctionDeclaration] = []
    tool_session_map: dict[str, ClientSession] = {}

    for _server_name, session in sessions.items():
        result = await session.list_tools()
        for tool in result.tools:
            schema = tool.inputSchema
            declarations.append(
                genai.types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters_json_schema=schema,
                )
            )
            tool_session_map[tool.name] = session
            print(f"    Tool: {tool.name}")

    if not declarations:
        return [], tool_session_map

    return [genai.types.Tool(function_declarations=declarations)], tool_session_map


async def execute_tool_call(
    tool_session_map: dict[str, ClientSession],
    function_call: genai.types.FunctionCall,
) -> genai.types.Part:
    """Execute a single MCP tool call and return a FunctionResponse Part."""
    name = function_call.name or ""
    args = dict(function_call.args or {})
    session = tool_session_map.get(name)

    if session is None:
        return genai.types.Part(
            function_response=genai.types.FunctionResponse(
                name=name, response={"error": f"Unknown tool: {name}"}
            )
        )

    print(f"  -> Calling tool: {name}({args})")
    result = await session.call_tool(name, arguments=args)

    texts: list[str] = []
    for block in result.content or []:
        if hasattr(block, "text") and block.text:
            texts.append(str(block.text))
    output = "\n".join(texts)
    if result.isError:
        return genai.types.Part(
            function_response=genai.types.FunctionResponse(name=name, response={"error": output})
        )
    return genai.types.Part(
        function_response=genai.types.FunctionResponse(name=name, response={"output": output})
    )


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
# Agentic chat loop
# ---------------------------------------------------------------------------


async def generate_response(
    client: genai.Client,
    history: list[genai.types.Content],
    tools: list[genai.types.Tool],
    tool_session_map: dict[str, ClientSession],
) -> str:
    """Send history to Gemini, handle tool calls in a loop, return final text."""
    config = genai.types.GenerateContentConfig(tools=list(tools)) if tools else None

    while True:
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=config,
        )

        # Check if the model wants to call functions
        candidates = response.candidates
        if not candidates or candidates[0].content is None:
            return response.text or "(empty response)"

        model_content = candidates[0].content
        parts = model_content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call]

        if not function_calls:
            # No tool calls — return the text response
            assistant_text = response.text or "(empty response)"
            history.append(
                genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
            )
            return assistant_text

        # Add the model's function-call turn to history
        history.append(model_content)

        # Execute all tool calls and collect responses
        response_parts: list[genai.types.Part] = []
        for fc in function_calls:
            part = await execute_tool_call(tool_session_map, fc)
            response_parts.append(part)

        # Add tool results to history and loop back
        history.append(genai.types.Content(role="user", parts=response_parts))


async def chat_loop(
    client: genai.Client,
    tools: list[genai.types.Tool],
    tool_session_map: dict[str, ClientSession],
) -> None:
    history: list[genai.types.Content] = []
    tool_count = sum(len(t.function_declarations or []) for t in tools)

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
            assistant_text = await generate_response(client, history, tools, tool_session_map)
        except Exception as e:
            print(f"\nError: {e}")
            history.pop()
            continue

        print(f"\nAssistant: {assistant_text}")

        if voice_mode:
            await asyncio.to_thread(speak_text, assistant_text)


async def main() -> None:
    client = get_client()

    mcp_config = load_mcp_config()
    mcp_server_names = list(mcp_config.get("mcpServers", {}).keys())

    if mcp_server_names:
        print(f"Connecting to MCP servers: {', '.join(mcp_server_names)}")

    async with AsyncExitStack() as stack:
        sessions = await connect_mcp_servers(mcp_config, stack)
        tools, tool_session_map = await gather_mcp_tools(sessions)
        await chat_loop(client, tools, tool_session_map)


if __name__ == "__main__":
    asyncio.run(main())
