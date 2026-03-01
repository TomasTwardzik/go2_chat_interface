from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai.errors import ServerError

if TYPE_CHECKING:
    from mcp import ClientSession

from chat.audio import AudioIO
from chat.config import (
    MAX_TOOL_CALLS,
    MODEL,
    ServiceUnavailableError,
    check_503,
)
from chat.workout_planner import WorkoutPlanner
from chat.workout_runner import WorkoutRunner

log = logging.getLogger("chat.loop")


class ChatLoop:
    """Main interactive REPL for the chatbot."""

    def __init__(
        self,
        client: genai.Client,
        sessions: list[ClientSession],
        tool_count: int,
    ) -> None:
        self._client = client
        self._sessions = sessions
        self._tool_count = tool_count
        self._audio = AudioIO()
        self._history: list[genai.types.Content] = []

    def _build_config(self) -> genai.types.GenerateContentConfig | None:
        """Build GenerateContentConfig with MCP sessions for AFC."""
        tools: list[Any] = list(self._sessions)
        log.debug("Building chat config with %d MCP session(s)", len(tools))
        if not tools:
            return None
        return genai.types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_TOOL_CALLS,
            ),
        )

    async def run(self) -> None:
        """Run the main chat REPL."""
        config = self._build_config()

        log.info("Chat with %s (%d MCP tools available)", MODEL, self._tool_count)
        log.info("Commands: 'quit'/'exit', 'clear', 'voice', 'start_workout', 'generate_workout'")
        log.info("-" * 60)

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nYou: ")
                user_input = user_input.strip()
            except (KeyboardInterrupt, EOFError):
                log.debug("Received interrupt/EOF, exiting chat loop")
                log.info("Goodbye!")
                break

            if not user_input:
                log.debug("Empty input, skipping")
                continue
            if user_input.lower() in ("quit", "exit"):
                log.debug("Quit command received")
                log.info("Goodbye!")
                break
            if user_input.lower() == "clear":
                self._history.clear()
                log.debug("Conversation history cleared")
                log.info("Conversation cleared.")
                continue
            try:
                if user_input.lower() == "start_workout":
                    log.debug("start_workout command received")
                    runner = WorkoutRunner(self._client, self._sessions)
                    await runner.run()
                    continue
                if user_input.lower() == "generate_workout":
                    log.debug("generate_workout command received")
                    planner = WorkoutPlanner(self._client)
                    await planner.generate()
                    continue

                voice_mode = user_input.lower() == "voice"
                if voice_mode:
                    log.debug("Voice mode activated")
                    try:
                        audio_bytes = await asyncio.to_thread(self._audio.record)
                        transcript = self._audio.transcribe(self._client, audio_bytes)
                        if not transcript:
                            log.warning("Could not transcribe audio")
                            continue
                        log.info("You (voice): %s", transcript)
                        user_input = transcript

                        if "start workout" in transcript.lower():
                            log.debug("Voice command triggered start_workout")
                            runner = WorkoutRunner(self._client, self._sessions)
                            await runner.run()
                            continue
                    except Exception as e:
                        log.error("Voice input failed: %s", e, exc_info=True)
                        continue

                log.debug("Sending user message: %r", user_input[:100])
                self._history.append(
                    genai.types.Content(role="user", parts=[genai.types.Part(text=user_input)])
                )
                try:
                    log.debug("Calling generate_content (history=%d)", len(self._history))
                    response = await self._client.aio.models.generate_content(
                        model=MODEL,
                        contents=self._history,
                        config=config,
                    )
                    assistant_text = response.text or "(empty response)"
                    log.debug("Response: %r", assistant_text[:200])

                    afc_history = response.automatic_function_calling_history
                    if afc_history:
                        log.debug("AFC history: %d entries", len(afc_history))
                        self._history.extend(afc_history)

                    self._history.append(
                        genai.types.Content(
                            role="model", parts=[genai.types.Part(text=assistant_text)]
                        )
                    )
                    log.debug("History size: %d entries", len(self._history))
                except ServerError as e:
                    check_503(e)
                    log.error("generate_content failed: %s", e, exc_info=True)
                    self._history.pop()
                    continue
                except Exception as e:
                    log.error("generate_content failed: %s", e, exc_info=True)
                    self._history.pop()
                    continue

                log.info("Assistant: %s", assistant_text)

                if voice_mode:
                    log.debug("Speaking response in voice mode")
                    await asyncio.to_thread(self._audio.speak, assistant_text)

            except ServiceUnavailableError as e:
                log.error("%s", e)
                break
