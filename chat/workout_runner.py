from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from google import genai
from google.genai.errors import ServerError

if TYPE_CHECKING:
    from mcp import ClientSession

from chat.config import (
    HISTORY_KEEP_STEPS,
    MAX_TOOL_CALLS,
    MODEL,
    WORKOUT_GENERATED_PATH,
    WORKOUT_PATH,
    check_503,
)

log = logging.getLogger("chat.workout.runner")


class WorkoutRunner:
    """Parses and executes scripted workout procedures."""

    def __init__(self, client: genai.Client, sessions: list[ClientSession]) -> None:
        self._client = client
        self._sessions = sessions

    @staticmethod
    def parse_procedure(text: str) -> dict[str, Any]:
        """Parse workout procedure markdown into header, prepare steps, and exercises.

        Returns:
            {
                "header": str,
                "prepare": list[str],
                "exercises": list[dict] — each with name, announcement,
                    prepare_steps, and optional movement dict.
            }
        """
        log.debug("Parsing workout procedure (%d chars)", len(text))
        header_lines: list[str] = []
        prepare_steps: list[str] = []
        exercises: list[dict[str, Any]] = []

        current_h2: str | None = None
        current_exercise: dict[str, Any] | None = None
        current_phase: str | None = None  # "prepare" or "movement"

        movement_re = re.compile(r"^- Movement \((reps|period): ([\d.]+)s?\):$")

        for line in text.splitlines():
            stripped = line.strip()

            # ## headings
            if stripped.startswith("## "):
                current_h2 = stripped[3:].strip()
                current_exercise = None
                current_phase = None
                continue

            # Before any ## section → header
            if current_h2 is None:
                if stripped and not stripped.startswith("#"):
                    header_lines.append(stripped)
                continue

            # ## Prepare phase → flat numbered steps
            if current_h2 == "Prepare phase":
                if stripped and stripped[0].isdigit():
                    step_text = stripped.split(".", 1)[1].strip() if "." in stripped else stripped
                    prepare_steps.append(step_text)
                continue

            # ## Workout Stages → exercise blocks
            if current_h2 != "Workout Stages":
                continue

            # ### heading → new exercise block
            if stripped.startswith("### "):
                if current_exercise is not None:
                    exercises.append(current_exercise)
                name = stripped[4:].strip()
                current_exercise = {
                    "name": name,
                    "announcement": "",
                    "prepare_steps": [],
                    "movement": None,
                }
                current_phase = None
                continue

            if current_exercise is None:
                continue

            # - Announcement: <command>
            if stripped.startswith("- Announcement:"):
                current_exercise["announcement"] = stripped[len("- Announcement:") :].strip()
                current_phase = None
                continue

            # - Prepare:
            if stripped == "- Prepare:":
                current_phase = "prepare"
                continue

            # - Movement (reps: N): or - Movement (period: Ns):
            m_match = movement_re.match(stripped)
            if m_match:
                m_type = m_match.group(1)
                m_value: int | float = float(m_match.group(2))
                if m_type == "reps":
                    m_value = int(m_value)
                current_exercise["movement"] = {
                    "type": m_type,
                    "value": m_value,
                    "steps": [],
                }
                current_phase = "movement"
                continue

            # Numbered steps under Prepare or Movement
            if stripped and stripped[0].isdigit() and current_phase:
                step_text = stripped.split(".", 1)[1].strip() if "." in stripped else stripped
                if current_phase == "prepare":
                    current_exercise["prepare_steps"].append(step_text)
                elif current_phase == "movement":
                    movement = current_exercise["movement"]
                    if isinstance(movement, dict):
                        cast("list[str]", movement["steps"]).append(step_text)

        # Append the last exercise
        if current_exercise is not None:
            exercises.append(current_exercise)

        log.debug(
            "Parsed: %d header lines, %d prepare steps, %d exercises",
            len(header_lines),
            len(prepare_steps),
            len(exercises),
        )
        for ex in exercises:
            movement = ex.get("movement")
            if movement:
                log.debug(
                    "  Exercise '%s': %s=%s, %d movement steps",
                    ex["name"],
                    movement["type"],
                    movement["value"],
                    len(movement["steps"]),
                )
            else:
                log.debug("  Exercise '%s': announcement only", ex["name"])
        return {
            "header": "\n".join(header_lines),
            "prepare": prepare_steps,
            "exercises": exercises,
        }

    def _build_config(self) -> genai.types.GenerateContentConfig | None:
        """Build GenerateContentConfig with MCP sessions for AFC."""
        tools: list[Any] = list(self._sessions)
        log.debug("Building GenerateContentConfig with %d MCP session(s)", len(tools))
        if not tools:
            return None
        return genai.types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_TOOL_CALLS,
            ),
        )

    async def _execute_steps(
        self,
        label: str,
        steps: list[str],
        system_prompt: list[genai.types.Content],
        config: genai.types.GenerateContentConfig | None,
    ) -> None:
        """Execute a list of workout steps with a sliding context window.

        Keeps the system prompt pair plus only the last HISTORY_KEEP_STEPS step
        exchanges to bound latency and token usage as the workout progresses.
        """
        wait_pattern = re.compile(r"\[WAIT:([\d.]+)]")
        log.debug("[%s] Executing %d step(s) (window=%d)", label, len(steps), HISTORY_KEEP_STEPS)

        step_exchanges: list[list[genai.types.Content]] = []

        for i, step in enumerate(steps, 1):
            log.info("[%s] Step %d/%d: %s", label, i, len(steps), step)

            user_msg = genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=f"Execute step {i}: {step}")],
            )

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
                response = await self._client.aio.models.generate_content(
                    model=MODEL,
                    contents=context,
                    config=config,
                )
                result_text = response.text or ""
                log.debug("[%s] Response text: %r", label, result_text[:200])

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

                # Immediately collapse to [user, model] to prevent AFC history bloat.
                step_exchanges.append([exchange[0], exchange[-1]])
                log.debug(
                    "[%s] Collapsed exchange: %d -> 2 entries",
                    label,
                    len(exchange),
                )
                if len(step_exchanges) > HISTORY_KEEP_STEPS:
                    trimmed = len(step_exchanges) - HISTORY_KEEP_STEPS
                    step_exchanges = step_exchanges[-HISTORY_KEEP_STEPS:]
                    log.debug(
                        "[%s] Trimmed %d old exchange(s), keeping %d",
                        label,
                        trimmed,
                        HISTORY_KEEP_STEPS,
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

            except ServerError as e:
                check_503(e)
                log.error("[%s] Step %d failed: %s", label, i, e, exc_info=True)
            except Exception as e:
                log.error("[%s] Step %d failed: %s", label, i, e, exc_info=True)

        log.debug("[%s] All %d step(s) complete", label, len(steps))

    async def _execute_exercise(
        self,
        exercise: dict[str, Any],
        system_prompt: list[genai.types.Content],
        config: genai.types.GenerateContentConfig | None,
    ) -> None:
        """Execute a single exercise: announcement -> prepare -> movement loop."""
        name = exercise["name"]
        log.info("--- Exercise: %s ---", name)

        # 1. Announcement
        announcement = exercise.get("announcement", "")
        if announcement:
            log.info("[%s] Announcement: %s", name, announcement)
            await self._execute_steps(name, [announcement], system_prompt, config)

        # 2. Prepare
        prepare_steps: list[str] = exercise.get("prepare_steps", [])
        if prepare_steps:
            log.info("[%s] Prepare: %d step(s)", name, len(prepare_steps))
            await self._execute_steps(name, prepare_steps, system_prompt, config)

        # 3. Movement
        movement: dict[str, Any] | None = exercise.get("movement")
        if movement is None:
            return

        m_type: str = movement["type"]
        m_value: int | float = movement["value"]
        steps: list[str] = movement["steps"]

        if m_type == "reps":
            log.info("[%s] Movement: %d reps, %d step(s)/rep", name, m_value, len(steps))
            for rep in range(1, int(m_value) + 1):
                log.info("[%s] Rep %d/%d", name, rep, int(m_value))
                await self._execute_steps(f"{name} rep {rep}", steps, system_prompt, config)
        elif m_type == "period":
            log.info("[%s] Movement: %.1fs period, %d step(s)", name, m_value, len(steps))
            await self._execute_steps(name, steps, system_prompt, config)
            log.info("[%s] Holding for %.1fs", name, m_value)
            await asyncio.sleep(float(m_value))

    async def run(self) -> None:
        """Load workout procedure, run prepare phase, then exercises."""
        log.info("Starting workout procedure")
        if WORKOUT_GENERATED_PATH.exists():
            workout_path = WORKOUT_GENERATED_PATH
            log.info("Using generated workout plan: %s", workout_path.name)
        elif WORKOUT_PATH.exists():
            workout_path = WORKOUT_PATH
            log.info("Using default workout plan: %s", workout_path.name)
        else:
            log.error(
                "No workout procedure found (checked %s and %s)",
                WORKOUT_GENERATED_PATH.name,
                WORKOUT_PATH.name,
            )
            return

        log.debug("Reading %s", workout_path)
        procedure = self.parse_procedure(workout_path.read_text())
        header = procedure["header"]
        prepare_steps: list[str] = procedure["prepare"]
        exercises: list[dict[str, Any]] = procedure["exercises"]

        log.info("Instructions:\n%s", header)
        log.info(
            "%d prepare step(s), %d exercise(s)",
            len(prepare_steps),
            len(exercises),
        )

        config = self._build_config()

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

        # --- Voice summary of the upcoming workout ---
        exercise_lines: list[str] = []
        for ex in exercises:
            movement = ex.get("movement")
            if movement:
                if movement["type"] == "reps":
                    exercise_lines.append(f"- {ex['name']} ({movement['value']} reps)")
                else:
                    exercise_lines.append(f"- {ex['name']} ({movement['value']}s hold)")
            else:
                exercise_lines.append(f"- {ex['name']}")
        workout_text = "\n".join(exercise_lines)
        summary_prompt = (
            "Before we begin, give the trainee a brief spoken summary of the workout "
            "they are about to do. Use the `say` MCP tool to speak it aloud.\n\n"
            "Mention: the number of exercises, a rough time estimate, list a few key "
            "exercises by name, and the overall theme or focus of the workout. "
            "Keep it to 2-3 sentences, upbeat and motivating.\n\n"
            f"Exercises:\n{workout_text}"
        )
        log.info("Generating workout summary")
        await self._client.aio.models.generate_content(
            model=MODEL,
            contents=system_prefix
            + [genai.types.Content(role="user", parts=[genai.types.Part(text=summary_prompt)])],
            config=config,
        )

        # --- Prepare phase ---
        if prepare_steps:
            log.info("--- Prepare phase (%d steps) ---", len(prepare_steps))
            await self._execute_steps("prepare", prepare_steps, system_prefix, config)
            log.info("Prepare phase complete")

        # --- Exercises ---
        for i, exercise in enumerate(exercises, 1):
            log.info("=== Exercise %d/%d ===", i, len(exercises))
            await self._execute_exercise(exercise, system_prefix, config)

        log.info("Workout complete!")
