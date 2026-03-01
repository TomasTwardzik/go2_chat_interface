from __future__ import annotations

import asyncio
import logging
import re

from google import genai

from chat.config import (
    MAX_REVIEW_RETRIES,
    MODEL,
    RULES_PATH,
    WORKOUT_GENERATED_PATH,
)

log = logging.getLogger("chat.workout.planner")


class WorkoutPlanner:
    """Interactive workout plan generator: gather user prefs, generate, review, save."""

    def __init__(self, client: genai.Client) -> None:
        self._client = client

    @staticmethod
    def _extract_markdown_block(text: str) -> str | None:
        """Extract content from a ```markdown fenced code block, if present."""
        pattern = re.compile(r"```markdown\s*\n(.*?)```", re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else None

    async def _review_plan(self, plan: str, rules: str) -> tuple[str, str | None]:
        """Review a generated workout plan against RULES.md.

        Returns a (severity, issues) tuple:
        - ("pass", None) — plan fully complies
        - ("minor", <issues>) — small/cosmetic issues, no rework needed
        - ("major", <issues>) — structural violations requiring rework
        """
        review_prompt = (
            "You are a strict reviewer. Check the following workout plan against the "
            "validation checklist and all rules below.\n\n"
            "Respond with ONE of these prefixes on the first line:\n"
            '- "PASS" — the plan fully complies with all rules.\n'
            '- "MINOR:" — there are only small cosmetic issues (formatting, minor wording, '
            "non-critical deviations). Follow with a numbered list of issues.\n"
            '- "MAJOR:" — there are strong inconsistencies (wrong keywords, missing required '
            "sections, structural violations, invalid exercise breakdowns). "
            "Follow with a numbered list of issues.\n\n"
            f"## Rules\n\n{rules}\n\n"
            f"## Workout Plan to Review\n\n{plan}"
        )
        log.debug("Sending plan for review (%d chars)", len(plan))
        response = await self._client.aio.models.generate_content(
            model=MODEL, contents=review_prompt
        )
        result = (response.text or "").strip()
        log.debug("Review result: %s", result[:200])
        upper = result.upper()
        if upper.startswith("PASS"):
            return ("pass", None)
        if upper.startswith("MINOR"):
            issues = result.split(":", 1)[1].strip() if ":" in result else result
            return ("minor", issues)
        # Default to major for any MAJOR: prefix or unrecognised format
        issues = result.split(":", 1)[1].strip() if ":" in result else result
        return ("major", issues)

    async def generate(self) -> None:
        """Run the interactive plan generation flow."""
        log.info("Starting workout plan generation")

        if not RULES_PATH.exists():
            log.error("RULES.md not found at %s", RULES_PATH)
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

        log.info("--- Workout Plan Generator ---")
        log.info("Answer the questions below to create a custom workout plan.")
        log.info("Type 'cancel' to abort.")

        # Step 1: Get initial model response (first question)
        response = await self._client.aio.models.generate_content(model=MODEL, contents=history)
        assistant_text = response.text or ""
        history.append(
            genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
        )
        log.info("Coach: %s", assistant_text)

        # Step 2: Multi-turn conversation until plan is generated
        plan_content: str | None = None
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nYou: ")
                user_input = user_input.strip()
            except (KeyboardInterrupt, EOFError):
                log.info("Plan generation cancelled.")
                return

            if not user_input:
                continue
            if user_input.lower() == "cancel":
                log.info("Plan generation cancelled.")
                return

            history.append(
                genai.types.Content(role="user", parts=[genai.types.Part(text=user_input)])
            )

            response = await self._client.aio.models.generate_content(model=MODEL, contents=history)
            assistant_text = response.text or ""
            history.append(
                genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
            )

            plan_content = self._extract_markdown_block(assistant_text)
            if plan_content:
                log.info("Plan generated (%d chars)", len(plan_content))
                log.info("Coach: %s", assistant_text)
                break

            log.info("Coach: %s", assistant_text)

        # Step 3: Save immediately
        WORKOUT_GENERATED_PATH.write_text(plan_content + "\n")
        log.info("Saved workout plan to %s", WORKOUT_GENERATED_PATH.name)

        # Compact context: drop full Q&A history, keep only the system prompt + plan.
        log.debug("Compacting context: %d entries -> 2 (system + plan)", len(history))
        history = [
            history[0],
            genai.types.Content(
                role="model",
                parts=[genai.types.Part(text=f"```markdown\n{plan_content}\n```")],
            ),
        ]

        # Step 4: Severity-aware review loop
        for attempt in range(1, MAX_REVIEW_RETRIES + 2):
            log.info("Reviewing plan (attempt %d)...", attempt)
            severity, issues = await self._review_plan(plan_content, rules)

            if severity == "pass":
                log.info("Plan passed review!")
                break

            if severity == "minor":
                log.warning("Minor issues found (keeping plan): %s", issues)
                break

            # severity == "major"
            if attempt > MAX_REVIEW_RETRIES:
                log.warning("Major issues but max retries reached. Keeping current plan.")
                log.warning("Issues: %s", issues)
                break

            log.warning("Major issues found, regenerating...")
            log.warning("Issues: %s", issues)

            fix_prompt = (
                f"The plan you generated has the following issues:\n\n{issues}\n\n"
                "Please fix all issues and output the corrected plan in a "
                "```markdown code block."
            )
            history.append(
                genai.types.Content(role="user", parts=[genai.types.Part(text=fix_prompt)])
            )
            response = await self._client.aio.models.generate_content(model=MODEL, contents=history)
            assistant_text = response.text or ""
            history.append(
                genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
            )

            fixed_plan = self._extract_markdown_block(assistant_text)
            if fixed_plan:
                plan_content = fixed_plan
                log.info("Regenerated plan (%d chars)", len(plan_content))
                WORKOUT_GENERATED_PATH.write_text(plan_content + "\n")
                log.info("Re-saved updated plan to %s", WORKOUT_GENERATED_PATH.name)
            else:
                log.warning("Regeneration did not produce a markdown block, keeping previous plan")

        log.info("Use 'start_workout' to run it.")
