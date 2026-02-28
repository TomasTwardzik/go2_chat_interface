# Workout Plan Generation Rules

You are generating a workout plan that will be parsed and executed by an automated runner. The plan **must** conform to the structure and syntax described below. Any deviation will cause the runner to fail.

## User Specification

Before generating a plan, collect the following from the user (use sensible defaults if not provided):

- **Fitness level** — beginner, intermediate, or advanced
- **Target muscle groups or workout type** — e.g. upper body, legs, full body, cardio, stretching
- **Available time** — total workout duration in minutes
- **Number of exercises** — how many distinct exercises to include
- **Repetitions / duration preference** — rep-based or time-based sets
- **Any injuries or limitations** — exercises to avoid

## Document Structure

The plan must be a Markdown file with exactly this structure:

```markdown
# Workout Procedure

<header lines — keyword definitions and global instructions>

## Prepare phase

<numbered prepare steps>

## Workout Stages

<numbered workout steps>
```

### Rules

1. The title must be exactly `# Workout Procedure`.
2. There must be exactly two sections: `## Prepare phase` and `## Workout Stages`, in that order.
3. Section names are case-sensitive and must match exactly.
4. Every step must be a numbered line (`1.`, `2.`, etc.), starting from 1 and incrementing without gaps.

## Header

The header sits between the title and the first section. It must contain these keyword definitions (copy them verbatim):

```
"Action(<mcp_action>)" key word mens using go2 mcp server.
"Say(<text>)" use mcp voice server to use speakers with the text argument
"Wait(time)" action and time duration defines a waiting in between actions.
"Voice(text)" use llm model to paraphrase the text and then use Say voice mcp command to use the speaker to say the text.

Don't use speakers for debug and step outputs, only use it for Say and Voice commands.
```

## Keywords

Every step must use exactly one of these four keywords:

### Action(`<mcp_action>`)

Calls a Go2 robot MCP tool.

- The argument must be a valid Go2 action name (e.g. `sit`, `stand_up`, `stand_down`, `stretch`, `hello`, `light_on`, `light_off`, `obstacle avoidance off`).
- Use `Action(list_actions)` only in the prepare phase to load available commands.
- An optional duration can follow: `Action(stretch) for 5 seconds`.

### Say(`<text>`)

Speaks the exact text through the voice MCP server.

- The text is spoken verbatim — use this for precise instructions, countdowns, or cues.
- Keep text short and clear (one or two sentences max).
- Do not include special characters or markup in the text.

### Voice(`<text>`)

The model rephrases the text into friendly, natural speech, then speaks it.

- Use this for greetings, encouragement, transitions, and sign-offs.
- The text argument is a description of what to say, not the exact words.
- Format: `Voice(<text>)` or `Voice: <text>` — both are accepted.

### Wait(`<time>`)

Pauses execution for the given duration.

- Format the time as a number followed by a unit: `Wait(0.5 sec)`, `Wait(3 sec)`, `Wait(10 sec)`.
- Use between exercises for rest periods and between actions for timing.
- Minimum wait: 0.5 sec. Maximum wait: 120 sec.

## Repeat Blocks

For repeated movements, use inline repeat notation on a single numbered step:

```
10. Repeat 5 times:( Action(stand_down), Action(stand_up))
```

- Format: `Repeat <N> times:( <step>, <step>, ... )`
- Each inner step must be a valid keyword call.
- Separate inner steps with commas.
- N must be between 1 and 20.

## Prepare Phase Rules

The prepare phase sets up the robot. It must:

1. Start with `Action(obstacle avoidance off)` as step 1.
2. Include `Action(list_actions)` as step 2 — described as: `use Action(list_actions) to load all available commands for an the go2 mcp server to be aware of them and keep them in context.`
3. Contain no Say, Voice, or Wait steps.
4. Have no more than 5 steps total.

## Workout Stages Rules

The workout section is the main routine. It must follow these rules:

1. **Start with a greeting** — begin with a Voice or Say step to greet and motivate the user.
2. **Announce each exercise** — before any Action that the user performs, use Say or Voice to tell them what is coming and how many reps or how long.
3. **Include rest periods** — add Wait steps between exercises (at least 1 sec between different exercises).
4. **End with a sign-off** — the final step must be a Say or Voice farewell.
5. **Exercise safety ordering**:
   - Begin with a warm-up (light movements or stretches).
   - Place high-intensity exercises in the middle.
   - End with a cool-down (stretches or light movements).
6. **Rep counts and durations must be appropriate** for the user's stated fitness level:
   - Beginner: 3–5 reps or 5–10 sec holds
   - Intermediate: 5–10 reps or 10–20 sec holds
   - Advanced: 10–20 reps or 20–30 sec holds
7. **Total step count** should stay between 8 and 40 steps.

## Common Exercise Breakdowns

The robot demonstrates exercises using its available actions. Below are the standard breakdowns for common exercises. When including an exercise in a workout plan, follow the listed prepare position and per-rep sequence exactly. Adjust rep counts and wait durations to the user's fitness level.

### Push-ups

Targets: chest, shoulders, triceps.

- **Prepare position:** `Action(stand_up)` — robot stands in high plank equivalent.
- **One rep:**
  1. `Action(stand_down)` — lower to low plank
  2. `Wait(0.5 sec)` — brief hold at bottom
  3. `Action(stand_up)` — push back up
  4. `Wait(0.5 sec)` — brief hold at top
- **As a Repeat block:** `Repeat N times:( Action(stand_down), Wait(0.5 sec), Action(stand_up), Wait(0.5 sec))`

### Squats

Targets: quadriceps, glutes, hamstrings.

- **Prepare position:** `Action(stand_up)` — robot stands upright.
- **One rep:**
  1. `Action(sit)` — lower into squat
  2. `Wait(1 sec)` — hold at bottom
  3. `Action(stand_up)` — rise back up
  4. `Wait(0.5 sec)` — brief pause at top
- **As a Repeat block:** `Repeat N times:( Action(sit), Wait(1 sec), Action(stand_up), Wait(0.5 sec))`

### Burpees

Targets: full body, cardio.

- **Prepare position:** `Action(stand_up)` — robot stands upright.
- **One rep:**
  1. `Action(stand_down)` — drop to ground
  2. `Wait(0.5 sec)` — brief ground contact
  3. `Action(stand_up)` — jump back up
  4. `Wait(0.5 sec)` — brief pause
- **As a Repeat block:** `Repeat N times:( Action(stand_down), Wait(0.5 sec), Action(stand_up), Wait(0.5 sec))`

### Stretching / Hold

Targets: flexibility, cool-down.

- **Execution:** `Action(stretch) for N seconds` — robot holds stretch position for the given duration.
- Typical durations: beginner 5 sec, intermediate 10 sec, advanced 20 sec.

### Up-downs (Beginner burpee alternative)

Targets: full body, low impact.

- **Prepare position:** `Action(stand_up)` — robot stands upright.
- **One rep:**
  1. `Action(sit)` — lower to ground
  2. `Wait(1 sec)` — rest at bottom
  3. `Action(stand_up)` — stand back up
  4. `Wait(1 sec)` — rest at top
- **As a Repeat block:** `Repeat N times:( Action(sit), Wait(1 sec), Action(stand_up), Wait(1 sec))`

### Plank hold

Targets: core, shoulders.

- **Execution:** `Action(stand_down)` followed by `Wait(N sec)` then `Action(stand_up)`.
- The wait duration is the hold time. Beginner 10 sec, intermediate 20 sec, advanced 30 sec.

### Greeting / Cool-down wave

Used as a warm-up intro or cool-down farewell.

- **Execution:** `Action(hello)` — robot waves. Pair with a `Voice()` or `Say()` greeting.

### Exercise selection guidelines

- Use only the robot actions listed above (`stand_up`, `stand_down`, `sit`, `stretch`, `hello`, `light_on`, `light_off`). Do not invent actions.
- Compose all exercises from these primitives following the breakdowns above.
- If the user requests an exercise not listed here, map it to the closest available breakdown and announce the substitution with a `Say()` step.
- Always include Wait steps between actions within a rep to give the user time to follow along.

## Validation Checklist

After generating the plan, verify every item:

- [ ] Title is exactly `# Workout Procedure`
- [ ] Header contains all keyword definitions verbatim
- [ ] Exactly two sections: `## Prepare phase` then `## Workout Stages`
- [ ] All steps are numbered sequentially starting from 1 in each section
- [ ] Every step uses exactly one valid keyword (Action, Say, Voice, Wait, or Repeat)
- [ ] Prepare phase starts with `Action(obstacle avoidance off)` and `Action(list_actions)`
- [ ] Prepare phase contains no Say, Voice, or Wait steps
- [ ] Workout starts with a greeting (Say or Voice)
- [ ] Workout ends with a farewell (Say or Voice)
- [ ] Every exercise Action is preceded by an announcement (Say or Voice)
- [ ] Rest Waits exist between different exercises
- [ ] Rep counts and durations match the user's fitness level
- [ ] No empty lines between numbered steps within a section
- [ ] Total workout stages between 8 and 40 steps
- [ ] All exercises follow the breakdowns from the Common Exercise Breakdowns section
- [ ] Only valid robot actions are used (`stand_up`, `stand_down`, `sit`, `stretch`, `hello`, `light_on`, `light_off`)
- [ ] Wait steps are included between actions within each rep
