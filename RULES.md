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

### <Exercise Name>
- Announcement: <Say or Voice command>
- Prepare:
  1. <Action step>
- Movement (reps: N):
  1. <Action/Wait steps for one rep>

### <Exercise Name>
- Announcement: <Say or Voice command>
- Prepare:
  1. <Action step>
- Movement (period: Ns):
  1. <Action steps>
```

### Rules

1. The title must be exactly `# Workout Procedure`.
2. There must be exactly two sections: `## Prepare phase` and `## Workout Stages`, in that order.
3. Section names are case-sensitive and must match exactly.
4. Prepare phase steps must be numbered lines (`1.`, `2.`, etc.), starting from 1.
5. Workout Stages contain `### ` exercise blocks (see Exercise Block Structure below).

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

Every step within an exercise block must use exactly one of these four keywords:

### Action(`<mcp_action>`)

Calls a Go2 robot MCP tool.

- The argument must be a valid Go2 action name (e.g. `sit`, `stand_up`, `stand_down`, `stretch`, `hello`, `light_on`, `light_off`, `obstacle avoidance off`).
- Use `Action(list_actions)` only in the prepare phase to load available commands.

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
- Use between actions within a movement for timing and rest.
- Minimum wait: 0.5 sec. Maximum wait: 120 sec.

## Prepare Phase Rules

The prepare phase sets up the robot. It must:

1. Start with `Action(obstacle avoidance off)` as step 1.
2. Include `Action(list_actions)` as step 2 — described as: `use Action(list_actions) to load all available commands for an the go2 mcp server to be aware of them and keep them in context.`
3. Contain no Say, Voice, or Wait steps.
4. Have no more than 5 steps total.

## Exercise Block Structure

Every exercise in the `## Workout Stages` section is a `### <Exercise Name>` block with three phases:

### 1. Announcement (required)

```
- Announcement: Say(<exercise name, rep count or duration>)
```

or

```
- Announcement: Voice(<exercise name, rep count or duration>)
```

- Must state the exercise name and either the number of repetitions or the time period.
- Use `Say()` for precise cues, `Voice()` for friendly/motivational phrasing.
- For greeting/farewell blocks, the Announcement is the only required field.

### 2. Prepare (optional)

```
- Prepare:
  1. Action(<action>)
  2. Action(<action>)
```

- Numbered Action steps to position the robot in the starting position for the exercise.
- Omit this section for announcement-only blocks (greeting, farewell).

### 3. Movement (optional)

For repetition-based exercises:

```
- Movement (reps: N):
  1. Action(<action>)
  2. Wait(<time>)
  3. Action(<action>)
  4. Wait(<time>)
```

For time-based exercises:

```
- Movement (period: Ns):
  1. Action(<action>)
```

- **reps: N** — the runner repeats the numbered steps N times. N must be between 1 and 20.
- **period: Ns** — the runner executes the steps once, then waits for the specified duration. Format: integer or decimal followed by `s` (e.g. `5s`, `10s`, `0.5s`).
- Steps within Movement must be numbered starting from 1.
- Include `Wait()` steps between actions within a rep to give the user time to follow along.
- Omit this section for announcement-only blocks (greeting, farewell).

## Workout Stages Rules

The Workout Stages section contains `### ` exercise blocks. It must follow these rules:

1. **Start with a greeting** — the first block should be an announcement-only block greeting and motivating the user.
2. **End with a farewell** — the last block should be an announcement-only block with a farewell message.
3. **Exercise safety ordering**:
   - Begin with warm-up exercises (light movements or stretches).
   - Place high-intensity exercises in the middle.
   - End with cool-down exercises (stretches or light movements).
4. **Rep counts and durations must be appropriate** for the user's stated fitness level:
   - Beginner: 3–5 reps or 5–10 sec holds
   - Intermediate: 5–10 reps or 10–20 sec holds
   - Advanced: 10–20 reps or 20–30 sec holds
5. **Exercise count** should be between 3 and 15 exercise blocks (including greeting/farewell).

## Common Exercise Breakdowns

The robot demonstrates exercises using its available actions. Below are the standard breakdowns for common exercises using the three-phase format. When including an exercise in a workout plan, follow these breakdowns exactly. Adjust rep counts, period durations, and wait times to the user's fitness level.

### Push-ups

Targets: chest, shoulders, triceps.

```markdown
### Push-ups
- Announcement: Say(Let's do push-ups, N repetitions)
- Prepare:
  1. Action(stand_up)
- Movement (reps: N):
  1. Action(stand_down)
  2. Wait(0.5 sec)
  3. Action(stand_up)
  4. Wait(0.5 sec)
```

### Squats

Targets: quadriceps, glutes, hamstrings.

```markdown
### Squats
- Announcement: Say(Time for squats, N repetitions)
- Prepare:
  1. Action(stand_up)
- Movement (reps: N):
  1. Action(sit)
  2. Wait(1 sec)
  3. Action(stand_up)
  4. Wait(0.5 sec)
```

### Burpees

Targets: full body, cardio.

```markdown
### Burpees
- Announcement: Say(Burpees, N repetitions)
- Prepare:
  1. Action(stand_up)
- Movement (reps: N):
  1. Action(stand_down)
  2. Wait(0.5 sec)
  3. Action(stand_up)
  4. Wait(0.5 sec)
```

### Stretching / Hold

Targets: flexibility, cool-down.

```markdown
### Stretching
- Announcement: Voice(Time to stretch and hold for N seconds)
- Prepare:
  1. Action(stand_up)
- Movement (period: Ns):
  1. Action(stretch)
```

### Up-downs (Beginner burpee alternative)

Targets: full body, low impact.

```markdown
### Up-downs
- Announcement: Say(Up-downs, N repetitions)
- Prepare:
  1. Action(stand_up)
- Movement (reps: N):
  1. Action(sit)
  2. Wait(1 sec)
  3. Action(stand_up)
  4. Wait(1 sec)
```

### Plank hold

Targets: core, shoulders.

```markdown
### Plank hold
- Announcement: Say(Plank hold for N seconds)
- Prepare:
  1. Action(stand_up)
- Movement (period: Ns):
  1. Action(stand_down)
```

### Greeting / Cool-down wave

Used as a warm-up intro or cool-down farewell. Announcement-only block.

```markdown
### Greeting
- Announcement: Voice(Greet the trainee and give them words of encouragement)
```

```markdown
### Farewell
- Announcement: Say(Great workout! See you next time!)
```

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
- [ ] Prepare phase steps are numbered sequentially starting from 1
- [ ] Prepare phase starts with `Action(obstacle avoidance off)` and `Action(list_actions)`
- [ ] Prepare phase contains no Say, Voice, or Wait steps
- [ ] Every exercise in Workout Stages is a `### <Name>` block
- [ ] Every exercise block has an `- Announcement:` line with Say or Voice
- [ ] Announcement states the exercise name and rep count or duration
- [ ] Exercise blocks with movement have a `- Prepare:` section with Action steps
- [ ] Movement uses either `reps: N` (1–20) or `period: Ns` format
- [ ] Movement steps are numbered sequentially starting from 1
- [ ] Wait steps exist between actions within each rep
- [ ] First block is a greeting (announcement-only)
- [ ] Last block is a farewell (announcement-only)
- [ ] Rep counts and durations match the user's fitness level
- [ ] Exercise count is between 3 and 15 blocks
- [ ] All exercises follow the breakdowns from Common Exercise Breakdowns
- [ ] Only valid robot actions are used (`stand_up`, `stand_down`, `sit`, `stretch`, `hello`, `light_on`, `light_off`)
