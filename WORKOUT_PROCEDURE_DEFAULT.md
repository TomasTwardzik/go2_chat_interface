# Workout Procedure

"Action(<mcp_action>)" key word mens using go2 mcp server.
"Say(<text>)" use mcp voice server to use speakers with the text argument
"Wait(time)" action and time duration defines a waiting in between actions.
"Voice(text)" use llm model to paraphrase the text and then use Say voice mcp command to use the speaker to say the text.

Don't use speakers for debug and step outputs, only use it for Say and Voice commands.

## Prepare phase

1. Action(obstacle avoidance off)
2. use Action(list_actions) to load all available commands for an the go2 mcp server to be aware of them and keep them in context.

## Workout Stages

### Greeting
- Announcement: Voice(Greet the trainee, welcome them to the workout and give them words of encouragement)

### Warm-up wave
- Announcement: Say(Let's warm up with a friendly wave)
- Prepare:
  1. Action(sit)
  2. Action(light_on)
- Movement (reps: 1):
  1. Action(hello)
  2. Wait(1 sec)
  3. Action(light_off)

### Push-ups
- Announcement: Say(Let's do some push-ups, 5 repetitions)
- Prepare:
  1. Action(stand_up)
- Movement (reps: 5):
  1. Action(stand_down)
  2. Wait(0.5 sec)
  3. Action(stand_up)
  4. Wait(0.5 sec)

### Stretching
- Announcement: Voice(Good job! Now let's stretch and cool down for 5 seconds)
- Prepare:
  1. Action(stand_up)
- Movement (period: 5s):
  1. Action(stretch)

### Farewell
- Announcement: Say(Awesome! See you at the next workout!)
