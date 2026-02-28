# Workout Procedure

"Action(<mcp_action>)" key word mens using go2 mcp server.
"Say(<text>)" keyword means use microphone to say exactly following text in quotes.
"Wait(time)" action and time duration defines a waiting in between actions.
"Voice" use speakers to say a paraphrased version of the following text

Don't use speakers for debug and step outputs, only use it for Say and Voice commands.
Load all available commands for an the go2 mcp server to be aware of them and keep them in context.
Make 1 second pause between the individual steps.

## Stages

1. Action(sit), Wait (0.5 sec) Action(light_on), Action(hello), action(light off)
2. Voice: Greet the trainee and give them words of encouragement.
3. Say(Let's do some pushups, 5 repetitions.)
4. Action(stand_up)
5. Say(Get ready)
6. Do 5 times following: (Action(stand_down), Wait(0.5s), Action(stand_up), Wait(0.5s))