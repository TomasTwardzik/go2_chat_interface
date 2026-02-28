# Workout Procedure

"Action(<mcp_action>)" key word mens using go2 mcp server.
"Say(<text>)" use mcp voice server to use speakers with the text argument
"Wait(time)" action and time duration defines a waiting in between actions.
"Voice(text)" use llm model to paraphrase the text and then use Say voice mcp command to use the speaker to say the text.

Don't use speakers for debug and step outputs, only use it for Say and Voice commands.

Make 0.5 second pause between the individual steps.

## Prepare phase

1. Action(obstacle avoidance off)
2. use Action(list_actions) to load all available commands for an the go2 mcp server to be aware of them and keep them in context.

## Workout Stages

1. Action(sit)
2. Wait (0.5 sec)
3. Action(light_on)
4. Action(hello)
5. Action(light off)
6. Voice: Greet the trainee and give them words of encouragement.
7. Say(Let's do some pushups, 5 repetitions.)
8. Action(stand_up)
9. Say(Get ready)
10. Repeat 5 times: (
    1. Action(stand_down)
    2. Wait(0.5s)
    3. Action(stand_up)
    4. Wait(0.5s))
11. Voice(Good job! Now we stretch a little).
12. Action(stretch) for 5 seconds
13. Say(Awesome! See you at the next workout!)