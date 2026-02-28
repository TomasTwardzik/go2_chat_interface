from __future__ import annotations

import io
import os
import sys
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from google import genai

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1


def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Export it with: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)
    return genai.Client(api_key=api_key)


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

    # Wait for Enter in a thread so recording continues
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


def transcribe_audio(client: genai.Client, model: str, audio_bytes: bytes) -> str:
    """Send audio to Gemini for transcription. Returns the transcript text."""
    audio_part = genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    text_part = genai.types.Part(
        text="Transcribe this audio exactly as spoken. Return only the transcription, nothing else."
    )
    response = client.models.generate_content(
        model=model,
        contents=[genai.types.Content(role="user", parts=[audio_part, text_part])],
    )
    return (response.text or "").strip()


def chat_loop(client: genai.Client) -> None:
    model = "gemini-2.5-flash"
    history: list[genai.types.Content] = []

    print(f"Chat with {model}")
    print("Commands: 'quit'/'exit', 'clear', 'voice' (push-to-talk)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
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

        if user_input.lower() == "voice":
            try:
                audio_bytes = record_audio()
                transcript = transcribe_audio(client, model, audio_bytes)
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
            response = client.models.generate_content(model=model, contents=history)
            assistant_text = response.text or "(empty response)"
        except Exception as e:
            print(f"\nError: {e}")
            history.pop()
            continue
        history.append(
            genai.types.Content(role="model", parts=[genai.types.Part(text=assistant_text)])
        )
        print(f"\nAssistant: {assistant_text}")


def main() -> None:
    client = get_client()
    chat_loop(client)


if __name__ == "__main__":
    main()
