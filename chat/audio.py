from __future__ import annotations

import io
import logging
import tempfile
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from google import genai
from gtts import gTTS

from chat.config import CHANNELS, MODEL, SAMPLE_RATE

log = logging.getLogger("chat.audio")


class AudioIO:
    """Handles audio recording, transcription, and text-to-speech playback."""

    def record(self) -> bytes:
        """Record audio from the microphone until Enter is pressed. Returns WAV bytes."""
        frames: list[np.ndarray] = []
        stop_event = threading.Event()

        def callback(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
            frames.append(indata.copy())

        log.debug("Starting audio recording (rate=%d, channels=%d)", SAMPLE_RATE, CHANNELS)
        log.info("Recording... press [Enter] to stop and send.")
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
        log.info("Recording stopped. Processing...")

        stream.stop()
        stream.close()

        audio_data = np.concatenate(frames, axis=0)
        log.debug(
            "Recorded %d frames, %.1f seconds", len(audio_data), len(audio_data) / SAMPLE_RATE
        )
        buf = io.BytesIO()
        sf.write(buf, audio_data, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        log.debug("Encoded WAV: %d bytes", len(wav_bytes))
        return wav_bytes

    def transcribe(self, client: genai.Client, audio_bytes: bytes) -> str:
        """Send audio to Gemini for transcription. Returns the transcript text."""
        log.debug("Transcribing %d bytes of audio via %s", len(audio_bytes), MODEL)
        audio_part = genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        text_part = genai.types.Part(
            text="Transcribe this audio exactly as spoken. Return only the transcription, "
            "nothing else."
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=[genai.types.Content(role="user", parts=[audio_part, text_part])],
        )
        transcript = (response.text or "").strip()
        log.debug("Transcription result: %r", transcript)
        return transcript

    def speak(self, text: str) -> None:
        """Convert text to speech and play it through speakers."""
        log.debug("TTS: generating speech for %d chars", len(text))
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            tts = gTTS(text=text)
            tts.write_to_fp(tmp)
            tmp.flush()
            tmp.seek(0)
            log.debug("TTS: reading audio from %s", tmp.name)
            data, samplerate = sf.read(tmp.name)
            log.debug("TTS: playing audio (samplerate=%d, samples=%d)", samplerate, len(data))
            sd.play(data, samplerate)
            sd.wait()
        log.debug("TTS: playback complete")
