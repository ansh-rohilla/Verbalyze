"""
verbalyze/agent/mic_listener.py

Real-time microphone audio capture and Speech-to-Text (STT) for macOS / Linux.
Features:
1. Push-to-Talk (Press Enter to start, Press Enter to finish)
2. Voice Activity Detection (VAD) / Silence auto-detection
3. Multi-lingual Indic STT (Google Speech API / Groq Whisper)
"""

import os
import sys
import time
import wave
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import speech_recognition as sr

# Regional Indic language codes for Google STT
GOOGLE_STT_LOCALES = {
    "hi": "hi-IN",
    "en": "en-IN",
    "gu": "gu-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-Guru-IN",
    "ur": "ur-IN",
    "or": "hi-IN", # fallback for Odia
    "as": "bn-IN", # fallback for Assamese
}


class MicrophoneListener:
    """Captures microphone audio and converts it to text across Indic languages."""

    def __init__(self, language: str = "hi", sample_rate: int = 16000):
        self.language = language
        self.locale = GOOGLE_STT_LOCALES.get(language, "hi-IN")
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()

    def record_push_to_talk(self) -> Optional[str]:
        """
        Records microphone audio while the user is speaking.
        User presses Enter to stop recording.
        """
        frames = []
        is_recording = True

        def callback(indata, frame_count, time_info, status):
            if is_recording:
                frames.append(indata.copy())

        # Start stream
        print("🎙️  [RECORDING...] Speak into your microphone now. Press [ENTER] when done: ", end="", flush=True)
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", callback=callback)
        with stream:
            try:
                input() # Wait for Enter key
            except (KeyboardInterrupt, EOFError):
                pass
            is_recording = False

        if not frames:
            return None

        # Concatenate and save to WAV
        audio_data = np.concatenate(frames, axis=0)
        
        # Check if empty/silent
        if len(audio_data) < self.sample_rate * 0.3:
            return None

        tmp_wav = tempfile.mktemp(suffix=".wav")
        with wave.open(tmp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        return tmp_wav

    def record_auto_vad(self, silence_seconds: float = 1.2, max_seconds: float = 12.0) -> Optional[str]:
        """
        Records automatically using Voice Activity Detection:
        Waits for voice onset, records until 1.2s of silence is observed.
        """
        block_size = int(self.sample_rate * 0.05) # 50ms blocks
        frames = []
        speech_started = False
        silence_blocks = 0
        silence_threshold_blocks = int(silence_seconds / 0.05)
        max_blocks = int(max_seconds / 0.05)
        total_blocks = 0

        # Ambient energy calibration
        print("🎙️  [Listening...] Calibrating ambient background noise...", end="", flush=True)
        ambient_blocks = []
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            for _ in range(6): # 300ms
                data, _ = stream.read(block_size)
                rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
                ambient_blocks.append(rms)

        ambient_level = max(150.0, np.mean(ambient_blocks))
        energy_threshold = max(350.0, ambient_level * 2.2)
        print(f"\r🎙️  [Listening...] Speak now in {self.language.upper()} (Auto-detecting speech)...              ", flush=True)

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            while total_blocks < max_blocks:
                data, _ = stream.read(block_size)
                total_blocks += 1
                rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

                if not speech_started:
                    if rms > energy_threshold:
                        speech_started = True
                        frames.append(data.copy())
                        print("🔴 [Speaking detected... recording]", end="\r", flush=True)
                else:
                    frames.append(data.copy())
                    if rms < energy_threshold:
                        silence_blocks += 1
                        if silence_blocks >= silence_threshold_blocks:
                            break
                    else:
                        silence_blocks = 0

        if not speech_started or not frames:
            return None

        audio_data = np.concatenate(frames, axis=0)
        tmp_wav = tempfile.mktemp(suffix=".wav")
        with wave.open(tmp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        return tmp_wav

    def transcribe(self, wav_path: str) -> Optional[str]:
        """Transcribes audio file using Google STT or Groq Whisper with fallback."""
        if not wav_path or not os.path.exists(wav_path):
            return None

        # 1. Try Groq Whisper if API key is present (sub-200ms latency)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                import requests
                headers = {"Authorization": f"Bearer {groq_api_key}"}
                with open(wav_path, "rb") as f:
                    files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
                    data = {
                        "model": "whisper-large-v3-turbo",
                        "language": self.language if self.language in ["hi", "en", "gu", "ta", "te", "mr", "bn"] else None,
                        "temperature": "0.0"
                    }
                    resp = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, files=files, data=data, timeout=8)
                    if resp.status_code == 200:
                        text = resp.json().get("text", "").strip()
                        if text:
                            return text
            except Exception:
                pass

        # 2. Fallback to Google Speech Recognition (free, built-in, 100% reliable across Indic languages)
        try:
            with sr.AudioFile(wav_path) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language=self.locale)
            return text.strip() if text else None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            # Silence error or network timeout
            return None
