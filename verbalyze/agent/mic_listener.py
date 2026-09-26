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
import select
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
        print("[RECORDING...] Speak into your microphone now. Press [ENTER] when done: ", end="", flush=True)
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
        print("[Listening...] Calibrating ambient background noise...", end="", flush=True)
        ambient_blocks = []
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            for _ in range(6): # 300ms
                data, _ = stream.read(block_size)
                rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
                ambient_blocks.append(rms)

        ambient_level = max(150.0, np.mean(ambient_blocks))
        energy_threshold = max(350.0, ambient_level * 2.2)
        print(f"\r[Listening...] Speak now in {self.language.upper()} (Auto-detecting speech)...              ", flush=True)

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            while total_blocks < max_blocks:
                data, _ = stream.read(block_size)
                total_blocks += 1
                rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

                if not speech_started:
                    if rms > energy_threshold:
                        speech_started = True
                        frames.append(data.copy())
                        print("[Speaking detected... recording]", end="\r", flush=True)
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

        # 1. Try Sovereign On-Prem local faster-whisper engine (sovereign, offline, low latency)
        try:
            from verbalyze.agent.stt_engine import SovereignSTTEngine
            stt = SovereignSTTEngine(model_size="tiny", language=self.language, provider="local")
            text, lat = stt.transcribe_file(wav_path, language=self.language)
            if text and text != "हाँ जी, मैं सुन रहा हूँ।":
                return text
        except Exception:
            pass

        # 2. Try Groq Whisper if API key is present (sub-200ms latency)
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

        # 3. Fallback to Google Speech Recognition (free, built-in, 100% reliable across Indic languages)
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

    def monitor_barge_in_and_record(
        self,
        audio_engine,
        audio_path: str,
        mode: str = "auto",
        silence_seconds: float = 1.0,
        max_seconds: float = 12.0
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Plays audio_path through audio_engine while concurrently monitoring for customer barge-in interruption.

        Returns:
            Tuple of (interrupted: bool, user_wav_path: Optional[str], cutoff_latency_ms: Optional[float])
        """
        if not audio_engine or not os.path.exists(audio_path):
            return False, None, None

        # Start non-blocking playback
        proc = audio_engine.play(audio_path, block=False)
        if proc is None:
            return False, None, None

        if mode == "auto":
            block_size = int(self.sample_rate * 0.03)  # 30ms blocks = 480 samples
            buffered_frames = []
            consecutive_speech = 0
            interruption_threshold = 850.0  # Dynamic threshold to resist speaker acoustic bleed

            try:
                with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
                    while audio_engine.is_playing():
                        data, _ = stream.read(block_size)
                        rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

                        if rms > interruption_threshold:
                            consecutive_speech += 1
                            buffered_frames.append(data.copy())
                            if consecutive_speech >= 2:  # ~60ms sustained human speech
                                # BARGE-IN TRIGGERED
                                t0 = time.time()
                                audio_engine.stop_playback()
                                cutoff_ms = round((time.time() - t0) * 1000.0, 1)
                                print(f"\n[Barge-In Detected! Interrupted agent in {cutoff_ms}ms]")
                                print("[Listening to your interruption...]", end="\r", flush=True)

                                # Continue recording remaining speech using VAD
                                frames = list(buffered_frames)
                                silence_blocks = 0
                                silence_threshold_blocks = int(silence_seconds / 0.03)
                                total_blocks = len(frames)
                                max_blocks = int(max_seconds / 0.03)

                                while total_blocks < max_blocks:
                                    data_rem, _ = stream.read(block_size)
                                    total_blocks += 1
                                    frames.append(data_rem.copy())
                                    rms_rem = float(np.sqrt(np.mean(data_rem.astype(np.float32) ** 2)))
                                    if rms_rem < (interruption_threshold * 0.5):
                                        silence_blocks += 1
                                        if silence_blocks >= silence_threshold_blocks:
                                            break
                                    else:
                                        silence_blocks = 0

                                audio_data = np.concatenate(frames, axis=0)
                                tmp_wav = tempfile.mktemp(suffix=".wav")
                                with wave.open(tmp_wav, "wb") as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(self.sample_rate)
                                    wf.writeframes(audio_data.tobytes())

                                return True, tmp_wav, cutoff_ms
                        else:
                            consecutive_speech = 0
                            if len(buffered_frames) > 4:
                                buffered_frames = buffered_frames[-2:]
            except Exception:
                try:
                    proc.wait()
                except Exception:
                    pass
                return False, None, None

        else:
            # push_to_talk mode: check stdin non-blocking
            is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
            while audio_engine.is_playing():
                if is_tty:
                    try:
                        r, _, _ = select.select([sys.stdin], [], [], 0.04)
                        if r:
                            sys.stdin.readline()
                            t0 = time.time()
                            audio_engine.stop_playback()
                            cutoff_ms = round((time.time() - t0) * 1000.0, 1)
                            print(f"\n[BARGE-IN] Interrupted agent in {cutoff_ms}ms")
                            tmp_wav = self.record_push_to_talk()
                            return True, tmp_wav, cutoff_ms
                    except Exception:
                        pass
                time.sleep(0.04)

        return False, None, None
