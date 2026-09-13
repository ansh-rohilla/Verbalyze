"""
verbalyze/agent/audio_engine.py

High-speed neural speech synthesis and audio playback for the voice agent.
Supports Microsoft Edge-TTS neural voices across Indian languages with fallback to gTTS.
"""

import os
import sys
import shutil
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

# Standard high-quality neural voices
NEURAL_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "bn": "bn-IN-TanishaaNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-OjasNeural",
    "as": "bn-IN-TanishaaNeural", # Closest neural fallback
    "or": "hi-IN-SwaraNeural",     # Fallback
    "ur": "ur-IN-GulNeural",       # Urdu
}


class AudioEngine:
    def __init__(self, language: str = "hi", voice: Optional[str] = None):
        self.language = language
        self.voice = voice or NEURAL_VOICES.get(language, "hi-IN-SwaraNeural")
        self._check_playback_player()

    def _check_playback_player(self):
        """Finds system player for playing back audio files."""
        if shutil.which("afplay"):
            self.player = "afplay"  # macOS native
        elif shutil.which("ffplay"):
            self.player = "ffplay"
        elif shutil.which("mpv"):
            self.player = "mpv"
        elif shutil.which("aplay"):
            self.player = "aplay"
        else:
            self.player = None

    async def _synthesize_edge_tts(self, text: str, output_path: str):
        """Synthesizes using edge-tts async API."""
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Synthesizes spoken text into an audio file (.mp3)."""
        if not text.strip():
            return None

        dest_path = output_path or tempfile.mktemp(suffix=".mp3")
        
        # Try Edge-TTS first (crystal clear neural voice)
        try:
            import edge_tts
            asyncio.run(self._synthesize_edge_tts(text, dest_path))
            return dest_path
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback to gTTS
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=self.language if self.language in ["hi", "en", "bn", "ta", "te", "gu", "mr", "kn", "ml"] else "hi")
            tts.save(dest_path)
            return dest_path
        except Exception:
            return None

    def play(self, audio_path: str, block: bool = True):
        """Plays audio file on speakers."""
        if not self.player or not os.path.exists(audio_path):
            return

        cmd = [self.player, audio_path]
        if self.player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]

        try:
            if block:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
