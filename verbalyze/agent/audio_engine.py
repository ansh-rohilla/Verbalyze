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

from verbalyze.agent.audio_quality import HumanLikenessScorer, AudioQualityReport

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
    def __init__(
        self,
        language: str = "hi",
        voice: Optional[str] = None,
        min_human_likeness: float = 0.80,
        simulate_telephony: bool = False
    ):
        self.language = language
        self.voice = voice or NEURAL_VOICES.get(language, "hi-IN-SwaraNeural")
        self.min_human_likeness = min_human_likeness
        self.simulate_telephony = simulate_telephony
        self.scorer = HumanLikenessScorer(min_threshold=min_human_likeness, simulate_telephony=simulate_telephony)
        self.last_quality_report: Optional[AudioQualityReport] = None
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

    async def _synthesize_edge_tts(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ):
        """Synthesizes using edge-tts async API with rate and pitch modulation."""
        import edge_tts
        v = voice or self.voice
        communicate = edge_tts.Communicate(text, v, rate=rate, pitch=pitch)
        await communicate.save(output_path)

    def _apply_telephony_effect(self, path: str):
        """Applies 8kHz G.711 A-law telephony degradation to an audio file in-place."""
        try:
            import pydub
            raw_seg = pydub.AudioSegment.from_file(path)
            deg_seg = self.scorer.telephony_sim.degrade_audio(raw_seg)
            deg_seg.export(path, format="mp3")
        except Exception:
            pass

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        min_human_likeness: Optional[float] = None,
        auto_heal: bool = True
    ) -> Optional[str]:
        """
        Synthesizes spoken text into an audio file (.mp3).
        Enforces a verified human-likeness quality threshold (default: >= 0.80 / MOS >= 4.2).
        Automatically attempts cadence & prosody auto-tuning if candidate audio falls below threshold.
        Rejects and drops audio if quality criteria cannot be satisfied.
        """
        if not text.strip():
            return None

        threshold = min_human_likeness if min_human_likeness is not None else self.min_human_likeness
        dest_path = output_path or tempfile.mktemp(suffix=".mp3")

        # Attempt 1: Standard synthesis
        try:
            import edge_tts
            asyncio.run(self._synthesize_edge_tts(text, dest_path))
            if self.simulate_telephony:
                self._apply_telephony_effect(dest_path)
            report = self.scorer.evaluate(dest_path, text, simulate_telephony=False if self.simulate_telephony else None)
            self.last_quality_report = report

            if report.score >= threshold:
                return dest_path

            # Attempt 2: Auto-healing if threshold was not reached
            if auto_heal:
                # Determine auto-tuning parameters based on report feedback
                target_rate = "+0%"
                if report.wpm > 155:
                    target_rate = "-8%"  # Slow down rushed speech
                elif report.wpm < 85:
                    target_rate = "+8%"  # Accelerate sluggish speech

                target_pitch = "+2Hz" if report.prosody_score < 0.85 else "+0Hz"
                heal_path = tempfile.mktemp(suffix=".mp3")
                asyncio.run(self._synthesize_edge_tts(text, heal_path, rate=target_rate, pitch=target_pitch))
                if self.simulate_telephony:
                    self._apply_telephony_effect(heal_path)
                heal_report = self.scorer.evaluate(heal_path, text, simulate_telephony=False if self.simulate_telephony else None)

                if heal_report.score >= threshold:
                    shutil.move(heal_path, dest_path)
                    self.last_quality_report = heal_report
                    return dest_path

                # Attempt 3: Alternative voice toggle for Hindi if applicable
                if self.language == "hi":
                    alt_voice = "hi-IN-MadhurNeural" if "Swara" in self.voice else "hi-IN-SwaraNeural"
                    alt_path = tempfile.mktemp(suffix=".mp3")
                    asyncio.run(self._synthesize_edge_tts(text, alt_path, voice=alt_voice, rate=target_rate))
                    if self.simulate_telephony:
                        self._apply_telephony_effect(alt_path)
                    alt_report = self.scorer.evaluate(alt_path, text, simulate_telephony=False if self.simulate_telephony else None)

                    if alt_report.score >= threshold:
                        shutil.move(alt_path, dest_path)
                        self.last_quality_report = alt_report
                        return dest_path

            # If still failing threshold, reject and drop audio
            print(f"⚠️  [Quality Gate REJECTED] Candidate audio score ({report.score*100:.1f}%) < threshold ({threshold*100:.1f}%). {report.feedback}")
            return None

        except ImportError:
            pass
        except Exception as e:
            pass

        # Fallback to gTTS if Edge-TTS failed
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=self.language if self.language in ["hi", "en", "bn", "ta", "te", "gu", "mr", "kn", "ml"] else "hi")
            tts.save(dest_path)
            report = self.scorer.evaluate(dest_path, text)
            self.last_quality_report = report
            if report.score >= threshold:
                return dest_path
            print(f"⚠️  [Quality Gate REJECTED] Fallback audio score ({report.score*100:.1f}%) < threshold ({threshold*100:.1f}%).")
            return None
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
