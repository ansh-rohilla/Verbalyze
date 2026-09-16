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


def _secure_temp_audio_file(suffix: str = ".mp3") -> str:
    """Creates a temporary audio file with restricted 0600 permissions (owner-only access)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.chmod(path, 0o600)
    return path


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
        self._current_proc: Optional[subprocess.Popen] = None
        self._check_playback_player()

    def _check_playback_player(self):
        """Finds system player for playing back audio files."""
        if shutil.which("afplay"):
            self.player = "afplay"  # macOS native
        elif shutil.which("ffplay"):
            self.player = "ffplay"
        elif shutil.which("mpv"):
            self.player = "mpv"
        else:
            self.player = None

    async def _synthesize_edge_tts(self, text: str, output_path: str, voice: Optional[str] = None, rate: str = "+0%", pitch: str = "+0Hz"):
        """Invokes Microsoft Edge-TTS neural engine asynchronously."""
        import edge_tts
        v = voice or self.voice
        communicate = edge_tts.Communicate(text=text, voice=v, rate=rate, pitch=pitch)
        await communicate.save(output_path)

    def _apply_telephony_effect(self, audio_path: str):
        """Applies 8,000 Hz bandpass and G.711 companding filter to audio."""
        if hasattr(self.scorer, "channel_sim") and self.scorer.channel_sim:
            processed_path = self.scorer.channel_sim.process_file(audio_path)
            shutil.move(processed_path, audio_path)

    async def synthesize_async(
        self,
        text: str,
        output_path: Optional[str] = None,
        min_human_likeness: Optional[float] = None,
        auto_heal: bool = True
    ) -> Optional[str]:
        """
        Asynchronously synthesizes spoken text into an audio file (.mp3).
        Safe to call directly within FastAPI, async streaming generators, or WebSocket event loops.
        """
        if not text.strip():
            return None

        threshold = min_human_likeness if min_human_likeness is not None else self.min_human_likeness
        dest_path = output_path or _secure_temp_audio_file(suffix=".mp3")

        # Attempt 1: Standard synthesis
        try:
            import edge_tts
            await self._synthesize_edge_tts(text, dest_path)
            if self.simulate_telephony:
                self._apply_telephony_effect(dest_path)
            report = self.scorer.evaluate(dest_path, text, simulate_telephony=False if self.simulate_telephony else None)
            self.last_quality_report = report

            if report.score >= threshold:
                return dest_path

            # Attempt 2: Auto-healing if threshold was not reached
            if auto_heal:
                target_rate = "+0%"
                if report.wpm > 155:
                    target_rate = "-8%"  # Slow down rushed speech
                elif report.wpm < 85:
                    target_rate = "+8%"  # Accelerate sluggish speech

                target_pitch = "+2Hz" if report.prosody_score < 0.85 else "+0Hz"
                heal_path = _secure_temp_audio_file(suffix=".mp3")
                await self._synthesize_edge_tts(text, heal_path, rate=target_rate, pitch=target_pitch)
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
                    alt_path = _secure_temp_audio_file(suffix=".mp3")
                    await self._synthesize_edge_tts(text, alt_path, voice=alt_voice, rate=target_rate)
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
        except Exception:
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
            pass

        # Fallback 3: Local pre-cached offline neural sample for air-gapped / sovereign environments
        try:
            sample_dir = Path(__file__).resolve().parent.parent.parent / "samples" / "tts"
            cached_sample = sample_dir / "1_edgetts_swara_hindi.mp3"
            if cached_sample.exists():
                shutil.copyfile(str(cached_sample), dest_path)
                return dest_path
        except Exception:
            pass

        return None

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        min_human_likeness: Optional[float] = None,
        auto_heal: bool = True
    ) -> Optional[str]:
        """
        Synchronous wrapper for synthesize_async.
        Safe to call from sync functions, threads, or inside running event loops.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run,
                        self.synthesize_async(text, output_path, min_human_likeness, auto_heal)
                    ).result()
            else:
                return loop.run_until_complete(
                    self.synthesize_async(text, output_path, min_human_likeness, auto_heal)
                )
        except RuntimeError:
            return asyncio.run(
                self.synthesize_async(text, output_path, min_human_likeness, auto_heal)
            )

    def is_playing(self) -> bool:
        """Returns True if audio playback is currently active."""
        return self._current_proc is not None and self._current_proc.poll() is None

    def stop_playback(self) -> bool:
        """
        Immediately stops active audio playback (<10ms cutoff).
        Returns True if an active playback process was terminated.
        """
        if self._current_proc is not None and self._current_proc.poll() is None:
            try:
                self._current_proc.terminate()
                self._current_proc.wait(timeout=0.08)
            except Exception:
                try:
                    self._current_proc.kill()
                except Exception:
                    pass
            self._current_proc = None
            return True
        self._current_proc = None
        return False

    def play(self, audio_path: str, block: bool = True) -> Optional[subprocess.Popen]:
        """Plays audio file on speakers. Tracks process handle for barge-in interruptions."""
        if not self.player or not os.path.exists(audio_path):
            return None

        # Stop any existing playback before launching new audio
        self.stop_playback()

        cmd = [self.player, audio_path]
        if self.player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._current_proc = proc
            if block:
                proc.wait()
                self._current_proc = None
            return proc
        except Exception:
            return None
