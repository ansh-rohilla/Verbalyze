"""
Verbalyze Real-Time Dual-Channel Active Speaker Diarization & Cross-Talk Energy Estimator.
Pure-Math Normalized Cross-Correlation (NCC) & Relative Energy DSP Engine.

Author: Verbalyze Telephony & Voice AI Team
Sovereignty: Section 65B Indian Evidence Act / ITU-T Compliant Telephony Signal Processing
Constraint: STRICT ZERO EMOJIS.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class DiarizationState(str, Enum):
    """Real-time speaker ownership state for a 20ms telephony audio frame."""
    SILENCE = "SILENCE"
    CALLER_ONLY = "CALLER_ONLY"
    AGENT_ONLY = "AGENT_ONLY"
    DOUBLE_TALK = "DOUBLE_TALK"
    CROSS_TALK_BLEED = "CROSS_TALK_BLEED"


@dataclass
class FrameDiarizationTelemetry:
    """Frame-level acoustic and speaker attribution telemetry."""
    frame_index: int
    start_ms: float
    end_ms: float
    state: DiarizationState
    caller_rms_dbov: float
    agent_rms_dbov: float
    cross_correlation: float
    cross_talk_bleed_detected: bool
    dominance_score: float
    processing_time_ms: float

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "start_ms": round(float(self.start_ms), 1),
            "end_ms": round(float(self.end_ms), 1),
            "state": self.state.value if isinstance(self.state, DiarizationState) else str(self.state),
            "caller_rms_dbov": round(float(self.caller_rms_dbov), 2),
            "agent_rms_dbov": round(float(self.agent_rms_dbov), 2),
            "cross_correlation": round(float(self.cross_correlation), 4),
            "cross_talk_bleed_detected": bool(self.cross_talk_bleed_detected),
            "dominance_score": round(float(self.dominance_score), 4),
            "processing_time_ms": round(float(self.processing_time_ms), 4),
        }


@dataclass
class SpeakerTurn:
    """Aggregated temporal speaker turn with cross-talk and dominance metrics."""
    turn_index: int
    start_ms: float
    end_ms: float
    duration_ms: float
    speaker: str
    dominant_speaker: str
    average_dominance: float
    cross_talk_bleed_frames: int
    cross_talk_bleed_ratio: float
    caller_rms_dbov: float
    agent_rms_dbov: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "turn_index": int(self.turn_index),
            "start_ms": round(float(self.start_ms), 1),
            "end_ms": round(float(self.end_ms), 1),
            "duration_ms": round(float(self.duration_ms), 1),
            "speaker": str(self.speaker),
            "dominant_speaker": str(self.dominant_speaker),
            "average_dominance": round(float(self.average_dominance), 4),
            "cross_talk_bleed_frames": int(self.cross_talk_bleed_frames),
            "cross_talk_bleed_ratio": round(float(self.cross_talk_bleed_ratio), 4),
            "caller_rms_dbov": round(float(self.caller_rms_dbov), 2),
            "agent_rms_dbov": round(float(self.agent_rms_dbov), 2),
            "confidence": round(float(self.confidence), 4),
        }


class DualChannelDiarizer:
    """
    Pure-Math Real-Time Dual-Channel Active Speaker Diarizer & Cross-Talk Estimator.
    
    Operates on 8kHz narrowband or 16kHz wideband linear PCM telephony streams:
      - Channel 0: Near-End Caller / Borrower
      - Channel 1: Far-End Agent / VoiceBot
      
    Features:
      - Real-time speech level estimation in dBov relative to digital full-scale.
      - Normalized Cross-Correlation (NCC) across delay search windows to reject
        loudspeaker acoustic bleed / echoed bot audio from caller microphone.
      - Relative energy ratio and speaker dominance metric D in [-1.0, +1.0].
      - Hysteresis and hangover debouncing to preserve vocal continuity across micro-pauses.
      - Automatic aggregation of consecutive frames into timestamped SpeakerTurn segments.
      - Diarized transcript formatting for downstream LLM context ingestion.
    """

    FULL_SCALE_RMS_SINE = 23170.47  # 32767 / sqrt(2)

    def __init__(
        self,
        sample_rate: int = 8000,
        vad_threshold_dbov: float = -45.0,
        cross_corr_threshold: float = 0.58,
        bleed_margin_db: float = 4.0,
        hangover_frames: int = 3,
        max_delay_ms: float = 20.0,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * 0.020)  # 160 samples at 8kHz, 320 at 16kHz
        self.vad_threshold_dbov = float(vad_threshold_dbov)
        self.cross_corr_threshold = float(cross_corr_threshold)
        self.bleed_margin_db = float(bleed_margin_db)
        self.hangover_frames = int(hangover_frames)
        self.max_delay_samples = int(sample_rate * (max_delay_ms / 1000.0))

        # Dynamic state registers
        self.frame_index = 0
        self.caller_hangover = 0
        self.agent_hangover = 0

        # Ring buffers for cross-correlation across frame boundaries (3 frames history)
        history_len = self.frame_size * 3
        self.caller_history = np.zeros(history_len, dtype=np.float32)
        self.agent_history = np.zeros(history_len, dtype=np.float32)

    def reset(self) -> None:
        """Resets all internal frame counters, state registers, and history buffers."""
        self.frame_index = 0
        self.caller_hangover = 0
        self.agent_hangover = 0
        self.caller_history.fill(0.0)
        self.agent_history.fill(0.0)

    @staticmethod
    def compute_rms_dbov(samples: np.ndarray) -> float:
        """Computes root-mean-square level in dBov relative to digital full scale."""
        if len(samples) == 0:
            return -70.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return float(20.0 * np.log10(max(rms, 1e-4) / DualChannelDiarizer.FULL_SCALE_RMS_SINE))

    def _calculate_normalized_cross_correlation(
        self,
        s0: np.ndarray,
        s1_window: np.ndarray,
    ) -> float:
        """
        Computes maximum normalized cross-correlation between s0 (current frame)
        and s1_window across the delay search range.
        Returns peak rho in [0.0, 1.0].
        """
        if len(s0) == 0 or len(s1_window) < len(s0):
            return 0.0

        e0 = float(np.sum(s0 ** 2))
        if e0 < 1e-3:
            return 0.0

        # Fast valid correlation
        corr = np.correlate(s1_window, s0, mode="valid")
        if len(corr) == 0:
            return 0.0

        # Compute running energy of s1 over windows of length len(s0)
        n0 = len(s0)
        s1_sq = s1_window ** 2
        s1_cumsum = np.pad(np.cumsum(s1_sq), (1, 0), mode="constant")
        s1_energies = s1_cumsum[n0:] - s1_cumsum[:-n0]

        denoms = np.sqrt(e0 * s1_energies + 1e-6)
        normalized_corrs = np.abs(corr) / np.maximum(denoms, 1e-4)

        return float(np.max(normalized_corrs))

    def process_frame(
        self,
        caller_pcm_20ms: bytes,
        agent_pcm_20ms: bytes,
    ) -> FrameDiarizationTelemetry:
        """
        Processes a single synchronized 20ms linear PCM audio frame from Caller (Ch0)
        and Agent (Ch1).
        
        Returns FrameDiarizationTelemetry with state, cross-correlation, bleed detection,
        and dominance metrics.
        """
        t0 = time.perf_counter()

        s0 = (
            np.frombuffer(caller_pcm_20ms, dtype=np.int16).astype(np.float32)
            if caller_pcm_20ms else np.zeros(self.frame_size, dtype=np.float32)
        )
        s1 = (
            np.frombuffer(agent_pcm_20ms, dtype=np.int16).astype(np.float32)
            if agent_pcm_20ms else np.zeros(self.frame_size, dtype=np.float32)
        )

        # Pad or truncate to standard frame size if necessary
        if len(s0) < self.frame_size:
            s0 = np.pad(s0, (0, self.frame_size - len(s0)), mode="constant")
        elif len(s0) > self.frame_size:
            s0 = s0[:self.frame_size]

        if len(s1) < self.frame_size:
            s1 = np.pad(s1, (0, self.frame_size - len(s1)), mode="constant")
        elif len(s1) > self.frame_size:
            s1 = s1[:self.frame_size]

        # Update history ring buffers (shift left, append current frame)
        self.caller_history = np.roll(self.caller_history, -self.frame_size)
        self.caller_history[-self.frame_size:] = s0

        self.agent_history = np.roll(self.agent_history, -self.frame_size)
        self.agent_history[-self.frame_size:] = s1

        # 1. Compute frame energy in dBov
        e0_dbov = self.compute_rms_dbov(s0)
        e1_dbov = self.compute_rms_dbov(s1)

        # 2. Raw speech activity detection with hangover debouncing
        raw_v0 = e0_dbov >= self.vad_threshold_dbov
        raw_v1 = e1_dbov >= self.vad_threshold_dbov

        if raw_v0:
            self.caller_hangover = self.hangover_frames
            v0 = True
        else:
            if self.caller_hangover > 0:
                self.caller_hangover -= 1
                v0 = True
            else:
                v0 = False

        if raw_v1:
            self.agent_hangover = self.hangover_frames
            v1 = True
        else:
            if self.agent_hangover > 0:
                self.agent_hangover -= 1
                v1 = True
            else:
                v1 = False

        # 3. Dominance Score D in [-1.0, +1.0]
        # Linear amplitude estimate: 10^(dBov / 20)
        lin0 = 10.0 ** (e0_dbov / 20.0)
        lin1 = 10.0 ** (e1_dbov / 20.0)
        lin_sum = lin0 + lin1
        if lin_sum > 1e-5:
            dominance = float((lin0 - lin1) / lin_sum)
        else:
            dominance = 0.0

        # 4. Cross-Talk & Bleed Detection via Normalized Cross-Correlation
        max_rho = 0.0
        bleed_detected = False

        # Check cross-correlation if both channels are active
        if v0 and v1:
            # We search agent history around the current caller frame
            # History buffer has 3 frames: [prev_2, prev_1, current]
            # Center of search is the end of the history buffer
            search_span = self.frame_size + (2 * self.max_delay_samples)
            if len(self.agent_history) >= search_span:
                agent_window = self.agent_history[-search_span:]
                max_rho = self._calculate_normalized_cross_correlation(s0, agent_window)

            # Cross-talk bleed condition:
            # High correlation (rho >= threshold) AND significant energy difference
            if max_rho >= self.cross_corr_threshold:
                if (e1_dbov - e0_dbov) >= self.bleed_margin_db:
                    # Agent speech bleeding into caller's microphone
                    bleed_detected = True
                elif (e0_dbov - e1_dbov) >= self.bleed_margin_db:
                    # Caller speech bleeding into agent's channel
                    bleed_detected = True

        # 5. Diarization State Classification
        if not v0 and not v1:
            state = DiarizationState.SILENCE
        elif v0 and not v1:
            state = DiarizationState.CALLER_ONLY
        elif v1 and not v0:
            state = DiarizationState.AGENT_ONLY
        else:
            # Both channels active
            if bleed_detected:
                state = DiarizationState.CROSS_TALK_BLEED
            else:
                state = DiarizationState.DOUBLE_TALK

        start_ms = (self.frame_index * self.frame_size / self.sample_rate) * 1000.0
        end_ms = ((self.frame_index + 1) * self.frame_size / self.sample_rate) * 1000.0
        self.frame_index += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return FrameDiarizationTelemetry(
            frame_index=self.frame_index - 1,
            start_ms=start_ms,
            end_ms=end_ms,
            state=state,
            caller_rms_dbov=e0_dbov,
            agent_rms_dbov=e1_dbov,
            cross_correlation=max_rho,
            cross_talk_bleed_detected=bleed_detected,
            dominance_score=dominance,
            processing_time_ms=elapsed_ms,
        )

    def _aggregate_frames_into_turns(
        self,
        telemetries: List[FrameDiarizationTelemetry],
    ) -> List[SpeakerTurn]:
        """
        Groups continuous frame-level classifications into semantic SpeakerTurn segments.
        Merges brief transitions and cross-talk bleeds into dominant speaker turns.
        """
        if not telemetries:
            return []

        turns: List[SpeakerTurn] = []
        current_frames: List[FrameDiarizationTelemetry] = []

        def resolve_turn_speaker(st: DiarizationState, bleed: bool, dom: float) -> str:
            if st == DiarizationState.SILENCE:
                return "SILENCE"
            elif st == DiarizationState.CALLER_ONLY:
                return "CALLER"
            elif st == DiarizationState.AGENT_ONLY:
                return "AGENT"
            elif st == DiarizationState.CROSS_TALK_BLEED:
                return "CALLER" if dom > 0 else "AGENT"
            else:  # DOUBLE_TALK
                return "DOUBLE_TALK"

        current_speaker = resolve_turn_speaker(
            telemetries[0].state,
            telemetries[0].cross_talk_bleed_detected,
            telemetries[0].dominance_score,
        )

        for tel in telemetries:
            spk = resolve_turn_speaker(tel.state, tel.cross_talk_bleed_detected, tel.dominance_score)
            if spk == current_speaker:
                current_frames.append(tel)
            else:
                # Close out current turn
                if current_frames:
                    turns.append(self._build_turn(len(turns), current_speaker, current_frames))
                current_frames = [tel]
                current_speaker = spk

        if current_frames:
            turns.append(self._build_turn(len(turns), current_speaker, current_frames))

        return turns

    @staticmethod
    def _build_turn(
        turn_idx: int,
        speaker: str,
        frames: List[FrameDiarizationTelemetry],
    ) -> SpeakerTurn:
        start_ms = frames[0].start_ms
        end_ms = frames[-1].end_ms
        dur_ms = end_ms - start_ms

        dom_scores = [f.dominance_score for f in frames]
        avg_dom = float(np.mean(dom_scores)) if dom_scores else 0.0

        if avg_dom > 0.25:
            dom_speaker = "CALLER"
        elif avg_dom < -0.25:
            dom_speaker = "AGENT"
        elif speaker == "SILENCE":
            dom_speaker = "NONE"
        else:
            dom_speaker = "BALANCED"

        bleed_count = sum(1 for f in frames if f.cross_talk_bleed_detected)
        bleed_ratio = float(bleed_count / len(frames)) if frames else 0.0

        caller_levels = [f.caller_rms_dbov for f in frames]
        agent_levels = [f.agent_rms_dbov for f in frames]
        avg_caller = float(np.mean(caller_levels)) if caller_levels else -70.0
        avg_agent = float(np.mean(agent_levels)) if agent_levels else -70.0

        # Confidence metric based on energy contrast or cross-correlation
        if speaker == "CALLER":
            conf = min(1.0, max(0.50, 0.50 + (avg_dom * 0.50)))
        elif speaker == "AGENT":
            conf = min(1.0, max(0.50, 0.50 + (-avg_dom * 0.50)))
        elif speaker == "DOUBLE_TALK":
            conf = 0.90
        else:
            conf = 0.95

        return SpeakerTurn(
            turn_index=turn_idx,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=dur_ms,
            speaker=speaker,
            dominant_speaker=dom_speaker,
            average_dominance=avg_dom,
            cross_talk_bleed_frames=bleed_count,
            cross_talk_bleed_ratio=bleed_ratio,
            caller_rms_dbov=avg_caller,
            agent_rms_dbov=avg_agent,
            confidence=conf,
        )

    def process_dual_streams(
        self,
        caller_pcm: bytes,
        agent_pcm: bytes,
    ) -> Tuple[List[FrameDiarizationTelemetry], List[SpeakerTurn]]:
        """
        Diarizes full arbitrary-length dual-channel linear PCM streams.
        Returns frame-by-frame telemetries and aggregated SpeakerTurn segments.
        """
        self.reset()
        frame_bytes = self.frame_size * 2
        max_len = max(len(caller_pcm), len(agent_pcm))

        telemetries: List[FrameDiarizationTelemetry] = []

        for offset in range(0, max_len, frame_bytes):
            c_chunk = caller_pcm[offset : offset + frame_bytes]
            a_chunk = agent_pcm[offset : offset + frame_bytes]

            if len(c_chunk) < frame_bytes:
                c_chunk = c_chunk + b"\x00" * (frame_bytes - len(c_chunk))
            if len(a_chunk) < frame_bytes:
                a_chunk = a_chunk + b"\x00" * (frame_bytes - len(a_chunk))

            tel = self.process_frame(c_chunk, a_chunk)
            telemetries.append(tel)

        turns = self._aggregate_frames_into_turns(telemetries)
        return telemetries, turns

    def process_stereo_pcm(
        self,
        stereo_pcm: bytes,
    ) -> Tuple[List[FrameDiarizationTelemetry], List[SpeakerTurn]]:
        """
        Diarizes standard 2-channel interleaved stereo 16-bit linear PCM audio
        (Channel 0 / Left = Caller, Channel 1 / Right = Agent).
        """
        if not stereo_pcm or len(stereo_pcm) < 4:
            return [], []

        samples = np.frombuffer(stereo_pcm, dtype=np.int16)
        # Even samples = Left (Caller), Odd samples = Right (Agent)
        caller_samples = samples[0::2]
        agent_samples = samples[1::2]

        return self.process_dual_streams(
            caller_samples.tobytes(),
            agent_samples.tobytes(),
        )

    def diarize_call_recorder(
        self,
        recorder: "DualChannelCallRecorder",  # type: ignore
    ) -> Tuple[List[FrameDiarizationTelemetry], List[SpeakerTurn]]:
        """
        Directly extracts and diarizes in-memory channels from a DualChannelCallRecorder.
        """
        caller_bytes = bytes(recorder._customer_samples)
        agent_bytes = bytes(recorder._agent_samples)
        return self.process_dual_streams(caller_bytes, agent_bytes)

    @staticmethod
    def format_diarized_transcript(
        turns: List[SpeakerTurn],
        stt_annotations: Optional[List[dict]] = None,
    ) -> str:
        """
        Formats speaker turns into clean human-readable dialogue transcripts for LLM context.
        Optionally aligns with STT transcription texts.
        """
        lines = []
        for turn in turns:
            if turn.speaker == "SILENCE" and turn.duration_ms < 500.0:
                continue

            start_sec = turn.start_ms / 1000.0
            end_sec = turn.end_ms / 1000.0
            time_str = f"[{int(start_sec // 60):02d}:{start_sec % 60:06.3f} - {int(end_sec // 60):02d}:{end_sec % 60:06.3f}]"

            if turn.speaker == "DOUBLE_TALK":
                spk_str = f"[DOUBLE_TALK (Dominant: {turn.dominant_speaker}, Dom: {turn.average_dominance:+.2f})]"
            elif turn.cross_talk_bleed_ratio > 0.40:
                spk_str = f"[{turn.speaker} (Cross-Talk Bleed: {turn.cross_talk_bleed_ratio * 100:.0f}%)]"
            else:
                spk_str = f"[{turn.speaker}]"

            # Match STT text if provided
            text = "..."
            if stt_annotations:
                matched_texts = [
                    a.get("text", "")
                    for a in stt_annotations
                    if not (a.get("end_ms", 0) < turn.start_ms or a.get("start_ms", 0) > turn.end_ms)
                ]
                if matched_texts:
                    text = " ".join(matched_texts)

            lines.append(f"{time_str} {spk_str}: {text}")

        return "\n".join(lines)
