"""
Verbalyze Dynamic Multi-Speaker Gain Normalizer & Automatic Level Control (ALC) Engine.
ITU-T G.169 & ITU-T P.56 Compliant Pure-Math DSP Implementation.

Author: Verbalyze Telephony & Voice AI Team
Sovereignty: Section 65B Indian Evidence Act / ITU-T G.169 Standard Telephony Leveling
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class ALCPreset(str, Enum):
    """Acoustic presets for Indian telephony automatic level control."""
    RURAL_WHISPER_BOOST = "RURAL_WHISPER_BOOST"
    LOUDSPEAKER_ANTI_CLIP = "LOUDSPEAKER_ANTI_CLIP"
    STUDIO_NATURAL = "STUDIO_NATURAL"
    BYPASS = "BYPASS"


@dataclass
class ALCTelemetry:
    """Real-time telemetry emitted by the Automatic Level Controller."""
    input_rms_dbov: float
    target_dbov: float
    gain_applied_db: float
    linear_gain: float
    is_speech_active: bool
    compression_active: bool
    expansion_active: bool
    limiting_active: bool
    processing_time_ms: float

    def to_dict(self) -> dict:
        return {
            "input_rms_dbov": round(float(self.input_rms_dbov), 2),
            "target_dbov": round(float(self.target_dbov), 2),
            "gain_applied_db": round(float(self.gain_applied_db), 2),
            "linear_gain": round(float(self.linear_gain), 4),
            "is_speech_active": bool(self.is_speech_active),
            "compression_active": bool(self.compression_active),
            "expansion_active": bool(self.expansion_active),
            "limiting_active": bool(self.limiting_active),
            "processing_time_ms": round(float(self.processing_time_ms), 4),
        }


class AutomaticLevelController:
    """
    ITU-T G.169 Compliant Pure-Math Automatic Level Controller (ALC).
    Operates on 8kHz narrowband or 16kHz wideband linear PCM frames (20ms).
    
    Provides:
      - Speech level estimation in dBov relative to digital full-scale (ITU-T P.56).
      - Dual-rate attack (<=10ms) and release (300-800ms) dynamics.
      - Speech activity detection and hangover timer to prevent noise pumping.
      - Downward noise gate expansion for ambient noise floors (< -46 dBov).
      - Sample-by-sample linear gain ramp across 20ms frames for zero clicks.
      - Soft-saturation limiter with tanh compression for zero 16-bit integer overflow.
    """

    # Reference digital full-scale RMS for 16-bit sine wave (32767 / sqrt(2))
    FULL_SCALE_RMS_SINE = 23170.47
    SOFT_LIMIT_THRESHOLD = 30000.0

    def __init__(
        self,
        sample_rate: int = 8000,
        target_dbov: Optional[float] = None,
        max_boost_db: Optional[float] = None,
        max_cut_db: Optional[float] = None,
        noise_gate_dbov: Optional[float] = None,
        silence_floor_dbov: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
        hangover_frames: Optional[int] = None,
        preset: ALCPreset = ALCPreset.STUDIO_NATURAL,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * 0.020)  # 160 samples at 8kHz, 320 at 16kHz
        self.preset = preset

        # Apply preset parameters
        self._apply_preset(preset)

        # Allow explicit parameter overrides
        if target_dbov is not None:
            self.target_dbov = float(target_dbov)
        if max_boost_db is not None:
            self.max_boost_db = float(max_boost_db)
        if max_cut_db is not None:
            self.max_cut_db = float(max_cut_db)
        if noise_gate_dbov is not None:
            self.noise_gate_dbov = float(noise_gate_dbov)
        if silence_floor_dbov is not None:
            self.silence_floor_dbov = float(silence_floor_dbov)
        if attack_ms is not None:
            self.attack_ms = float(attack_ms)
        if release_ms is not None:
            self.release_ms = float(release_ms)
        if hangover_frames is not None:
            self.hangover_frames = int(hangover_frames)

        # Dynamic state registers
        self.current_gain_db = 0.0
        self.prev_linear_gain = 1.0
        self.hangover_counter = 0

    def _apply_preset(self, preset: ALCPreset) -> None:
        """Configures controller parameters according to the selected telephony preset."""
        self.preset = preset
        if preset == ALCPreset.RURAL_WHISPER_BOOST:
            # High boost for weak/faint rural cellular signals
            self.target_dbov = -18.0
            self.max_boost_db = 16.0
            self.max_cut_db = -12.0
            self.noise_gate_dbov = -48.0
            self.silence_floor_dbov = -56.0
            self.attack_ms = 8.0
            self.release_ms = 350.0
            self.hangover_frames = 10
        elif preset == ALCPreset.LOUDSPEAKER_ANTI_CLIP:
            # Fast attenuation for shouted or speakerphone speech
            self.target_dbov = -22.0
            self.max_boost_db = 6.0
            self.max_cut_db = -20.0
            self.noise_gate_dbov = -44.0
            self.silence_floor_dbov = -54.0
            self.attack_ms = 4.0
            self.release_ms = 400.0
            self.hangover_frames = 6
        elif preset == ALCPreset.BYPASS:
            # Pass-through without leveling
            self.target_dbov = -20.0
            self.max_boost_db = 0.0
            self.max_cut_db = 0.0
            self.noise_gate_dbov = -60.0
            self.silence_floor_dbov = -70.0
            self.attack_ms = 20.0
            self.release_ms = 20.0
            self.hangover_frames = 0
        else:  # STUDIO_NATURAL
            # Balanced natural leveling
            self.target_dbov = -20.0
            self.max_boost_db = 12.0
            self.max_cut_db = -15.0
            self.noise_gate_dbov = -46.0
            self.silence_floor_dbov = -55.0
            self.attack_ms = 10.0
            self.release_ms = 500.0
            self.hangover_frames = 8

    def reset_state(self) -> None:
        """Resets dynamic filter and gain state registers."""
        self.current_gain_db = 0.0
        self.prev_linear_gain = 1.0
        self.hangover_counter = 0

    def compute_dbov(self, rms: float) -> float:
        """Computes signal level in dBov relative to digital full-scale sine RMS."""
        rms_val = max(1e-4, float(rms))
        return float(20.0 * np.log10(rms_val / self.FULL_SCALE_RMS_SINE))

    def process_frame(self, pcm_bytes: bytes) -> Tuple[bytes, ALCTelemetry]:
        """
        Processes a single 20ms linear PCM audio frame through the ALC engine.
        Applies level estimation, dual-rate ballistics, noise gate, gain interpolation,
        and soft-saturation limiting.
        """
        t0 = time.perf_counter()

        if not pcm_bytes or len(pcm_bytes) < 4:
            return pcm_bytes, ALCTelemetry(
                input_rms_dbov=-70.0,
                target_dbov=self.target_dbov,
                gain_applied_db=0.0,
                linear_gain=1.0,
                is_speech_active=False,
                compression_active=False,
                expansion_active=False,
                limiting_active=False,
                processing_time_ms=0.0,
            )

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n_samples = len(samples)

        # 1. Compute Input RMS and dBov
        rms = float(np.sqrt(np.mean(samples ** 2)))
        input_dbov = self.compute_dbov(rms)

        # 2. Speech Activity Detection and Hangover Logic
        if input_dbov >= self.noise_gate_dbov:
            is_speech_active = True
            self.hangover_counter = self.hangover_frames
        else:
            if self.hangover_counter > 0:
                self.hangover_counter -= 1
                is_speech_active = True
            else:
                is_speech_active = False

        # 3. Determine Target Gain (dB)
        compression_active = False
        expansion_active = False

        if self.preset == ALCPreset.BYPASS:
            target_gain_db = 0.0
        elif is_speech_active:
            # Active speech: compute error relative to target dBov
            error_db = self.target_dbov - input_dbov

            # Deadband (+-1.0 dB) to eliminate hunting / micro-oscillations
            if abs(error_db) <= 1.0:
                target_gain_db = self.current_gain_db
            else:
                target_gain_db = float(np.clip(error_db, self.max_cut_db, self.max_boost_db))

            if target_gain_db < 0.0:
                compression_active = True
        else:
            # Inactive speech: ambient background noise or silence
            if input_dbov <= self.silence_floor_dbov:
                # Dead silence: decay smoothly toward 0.0 dB
                target_gain_db = 0.0
            else:
                # Downward expansion region (-55 dBov to -46 dBov)
                # Attenuate noise floor mildly to prevent noise pumping
                expansion_db = (input_dbov - self.noise_gate_dbov) * 0.45
                target_gain_db = float(min(0.0, expansion_db))
                expansion_active = True

        # 4. Ballistics: Dual-Rate Attack / Release Smoothing
        # Frame duration is 20ms
        dt_sec = float(n_samples / self.sample_rate)

        if target_gain_db < self.current_gain_db:
            # Fast attack (gain reduction to suppress loud bursts)
            tau = max(0.001, self.attack_ms / 1000.0)
            alpha = float(1.0 - np.exp(-dt_sec / tau))
        else:
            # Slow release (gain increase to avoid volume pumping)
            tau = max(0.001, self.release_ms / 1000.0)
            alpha = float(1.0 - np.exp(-dt_sec / tau))

        self.current_gain_db += alpha * (target_gain_db - self.current_gain_db)
        new_linear_gain = float(10.0 ** (self.current_gain_db / 20.0))

        # 5. Sample-by-Sample Gain Interpolation (Zero Frame-Boundary Clicks)
        # Ramp linear gain from previous frame's end gain to new frame's target gain
        gain_ramp = np.linspace(
            self.prev_linear_gain,
            new_linear_gain,
            n_samples,
            endpoint=True,
            dtype=np.float32,
        )
        self.prev_linear_gain = new_linear_gain

        # Apply continuous interpolated gain
        scaled_samples = samples * gain_ramp

        # 6. Soft-Saturation Peak Limiter
        # For samples exceeding 30,000, smoothly compress into remaining headroom
        limiting_active = False
        abs_samples = np.abs(scaled_samples)
        over_mask = abs_samples > self.SOFT_LIMIT_THRESHOLD

        if np.any(over_mask):
            limiting_active = True
            headroom = 32767.0 - self.SOFT_LIMIT_THRESHOLD
            overshoot = abs_samples[over_mask] - self.SOFT_LIMIT_THRESHOLD
            compressed = self.SOFT_LIMIT_THRESHOLD + headroom * np.tanh(overshoot / headroom)
            scaled_samples[over_mask] = np.sign(scaled_samples[over_mask]) * compressed

        # Strict 16-bit PCM integer clipping
        out_samples = np.clip(scaled_samples, -32768, 32767).astype(np.int16)
        out_pcm = out_samples.tobytes()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = ALCTelemetry(
            input_rms_dbov=input_dbov,
            target_dbov=self.target_dbov,
            gain_applied_db=self.current_gain_db,
            linear_gain=new_linear_gain,
            is_speech_active=is_speech_active,
            compression_active=compression_active,
            expansion_active=expansion_active,
            limiting_active=limiting_active,
            processing_time_ms=elapsed_ms,
        )

        return out_pcm, telemetry

    def process_stream(
        self,
        pcm_bytes: bytes,
    ) -> Tuple[bytes, List[ALCTelemetry]]:
        """
        Processes an arbitrary-length linear PCM stream in continuous 20ms frames,
        maintaining filter and ballistics states across the entire stream.
        """
        frame_bytes_len = self.frame_size * 2
        total_len = len(pcm_bytes)

        out_chunks = []
        telemetries = []

        for offset in range(0, total_len, frame_bytes_len):
            chunk = pcm_bytes[offset : offset + frame_bytes_len]
            if len(chunk) < frame_bytes_len:
                # Pad final partial frame with zeros if necessary
                chunk = chunk + b"\x00" * (frame_bytes_len - len(chunk))

            proc_chunk, tel = self.process_frame(chunk)
            out_chunks.append(proc_chunk)
            telemetries.append(tel)

        return b"".join(out_chunks), telemetries
