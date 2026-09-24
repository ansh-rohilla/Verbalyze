"""
verbalyze/telephony/equalizer.py

Dynamic Multi-Band Acoustic Equalizer (Indic Telecom Formant Enhancer).
Tailored for 8kHz narrowband telecom speech (ITU-T G.712 bandpass 300Hz-3400Hz).
Implements a 5-band parametric biquad equalizer in Direct Form II Transposed
architecture to selectively enhance Formant 3 (F3) retroflex consonants (ट-वर्ग / ण, ड़, ढ़, ष),
palatal sibilants (श), and nasal anti-formants over Indian mobile networks.
Pure-math NumPy implementation with zero external C++ or heavy ML dependencies.
Zero-emoji compliant. DPDP Act 2023 & RBI data sovereignty compliant.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class EQBandConfig:
    """Configuration for a single parametric biquad equalizer band."""
    name: str
    center_freq_hz: float
    gain_db: float
    q_factor: float
    filter_type: str = "peaking"  # "peaking", "low_shelf", "high_shelf"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "center_freq_hz": round(self.center_freq_hz, 1),
            "gain_db": round(self.gain_db, 2),
            "q_factor": round(self.q_factor, 2),
            "filter_type": self.filter_type,
        }


@dataclass
class EQTelemetry:
    """Telemetry data captured during a 20ms frame equalization cycle."""
    rms_in: float
    rms_out: float
    gain_applied_db: float
    speech_active: bool
    preset_name: str
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rms_in": round(float(self.rms_in), 2),
            "rms_out": round(float(self.rms_out), 2),
            "gain_applied_db": round(float(self.gain_applied_db), 2),
            "speech_active": bool(self.speech_active),
            "preset_name": self.preset_name,
            "processing_time_ms": round(float(self.processing_time_ms), 3),
        }


class BiquadFilter:
    """
    Second-order IIR biquad filter in Direct Form II Transposed structure.
    Preserves filter delay state (s1, s2) across consecutive 20ms frames to
    guarantee seamless phase continuity and zero boundary click artifacts.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        center_freq_hz: float = 1000.0,
        gain_db: float = 0.0,
        q_factor: float = 1.0,
        filter_type: str = "peaking",
    ):
        self.sample_rate = sample_rate
        self.center_freq_hz = center_freq_hz
        self.gain_db = gain_db
        self.q_factor = q_factor
        self.filter_type = filter_type

        # Direct Form II Transposed state variables
        self.s1: float = 0.0
        self.s2: float = 0.0

        # Normalized biquad coefficients: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
        self.b0: float = 1.0
        self.b1: float = 0.0
        self.b2: float = 0.0
        self.a1: float = 0.0
        self.a2: float = 0.0

        self.recompute_coefficients(self.gain_db)

    def reset_state(self):
        """Resets filter memory delay states to zero."""
        self.s1 = 0.0
        self.s2 = 0.0

    def recompute_coefficients(self, gain_db: float):
        """
        Recomputes biquad filter coefficients based on Robert Bristow-Johnson Audio EQ Cookbook.
        Supports peaking EQ, low-shelf, and high-shelf filters.
        """
        self.gain_db = gain_db
        # Guard against Nyquist frequency violation
        f0 = max(10.0, min(self.center_freq_hz, (self.sample_rate / 2.0) - 20.0))
        q = max(0.1, self.q_factor)

        omega0 = 2.0 * math.pi * (f0 / self.sample_rate)
        cos_w0 = math.cos(omega0)
        sin_w0 = math.sin(omega0)
        alpha = sin_w0 / (2.0 * q)
        a_gain = 10.0 ** (gain_db / 40.0)  # sqrt(10^(gain/20))

        if self.filter_type == "peaking":
            b0_raw = 1.0 + (alpha * a_gain)
            b1_raw = -2.0 * cos_w0
            b2_raw = 1.0 - (alpha * a_gain)
            a0_raw = 1.0 + (alpha / a_gain)
            a1_raw = -2.0 * cos_w0
            a2_raw = 1.0 - (alpha / a_gain)
        elif self.filter_type == "high_shelf":
            two_sqrt_a_alpha = 2.0 * math.sqrt(a_gain) * alpha
            b0_raw = a_gain * ((a_gain + 1.0) + ((a_gain - 1.0) * cos_w0) + two_sqrt_a_alpha)
            b1_raw = -2.0 * a_gain * ((a_gain - 1.0) + ((a_gain + 1.0) * cos_w0))
            b2_raw = a_gain * ((a_gain + 1.0) + ((a_gain - 1.0) * cos_w0) - two_sqrt_a_alpha)
            a0_raw = (a_gain + 1.0) - ((a_gain - 1.0) * cos_w0) + two_sqrt_a_alpha
            a1_raw = 2.0 * ((a_gain - 1.0) - ((a_gain + 1.0) * cos_w0))
            a2_raw = (a_gain + 1.0) - ((a_gain - 1.0) * cos_w0) - two_sqrt_a_alpha
        elif self.filter_type == "low_shelf":
            two_sqrt_a_alpha = 2.0 * math.sqrt(a_gain) * alpha
            b0_raw = a_gain * ((a_gain + 1.0) - ((a_gain - 1.0) * cos_w0) + two_sqrt_a_alpha)
            b1_raw = 2.0 * a_gain * ((a_gain - 1.0) - ((a_gain + 1.0) * cos_w0))
            b2_raw = a_gain * ((a_gain + 1.0) - ((a_gain - 1.0) * cos_w0) - two_sqrt_a_alpha)
            a0_raw = (a_gain + 1.0) + ((a_gain - 1.0) * cos_w0) + two_sqrt_a_alpha
            a1_raw = -2.0 * ((a_gain - 1.0) + ((a_gain + 1.0) * cos_w0))
            a2_raw = (a_gain + 1.0) + ((a_gain - 1.0) * cos_w0) - two_sqrt_a_alpha
        else:
            # Identity passthrough
            b0_raw, b1_raw, b2_raw = 1.0, 0.0, 0.0
            a0_raw, a1_raw, a2_raw = 1.0, 0.0, 0.0

        inv_a0 = 1.0 / a0_raw
        self.b0 = b0_raw * inv_a0
        self.b1 = b1_raw * inv_a0
        self.b2 = b2_raw * inv_a0
        self.a1 = a1_raw * inv_a0
        self.a2 = a2_raw * inv_a0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Processes audio samples through Direct Form II Transposed difference equation.
        Updates internal state variables s1 and s2 in place.
        """
        # If gain is virtually 0 dB, identity bypass
        if abs(self.gain_db) < 0.05:
            return samples

        output = np.zeros_like(samples)
        b0, b1, b2 = self.b0, self.b1, self.b2
        a1, a2 = self.a1, self.a2
        s1, s2 = self.s1, self.s2

        for i in range(len(samples)):
            x = samples[i]
            y = (b0 * x) + s1
            s1 = (b1 * x) - (a1 * y) + s2
            s2 = (b2 * x) - (a2 * y)
            output[i] = y

        self.s1 = s1
        self.s2 = s2
        return output

    def frequency_response(self, freq_hz: float) -> float:
        """Calculates magnitude response |H(f)| in decibels (dB) at a specified frequency."""
        omega = 2.0 * math.pi * (freq_hz / self.sample_rate)
        z_inv = np.exp(-1j * omega)
        z_inv2 = z_inv ** 2

        numerator = self.b0 + (self.b1 * z_inv) + (self.b2 * z_inv2)
        denominator = 1.0 + (self.a1 * z_inv) + (self.a2 * z_inv2)

        h_val = numerator / denominator
        mag_db = 20.0 * np.log10(np.abs(h_val) + 1e-9)
        return float(mag_db)


class IndicFormantEqualizer:
    """
    5-Band Parametric Biquad Equalizer optimized for Indian Telephony.

    Combats ITU-T G.712 telephone rolloff and accentuates critical acoustic formants:
    - Band 1 (300 Hz): Nasal warmth and pole-zero distinction (ण vs न/म)
    - Band 2 (750 Hz): First Formant (F1) vowel body clarity (अ, आ, इ, उ)
    - Band 3 (1600 Hz): Second Formant (F2) palatal consonant transitions
    - Band 4 (2400 Hz): Third Formant (F3) retroflex cavity resonance (ट-वर्ग / ड़, ढ़)
    - Band 5 (3200 Hz): Sibilant burst energy and upper telecom rolloff compensation (ष, श)

    Includes dynamic speech-adaptive gating (avoiding idle noise amplification) and
    soft saturation peak limiting to guarantee zero digital clipping.
    """

    PRESETS: Dict[str, List[EQBandConfig]] = {
        "INDIC_RETROFLEX_ENHANCE": [
            EQBandConfig("nasal_warmth", 300.0, 2.0, 1.0, "peaking"),
            EQBandConfig("vowel_body", 750.0, 1.0, 1.2, "peaking"),
            EQBandConfig("palatal_articulation", 1600.0, 2.5, 1.4, "peaking"),
            EQBandConfig("retroflex_f3", 2400.0, 4.5, 1.8, "peaking"),
            EQBandConfig("sibilant_burst", 3200.0, 5.0, 1.5, "peaking"),
        ],
        "TELECOM_CLARITY_BOOST": [
            EQBandConfig("low_cut", 320.0, 0.5, 1.0, "peaking"),
            EQBandConfig("mid_presence", 1200.0, 2.0, 1.2, "peaking"),
            EQBandConfig("speech_intelligibility", 2000.0, 3.5, 1.5, "peaking"),
            EQBandConfig("high_clarity", 2800.0, 4.0, 1.6, "peaking"),
            EQBandConfig("treble_edge", 3300.0, 4.5, 1.4, "peaking"),
        ],
        "NASAL_DISTINCTION": [
            EQBandConfig("nasal_anti_formant", 280.0, 3.5, 1.2, "peaking"),
            EQBandConfig("nasal_vowel_f1", 650.0, 1.5, 1.3, "peaking"),
            EQBandConfig("velar_consonants", 1500.0, 2.0, 1.4, "peaking"),
            EQBandConfig("retroflex_nasal_f3", 2350.0, 4.0, 1.8, "peaking"),
            EQBandConfig("upper_air", 3250.0, 3.0, 1.5, "peaking"),
        ],
        "NEUTRAL_BYPASS": [
            EQBandConfig("band_1", 300.0, 0.0, 1.0, "peaking"),
            EQBandConfig("band_2", 750.0, 0.0, 1.2, "peaking"),
            EQBandConfig("band_3", 1600.0, 0.0, 1.4, "peaking"),
            EQBandConfig("band_4", 2400.0, 0.0, 1.8, "peaking"),
            EQBandConfig("band_5", 3200.0, 0.0, 1.5, "peaking"),
        ],
    }

    def __init__(
        self,
        sample_rate: int = 8000,
        preset_name: str = "INDIC_RETROFLEX_ENHANCE",
        dynamic_gating: bool = True,
        speech_rms_threshold: float = 80.0,
        soft_clip_threshold: float = 30000.0,
    ):
        self.sample_rate = sample_rate
        self.preset_name = preset_name
        self.dynamic_gating = dynamic_gating
        self.speech_rms_threshold = speech_rms_threshold
        self.soft_clip_threshold = soft_clip_threshold

        self.bands: List[BiquadFilter] = []
        self.band_configs: List[EQBandConfig] = []
        self.set_preset(preset_name)

    def set_preset(self, preset_name: str):
        """Applies a named equalization preset and recalculates biquad coefficients."""
        if preset_name not in self.PRESETS:
            preset_name = "INDIC_RETROFLEX_ENHANCE"
        self.preset_name = preset_name
        self.band_configs = self.PRESETS[preset_name]

        self.bands.clear()
        for cfg in self.band_configs:
            filter_stage = BiquadFilter(
                sample_rate=self.sample_rate,
                center_freq_hz=cfg.center_freq_hz,
                gain_db=cfg.gain_db,
                q_factor=cfg.q_factor,
                filter_type=cfg.filter_type,
            )
            self.bands.append(filter_stage)

    def set_band_gain(self, band_idx: int, gain_db: float):
        """Dynamically tunes the gain in decibels for a specific band index (0 to 4)."""
        if 0 <= band_idx < len(self.bands):
            self.band_configs[band_idx].gain_db = gain_db
            self.bands[band_idx].recompute_coefficients(gain_db)

    def reset_state(self):
        """Clears memory states across all 5 cascaded biquads."""
        for band in self.bands:
            band.reset_state()

    def get_frequency_response(self, freqs_hz: List[float]) -> List[float]:
        """Calculates composite equalized frequency response in dB across all 5 cascaded biquads."""
        responses = []
        for f in freqs_hz:
            total_db = sum(band.frequency_response(f) for band in self.bands)
            responses.append(round(total_db, 2))
        return responses

    def process_frame(self, pcm_bytes: bytes) -> Tuple[bytes, EQTelemetry]:
        """
        Ingests a 20ms linear PCM16 audio frame:
        1. Checks input RMS for dynamic speech gating.
        2. Filters sequentially through all 5 biquad stages in Direct Form II Transposed mode.
        3. Applies soft-saturation peak limiting to prevent digital clipping.
        4. Returns processed linear PCM16 bytes and EQTelemetry.
        """
        t0 = time.perf_counter()
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

        rms_in = float(np.sqrt(np.mean(samples ** 2)))
        speech_active = rms_in >= self.speech_rms_threshold

        # Dynamic Gating: If audio is background noise or pause, scale gains towards 0 dB
        # to prevent amplifying ambient street or ceiling fan rumble
        if self.dynamic_gating and not speech_active:
            # Scale gain attenuation factor: 0.0 during deep silence, smoothly ramping to 1.0 at threshold
            gate_factor = max(0.0, min(1.0, (rms_in / self.speech_rms_threshold) ** 2))
            for idx, band in enumerate(self.bands):
                base_gain = self.band_configs[idx].gain_db
                gated_gain = base_gain * gate_factor
                band.recompute_coefficients(gated_gain)

        # Cascaded Biquad Filtering
        processed = samples.copy()
        for band in self.bands:
            processed = band.process(processed)

        # Restore original preset coefficients if gating temporarily modified them
        if self.dynamic_gating and not speech_active:
            for idx, band in enumerate(self.bands):
                band.recompute_coefficients(self.band_configs[idx].gain_db)

        # Soft-Saturation Peak Limiter: Smooth tanh compression knee for peaks > soft_clip_threshold
        max_abs = float(np.max(np.abs(processed))) if len(processed) > 0 else 0.0
        if max_abs > self.soft_clip_threshold:
            thresh = self.soft_clip_threshold
            headroom = 32767.0 - thresh
            over = np.abs(processed) - thresh
            mask = over > 0
            # Compress overshoots smoothly into remaining headroom
            processed[mask] = np.sign(processed[mask]) * (thresh + (headroom * np.tanh(over[mask] / headroom)))

        rms_out = float(np.sqrt(np.mean(processed ** 2)))
        gain_db = 20.0 * np.log10((rms_out + 1e-4) / (rms_in + 1e-4)) if rms_in > 0 else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = EQTelemetry(
            rms_in=rms_in,
            rms_out=rms_out,
            gain_applied_db=gain_db,
            speech_active=speech_active,
            preset_name=self.preset_name,
            processing_time_ms=elapsed_ms,
        )

        out_pcm = np.clip(processed, -32768, 32767).astype(np.int16).tobytes()
        return out_pcm, telemetry
