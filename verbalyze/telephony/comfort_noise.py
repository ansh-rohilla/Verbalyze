"""
verbalyze/telephony/comfort_noise.py

Adaptive Comfort Noise Generator (CNG) Engine.
Compliant with ITU-T G.711 Appendix II and RFC 3389.
Models background noise environments (ceiling fan hum, street noise, cellular line hiss)
via Levinson-Durbin Linear Predictive Coding (LPC) and all-pole synthesis filtering.
Generates stationary pseudo-random comfort noise during Discontinuous Transmission (DTX)
speech pauses to eradicate unnatural "dead-line" silence without digital clicks.
Pure-math NumPy implementation with zero external C++ or heavy ML dependencies.
Zero-emoji compliant. DPDP Act 2023 & RBI data sovereignty compliant.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class SIDPacket:
    """
    RFC 3389 / ITU-T G.711 Appendix II Silence Insertion Descriptor (SID) Frame.
    Carries compact background noise level and spectral reflection coefficients.
    """
    noise_level_dbov: int  # 0 to 127 (-dBov)
    reflection_coefficients: List[float] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        """Serializes SID packet into standard RFC 3389 binary representation."""
        level_byte = max(0, min(127, int(self.noise_level_dbov)))
        payload = bytearray([level_byte])
        for k in self.reflection_coefficients:
            q = max(-128, min(127, int(round(k * 127.0))))
            payload.append(q & 0xFF)
        return bytes(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SIDPacket":
        """Deserializes RFC 3389 binary payload into SIDPacket."""
        if not data:
            return cls(noise_level_dbov=60, reflection_coefficients=[0.0, 0.0, 0.0, 0.0])
        level = int(data[0])
        refl = []
        for b in data[1:]:
            val = b if b < 128 else (b - 256)
            refl.append(float(val) / 127.0)
        return cls(noise_level_dbov=level, reflection_coefficients=refl)


@dataclass
class CNGTelemetry:
    """Telemetry data captured during a 20ms comfort noise generation cycle."""
    is_cng_active: bool
    noise_level_dbov: float
    lpc_order: int
    lpc_coefficients: List[float]
    reflection_coefficients: List[float]
    rms: float
    preset_name: str
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_cng_active": bool(self.is_cng_active),
            "noise_level_dbov": round(float(self.noise_level_dbov), 1),
            "lpc_order": int(self.lpc_order),
            "lpc_coefficients": [round(float(c), 4) for c in self.lpc_coefficients],
            "reflection_coefficients": [round(float(k), 4) for k in self.reflection_coefficients],
            "rms": round(float(self.rms), 2),
            "preset_name": self.preset_name,
            "processing_time_ms": round(float(self.processing_time_ms), 3),
        }


class ComfortNoiseGenerator:
    """
    Adaptive Comfort Noise Generator (ITU-T G.711 Appendix II & RFC 3389).

    Features:
    1. Real-time background noise autocorrelation and Levinson-Durbin LPC estimation.
    2. Direct Form II Transposed all-pole synthesis filtering for seamless click-free playout.
    3. RFC 3389 Silence Insertion Descriptor (SID) formatting and decoding.
    4. Smooth raised-cosine / linear cross-fading for speech-to-CNG and CNG-to-speech transitions.
    5. Presets for Indian telephony acoustic environments (ceiling fan rumble, traffic, line hiss).
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        "INDIAN_ROOM_CEILING_FAN": {
            "level_dbov": -50.0,
            "lpc": [-1.25, 0.55, -0.15, 0.05],
            "description": "Low-frequency ceiling fan hum and motor resonance (100-300 Hz)",
        },
        "URBAN_STREET_TRAFFIC": {
            "level_dbov": -45.0,
            "lpc": [-0.75, 0.30, -0.10, 0.02],
            "description": "Broadband pink-sloped street drone and distant vehicle hum",
        },
        "CELLULAR_LINE_HISS": {
            "level_dbov": -58.0,
            "lpc": [0.0, 0.0, 0.0, 0.0],
            "description": "Flat high-frequency cellular quantization noise and line static",
        },
        "CLEAN_OFFICE_QUIET": {
            "level_dbov": -65.0,
            "lpc": [-0.35, 0.10, 0.0, 0.0],
            "description": "Subtle ambient room presence with clean low noise floor",
        },
    }

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_duration_ms: float = 20.0,
        lpc_order: int = 4,
        adaptation_rate: float = 0.08,
        preset_name: str = "INDIAN_ROOM_CEILING_FAN",
        speech_rms_threshold: float = 450.0,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))
        self.lpc_order = lpc_order
        self.adaptation_rate = adaptation_rate
        self.preset_name = preset_name
        self.speech_rms_threshold = speech_rms_threshold

        # Direct Form II Transposed synthesis filter state
        self.filter_state = np.zeros(self.lpc_order, dtype=np.float32)

        # Active noise model parameters
        self.lpc_coeffs = np.zeros(self.lpc_order, dtype=np.float32)
        self.reflection_coeffs = np.zeros(self.lpc_order, dtype=np.float32)
        self.residual_sigma: float = 50.0
        self.noise_level_dbov: float = -50.0

        # State tracking
        self.consecutive_cng_frames: int = 0
        self.is_active: bool = False
        self.noise_frames_analyzed: int = 0

        self.set_preset(preset_name)

    def set_preset(self, preset_name: str):
        """Initializes noise model from an acoustic preset profile."""
        if preset_name not in self.PRESETS:
            preset_name = "INDIAN_ROOM_CEILING_FAN"
        self.preset_name = preset_name
        p = self.PRESETS[preset_name]

        self.noise_level_dbov = float(p["level_dbov"])
        self.lpc_coeffs = np.array(p["lpc"][:self.lpc_order], dtype=np.float32)
        if len(self.lpc_coeffs) < self.lpc_order:
            padded = np.zeros(self.lpc_order, dtype=np.float32)
            padded[:len(self.lpc_coeffs)] = self.lpc_coeffs
            self.lpc_coeffs = padded

        self.reflection_coeffs = self._lpc_to_reflection(self.lpc_coeffs)

        # Calculate residual sigma from level_dbov and reflection coefficients:
        # sigma_e^2 = sigma_y^2 * prod(1 - k_i^2)
        target_signal_sigma = float(32767.0 * (10.0 ** (self.noise_level_dbov / 20.0)))
        gain_factor = float(np.sqrt(np.clip(np.prod(1.0 - (self.reflection_coeffs ** 2)), 0.01, 1.0)))
        self.residual_sigma = max(1.0, target_signal_sigma * gain_factor)

    def reset_state(self):
        """Clears synthesis filter memory and consecutive frame counter."""
        self.filter_state.fill(0.0)
        self.consecutive_cng_frames = 0
        self.is_active = False

    def update_noise_model(self, pcm_frame: bytes) -> bool:
        """
        Analyzes a background noise frame using autocorrelation and Levinson-Durbin recursion.
        Updates smoothed LPC synthesis coefficients and residual excitation sigma.
        Returns True if model was updated, False if frame was rejected as speech.
        """
        samples = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32)
        if len(samples) < self.frame_samples:
            return False

        rms = float(np.sqrt(np.mean(samples ** 2)))
        # Guard: Ignore frames above speech threshold
        if rms >= self.speech_rms_threshold or rms < 1.0:
            return False

        # Compute autocorrelation sequence R[0 ... lpc_order]
        r = np.zeros(self.lpc_order + 1, dtype=np.float32)
        for k in range(self.lpc_order + 1):
            r[k] = float(np.sum(samples[k:] * samples[:len(samples) - k]))

        # Run Levinson-Durbin recursion
        new_lpc, new_refl, residual_energy = self._levinson_durbin(r, self.lpc_order)
        new_sigma = float(np.sqrt(max(1.0, residual_energy / len(samples))))
        new_dbov = float(10.0 * np.log10(max(1e-9, (rms ** 2) / (32767.0 ** 2))))

        # Apply exponential moving average smoothing
        alpha = self.adaptation_rate
        if self.noise_frames_analyzed == 0:
            self.lpc_coeffs = new_lpc
            self.reflection_coeffs = new_refl
            self.residual_sigma = new_sigma
            self.noise_level_dbov = new_dbov
        else:
            self.lpc_coeffs = (alpha * new_lpc) + ((1.0 - alpha) * self.lpc_coeffs)
            self.reflection_coeffs = (alpha * new_refl) + ((1.0 - alpha) * self.reflection_coeffs)
            self.residual_sigma = (alpha * new_sigma) + ((1.0 - alpha) * self.residual_sigma)
            self.noise_level_dbov = (alpha * new_dbov) + ((1.0 - alpha) * self.noise_level_dbov)

        self.noise_frames_analyzed += 1
        return True

    def generate_comfort_noise_frame(self) -> Tuple[bytes, CNGTelemetry]:
        """
        Synthesizes a 20ms comfort noise frame using all-pole LPC synthesis filtering
        in Direct Form II Transposed structure.
        """
        t0 = time.perf_counter()
        self.consecutive_cng_frames += 1
        self.is_active = True

        # Generate zero-mean white Gaussian excitation
        excitation = np.random.normal(0.0, self.residual_sigma, self.frame_samples).astype(np.float32)

        # Direct Form II Transposed All-Pole IIR Filtering:
        # H(z) = 1 / (1 + sum_{i=1}^M a_i * z^-i)
        output = np.zeros(self.frame_samples, dtype=np.float32)
        s = self.filter_state
        m = self.lpc_order
        a = self.lpc_coeffs

        for n in range(self.frame_samples):
            yn = excitation[n] + s[0]
            for j in range(m - 1):
                s[j] = (-a[j] * yn) + s[j + 1]
            s[m - 1] = -a[m - 1] * yn
            output[n] = yn

        self.filter_state = s

        # Soft limiter for peaks exceeding 30000
        max_abs = float(np.max(np.abs(output)))
        if max_abs > 30000.0:
            thresh = 30000.0
            headroom = 32767.0 - thresh
            over = np.abs(output) - thresh
            mask = over > 0
            output[mask] = np.sign(output[mask]) * (thresh + (headroom * np.tanh(over[mask] / headroom)))

        out_rms = float(np.sqrt(np.mean(output ** 2)))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = CNGTelemetry(
            is_cng_active=True,
            noise_level_dbov=self.noise_level_dbov,
            lpc_order=self.lpc_order,
            lpc_coefficients=list(self.lpc_coeffs),
            reflection_coefficients=list(self.reflection_coeffs),
            rms=out_rms,
            preset_name=self.preset_name,
            processing_time_ms=elapsed_ms,
        )

        out_pcm = np.clip(output, -32768, 32767).astype(np.int16).tobytes()
        return out_pcm, telemetry

    def cross_fade(self, speech_pcm: bytes, cng_pcm: bytes, overlap_samples: int = 32) -> bytes:
        """
        Smoothly blends speech and comfort noise frames across overlap_samples (default 32 = 4ms at 8kHz)
        to prevent abrupt energy cliffs at speech onset and offset boundaries.
        """
        s_arr = np.frombuffer(speech_pcm, dtype=np.int16).astype(np.float32)
        c_arr = np.frombuffer(cng_pcm, dtype=np.int16).astype(np.float32)

        n = min(len(s_arr), len(c_arr), overlap_samples)
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)

        # Cross fade: cng fades in, speech fades out
        blended = c_arr.copy()
        blended[:n] = ((1.0 - ramp) * s_arr[:n]) + (ramp * c_arr[:n])

        return np.clip(blended, -32768, 32767).astype(np.int16).tobytes()

    def to_sid_packet(self) -> SIDPacket:
        """Creates an RFC 3389 SID packet representing current background noise state."""
        # Convert -dBov (e.g. -52.0) to positive integer level 52
        level = int(round(abs(self.noise_level_dbov)))
        level = max(0, min(127, level))
        return SIDPacket(
            noise_level_dbov=level,
            reflection_coefficients=list(self.reflection_coeffs),
        )

    def apply_sid_packet(self, sid: SIDPacket):
        """Updates internal comfort noise model from a received RFC 3389 SID frame."""
        self.noise_level_dbov = -float(sid.noise_level_dbov)

        if sid.reflection_coefficients:
            refl = np.array(sid.reflection_coefficients[:self.lpc_order], dtype=np.float32)
            if len(refl) < self.lpc_order:
                padded = np.zeros(self.lpc_order, dtype=np.float32)
                padded[:len(refl)] = refl
                refl = padded
            self.reflection_coeffs = refl
            self.lpc_coeffs = self._reflection_to_lpc(refl)

        target_signal_sigma = float(32767.0 * (10.0 ** (self.noise_level_dbov / 20.0)))
        gain_factor = float(np.sqrt(np.clip(np.prod(1.0 - (self.reflection_coeffs ** 2)), 0.01, 1.0)))
        self.residual_sigma = max(1.0, target_signal_sigma * gain_factor)

    @staticmethod
    def _levinson_durbin(r: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Levinson-Durbin algorithm for Linear Predictive Coding (LPC).
        Solves Toeplitz system R a = -r to extract predictor coefficients a_i,
        reflection coefficients k_i, and prediction error energy E.
        """
        a = np.zeros(order, dtype=np.float32)
        k = np.zeros(order, dtype=np.float32)
        e = float(r[0])

        if e < 1e-4:
            return a, k, e

        for i in range(1, order + 1):
            idx = i - 1
            acc = float(r[i])
            for j in range(1, i):
                acc += float(a[j - 1] * r[i - j])

            ki = -acc / max(1e-6, e)
            # Bound reflection coefficient for filter stability
            ki = max(-0.995, min(0.995, ki))
            k[idx] = ki

            a_new = a.copy()
            a_new[idx] = ki
            for j in range(1, i):
                a_new[j - 1] = a[j - 1] + (ki * a[i - j - 1])
            a = a_new

            e = max(1e-4, e * (1.0 - (ki * ki)))

        return a, k, e

    @staticmethod
    def _lpc_to_reflection(a: np.ndarray) -> np.ndarray:
        """Converts LPC predictor coefficients to reflection coefficients via step-down recursion."""
        order = len(a)
        k = np.zeros(order, dtype=np.float32)
        curr_a = a.copy()

        for i in range(order, 0, -1):
            idx = i - 1
            ki = curr_a[idx]
            ki = max(-0.995, min(0.995, ki))
            k[idx] = ki

            denom = 1.0 - (ki * ki)
            if denom < 1e-6:
                break
            prev_a = np.zeros(idx, dtype=np.float32)
            for j in range(idx):
                prev_a[j] = (curr_a[j] - (ki * curr_a[idx - 1 - j])) / denom
            curr_a = prev_a

        return k

    @staticmethod
    def _reflection_to_lpc(k: np.ndarray) -> np.ndarray:
        """Converts reflection coefficients to LPC predictor coefficients via step-up recursion."""
        order = len(k)
        a = np.zeros(order, dtype=np.float32)

        for i in range(1, order + 1):
            idx = i - 1
            ki = k[idx]
            a_new = a.copy()
            a_new[idx] = ki
            for j in range(1, i):
                a_new[j - 1] = a[j - 1] + (ki * a[i - j - 1])
            a = a_new

        return a
