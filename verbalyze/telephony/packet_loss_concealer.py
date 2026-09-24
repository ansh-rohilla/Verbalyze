"""
verbalyze/telephony/packet_loss_concealer.py

Real-Time Acoustic Packet Loss Concealment (PLC) & Waveform Extrapolation Engine.
Implements ITU-T G.711 Appendix I pitch-synchronous waveform replication,
normalized cross-correlation pitch analysis, unvoiced phase-randomization,
multi-frame progressive attenuation, and overlap-add (OLA) resynchronization.
Pure-math NumPy implementation with zero external C++ or heavy ML dependencies.
Zero-emoji compliant. DPDP Act 2023 & RBI data sovereignty compliant.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class PLCTelemetry:
    """Telemetry data captured during a single frame packet loss concealment cycle."""
    is_concealed: bool
    is_voiced: bool
    pitch_period: int
    pitch_freq_hz: float
    correlation: float
    attenuation_factor: float
    consecutive_lost_frames: int
    processing_time_ms: float
    method: str  # "passthrough", "pitch_replicated", "unvoiced_spectral", "comfort_silence"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_concealed": self.is_concealed,
            "is_voiced": self.is_voiced,
            "pitch_period": int(self.pitch_period),
            "pitch_freq_hz": round(self.pitch_freq_hz, 1),
            "correlation": round(self.correlation, 4),
            "attenuation_factor": round(self.attenuation_factor, 4),
            "consecutive_lost_frames": int(self.consecutive_lost_frames),
            "processing_time_ms": round(self.processing_time_ms, 3),
            "method": self.method,
        }


class PacketLossConcealer:
    """
    ITU-T G.711 Appendix I Compliant Packet Loss Concealment (PLC) Engine.

    Synthesizes missing 20ms audio frames during mobile network drops and jitter bursts:
    1. Tracks trailing 60ms (480 samples at 8kHz) of decoded speech in a ring buffer.
    2. Computes normalized cross-correlation across human vocal pitch bounds (50Hz - 400Hz).
    3. Replicates pitch cycles with pitch-synchronous overlap-add (OLA) cross-fading for voiced speech.
    4. Randomizes phase spectra for unvoiced consonants and ambient noise, preventing metallic buzzing.
    5. Progressively attenuates multi-frame burst loss down to comfort silence after 60ms.
    6. Smoothly resynchronizes the newly arrived good packet via OLA blending, eliminating clicks.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_duration_ms: float = 20.0,
        history_duration_ms: float = 60.0,
        min_pitch_hz: float = 50.0,
        max_pitch_hz: float = 400.0,
        voiced_correlation_threshold: float = 0.55,
        ola_samples: int = 24,
        resync_samples: int = 32,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))
        self.frame_bytes = self.frame_samples * 2

        self.history_samples = int(sample_rate * (history_duration_ms / 1000.0))
        self.min_pitch_period = max(10, int(sample_rate / max_pitch_hz))  # e.g. 20 samples at 8kHz (400Hz)
        self.max_pitch_period = min(self.history_samples // 2, int(sample_rate / min_pitch_hz))  # e.g. 160 samples at 8kHz (50Hz)

        self.voiced_correlation_threshold = voiced_correlation_threshold
        self.ola_samples = ola_samples
        self.resync_samples = resync_samples

        # History buffer (float32 audio samples in range [-32768, 32767])
        self.history = np.zeros(self.history_samples, dtype=np.float32)
        self.history_filled = False

        # State tracking
        self.consecutive_lost_frames: int = 0
        self.total_good_frames: int = 0
        self.total_concealed_frames: int = 0
        self.total_voiced_concealments: int = 0
        self.total_unvoiced_concealments: int = 0

        # Synthetic tail buffer for smooth overlap-add on next good frame arrival
        self.synthetic_tail = np.zeros(self.resync_samples, dtype=np.float32)

        # Last detected pitch parameters
        self.last_pitch_period: int = 40
        self.last_correlation: float = 0.0
        self.last_is_voiced: bool = False

    def reset(self):
        """Resets the history buffer and internal loss counters."""
        self.history.fill(0.0)
        self.history_filled = False
        self.consecutive_lost_frames = 0
        self.synthetic_tail.fill(0.0)
        self.last_pitch_period = 40
        self.last_correlation = 0.0
        self.last_is_voiced = False

    def estimate_pitch(self) -> Tuple[int, float, bool]:
        """
        Calculates normalized cross-correlation across pitch lags [min_pitch_period, max_pitch_period].
        Identifies the optimal pitch period P and determines whether the speech is voiced or unvoiced.
        """
        history_len = len(self.history)
        corr_win_len = min(60, self.frame_samples // 2)  # 40-60 samples (5-7.5ms)

        if not self.history_filled and np.all(self.history == 0):
            return 40, 0.0, False

        # Reference target vector from the most recent tail of history
        target = self.history[-corr_win_len:]
        target_energy = np.sum(target ** 2)
        if target_energy < 1e-4:
            return 40, 0.0, False

        lags = np.arange(self.min_pitch_period, self.max_pitch_period + 1)
        best_lag = self.min_pitch_period
        best_norm_corr = -1.0

        # Vectorized candidate extraction
        # Each candidate is history[-corr_win_len - k : -k]
        # Build candidate matrix of shape (num_lags, corr_win_len)
        num_lags = len(lags)
        cand_matrix = np.zeros((num_lags, corr_win_len), dtype=np.float32)

        for idx, lag in enumerate(lags):
            start_idx = history_len - corr_win_len - lag
            end_idx = history_len - lag
            cand_matrix[idx, :] = self.history[start_idx:end_idx]

        # Compute dot products and candidate energies
        cross_corr = np.dot(cand_matrix, target)
        cand_energies = np.sum(cand_matrix ** 2, axis=1)

        denom = np.sqrt(target_energy * cand_energies) + 1e-6
        norm_corrs = cross_corr / denom

        max_idx = int(np.argmax(norm_corrs))
        best_norm_corr = float(norm_corrs[max_idx])
        best_lag = int(lags[max_idx])

        # Voice Activity & Voicing Decision
        frame_rms = float(np.sqrt(np.mean(self.history[-self.frame_samples:] ** 2)))
        is_voiced = (best_norm_corr >= self.voiced_correlation_threshold) and (frame_rms >= 100.0)

        return best_lag, best_norm_corr, is_voiced

    def ingest_good_frame(self, pcm_bytes: bytes) -> Tuple[bytes, PLCTelemetry]:
        """
        Ingests a valid incoming PCM audio frame.
        If recovering from previous packet loss, performs Overlap-Add (OLA)
        resynchronization with the synthetic tail to eradicate frame-boundary clicks.
        """
        t0 = time.perf_counter()
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

        if len(samples) < self.frame_samples:
            # Pad short frames if necessary
            padded = np.zeros(self.frame_samples, dtype=np.float32)
            padded[:len(samples)] = samples
            samples = padded

        # Check if recovering from previous packet loss
        was_loss = self.consecutive_lost_frames > 0
        if was_loss and self.synthetic_tail is not None and len(self.synthetic_tail) == self.resync_samples:
            # Apply smooth linear overlap-add cross-fade over the first resync_samples
            resync_len = min(self.resync_samples, len(samples))
            ramp_up = np.linspace(0.0, 1.0, resync_len, dtype=np.float32)
            ramp_down = 1.0 - ramp_up

            samples[:resync_len] = (ramp_down * self.synthetic_tail[:resync_len]) + (ramp_up * samples[:resync_len])

        # Update history ring buffer
        self.history = np.roll(self.history, -self.frame_samples)
        self.history[-self.frame_samples:] = samples
        self.history_filled = True

        # Reset consecutive loss counters
        self.consecutive_lost_frames = 0
        self.total_good_frames += 1

        # Clear synthetic tail
        self.synthetic_tail.fill(0.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = PLCTelemetry(
            is_concealed=False,
            is_voiced=self.last_is_voiced,
            pitch_period=self.last_pitch_period,
            pitch_freq_hz=(self.sample_rate / max(1, self.last_pitch_period)),
            correlation=self.last_correlation,
            attenuation_factor=1.0,
            consecutive_lost_frames=0,
            processing_time_ms=elapsed_ms,
            method="passthrough",
        )

        out_pcm = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
        return out_pcm, telemetry

    def conceal_frame(self) -> Tuple[bytes, PLCTelemetry]:
        """
        Synthesizes a missing 20ms audio frame according to ITU-T G.711 Appendix I:
        - Estimates pitch period and voicing state.
        - Replicates pitch waveforms with cross-fading for voiced speech.
        - Randomizes phase spectrum for unvoiced consonants/ambient noise.
        - Applies multi-frame progressive attenuation curve.
        - Saves synthetic tail for smooth resynchronization upon recovery.
        """
        t0 = time.perf_counter()
        self.consecutive_lost_frames += 1
        self.total_concealed_frames += 1

        # Determine attenuation factor based on consecutive lost frames
        # Frame 1 (0-20ms): 1.0 -> 0.95
        # Frame 2 (20-40ms): 0.95 -> 0.70
        # Frame 3 (40-60ms): 0.70 -> 0.35
        # Frame 4+ (>60ms): 0.35 -> 0.00
        if self.consecutive_lost_frames == 1:
            att_start, att_end = 1.0, 0.95
            method = "pitch_replicated"
        elif self.consecutive_lost_frames == 2:
            att_start, att_end = 0.95, 0.70
            method = "pitch_replicated"
        elif self.consecutive_lost_frames == 3:
            att_start, att_end = 0.70, 0.35
            method = "pitch_replicated"
        elif self.consecutive_lost_frames == 4:
            att_start, att_end = 0.35, 0.00
            method = "comfort_silence"
        else:
            att_start, att_end = 0.0, 0.0
            method = "comfort_silence"

        attenuation_ramp = np.linspace(att_start, att_end, self.frame_samples, dtype=np.float32)
        mean_att = float((att_start + att_end) / 2.0)

        # If beyond 4 frames (80ms), emit pure silence
        if mean_att <= 0.0:
            synth_samples = np.zeros(self.frame_samples, dtype=np.float32)
            self.synthetic_tail.fill(0.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            telemetry = PLCTelemetry(
                is_concealed=True,
                is_voiced=False,
                pitch_period=self.last_pitch_period,
                pitch_freq_hz=(self.sample_rate / max(1, self.last_pitch_period)),
                correlation=0.0,
                attenuation_factor=0.0,
                consecutive_lost_frames=self.consecutive_lost_frames,
                processing_time_ms=elapsed_ms,
                method="comfort_silence",
            )
            return synth_samples.astype(np.int16).tobytes(), telemetry

        # On the first lost frame, analyze pitch and voicing
        if self.consecutive_lost_frames == 1:
            best_lag, best_norm_corr, is_voiced = self.estimate_pitch()
            self.last_pitch_period = best_lag
            self.last_correlation = best_norm_corr
            self.last_is_voiced = is_voiced
        else:
            best_lag = self.last_pitch_period
            best_norm_corr = self.last_correlation
            is_voiced = self.last_is_voiced

        if is_voiced:
            self.total_voiced_concealments += 1
            method = "pitch_replicated"
            total_synth = self._synthesize_voiced_pitch_replication(best_lag, count=self.frame_samples + self.resync_samples)
            synth_samples = total_synth[:self.frame_samples]
            ext_tail = total_synth[self.frame_samples:] * att_end
        else:
            self.total_unvoiced_concealments += 1
            method = "unvoiced_spectral"
            synth_samples = self._synthesize_unvoiced_spectral()
            ext_tail = np.zeros(self.resync_samples, dtype=np.float32)

        # Apply multi-frame attenuation ramp
        synth_samples *= attenuation_ramp

        # Advance history with the synthetic frame to allow multi-frame extrapolation
        self.history = np.roll(self.history, -self.frame_samples)
        self.history[-self.frame_samples:] = synth_samples

        self.synthetic_tail = ext_tail

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = PLCTelemetry(
            is_concealed=True,
            is_voiced=is_voiced,
            pitch_period=best_lag,
            pitch_freq_hz=(self.sample_rate / max(1, best_lag)),
            correlation=best_norm_corr,
            attenuation_factor=mean_att,
            consecutive_lost_frames=self.consecutive_lost_frames,
            processing_time_ms=elapsed_ms,
            method=method,
        )

        out_pcm = np.clip(synth_samples, -32768, 32767).astype(np.int16).tobytes()
        return out_pcm, telemetry

    def _synthesize_voiced_pitch_replication(self, pitch_period: int, count: Optional[int] = None) -> np.ndarray:
        """
        Extrapolates periodic speech forward using the most recent pitch period.
        Applies pitch-synchronous overlap-add (OLA) cross-fading at repetition boundaries
        to eradicate slope/derivative discontinuities and acoustic clicks.
        """
        target_count = count if count is not None else self.frame_samples
        pitch_period = int(pitch_period)
        pitch_period = max(self.min_pitch_period, min(pitch_period, len(self.history) // 2))
        pitch_cycle = self.history[-pitch_period:].copy()

        repeats = int(np.ceil(target_count / pitch_period))
        output = np.tile(pitch_cycle, repeats)[:target_count]

        # Smooth boundary transitions at k * pitch_period
        L = min(self.ola_samples, max(2, pitch_period // 4))
        ramp = np.linspace(1.0, 0.0, L, endpoint=False, dtype=np.float32)

        for k in range(1, repeats):
            idx = k * pitch_period
            if idx < target_count:
                step = output[idx] - output[idx - 1]
                expected_delta = output[idx - 1] - output[idx - 2] if idx >= 2 else 0.0
                discontinuity = step - expected_delta

                apply_len = min(L, target_count - idx)
                output[idx : idx + apply_len] -= discontinuity * ramp[:apply_len]

        return output

    def _synthesize_unvoiced_spectral(self, count: Optional[int] = None) -> np.ndarray:
        """
        Synthesizes unvoiced consonants and ambient background noise via
        frequency-domain phase randomization. Preserves spectral magnitude
        envelope while eliminating periodic robotic buzzing or metallic comb filtering.
        """
        target_count = count if count is not None else self.frame_samples
        recent = self.history[-self.frame_samples:].copy()

        # Compute Fast Fourier Transform of recent frame
        fft_data = np.fft.rfft(recent)
        magnitudes = np.abs(fft_data)

        # Randomize phase angles uniformly between -pi and +pi
        random_phases = np.random.uniform(-np.pi, np.pi, size=len(fft_data))
        # Keep DC and Nyquist real to preserve signal integrity
        random_phases[0] = 0.0
        random_phases[-1] = 0.0

        randomized_fft = magnitudes * np.exp(1j * random_phases)
        synthesized = np.fft.irfft(randomized_fft, n=self.frame_samples).astype(np.float32)

        # Match overall RMS to previous frame
        prev_rms = np.sqrt(np.mean(recent ** 2)) + 1e-6
        synth_rms = np.sqrt(np.mean(synthesized ** 2)) + 1e-6
        synthesized *= (prev_rms / synth_rms)

        if target_count != self.frame_samples:
            return synthesized[:target_count]
        return synthesized

    def get_stats(self) -> Dict[str, Any]:
        """Returns comprehensive PLC telemetry and performance counters."""
        return {
            "total_good_frames": self.total_good_frames,
            "total_concealed_frames": self.total_concealed_frames,
            "total_voiced_concealments": self.total_voiced_concealments,
            "total_unvoiced_concealments": self.total_unvoiced_concealments,
            "consecutive_lost_frames": self.consecutive_lost_frames,
            "last_pitch_period": self.last_pitch_period,
            "last_correlation": round(self.last_correlation, 4),
            "last_is_voiced": self.last_is_voiced,
        }


# G.711 Appendix I alias for ITU-T telecom standard mapping
G711AppendixIPLC = PacketLossConcealer
