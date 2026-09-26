"""
Verbalyze Real-Time Cellular Line Impairment & Acoustic Quality Classifier.
Pure-Math Single-Ended ITU-T P.862 PESQ & POLQA-MOS Non-Intrusive Quality Estimator.

Author: Verbalyze Telephony & Voice AI Team
Sovereignty: Section 65B Indian Evidence Act / ITU-T P.862 & P.563 Telephony Standards
Constraint: STRICT ZERO EMOJIS.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class LineImpairmentType(str, Enum):
    """Categorization of physical cellular and telephony line impairments."""
    CLEAN = "CLEAN"
    CARRIER_CLIPPING = "CARRIER_CLIPPING"
    MAINS_50HZ_HUM = "MAINS_50HZ_HUM"
    RF_FADING_DROPOUT = "RF_FADING_DROPOUT"
    HIGH_NOISE_FLOOR = "HIGH_NOISE_FLOOR"
    PHASE_JITTER_METALLIC = "PHASE_JITTER_METALLIC"
    SEVERELY_DEGRADED = "SEVERELY_DEGRADED"


@dataclass
class AcousticQualityTelemetry:
    """Frame-level acoustic quality and physical impairment telemetry."""
    frame_index: int
    start_ms: float
    end_ms: float
    estimated_pesq_score: float
    estimated_mos: float
    snr_db: float
    clipping_ratio: float
    mains_50hz_hum_ratio_db: float
    dropout_severity: float
    spectral_centroid_hz: float
    primary_impairment: LineImpairmentType
    failover_recommended: bool
    failover_reason: Optional[str]
    processing_time_ms: float

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "start_ms": round(float(self.start_ms), 1),
            "end_ms": round(float(self.end_ms), 1),
            "estimated_pesq_score": round(float(self.estimated_pesq_score), 2),
            "estimated_mos": round(float(self.estimated_mos), 2),
            "snr_db": round(float(self.snr_db), 1),
            "clipping_ratio": round(float(self.clipping_ratio), 4),
            "mains_50hz_hum_ratio_db": round(float(self.mains_50hz_hum_ratio_db), 1),
            "dropout_severity": round(float(self.dropout_severity), 3),
            "spectral_centroid_hz": round(float(self.spectral_centroid_hz), 1),
            "primary_impairment": (
                self.primary_impairment.value
                if isinstance(self.primary_impairment, LineImpairmentType)
                else str(self.primary_impairment)
            ),
            "failover_recommended": bool(self.failover_recommended),
            "failover_reason": self.failover_reason,
            "processing_time_ms": round(float(self.processing_time_ms), 4),
        }


@dataclass
class AcousticQualityReport:
    """Aggregated stream/call acoustic quality and trunk health report."""
    total_frames: int
    duration_seconds: float
    average_mos: float
    min_mos: float
    p95_mos: float
    average_pesq: float
    dominant_impairment: LineImpairmentType
    impairment_breakdown: Dict[str, float]
    trunk_health_status: str
    failover_event_count: int

    def to_dict(self) -> dict:
        return {
            "total_frames": int(self.total_frames),
            "duration_seconds": round(float(self.duration_seconds), 3),
            "average_mos": round(float(self.average_mos), 2),
            "min_mos": round(float(self.min_mos), 2),
            "p95_mos": round(float(self.p95_mos), 2),
            "average_pesq": round(float(self.average_pesq), 2),
            "dominant_impairment": (
                self.dominant_impairment.value
                if isinstance(self.dominant_impairment, LineImpairmentType)
                else str(self.dominant_impairment)
            ),
            "impairment_breakdown": {
                k: round(float(v), 2) for k, v in self.impairment_breakdown.items()
            },
            "trunk_health_status": str(self.trunk_health_status),
            "failover_event_count": int(self.failover_event_count),
        }


class CellularLineQualityClassifier:
    """
    Pure-Math Real-Time Cellular Line Impairment & Speech Quality Classifier.
    Non-intrusive single-ended estimator for ITU-T P.862 PESQ and POLQA-MOS.
    
    Operates on 8kHz narrowband or 16kHz wideband linear PCM telephony frames (20ms).
    
    Detects:
      1. Carrier hard/soft clipping and flat-topped sample saturation.
      2. 50Hz / 100Hz Indian mains ground hum from cheap chargers and inverters.
      3. RF multipath fading and sudden packet burst dropouts.
      4. Low SNR and elevated background noise floors.
      5. High-frequency phase jitter and comb-filtering transcoding artifacts.
    """

    FULL_SCALE_RMS_SINE = 23170.47

    def __init__(
        self,
        sample_rate: int = 8000,
        failover_mos_threshold: float = 2.80,
        failover_consecutive_frames: int = 4,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * 0.020)  # 160 at 8k, 320 at 16k
        self.failover_mos_threshold = float(failover_mos_threshold)
        self.failover_consecutive_frames = int(failover_consecutive_frames)

        # Precomputed Hann window for spectral analysis
        self.hann_window = np.hanning(self.frame_size).astype(np.float32)

        # Dynamic state registers
        self.frame_index = 0
        self.noise_floor_estimate = 30.0  # RMS
        self.prev_frame_rms = 0.0
        self.prev_frame_was_speech = False
        self.consecutive_failover_count = 0

    def reset(self) -> None:
        """Resets all internal frame counters, tracking registers, and state flags."""
        self.frame_index = 0
        self.noise_floor_estimate = 30.0
        self.prev_frame_rms = 0.0
        self.prev_frame_was_speech = False
        self.consecutive_failover_count = 0

    def compute_rms_dbov(self, samples: np.ndarray) -> float:
        """Computes signal level in dBov relative to digital full-scale sine RMS."""
        if len(samples) == 0:
            return -70.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return float(20.0 * np.log10(max(rms, 1e-4) / self.FULL_SCALE_RMS_SINE))

    def process_frame(self, pcm_bytes: bytes) -> AcousticQualityTelemetry:
        """
        Evaluates a single 20ms audio frame, extracts physical impairment metrics,
        estimates ITU-T P.862 PESQ and POLQA-MOS, and evaluates LCR failover recommendations.
        """
        t0 = time.perf_counter()

        if not pcm_bytes or len(pcm_bytes) < 4:
            return AcousticQualityTelemetry(
                frame_index=self.frame_index,
                start_ms=(self.frame_index * 20.0),
                end_ms=((self.frame_index + 1) * 20.0),
                estimated_pesq_score=1.0,
                estimated_mos=1.0,
                snr_db=0.0,
                clipping_ratio=0.0,
                mains_50hz_hum_ratio_db=-60.0,
                dropout_severity=0.0,
                spectral_centroid_hz=0.0,
                primary_impairment=LineImpairmentType.CLEAN,
                failover_recommended=False,
                failover_reason=None,
                processing_time_ms=0.0,
            )

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) < self.frame_size:
            samples = np.pad(samples, (0, self.frame_size - len(samples)), mode="constant")
        elif len(samples) > self.frame_size:
            samples = samples[:self.frame_size]

        rms = float(np.sqrt(np.mean(samples ** 2)))
        level_dbov = self.compute_rms_dbov(samples)

        # 1. Carrier Clipping & Saturation Analysis
        abs_samples = np.abs(samples)
        near_full_scale = abs_samples >= 31500.0
        # Flat-top detection (consecutive samples with near-zero derivative at high amplitude)
        diffs = np.abs(np.diff(samples))
        flat_topped = (diffs <= 2.0) & (abs_samples[:-1] >= 28000.0)
        clipped_count = int(np.sum(near_full_scale) + np.sum(flat_topped))
        clipping_ratio = float(min(1.0, clipped_count / self.frame_size))

        # 2. Spectral Analysis via FFT (Hann Windowed)
        windowed = samples * self.hann_window
        fft_vals = np.abs(np.fft.rfft(windowed))
        power_spec = fft_vals ** 2
        total_power = float(np.sum(power_spec))

        # Frequencies corresponding to rfft bins
        freq_bins = np.fft.rfftfreq(self.frame_size, d=1.0 / self.sample_rate)

        # 50Hz and 100Hz Indian Mains Hum Detection
        # Find bins nearest to 50Hz and 100Hz (+-15Hz band)
        hum_mask_50 = (freq_bins >= 40.0) & (freq_bins <= 60.0)
        hum_mask_100 = (freq_bins >= 90.0) & (freq_bins <= 110.0)
        hum_mask = hum_mask_50 | hum_mask_100
        num_hum_bins = int(np.sum(hum_mask))
        hum_power = float(np.sum(power_spec[hum_mask]))

        if total_power > 1e-4:
            hum_ratio = hum_power / total_power
            hum_ratio_db = float(10.0 * np.log10(max(hum_ratio, 1e-6)))
            mean_hum_bin_power = hum_power / max(num_hum_bins, 1)
            mean_total_bin_power = total_power / len(power_spec)
            hum_prominence = mean_hum_bin_power / max(mean_total_bin_power, 1e-6)
        else:
            hum_ratio_db = -60.0
            hum_prominence = 0.0

        # Spectral Centroid Calculation
        if total_power > 1e-4:
            centroid_hz = float(np.sum(freq_bins * fft_vals) / np.maximum(np.sum(fft_vals), 1e-4))
        else:
            centroid_hz = 0.0

        # 3. Stationary Noise Floor Estimation via 20th Percentile Spectral Floor
        pct20 = float(np.percentile(power_spec, 20))
        denom = 0.22314 * (3.0 / 8.0) * self.frame_size
        instant_noise_sigma = float(np.sqrt(pct20 / max(denom, 1e-6)))

        # Dynamic Noise Floor Tracking
        if instant_noise_sigma < self.noise_floor_estimate:
            self.noise_floor_estimate += 0.35 * (instant_noise_sigma - self.noise_floor_estimate)
        else:
            self.noise_floor_estimate += 0.15 * (instant_noise_sigma - self.noise_floor_estimate)
        self.noise_floor_estimate = max(10.0, self.noise_floor_estimate)

        snr_db = float(
            max(0.0, 20.0 * np.log10(max(rms, 1.0) / max(self.noise_floor_estimate, 1.0)))
        )

        is_speech = (level_dbov > -45.0) and (rms > self.noise_floor_estimate * 1.25)

        # 4. RF Multipath Fading & Sudden Dropout Detection
        dropout_severity = 0.0
        if self.prev_frame_was_speech and not is_speech and self.prev_frame_rms > 400.0:
            energy_drop_db = 20.0 * np.log10(self.prev_frame_rms / max(rms, 1.0))
            if energy_drop_db > 16.0:
                # Sudden cliff drop in active speech frame
                dropout_severity = float(min(1.0, (energy_drop_db - 16.0) / 14.0))

        self.prev_frame_rms = rms
        self.prev_frame_was_speech = is_speech

        # 5. Perceptual Quality Model (ITU-T P.862 PESQ & POLQA-MOS Proxy)
        # Base pristine MOS: 4.50
        penalty_noise = 0.0
        if is_speech and snr_db < 24.0:
            penalty_noise = float(np.clip((24.0 - snr_db) / 24.0 * 1.5, 0.0, 1.5))
        elif self.noise_floor_estimate > 250.0:
            penalty_noise = float(np.clip((self.noise_floor_estimate - 250.0) / 500.0 * 1.5, 0.0, 1.5))

        penalty_clip = 0.0
        if clipping_ratio > 0.008:
            penalty_clip = float(np.clip(clipping_ratio * 30.0, 0.0, 2.2))

        penalty_hum = 0.0
        if hum_ratio_db > -14.0 and hum_prominence > 2.5 and rms > 60.0:
            penalty_hum = float(np.clip((hum_ratio_db + 14.0) / 8.0 * 1.4, 0.0, 1.6))

        penalty_dropout = float(np.clip(dropout_severity * 2.0, 0.0, 2.2))

        penalty_centroid = 0.0
        if is_speech:
            if centroid_hz < 450.0 and centroid_hz > 0 and penalty_hum == 0.0:
                penalty_centroid = 0.50  # Heavy lowpass muffle
            elif centroid_hz > 2800.0:
                penalty_centroid = 0.60  # Comb-filtered / phase jitter metallic

        total_penalty = penalty_noise + penalty_clip + penalty_hum + penalty_dropout + penalty_centroid
        estimated_mos = float(np.clip(4.50 - total_penalty, 1.0, 4.5))

        # Standard linear mapping from MOS to PESQ raw score (-0.5 to 4.5)
        estimated_pesq = float(np.clip(-0.5 + 5.0 * ((estimated_mos - 1.0) / 3.5), -0.5, 4.5))

        # 6. Primary Impairment Classification
        penalties = {
            LineImpairmentType.HIGH_NOISE_FLOOR: penalty_noise,
            LineImpairmentType.CARRIER_CLIPPING: penalty_clip,
            LineImpairmentType.MAINS_50HZ_HUM: penalty_hum,
            LineImpairmentType.RF_FADING_DROPOUT: penalty_dropout,
            LineImpairmentType.PHASE_JITTER_METALLIC: penalty_centroid,
        }

        significant_penalties = sum(1 for p in penalties.values() if p >= 0.50)
        if estimated_mos >= 3.80:
            primary_impairment = LineImpairmentType.CLEAN
        elif significant_penalties >= 2 and estimated_mos < 2.60:
            primary_impairment = LineImpairmentType.SEVERELY_DEGRADED
        else:
            primary_impairment = max(penalties.items(), key=lambda item: item[1])[0]

        # 7. LCR Auto-Failover Recommendation Trigger
        if estimated_mos < self.failover_mos_threshold:
            self.consecutive_failover_count += 1
        else:
            self.consecutive_failover_count = max(0, self.consecutive_failover_count - 1)

        failover_recommended = self.consecutive_failover_count >= self.failover_consecutive_frames
        failover_reason = primary_impairment.value if failover_recommended else None

        start_ms = (self.frame_index * 20.0)
        end_ms = ((self.frame_index + 1) * 20.0)
        self.frame_index += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return AcousticQualityTelemetry(
            frame_index=self.frame_index - 1,
            start_ms=start_ms,
            end_ms=end_ms,
            estimated_pesq_score=estimated_pesq,
            estimated_mos=estimated_mos,
            snr_db=snr_db,
            clipping_ratio=clipping_ratio,
            mains_50hz_hum_ratio_db=hum_ratio_db,
            dropout_severity=dropout_severity,
            spectral_centroid_hz=centroid_hz,
            primary_impairment=primary_impairment,
            failover_recommended=failover_recommended,
            failover_reason=failover_reason,
            processing_time_ms=elapsed_ms,
        )

    def analyze_stream(
        self,
        pcm_bytes: bytes,
        reset_state: bool = True,
    ) -> Tuple[List[AcousticQualityTelemetry], AcousticQualityReport]:
        """
        Analyzes an entire audio stream frame-by-frame, tracking quality evolution
        and emitting an overarching AcousticQualityReport.
        """
        if reset_state:
            self.reset()
        frame_bytes = self.frame_size * 2
        total_len = len(pcm_bytes)

        telemetries: List[AcousticQualityTelemetry] = []
        failover_events = 0
        in_failover_state = False

        for offset in range(0, total_len, frame_bytes):
            chunk = pcm_bytes[offset : offset + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))

            tel = self.process_frame(chunk)
            telemetries.append(tel)

            if tel.failover_recommended and not in_failover_state:
                failover_events += 1
                in_failover_state = True
            elif not tel.failover_recommended:
                in_failover_state = False

        total_frames = len(telemetries)
        if total_frames == 0:
            report = AcousticQualityReport(
                total_frames=0,
                duration_seconds=0.0,
                average_mos=4.5,
                min_mos=4.5,
                p95_mos=4.5,
                average_pesq=4.5,
                dominant_impairment=LineImpairmentType.CLEAN,
                impairment_breakdown={},
                trunk_health_status="HEALTHY",
                failover_event_count=0,
            )
            return [], report

        mos_vals = [t.estimated_mos for t in telemetries]
        pesq_vals = [t.estimated_pesq_score for t in telemetries]

        avg_mos = float(np.mean(mos_vals))
        min_mos = float(np.min(mos_vals))
        p95_mos = float(np.percentile(mos_vals, 95))
        avg_pesq = float(np.mean(pesq_vals))

        # Impairment breakdown (% of frames)
        impairment_counts: Dict[str, int] = {}
        for t in telemetries:
            key = t.primary_impairment.value
            impairment_counts[key] = impairment_counts.get(key, 0) + 1

        breakdown = {
            k: (v / total_frames) * 100.0 for k, v in impairment_counts.items()
        }

        # Determine dominant non-clean impairment if present, otherwise CLEAN
        non_clean = {k: v for k, v in breakdown.items() if k != "CLEAN"}
        if non_clean:
            dominant_name = max(non_clean.items(), key=lambda x: x[1])[0]
            dominant_impairment = LineImpairmentType(dominant_name)
        else:
            dominant_impairment = LineImpairmentType.CLEAN

        # Trunk health classification
        if avg_mos >= 3.60 and failover_events == 0:
            trunk_health = "HEALTHY"
        elif avg_mos >= 2.80 and failover_events <= 1:
            trunk_health = "DEGRADED"
        else:
            trunk_health = "CRITICAL_FAILOVER"

        report = AcousticQualityReport(
            total_frames=total_frames,
            duration_seconds=round(total_frames * 0.020, 3),
            average_mos=avg_mos,
            min_mos=min_mos,
            p95_mos=p95_mos,
            average_pesq=avg_pesq,
            dominant_impairment=dominant_impairment,
            impairment_breakdown=breakdown,
            trunk_health_status=trunk_health,
            failover_event_count=failover_events,
        )

        return telemetries, report
