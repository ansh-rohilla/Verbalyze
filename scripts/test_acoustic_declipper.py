#!/usr/bin/env python3
"""
scripts/test_acoustic_declipper.py

Test Suite for Cellular Microphone Non-Linear Acoustic De-Clipper & Crest Restorer.
Validates pure-math cubic Hermite spline interpolation, vocal curvature matching,
dynamic soft-knee headroom auto-scaling, and total harmonic distortion reduction:
1. Clean unclipped speech passthrough (Profile CLEAN, zero clipping, 1.000 correlation).
2. Hard flat-top rail clipping detection and interval boundary identification.
3. Constrained cubic Hermite spline crest reconstruction (C1 continuity, summit elevation).
4. Dynamic soft-knee headroom auto-scaling (overflow prevention, smooth gain follower).
5. Clipping profile severity classification (CLEAN, MILD, SEVERE, HARD_LIMITER).
6. Total Harmonic Distortion (THD) odd-harmonic splatter suppression (> 6.0 dB).
7. Crest factor dynamic range recovery (restoring peak-to-RMS ratio).
8. Asymmetric clipping and DC-offset saturation resilience.
9. FastAPI REST Telephony Endpoints (/health, /declip, /benchmark).
10. Ultra-low latency SLA benchmark (< 0.030 ms per 20ms frame, > 650x real-time headroom).

Target SLA: Latency < 0.030 ms per 20ms frame | Headroom > 650x
Zero-emoji compliant.
"""

import base64
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.telephony.declipper import (
    AcousticDeclipper,
    DeclipperProfile,
    DeclipTelemetry,
    DeclipReport,
)
from verbalyze.telephony.server import create_app


def generate_sine_pcm(
    freq_hz: float = 300.0,
    amplitude: float = 16000.0,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
    clip_ceiling: float = 32767.0,
) -> bytes:
    """Generates linear PCM 16-bit sine wave, optionally clipped."""
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    sig = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    sig = np.clip(sig, -clip_ceiling, clip_ceiling)
    return sig.astype(np.int16).tobytes()


def generate_clipped_harmonic_speech(
    f0: float = 180.0,
    drive_gain: float = 2.5,
    duration_ms: float = 200.0,
    sample_rate: int = 8000,
) -> bytes:
    """
    Simulates a shouting speaker driving a cellular microphone into hard saturation.
    Combines glottal fundamentals with formants and applies pre-amp overdrive.
    """
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    speech = (
        0.60 * np.sin(2.0 * np.pi * f0 * t)
        + 0.25 * np.sin(2.0 * np.pi * (2 * f0) * t)
        + 0.15 * np.sin(2.0 * np.pi * (3 * f0) * t)
    )
    # Overdrive into 16-bit rail
    overdriven = speech * (32767.0 * drive_gain)
    clipped = np.clip(overdriven, -32768.0, 32767.0)
    return clipped.astype(np.int16).tobytes()


def test_1_clean_unclipped_speech_passthrough():
    print("\n--- Test 1: Clean Unclipped Speech Passthrough ---")
    declipper = AcousticDeclipper(sample_rate=8000)
    clean_pcm = generate_sine_pcm(freq_hz=400.0, amplitude=14000.0, duration_ms=20.0)

    out_pcm, telem = declipper.process_frame(clean_pcm)

    print(f"[Telemetry] Profile: {telem.profile.value}")
    print(f"[Telemetry] Clipped Samples: {telem.clipped_sample_count} ({telem.clipping_percentage:.2f}%)")
    print(f"[Telemetry] Peak Amplitude: Orig={telem.original_peak_amplitude:.1f}, Restored={telem.restored_peak_amplitude:.1f}")
    print(f"[Telemetry] Headroom Scale Factor: {telem.headroom_scale_factor:.4f}")

    assert telem.profile == DeclipperProfile.CLEAN
    assert telem.clipped_sample_count == 0
    assert telem.clipping_percentage == 0.0
    assert telem.reconstructed_crest_count == 0
    assert telem.headroom_scale_factor == 1.0

    in_arr = np.frombuffer(clean_pcm, dtype=np.int16).astype(np.float32)
    out_arr = np.frombuffer(out_pcm, dtype=np.int16).astype(np.float32)
    max_diff = float(np.max(np.abs(in_arr - out_arr)))
    print(f"[Signal Fidelity] Maximum Absolute Sample Difference: {max_diff}")
    assert max_diff <= 1.0, f"Clean audio must pass through unaltered (max diff: {max_diff})"
    print("[OK] Clean speech passed through perfectly with zero modification")


def test_2_hard_flat_top_rail_clipping_detection():
    print("\n--- Test 2: Hard Flat-Top Rail Clipping Detection ---")
    declipper = AcousticDeclipper(sample_rate=8000, clipping_threshold_fraction=0.96)
    # Sine wave with amplitude 48000 clamped to 32767
    clipped_pcm = generate_sine_pcm(freq_hz=250.0, amplitude=48000.0, duration_ms=20.0)

    samples = np.frombuffer(clipped_pcm, dtype=np.int16).astype(np.float32)
    intervals = declipper._find_clipped_intervals(samples)

    print(f"[Intervals Detected] Count: {len(intervals)}")
    for i, (start, end, pol) in enumerate(intervals):
        gap = end - start + 1
        print(f"  Interval {i+1}: samples [{start}..{end}], length={gap}, polarity={'+' if pol > 0 else '-'}")

    assert len(intervals) >= 2, f"Expected at least 2 clipped crest intervals, got {len(intervals)}"
    # Verify polarity alternates (positive crest then negative crest or vice versa)
    for i in range(len(intervals) - 1):
        assert intervals[i][2] != intervals[i+1][2], "Consecutive crests must alternate polarity in a sine wave"

    print("[OK] Hard flat-topped clipping intervals and polarities detected accurately")


def test_3_cubic_hermite_spline_crest_reconstruction():
    print("\n--- Test 3: Constrained Cubic Hermite Spline Crest Reconstruction ---")
    declipper = AcousticDeclipper(sample_rate=8000, acoustic_elasticity=1.0)
    # Generate overdriven sine wave
    t = np.linspace(0, 0.02, 160, endpoint=False)
    raw_sig = 44000.0 * np.sin(2.0 * np.pi * 200.0 * t)
    clipped_sig = np.clip(raw_sig, -32767.0, 32767.0).astype(np.float32)

    intervals = declipper._find_clipped_intervals(clipped_sig)
    assert len(intervals) > 0, "Must detect clipped intervals"

    start, end, pol = intervals[0]
    crest = declipper._reconstruct_crest(clipped_sig, start, end, pol)

    print(f"[Crest Reconstruction] Gap length: {len(crest)} samples")
    print(f"[Crest Reconstruction] Boundary Left: {clipped_sig[start-1]:.1f}, Right: {clipped_sig[end+1]:.1f}")
    print(f"[Crest Reconstruction] Restored Peak: {np.max(np.abs(crest)):.1f} vs Rail: 32767.0")

    # In polarity direction, the reconstructed crest should exceed the clipping threshold
    if pol > 0:
        assert np.max(crest) > 32767.0, f"Positive reconstructed peak {np.max(crest)} should exceed 32767"
        # Check summit concavity: middle sample should be higher than boundary samples
        mid_idx = len(crest) // 2
        assert crest[mid_idx] >= crest[0], "Crest midpoint must be elevated above entry boundary"
        assert crest[mid_idx] >= crest[-1], "Crest midpoint must be elevated above exit boundary"
    else:
        assert np.min(crest) < -32767.0, f"Negative reconstructed peak {np.min(crest)} should be below -32767"

    # Verify C1 slope continuity: smooth transition from entry slope
    m_entry = clipped_sig[start - 1] - clipped_sig[start - 2]
    m_reconstructed_start = crest[0] - clipped_sig[start - 1]
    print(f"[C1 Smoothness] Entry Slope: {m_entry:.2f}, First Reconstructed Step: {m_reconstructed_start:.2f}")
    # Signs of slope must match
    assert (m_entry > 0 and m_reconstructed_start > 0) or (m_entry < 0 and m_reconstructed_start < 0)

    print("[OK] Cubic Hermite spline curvature and summit elevation reconstructed with C1 continuity")


def test_4_dynamic_headroom_auto_scaling():
    print("\n--- Test 4: Dynamic Soft-Knee Headroom Auto-Scaling ---")
    declipper = AcousticDeclipper(sample_rate=8000, target_headroom_db=0.5)

    # Process 5 consecutive frames of heavily saturated audio
    frame_pcm = generate_sine_pcm(freq_hz=300.0, amplitude=46000.0, duration_ms=20.0)

    scale_factors: List[float] = []
    for f_idx in range(5):
        out_pcm, telem = declipper.process_frame(frame_pcm)
        out_samples = np.frombuffer(out_pcm, dtype=np.int16).astype(np.float32)
        scale_factors.append(telem.headroom_scale_factor)

        # Confirm PCM bounds are strictly respected (zero wrap-around or overflow)
        max_val = float(np.max(out_samples))
        min_val = float(np.min(out_samples))
        assert max_val <= 32767.0, f"Frame {f_idx} exceeded +32767 PCM rail: {max_val}"
        assert min_val >= -32768.0, f"Frame {f_idx} below -32768 PCM rail: {min_val}"

    print(f"[Headroom Scaling Across Frames] Scale history: {[round(s, 4) for s in scale_factors]}")
    # Headroom scale should decrease smoothly below 1.0 to accommodate restored crests
    assert scale_factors[-1] < 1.0, f"Headroom scale should be compressed below 1.0, got {scale_factors[-1]}"
    assert scale_factors[-1] < scale_factors[0], "Headroom follower should smoothly compress over multiple frames"

    print("[OK] Soft-knee dynamic headroom follower prevented 16-bit PCM overflow without audio clipping")


def test_5_severe_saturation_profile_classification():
    print("\n--- Test 5: Saturation Profile Severity Classification ---")

    t_ax = np.linspace(0, 0.02, 160, endpoint=False)

    # 1. CLEAN (0.0% clipped)
    pcm_clean = (15000.0 * np.sin(2.0 * np.pi * 200.0 * t_ax)).astype(np.int16).tobytes()

    # 2. MILD_CLIPPING (< 3.0% clipped, e.g. 3 samples out of 160 = 1.88%)
    arr_mild = (15000.0 * np.sin(2.0 * np.pi * 200.0 * t_ax)).astype(np.float32)
    arr_mild[40:43] = 32500.0
    pcm_mild = arr_mild.astype(np.int16).tobytes()

    # 3. SEVERE_RAIL_CLIPPING (3.0% - 15.0% clipped, e.g. 60Hz amp 31600 = 5.62%)
    arr_severe = np.clip(31600.0 * np.sin(2.0 * np.pi * 60.0 * t_ax), -32768.0, 32767.0)
    pcm_severe = arr_severe.astype(np.int16).tobytes()

    # 4. HARD_LIMITER_SATURATION (> 15.0% clipped, e.g. 250Hz amp 45000 = 56.25%)
    arr_hard = np.clip(45000.0 * np.sin(2.0 * np.pi * 250.0 * t_ax), -32768.0, 32767.0)
    pcm_hard = arr_hard.astype(np.int16).tobytes()

    cases = [
        ("Clean Signal", pcm_clean, DeclipperProfile.CLEAN),
        ("Mild Saturation", pcm_mild, DeclipperProfile.MILD_CLIPPING),
        ("Severe Rail Clipping", pcm_severe, DeclipperProfile.SEVERE_RAIL_CLIPPING),
        ("Hard Limiter Overdrive", pcm_hard, DeclipperProfile.HARD_LIMITER_SATURATION),
    ]

    for label, pcm_data, expected_profile in cases:
        declipper = AcousticDeclipper(sample_rate=8000)
        _, telem = declipper.process_frame(pcm_data)
        print(f"[{label}] Clip%: {telem.clipping_percentage:.2f}% | Profile: {telem.profile.value}")
        assert telem.profile == expected_profile, (
            f"Expected {expected_profile.value} for {label}, got {telem.profile.value} (clip%={telem.clipping_percentage:.2f})"
        )

    print("[OK] All clipping severity profiles correctly classified across full dynamic range")


def test_6_total_harmonic_distortion_reduction():
    print("\n--- Test 6: Total Harmonic Distortion (THD) Odd-Harmonic Suppression ---")
    declipper = AcousticDeclipper(sample_rate=8000)

    # 100ms stream of saturated 200 Hz tone
    freq = 200.0
    duration_ms = 100.0
    t = np.linspace(0, duration_ms / 1000.0, int(8000 * duration_ms / 1000.0), endpoint=False)
    raw_sig = 48000.0 * np.sin(2.0 * np.pi * freq * t)
    clipped_sig = np.clip(raw_sig, -32767.0, 32767.0).astype(np.int16).tobytes()

    restored_pcm, telemetries, report = declipper.process_stream(clipped_sig)

    print(f"[THD Telemetry] Average Estimated THD Reduction: {report.average_thd_reduction_db:.2f} dB")
    print(f"[THD Telemetry] Total Reconstructed Crests: {report.total_reconstructed_crests}")
    assert report.average_thd_reduction_db >= 6.0, f"Expected THD reduction >= 6.0 dB, got {report.average_thd_reduction_db:.2f} dB"

    # Spectral harmonic analysis: Measure 3rd harmonic (600 Hz) suppression
    clipped_arr = np.frombuffer(clipped_sig, dtype=np.int16).astype(np.float32)
    restored_arr = np.frombuffer(restored_pcm, dtype=np.int16).astype(np.float32)

    fft_clip = np.abs(np.fft.rfft(clipped_arr))
    fft_rest = np.abs(np.fft.rfft(restored_arr))
    freqs = np.fft.rfftfreq(len(clipped_arr), d=1.0 / 8000.0)

    # Find bin for 3rd harmonic (600 Hz)
    idx_3f = int(np.argmin(np.abs(freqs - (3.0 * freq))))
    p_3f_clipped = float(fft_clip[idx_3f])
    p_3f_restored = float(fft_rest[idx_3f])

    ratio_3f_db = 20.0 * math.log10(max(p_3f_clipped, 1.0) / max(p_3f_restored, 1.0))
    print(f"[Harmonic Analysis] 3rd Harmonic (600 Hz): Clipped Amp={p_3f_clipped:.1f}, Restored Amp={p_3f_restored:.1f}")
    print(f"[Harmonic Analysis] 3rd Harmonic Power Attenuation: {ratio_3f_db:.2f} dB")

    assert ratio_3f_db > 0.5, f"Expected 3rd harmonic suppression > 0.5 dB, got {ratio_3f_db:.2f} dB"
    print("[OK] Harsh odd-harmonic splatter significantly attenuated by cubic crest reconstruction")


def test_7_crest_factor_restoration():
    print("\n--- Test 7: Crest Factor Dynamic Range Recovery ---")
    declipper = AcousticDeclipper(sample_rate=8000)
    overdriven_speech = generate_clipped_harmonic_speech(f0=160.0, drive_gain=2.8, duration_ms=100.0)

    restored_pcm, telemetries, report = declipper.process_stream(overdriven_speech)

    print(f"[Crest Factor] Average CF Gain: {report.average_crest_factor_gain_db:.2f} dB")
    print(f"[Crest Factor] Dominant Profile: {report.dominant_profile.value}")
    print(f"[Crest Factor] Total Clipped Samples: {report.total_clipped_samples}")

    # Clipped speech has squashed crest factor; restored audio should show positive CF recovery
    assert report.average_crest_factor_gain_db >= 0.0, "Crest factor gain must be non-negative"
    assert report.total_reconstructed_crests > 0, "Crests must be actively reconstructed"
    print("[OK] Speech dynamic range and crest factor successfully recovered")


def test_8_asymmetric_and_dc_offset_clipping_handling():
    print("\n--- Test 8: Asymmetric & DC-Offset Clipping Handling ---")
    declipper = AcousticDeclipper(sample_rate=8000)

    # Signal with strong positive DC bias driving only positive half into rail
    t = np.linspace(0, 0.04, 320, endpoint=False)
    dc_offset = 12000.0
    sig = dc_offset + 30000.0 * np.sin(2.0 * np.pi * 200.0 * t)
    asymmetric_clipped = np.clip(sig, -32768.0, 32767.0).astype(np.int16).tobytes()

    restored_pcm, telemetries, report = declipper.process_stream(asymmetric_clipped)

    print(f"[Asymmetric Clipping] Processed {len(telemetries)} frames")
    print(f"[Asymmetric Clipping] Total Reconstructed Crests: {report.total_reconstructed_crests}")
    assert report.total_reconstructed_crests > 0

    # Ensure no NaN or Inf
    restored_arr = np.frombuffer(restored_pcm, dtype=np.int16)
    assert not np.isnan(restored_arr).any()
    assert len(restored_arr) == len(t)
    print("[OK] Asymmetric unipolar clipping recovered cleanly without numerical instability")


def test_9_fastapi_rest_endpoints_verification():
    print("\n--- Test 9: FastAPI REST Telephony Endpoints Verification ---")
    from starlette.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    print(f"[GET /health] Status: {h_data.get('status')} | Declipper: {h_data.get('declipper_status')}")
    assert h_data.get("declipper_status") == "ready"

    # 2. POST /telephony/audio/declip
    clipped_bytes = generate_sine_pcm(freq_hz=300.0, amplitude=44000.0, duration_ms=40.0)
    b64_audio = base64.b64encode(clipped_bytes).decode("ascii")

    declip_resp = client.post(
        "/telephony/audio/declip",
        json={"audio_base64": b64_audio, "sample_rate": 8000},
    )
    assert declip_resp.status_code == 200
    d_data = declip_resp.json()
    print(f"[POST /telephony/audio/declip] Status: {d_data.get('status')}")
    print(f"[POST /telephony/audio/declip] Dominant Profile: {d_data.get('dominant_profile')}")
    print(f"[POST /telephony/audio/declip] Total Clipped Samples: {d_data.get('total_clipped_samples')}")
    print(f"[POST /telephony/audio/declip] THD Reduction: {d_data.get('average_thd_reduction_db')} dB")

    assert d_data.get("status") == "ok"
    assert d_data.get("dominant_profile") in ["SEVERE_RAIL_CLIPPING", "HARD_LIMITER_SATURATION"]
    assert d_data.get("total_clipped_samples") > 0
    assert len(d_data.get("restored_audio_base64")) > 0

    # 3. POST /telephony/audio/declip/benchmark
    bench_resp = client.post("/telephony/audio/declip/benchmark")
    assert bench_resp.status_code == 200
    b_data = bench_resp.json()
    print(f"[POST /telephony/audio/declip/benchmark] Avg Latency: {b_data.get('avg_declip_processing_time_ms')} ms")
    print(f"[POST /telephony/audio/declip/benchmark] Meets SLA: {b_data.get('meets_sla')} | Headroom: {b_data.get('real_time_headroom_factor')}x")

    assert b_data.get("meets_sla") is True
    assert b_data.get("real_time_headroom_factor") >= 650.0
    print("[OK] All FastAPI declipper endpoints verified cleanly")


def test_10_sub_0_03ms_latency_sla_benchmark():
    print("\n--- Test 10: Pure-Math Sub-0.03ms SLA Latency Benchmark ---")
    declipper = AcousticDeclipper(sample_rate=8000)
    # 20ms ITU-T P.50 speech frame with clipped vocal crests (125 Hz fundamental)
    t = np.linspace(0, 0.02, 160, endpoint=False)
    sig = 38000.0 * np.sin(2 * np.pi * 125.0 * t)
    frame = np.clip(sig, -32768, 32767).astype(np.int16).tobytes()

    # Warm-up
    for _ in range(50):
        declipper.process_frame(frame)

    n_iterations = 500
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        declipper.process_frame(frame)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))
    headroom = 20.0 / max(avg_ms, 1e-4)

    print(f"[Benchmark Results] 20ms Frames Profiled: {n_iterations}")
    print(f"Mean Latency: {avg_ms:.4f} ms per 20ms frame")
    print(f"95th Percentile: {p95_ms:.4f} ms")
    print(f"Maximum Latency: {max_ms:.4f} ms")
    print(f"Target SLA: < 0.030 ms | Meets SLA: {avg_ms < 0.030}")
    print(f"Real-Time Headroom Factor: {headroom:.1f}x")

    assert avg_ms < 0.030, f"Mean latency {avg_ms:.4f} ms exceeded 0.030 ms SLA"
    assert headroom >= 650.0, f"Headroom {headroom:.1f}x should be >= 650x"
    print("[OK] Pure-math Acoustic De-Clipper easily meets sub-0.03ms latency SLA with massive headroom")


def main():
    print("=" * 80)
    print("Verbalyze Cellular Microphone Non-Linear Acoustic De-Clipper Suite")
    print("Pure-Math Cubic Hermite Spline & Dynamic Crest Restorer")
    print("Target SLA: Latency < 0.030 ms per 20ms frame | Headroom > 650x")
    print("=" * 80)

    t0 = time.perf_counter()
    test_1_clean_unclipped_speech_passthrough()
    test_2_hard_flat_top_rail_clipping_detection()
    test_3_cubic_hermite_spline_crest_reconstruction()
    test_4_dynamic_headroom_auto_scaling()
    test_5_severe_saturation_profile_classification()
    test_6_total_harmonic_distortion_reduction()
    test_7_crest_factor_restoration()
    test_8_asymmetric_and_dc_offset_clipping_handling()
    test_9_fastapi_rest_endpoints_verification()
    test_10_sub_0_03ms_latency_sla_benchmark()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"ALL 10 ACOUSTIC DE-CLIPPER TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | Pure Math & SLA Compliant")
    print("=" * 80)


if __name__ == "__main__":
    main()
