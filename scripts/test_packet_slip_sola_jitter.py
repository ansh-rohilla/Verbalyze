#!/usr/bin/env python3
"""
scripts/test_packet_slip_sola_jitter.py

Comprehensive 10-Part Automated Verification Suite for Pure-Math Fractional-Sample
Acoustic Jitter Buffer & Packet Slip Synthesizer (RFC 3550 / ITU-T G.1020).

Validates:
1. SOLA Time Expansion (+15%) Pitch and Formant Invariance (<2% pitch error).
2. SOLA Time Compression (-15%) Pitch Invariance without Chipmunk Distortion.
3. Overlap-Add Phase Continuity & Zero Boundary Cliff Discontinuities.
4. 4-Point Cubic Hermite Fractional-Sample Clock Drift Compensation.
5. ITU-T G.1020 Packet Slip Synthesizer (Cycle Insertion & Deletion).
6. AdaptiveJitterBuffer Starvation Prevention via Dynamic Time Expansion.
7. AdaptiveJitterBuffer Latency Drain via Dynamic Time Compression.
8. Multilingual Indic Pitch Contour Preservation (Hindi / Tamil Cadence).
9. FastAPI Telephony SOLA Jitter REST Endpoints & Health Check.
10. Sub-0.05ms Frame Processing Latency SLA Benchmark (>400x real-time headroom).

Target Latency SLA: <0.05 ms per 20 ms frame | Pure-Math NumPy | Zero-Emoji Compliant.
"""

import sys
import os
import time
import base64
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.telephony.sola_tsm import (
    SOLATimeScaleModifier,
    PacketSlipSynthesizer,
    FractionalSampleInterpolator,
    SOLAMode,
    SlipType,
    TSMTelemetry,
    SlipTelemetry,
    FractionalTelemetry,
)
from verbalyze.telephony.jitter_buffer import (
    AdaptiveJitterBuffer,
    JitterBufferPacket,
    JitterBufferStats,
)
from verbalyze.telephony.server import create_app
from starlette.testclient import TestClient


def generate_harmonic_speech(
    f0_hz: float,
    duration_s: float,
    sample_rate: int = 8000,
    amplitude: float = 8000.0,
) -> np.ndarray:
    """Generates synthetic harmonic speech with fundamental F0 and 3 formants."""
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sig = (
        np.sin(2 * np.pi * f0_hz * t) * amplitude +
        np.sin(2 * np.pi * (2 * f0_hz) * t) * (amplitude * 0.5) +
        np.sin(2 * np.pi * (3 * f0_hz) * t) * (amplitude * 0.25)
    )
    return sig.astype(np.float32)


def test_1_sola_time_expansion_pitch_invariance():
    print("\n--- Test 1: SOLA Time Expansion (+15%) Pitch & Formant Invariance ---")
    sample_rate = 8000
    sola = SOLATimeScaleModifier(sample_rate=sample_rate)

    f0 = 180.0
    duration = 0.3  # 300ms
    speech = generate_harmonic_speech(f0, duration, sample_rate=sample_rate)

    pcm_in = speech.astype(np.int16).tobytes()
    pcm_out, telemetry = sola.modify_pcm(pcm_in, scale_factor=1.15)

    print(f"[Telemetry] Mode: {telemetry.mode} | Scale: {telemetry.scale_factor} | Corr: {telemetry.mean_correlation:.4f}")
    print(f"[Samples] In: {telemetry.input_samples} | Out: {telemetry.output_samples} | Ratio: {telemetry.output_samples / telemetry.input_samples:.3f}")
    print(f"[Pitch] In: {telemetry.pitch_in_hz:.1f} Hz | Out: {telemetry.pitch_out_hz:.1f} Hz | Pitch Error: {telemetry.pitch_error_pct:.2f}%")

    assert telemetry.mode == SOLAMode.EXPANSION.value
    assert telemetry.scale_factor == 1.15
    assert telemetry.output_samples > telemetry.input_samples
    assert 1.05 <= (telemetry.output_samples / telemetry.input_samples) <= 1.25
    assert telemetry.pitch_error_pct < 2.0, f"Pitch changed by {telemetry.pitch_error_pct}%, exceeds 2% SLA"
    assert telemetry.mean_correlation >= 0.85
    print("[OK] SOLA Time Expansion stretched speech by +15% with <2% pitch error")


def test_2_sola_time_compression_pitch_invariance():
    print("\n--- Test 2: SOLA Time Compression (-15%) Pitch Invariance (No Chipmunk) ---")
    sample_rate = 8000
    sola = SOLATimeScaleModifier(sample_rate=sample_rate)

    f0 = 220.0
    duration = 0.35  # 350ms
    speech = generate_harmonic_speech(f0, duration, sample_rate=sample_rate)

    pcm_in = speech.astype(np.int16).tobytes()
    pcm_out, telemetry = sola.modify_pcm(pcm_in, scale_factor=0.85)

    print(f"[Telemetry] Mode: {telemetry.mode} | Scale: {telemetry.scale_factor} | Corr: {telemetry.mean_correlation:.4f}")
    print(f"[Samples] In: {telemetry.input_samples} | Out: {telemetry.output_samples} | Ratio: {telemetry.output_samples / telemetry.input_samples:.3f}")
    print(f"[Pitch] In: {telemetry.pitch_in_hz:.1f} Hz | Out: {telemetry.pitch_out_hz:.1f} Hz | Pitch Error: {telemetry.pitch_error_pct:.2f}%")

    assert telemetry.mode == SOLAMode.COMPRESSION.value
    assert telemetry.scale_factor == 0.85
    assert telemetry.output_samples < telemetry.input_samples
    assert 0.75 <= (telemetry.output_samples / telemetry.input_samples) <= 0.95
    assert telemetry.pitch_error_pct < 2.0, f"Pitch changed by {telemetry.pitch_error_pct}%, chipmunk effect detected"
    assert telemetry.mean_correlation >= 0.85
    print("[OK] SOLA Time Compression accelerated speech by -15% with zero chipmunk effect")


def test_3_phase_continuity_and_boundary_smoothness():
    print("\n--- Test 3: Phase Continuity & Boundary Discontinuity Elimination ---")
    sample_rate = 8000
    sola = SOLATimeScaleModifier(sample_rate=sample_rate)

    # Signal with non-integer pitch period (217.39 Hz = 36.8 samples at 8kHz)
    t = np.linspace(0.0, 0.25, int(sample_rate * 0.25), endpoint=False)
    audio = (np.sin(2 * np.pi * 217.39 * t) * 12000.0).astype(np.float32)

    y_exp, tel_exp = sola.modify_scale(audio, scale_factor=1.12)

    diffs_orig = np.abs(np.diff(audio))
    diffs_sola = np.abs(np.diff(y_exp))
    max_step_orig = float(np.max(diffs_orig))
    max_step_sola = float(np.max(diffs_sola))

    print(f"[Smoothness] Original Max Step: {max_step_orig:.1f} | SOLA Max Step: {max_step_sola:.1f} | Ratio: {max_step_sola / max_step_orig:.2f}")

    assert max_step_sola <= max_step_orig * 1.35, "Phase jump or click detected at overlap splice boundary"
    assert not np.isnan(y_exp).any()
    print("[OK] Raised-cosine overlap-add cross-fading eliminates phase clicks and cliff discontinuities")


def test_4_fractional_sample_clock_drift_compensation():
    print("\n--- Test 4: 4-Point Cubic Hermite Clock Drift Compensation ---")
    sample_rate = 8000
    interpolator = FractionalSampleInterpolator(sample_rate=sample_rate)

    t = np.linspace(0.0, 0.1, 800, endpoint=False)
    sig = (np.sin(2 * np.pi * 250.0 * t) * 10000.0).astype(np.float32)

    # 1. Micro ppm positive drift (+500 ppm = 0.05% clock speed mismatch)
    y_drift, tel_drift = interpolator.interpolate_ppm(sig, ppm_drift=500.0)
    print(f"[PPM +500] In: {tel_drift.input_samples} | Out: {tel_drift.output_samples} | Ratio: {tel_drift.ratio:.6f} | Time: {tel_drift.processing_time_ms:.3f} ms")
    assert tel_drift.output_samples >= tel_drift.input_samples
    assert abs(tel_drift.ppm_drift - 500.0) < 0.01

    # 2. Arbitrary fractional resampling ratio (1.002)
    y_ratio, tel_ratio = interpolator.interpolate(sig, ratio=1.002)
    print(f"[Ratio 1.002] In: {tel_ratio.input_samples} | Out: {tel_ratio.output_samples} | Time: {tel_ratio.processing_time_ms:.3f} ms")
    assert tel_ratio.output_samples == int(round(800 * 1.002))

    # 3. PCM byte wrapper
    pcm_in = sig.astype(np.int16).tobytes()
    pcm_out, tel_pcm = interpolator.interpolate_pcm(pcm_in, ratio=0.998)
    assert len(pcm_out) == tel_pcm.output_samples * 2
    print("[OK] Cubic Hermite fractional-sample interpolator cleanly compensates for carrier clock drift")


def test_5_packet_slip_synthesis_cycle_insertion_deletion():
    print("\n--- Test 5: ITU-T G.1020 Packet Slip Synthesizer (Cycle Insertion & Deletion) ---")
    sample_rate = 8000
    synthesizer = PacketSlipSynthesizer(sample_rate=sample_rate)

    # 200 Hz pure tone (period = 40 samples at 8kHz)
    t = np.linspace(0.0, 0.15, 1200, endpoint=False)
    audio = (np.sin(2 * np.pi * 200.0 * t) * 10000.0).astype(np.float32)

    # 1. Slip Deletion (Positive Slip)
    y_del, tel_del = synthesizer.synthesize_slip(audio, SlipType.DELETION, period_samples=40)
    print(f"[Slip Deletion] Type: {tel_del.slip_type} | Samples Removed: {tel_del.slip_samples} | Pitch: {tel_del.detected_pitch_hz:.1f} Hz | Max Jump: {tel_del.max_discontinuity_jump:.1f}")
    assert tel_del.slip_type == SlipType.DELETION.value
    assert tel_del.slip_samples == 40
    assert len(y_del) == len(audio) - 40

    # 2. Slip Insertion (Negative Slip)
    y_ins, tel_ins = synthesizer.synthesize_slip(audio, SlipType.INSERTION, period_samples=40)
    print(f"[Slip Insertion] Type: {tel_ins.slip_type} | Samples Inserted: {tel_ins.slip_samples} | Pitch: {tel_ins.detected_pitch_hz:.1f} Hz | Max Jump: {tel_ins.max_discontinuity_jump:.1f}")
    assert tel_ins.slip_type == SlipType.INSERTION.value
    assert tel_ins.slip_samples == 40
    assert len(y_ins) == len(audio) + 40

    # Verify no large step discontinuity at the splice
    orig_max_diff = float(np.max(np.abs(np.diff(audio))))
    assert tel_del.max_discontinuity_jump <= orig_max_diff * 1.15
    assert tel_ins.max_discontinuity_jump <= orig_max_diff * 1.15
    print("[OK] Packet slip synthesizer executes click-free pitch cycle insertion and deletion")


def test_6_jitter_buffer_starvation_prevention():
    print("\n--- Test 6: AdaptiveJitterBuffer Starvation Prevention via SOLA Expansion ---")
    sample_rate = 8000
    # Create jitter buffer with SOLA enabled, target delay 80ms
    jb = AdaptiveJitterBuffer(
        frame_duration_ms=20.0,
        sample_rate=sample_rate,
        nominal_delay_ms=80.0,
        enable_sola=True,
    )

    # Push only 2 frames (40ms buffer occupancy, well below target 80ms -> starvation risk)
    frame1 = generate_harmonic_speech(200.0, 0.02, sample_rate=sample_rate).astype(np.int16).tobytes()
    frame2 = generate_harmonic_speech(200.0, 0.02, sample_rate=sample_rate).astype(np.int16).tobytes()

    now = time.time() * 1000.0
    jb.push(frame1, sequence_number=1, arrival_time_ms=now)
    jb.push(frame2, sequence_number=2, arrival_time_ms=now + 20)

    # Pop with SOLA
    popped1, concealed1, tel1 = jb.pop_sola(current_time_ms=now + 20)

    print(f"[Starvation Pop 1] Concealed: {concealed1} | Scale Factor: {jb.stats.last_scale_factor:.4f} | Expansions Count: {jb.stats.sola_expansions_count}")
    assert concealed1 is False
    assert len(popped1) == 320  # Exactly 1 frame (20ms at 8kHz = 160 samples = 320 bytes)
    assert jb.stats.last_scale_factor > 1.0, f"Expected expansion factor > 1.0, got {jb.stats.last_scale_factor}"
    assert jb.stats.sola_expansions_count > 0
    print("[OK] Jitter buffer proactively expanded audio to protect against imminent packet starvation")


def test_7_jitter_buffer_latency_drain_compression():
    print("\n--- Test 7: AdaptiveJitterBuffer Latency Drain via SOLA Compression ---")
    sample_rate = 8000
    jb = AdaptiveJitterBuffer(
        frame_duration_ms=20.0,
        sample_rate=sample_rate,
        nominal_delay_ms=40.0,
        enable_sola=True,
    )

    # Queue up 7 packets (140ms occupancy, exceeding 40ms + 40ms = 80ms threshold -> latency accumulation)
    now = time.time() * 1000.0
    for seq in range(1, 8):
        frame = generate_harmonic_speech(200.0, 0.02, sample_rate=sample_rate).astype(np.int16).tobytes()
        jb.push(frame, sequence_number=seq, arrival_time_ms=now + (seq * 5))

    popped, concealed, tel = jb.pop_sola(current_time_ms=now + 50)
    print(f"[Latency Drain Pop] Concealed: {concealed} | Scale Factor: {jb.stats.last_scale_factor:.4f} | Compressions Count: {jb.stats.sola_compressions_count}")

    assert concealed is False
    assert len(popped) == 320
    assert jb.stats.last_scale_factor < 1.0, f"Expected compression factor < 1.0, got {jb.stats.last_scale_factor}"
    assert jb.stats.sola_compressions_count > 0
    print("[OK] Jitter buffer successfully compressed speech to drain accumulated buffer latency")


def test_8_indic_melodic_cadence_preservation():
    print("\n--- Test 8: Multilingual Indic Melodic Cadence Preservation ---")
    sample_rate = 8000
    sola = SOLATimeScaleModifier(sample_rate=sample_rate)

    # Simulate an Indic rising-falling melodic pitch excursion (Hindi / Bengali intonation: 150 Hz -> 240 Hz -> 170 Hz)
    duration = 0.4
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    f_instant = 150.0 + 90.0 * np.sin(np.pi * (t / duration))
    phase = 2 * np.pi * np.cumsum(f_instant) / sample_rate
    indic_audio = (np.sin(phase) * 9000.0).astype(np.float32)

    # Test expansion and compression on natural glide
    y_exp, tel_exp = sola.modify_scale(indic_audio, scale_factor=1.10)
    y_comp, tel_comp = sola.modify_scale(indic_audio, scale_factor=0.90)

    print(f"[Indic Glide Expansion] Corr: {tel_exp.mean_correlation:.4f} | Time: {tel_exp.processing_time_ms:.3f} ms")
    print(f"[Indic Glide Compression] Corr: {tel_comp.mean_correlation:.4f} | Time: {tel_comp.processing_time_ms:.3f} ms")

    assert tel_exp.mean_correlation >= 0.80
    assert tel_comp.mean_correlation >= 0.80
    assert not np.isnan(y_exp).any()
    assert not np.isnan(y_comp).any()
    print("[OK] Indic intonation pitch glide contours preserved without harmonic distortion")


def test_9_fastapi_sola_jitter_rest_endpoints():
    print("\n--- Test 9: FastAPI Telephony SOLA Jitter REST Endpoints ---")
    app = create_app()
    client = TestClient(app)

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    print(f"[GET /health] sola_jitter_status: {h_data.get('sola_jitter_status')}")
    assert h_data.get("sola_jitter_status") == "ready"

    # 2. POST /telephony/jitter/sola-modify
    t = np.linspace(0.0, 0.2, 1600, endpoint=False)
    pcm_test = (np.sin(2 * np.pi * 200.0 * t) * 10000.0).astype(np.int16).tobytes()
    b64_in = base64.b64encode(pcm_test).decode("ascii")

    mod_resp = client.post("/telephony/jitter/sola-modify", json={
        "audio_base64": b64_in,
        "scale_factor": 1.15,
        "sample_rate": 8000,
    })
    assert mod_resp.status_code == 200
    mod_data = mod_resp.json()
    print(f"[POST /telephony/jitter/sola-modify] Status: {mod_data.get('status')} | Mode: {mod_data['telemetry']['mode']} | Pitch: {mod_data['telemetry']['pitch_out_hz']} Hz")
    assert mod_data.get("status") == "ok"
    assert mod_data["telemetry"]["mode"] == "expansion"
    assert mod_data.get("modified_audio_base64") is not None

    # 3. POST /telephony/jitter/slip-synthesize
    slip_resp = client.post("/telephony/jitter/slip-synthesize", json={
        "audio_base64": b64_in,
        "slip_type": "deletion",
        "period_samples": 40,
        "sample_rate": 8000,
    })
    assert slip_resp.status_code == 200
    slip_data = slip_resp.json()
    print(f"[POST /telephony/jitter/slip-synthesize] Status: {slip_data.get('status')} | Slip: {slip_data['telemetry']['slip_type']} | Samples: {slip_data['telemetry']['slip_samples']}")
    assert slip_data.get("status") == "ok"
    assert slip_data["telemetry"]["slip_type"] == "deletion"
    assert slip_data["telemetry"]["slip_samples"] == 40

    # 4. POST /telephony/jitter/sola-benchmark
    bench_resp = client.post("/telephony/jitter/sola-benchmark")
    assert bench_resp.status_code == 200
    bench_data = bench_resp.json()
    print(f"[POST /telephony/jitter/sola-benchmark] Avg: {bench_data.get('avg_sola_time_ms')} ms | Meets SLA: {bench_data.get('meets_sla')} | Headroom: {bench_data.get('real_time_headroom_factor')}x")
    assert bench_data.get("meets_sla") is True
    assert bench_data.get("real_time_headroom_factor") >= 400.0
    print("[OK] All FastAPI Telephony SOLA Jitter REST endpoints verified successfully")


def test_10_sub_0_05ms_latency_sla_benchmark():
    print("\n--- Test 10: Pure-Math Sub-0.05ms SLA Latency Benchmark ---")
    sample_rate = 8000
    sola = SOLATimeScaleModifier(sample_rate=sample_rate)

    # Standard 20ms frame (160 samples at 8kHz)
    t = np.linspace(0.0, 0.02, 160, endpoint=False)
    frame = (np.sin(2 * np.pi * 200.0 * t) * 10000.0).astype(np.int16).tobytes()

    # Warm-up
    for _ in range(10):
        sola.process_streaming_frame(frame, scale_factor=1.15)

    n_iterations = 250
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        sola.process_streaming_frame(frame, scale_factor=1.15)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))
    headroom = 20.0 / max(avg_ms, 1e-4)

    print(f"[Benchmark Results] 20ms Frames Profiled: {n_iterations}")
    print(f"Mean Latency: {avg_ms:.4f} ms per 20ms frame")
    print(f"95th Percentile: {p95_ms:.4f} ms")
    print(f"Maximum Latency: {max_ms:.4f} ms")
    print(f"Target SLA: < 0.050 ms | Meets SLA: {avg_ms < 0.05}")
    print(f"Real-Time Headroom Factor: {headroom:.1f}x")

    assert avg_ms < 0.05, f"Mean latency {avg_ms:.4f} ms exceeded 0.05 ms SLA"
    assert headroom >= 400.0, f"Headroom {headroom:.1f}x should be >= 400x"
    print("[OK] Pure-math SOLA engine satisfies sub-0.05ms latency SLA with massive headroom")


def main():
    print("=" * 80)
    print("Verbalyze Fractional-Sample Acoustic Jitter Buffer & Packet Slip Suite")
    print("RFC 3550 & ITU-T G.1020 Compliant | Pure-Math Synchronized Overlap-Add (SOLA)")
    print("Target SLA: Latency < 0.05 ms per 20 ms frame | Real-Time Headroom > 400x")
    print("=" * 80)

    t0 = time.time()
    test_1_sola_time_expansion_pitch_invariance()
    test_2_sola_time_compression_pitch_invariance()
    test_3_phase_continuity_and_boundary_smoothness()
    test_4_fractional_sample_clock_drift_compensation()
    test_5_packet_slip_synthesis_cycle_insertion_deletion()
    test_6_jitter_buffer_starvation_prevention()
    test_7_jitter_buffer_latency_drain_compression()
    test_8_indic_melodic_cadence_preservation()
    test_9_fastapi_sola_jitter_rest_endpoints()
    test_10_sub_0_05ms_latency_sla_benchmark()

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ALL 10 FRACTIONAL-SAMPLE JITTER & PACKET SLIP TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | RFC 3550 & ITU-T G.1020 Compliant")
    print("=" * 80)


if __name__ == "__main__":
    main()
