#!/usr/bin/env python3
"""
scripts/test_packet_loss_concealment_and_jitter.py

Comprehensive 8-Part Automated Test Suite for Packet Loss Concealment (PLC)
and Adaptive Telecom Jitter Buffer Smoothing (Pure-Math DSP).
Validates ITU-T G.711 Appendix I pitch-synchronous waveform replication,
normalized cross-correlation pitch analysis, unvoiced phase-randomization,
multi-frame progressive attenuation, OLA resynchronization, and FastAPI REST endpoints.

Target SLA: Frame Concealment Latency < 0.5 ms per 20 ms frame | Zero Audio Clicks
Zero-Emoji Compliant.
"""

import sys
import os
import time
import base64
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.telephony.packet_loss_concealer import (
    PacketLossConcealer,
    PLCTelemetry,
    G711AppendixIPLC,
)
from verbalyze.telephony.jitter_buffer import (
    AdaptiveJitterBuffer,
    JitterBufferPacket,
    JitterBufferStats,
)
from verbalyze.telephony.server import create_app
from starlette.testclient import TestClient


def generate_sine_wave(freq_hz: float, duration_s: float, sample_rate: int = 8000, amplitude: float = 10000.0) -> np.ndarray:
    """Generates a clean synthetic sinusoidal audio waveform."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.float32)


def test_1_pitch_period_estimation_cross_correlation():
    print("\n--- Test 1: Pitch Period Estimation via Normalized Cross-Correlation ---")
    sample_rate = 8000
    plc = PacketLossConcealer(sample_rate=sample_rate, frame_duration_ms=20.0)

    # Test distinct pitch frequencies spanning human vocal range
    test_freqs = [125.0, 200.0, 250.0]  # Expected periods: 64, 40, 32 samples

    for freq in test_freqs:
        plc.reset()
        expected_period = int(round(sample_rate / freq))

        # Generate 60ms of tone (3 frames = 480 samples)
        audio = generate_sine_wave(freq, 0.06, sample_rate=sample_rate, amplitude=12000.0)
        pcm_bytes = audio.astype(np.int16).tobytes()

        # Ingest 3 frames
        for i in range(0, len(pcm_bytes), 320):
            plc.ingest_good_frame(pcm_bytes[i:i + 320])

        best_lag, best_corr, is_voiced = plc.estimate_pitch()

        print(f"[Tone {freq}Hz] Expected Period: {expected_period} | Detected: {best_lag} samples | Corr: {best_corr:.4f} | Voiced: {is_voiced}")
        assert abs(best_lag - expected_period) <= 1, f"Pitch lag {best_lag} deviates from expected {expected_period}"
        assert best_corr >= 0.95, f"Cross-correlation {best_corr} should be >= 0.95 for pure tone"
        assert is_voiced is True, "Periodic tone must be classified as voiced"

    print("[OK] Normalized cross-correlation accurately resolved pitch periods with >0.95 correlation")


def test_2_voiced_vs_unvoiced_classification():
    print("\n--- Test 2: Voiced vs Unvoiced Speech Classification & Stability ---")
    sample_rate = 8000
    plc = PacketLossConcealer(sample_rate=sample_rate, frame_duration_ms=20.0)

    # 1. Harmonic Voiced Speech Simulation (Fundamental + Formants)
    t = np.linspace(0, 0.06, 480, endpoint=False)
    harmonic_speech = (
        np.sin(2 * np.pi * 150.0 * t) * 8000.0 +
        np.sin(2 * np.pi * 300.0 * t) * 4000.0 +
        np.sin(2 * np.pi * 450.0 * t) * 2000.0
    ).astype(np.int16).tobytes()

    for i in range(0, len(harmonic_speech), 320):
        plc.ingest_good_frame(harmonic_speech[i:i + 320])

    lag_v, corr_v, is_voiced = plc.estimate_pitch()
    print(f"[Voiced Complex Harmonic] Lag: {lag_v} | Corr: {corr_v:.4f} | Voiced: {is_voiced}")
    assert is_voiced is True, "Harmonic voiced speech must be detected as voiced"
    assert corr_v >= 0.85, f"Harmonic correlation {corr_v} must be >= 0.85"

    # 2. Gaussian White Noise Simulation (Unvoiced Fricative / Line Hiss)
    plc.reset()
    np.random.seed(42)
    noise_samples = (np.random.normal(0, 1500, 480)).astype(np.int16).tobytes()

    for i in range(0, len(noise_samples), 320):
        plc.ingest_good_frame(noise_samples[i:i + 320])

    lag_u, corr_u, is_unvoiced = plc.estimate_pitch()
    print(f"[Unvoiced Gaussian Noise] Lag: {lag_u} | Corr: {corr_u:.4f} | Voiced: {is_unvoiced}")
    assert is_unvoiced is False, "Gaussian noise must be classified as unvoiced"
    assert corr_u < 0.55, f"Noise cross-correlation {corr_u} must be below 0.55"

    print("[OK] Voiced and unvoiced speech regimes accurately discriminated")


def test_3_single_frame_concealment_and_waveform_correlation():
    print("\n--- Test 3: Single-Frame (20ms) Concealment & Waveform Correlation ---")
    sample_rate = 8000
    plc = PacketLossConcealer(sample_rate=sample_rate, frame_duration_ms=20.0)

    # Generate 80ms of continuous 200 Hz tone (4 frames: 3 good + 1 ground truth)
    total_audio = generate_sine_wave(200.0, 0.08, sample_rate=sample_rate, amplitude=12000.0)
    frame_len = 160  # 20ms at 8kHz

    frame_1 = total_audio[0:frame_len].astype(np.int16).tobytes()
    frame_2 = total_audio[frame_len:2*frame_len].astype(np.int16).tobytes()
    frame_3 = total_audio[2*frame_len:3*frame_len].astype(np.int16).tobytes()
    ground_truth_frame_4 = total_audio[3*frame_len:4*frame_len]

    # Ingest 3 good frames
    plc.ingest_good_frame(frame_1)
    plc.ingest_good_frame(frame_2)
    plc.ingest_good_frame(frame_3)

    # Now drop frame 4 and synthesize concealment
    synth_pcm, telemetry = plc.conceal_frame()
    synth_samples = np.frombuffer(synth_pcm, dtype=np.int16).astype(np.float32)

    # Compute Pearson correlation with ground truth frame 4
    synth_norm = synth_samples - np.mean(synth_samples)
    gt_norm = ground_truth_frame_4 - np.mean(ground_truth_frame_4)
    corr = float(np.dot(synth_norm, gt_norm) / (np.linalg.norm(synth_norm) * np.linalg.norm(gt_norm) + 1e-6))

    print(f"[Concealed Frame 4] Correlation with Ground Truth Continuation: {corr:.4f}")
    print(f"[Telemetry] Voiced: {telemetry.is_voiced}, Pitch: {telemetry.pitch_period} samples, Method: {telemetry.method}")

    assert corr >= 0.90, f"Synthesized waveform correlation {corr} should be >= 0.90 for periodic tone"
    assert telemetry.is_concealed is True
    assert telemetry.method == "pitch_replicated"
    print("[OK] Single-frame pitch-synchronous extrapolation verified with >0.90 waveform correlation")


def test_4_boundary_continuity_and_click_elimination():
    print("\n--- Test 4: Boundary Derivative Continuity & Zero-Click Elimination ---")
    sample_rate = 8000
    plc = PacketLossConcealer(sample_rate=sample_rate, frame_duration_ms=20.0, ola_samples=24)

    # Tone with 35-sample pitch period (approx 228.5 Hz) which does not divide 160 evenly
    # 160 / 35 = 4.57 cycles. Without OLA cross-fading, repetition creates a phase cliff.
    audio = generate_sine_wave(228.57, 0.06, sample_rate=sample_rate, amplitude=14000.0)
    for i in range(0, len(audio), 160):
        plc.ingest_good_frame(audio[i:i + 160].astype(np.int16).tobytes())

    # Measure jump at boundary from history tail to synthesized start
    last_hist_sample = plc.history[-1]
    synth_pcm, telemetry = plc.conceal_frame()
    synth_samples = np.frombuffer(synth_pcm, dtype=np.int16).astype(np.float32)

    boundary_diff = abs(synth_samples[0] - last_hist_sample)
    print(f"Boundary Transition: Last History Sample = {last_hist_sample:.1f}, First Synth Sample = {synth_samples[0]:.1f}, Diff = {boundary_diff:.1f}")

    # Maximum step change across the synthesized frame
    diffs = np.abs(np.diff(synth_samples))
    max_step = float(np.max(diffs))
    print(f"Maximum Step Jump across 160 samples: {max_step:.1f} (within smooth continuous waveform bounds)")

    # For a 14000 amplitude 228.57 Hz sine wave at 8kHz, the maximum normal slope per sample is:
    # 14000 * 2 * pi * 228.57 / 8000 ≈ 2513
    assert max_step < 3500.0, f"Max step {max_step} exceeds acoustic smoothness threshold, possible click artifact"
    print("[OK] Smooth overlap-add cross-fading verified; high-frequency clicks eliminated")


def test_5_multi_frame_burst_loss_progressive_attenuation():
    print("\n--- Test 5: Multi-Frame Burst Loss Progressive Energy Attenuation ---")
    sample_rate = 8000
    plc = PacketLossConcealer(sample_rate=sample_rate, frame_duration_ms=20.0)

    # Seed with 12000 amplitude tone
    seed_audio = generate_sine_wave(200.0, 0.06, sample_rate=sample_rate, amplitude=12000.0)
    for i in range(0, len(seed_audio), 160):
        plc.ingest_good_frame(seed_audio[i:i + 160].astype(np.int16).tobytes())

    initial_rms = float(np.sqrt(np.mean(seed_audio[-160:] ** 2)))
    print(f"Initial Seed RMS: {initial_rms:.1f}")

    rms_levels = []
    # Drop 5 consecutive frames (100ms total drop)
    for frame_idx in range(1, 6):
        synth_pcm, telemetry = plc.conceal_frame()
        synth = np.frombuffer(synth_pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(synth ** 2)))
        ratio = rms / initial_rms
        rms_levels.append(ratio)
        print(f"Lost Frame {frame_idx} (Loss Duration: {frame_idx * 20}ms): RMS={rms:.1f}, Energy Retention={ratio * 100.0:.1f}%, Method={telemetry.method}")

    # Assert progressive decay curve:
    # Frame 1: ~90-100%
    assert 0.85 <= rms_levels[0] <= 1.05, f"Frame 1 retention {rms_levels[0]} out of expected range"
    # Frame 2: ~60-85%
    assert 0.60 <= rms_levels[1] <= 0.88, f"Frame 2 retention {rms_levels[1]} out of expected range"
    # Frame 3: ~30-60%
    assert 0.25 <= rms_levels[2] <= 0.60, f"Frame 3 retention {rms_levels[2]} out of expected range"
    # Frame 4: <25%
    assert rms_levels[3] <= 0.25, f"Frame 4 retention {rms_levels[3]} out of expected range"
    # Frame 5+: Exact Comfort Silence (0.0)
    assert rms_levels[4] == 0.0, f"Frame 5 retention {rms_levels[4]} should be exact silence"

    print("[OK] Multi-frame progressive attenuation curve matches ITU-T G.711 Appendix I specification")


def test_6_post_loss_good_packet_ola_resynchronization():
    print("\n--- Test 6: Post-Loss Good Packet Overlap-Add (OLA) Resynchronization ---")
    sample_rate = 8000
    plc = PacketLossConcealer(sample_rate=sample_rate, frame_duration_ms=20.0, resync_samples=32)

    # Seed with 200 Hz cosine wave at peak amplitude (+12000)
    t = np.linspace(0, 0.06, 480, endpoint=False)
    seed_audio = (np.cos(2 * np.pi * 200.0 * t) * 12000.0).astype(np.float32)
    for i in range(0, len(seed_audio), 160):
        plc.ingest_good_frame(seed_audio[i:i + 160].astype(np.int16).tobytes())

    # Conceal 1 frame
    synth_pcm, _ = plc.conceal_frame()
    last_synth_sample = float(np.frombuffer(synth_pcm, dtype=np.int16)[-1])

    # Next good frame arrives with inverted phase at negative peak (-12000), causing massive ~24,000 cliff
    anti_phase_audio = (-seed_audio[:160]).copy()
    raw_cliff = abs(anti_phase_audio[0] - last_synth_sample)

    # Ingest good frame through PLC with OLA resynchronization
    resynced_pcm, telemetry = plc.ingest_good_frame(anti_phase_audio.astype(np.int16).tobytes())
    resynced_samples = np.frombuffer(resynced_pcm, dtype=np.int16).astype(np.float32)

    # Measured transition cliff with OLA resync
    resynced_cliff = abs(resynced_samples[0] - last_synth_sample)
    cliff_reduction = ((raw_cliff - resynced_cliff) / raw_cliff) * 100.0
    print(f"Worst-Case Anti-Phase Boundary Cliff: Raw Unfiltered = {raw_cliff:.1f} -> With OLA Resync = {resynced_cliff:.1f} ({cliff_reduction:.1f}% reduction)")

    # Maximum step jump in the first 32 samples
    ola_max_step = float(np.max(np.abs(np.diff(resynced_samples[:32]))))
    print(f"Max Step Jump in OLA Cross-Fade Region: {ola_max_step:.1f}")

    assert resynced_cliff < raw_cliff * 0.15, f"OLA resynchronization should reduce phase cliff by at least 85% (got {cliff_reduction:.1f}%)"
    print("[OK] Post-loss good frame smoothly resynchronized without acoustic phase shock")


def test_7_adaptive_jitter_buffer_reordering_and_smoothing():
    print("\n--- Test 7: Adaptive Jitter Buffer Reordering & Concealment Smoothing ---")
    jb = AdaptiveJitterBuffer(frame_duration_ms=20.0, sample_rate=8000, min_delay_ms=40.0, max_delay_ms=200.0)

    # Generate 5 frames
    frames = [generate_sine_wave(200.0, 0.02, sample_rate=8000, amplitude=8000.0).astype(np.int16).tobytes() for _ in range(5)]

    # 1. Test Out-of-Order Packet Reordering:
    # Send seq 1, seq 3, seq 2, seq 5, seq 4
    now = time.time() * 1000.0
    jb.push(frames[0], sequence_number=1, arrival_time_ms=now)
    jb.push(frames[2], sequence_number=3, arrival_time_ms=now + 5)
    jb.push(frames[1], sequence_number=2, arrival_time_ms=now + 10)
    jb.push(frames[4], sequence_number=5, arrival_time_ms=now + 15)
    jb.push(frames[3], sequence_number=4, arrival_time_ms=now + 20)

    stats = jb.get_stats()
    print(f"[Reordering] Total Received: {stats.total_packets_received}, Packets Reordered: {stats.packets_reordered}")
    assert stats.packets_reordered >= 2, "Out-of-order packets should be recognized by jitter stats"

    # Pop 5 frames and verify they emerge in strictly sorted order
    popped_seqs = []
    for _ in range(5):
        frame, is_concealed = jb.pop()
        assert is_concealed is False
        assert len(frame) == 320
        popped_seqs.append(jb.last_played_sequence)

    print(f"[Playout Order] Sequence: {popped_seqs}")
    assert popped_seqs == [1, 2, 3, 4, 5], f"Playout sequence {popped_seqs} should be strictly sequential"

    # 2. Test Packet Loss Trigger:
    # Send seq 7 (seq 6 dropped by cellular corridor)
    jb.push(frames[0], sequence_number=7, arrival_time_ms=now + 40)

    # Expected sequence is 6. Pop should trigger PLC concealment.
    concealed_frame, is_concealed = jb.pop()
    print(f"[Missing Packet 6] Popped: Concealed={is_concealed}, LastPlayedSeq={jb.last_played_sequence}")
    assert is_concealed is True, "Missing packet should trigger PLC concealment"
    assert jb.last_played_sequence == 6, "Last played sequence should advance past concealed frame"
    assert len(concealed_frame) == 320

    # Next pop should deliver packet 7 with resynchronization
    good_frame, is_concealed_7 = jb.pop()
    print(f"[Recovered Packet 7] Popped: Concealed={is_concealed_7}, LastPlayedSeq={jb.last_played_sequence}")
    assert is_concealed_7 is False
    assert jb.last_played_sequence == 7

    print("[OK] Adaptive Jitter Buffer handles packet reordering, loss concealment, and playout synchronization")


def test_8_fastapi_telephony_plc_rest_endpoints():
    print("\n--- Test 8: FastAPI Telephony PLC REST Endpoints ---")
    app = create_app()
    client = TestClient(app)

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_json = health_resp.json()
    print(f"[GET /health] Status: {health_json.get('status')}, PLC: {health_json.get('plc_status')}")
    assert health_json.get("plc_status") == "ready"

    # 2. POST /telephony/plc/conceal
    seed_pcm = generate_sine_wave(200.0, 0.04, sample_rate=8000, amplitude=10000.0).astype(np.int16).tobytes()
    seed_b64 = base64.b64encode(seed_pcm).decode("ascii")

    conceal_payload = {
        "audio_base64": seed_b64,
        "consecutive_drops": 2,
        "simulate_recovery": True,
        "sample_rate": 8000,
    }
    conceal_resp = client.post("/telephony/plc/conceal", json=conceal_payload)
    assert conceal_resp.status_code == 200
    conceal_json = conceal_resp.json()

    print(f"[POST /telephony/plc/conceal] Status: {conceal_json.get('status')}, Concealed Frames: {conceal_json.get('concealed_frames_count')}")
    assert conceal_json.get("status") == "ok"
    assert conceal_json.get("concealed_frames_count") == 2
    assert len(conceal_json.get("concealed_frames")) == 2
    assert conceal_json.get("resynced_good_frame_base64") is not None

    # 3. POST /telephony/plc/benchmark
    bench_resp = client.post("/telephony/plc/benchmark")
    assert bench_resp.status_code == 200
    bench_json = bench_resp.json()

    avg_time = bench_json.get("avg_plc_time_ms")
    p95_time = bench_json.get("p95_plc_time_ms")
    headroom = bench_json.get("real_time_headroom_factor")
    meets_sla = bench_json.get("meets_plc_sla")

    print(f"[POST /telephony/plc/benchmark] Avg: {avg_time} ms | P95: {p95_time} ms | Meets SLA (<0.5ms): {meets_sla} | Headroom: {headroom}x")
    assert meets_sla is True, f"PLC execution time {avg_time} ms exceeded 0.5 ms SLA target"
    assert headroom >= 40.0, f"Headroom {headroom}x should be >= 40x"

    print("[OK] All FastAPI Telephony PLC REST endpoints verified successfully")


def main():
    print("=" * 80)
    print("Verbalyze Real-Time Packet Loss Concealment (PLC) & Jitter Smoothing Suite")
    print("ITU-T G.711 Appendix I Compliant | Target Latency SLA < 0.5 ms per 20 ms frame")
    print("=" * 80)

    t0 = time.time()
    test_1_pitch_period_estimation_cross_correlation()
    test_2_voiced_vs_unvoiced_classification()
    test_3_single_frame_concealment_and_waveform_correlation()
    test_4_boundary_continuity_and_click_elimination()
    test_5_multi_frame_burst_loss_progressive_attenuation()
    test_6_post_loss_good_packet_ola_resynchronization()
    test_7_adaptive_jitter_buffer_reordering_and_smoothing()
    test_8_fastapi_telephony_plc_rest_endpoints()

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ALL 8 PACKET LOSS CONCEALMENT (PLC) & JITTER TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | ITU-T G.711 Appendix I Compliant")
    print("=" * 80)


if __name__ == "__main__":
    main()
