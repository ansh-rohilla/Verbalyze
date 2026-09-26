#!/usr/bin/env python3
"""
scripts/test_voice_boundary_predictor.py

Comprehensive Test Suite for Pure-Math Acoustic End-of-Turn (EoT) & Voice Boundary Predictor:
1. ITU-T P.56 Active Speech Level Estimation & Activity Factor.
2. Pitch Declination & Parabolic NACF Tracking (Falling Terminal Statement).
3. Rising Pitch Intonation (Query / Mid-Clause Floor Holding).
4. Short-Time Energy Decay Rate (Terminal Consonant Cutoff).
5. Spectral Flux & Inhalation Breath Intake Detection.
6. Dynamic Variable Silence Decision Window (120ms Snappy vs 650ms Protection).
7. FastAPI REST Endpoints (/health, /telephony/eot/analyze).
8. Sub-0.1ms Execution Throughput & Real-Time Headroom Benchmark.

Zero-emoji compliant.
"""

import sys
import math
import time
import base64
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.telephony.voice_boundary import (
    VoiceBoundaryPredictor,
    VoiceBoundaryTelemetry,
    PitchTrend,
    TurnBoundaryDecision,
    ITUTP56SpeechLevelEstimator,
    PitchDeclinationTracker,
    EnergyAndFluxTracker,
)


def generate_sine_pcm(freq_hz: float, duration_sec: float, sample_rate: int = 8000, amplitude: float = 12000.0) -> bytes:
    """Generates continuous sine wave 16-bit linear PCM bytes."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    signal = (np.sin(2.0 * np.pi * freq_hz * t) * amplitude).astype(np.int16)
    return signal.tobytes()


def generate_sweep_pcm(start_hz: float, end_hz: float, duration_sec: float, sample_rate: int = 8000, amplitude: float = 12000.0) -> bytes:
    """Generates linear frequency sweep 16-bit linear PCM bytes."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    # Phase integral of linear frequency sweep
    instantaneous_freq = start_hz + (end_hz - start_hz) * (t / duration_sec)
    phase = 2.0 * np.pi * (start_hz * t + 0.5 * (end_hz - start_hz) * (t ** 2) / duration_sec)
    signal = (np.sin(phase) * amplitude).astype(np.int16)
    return signal.tobytes()


def test_1_itut_p56_speech_level():
    print("=" * 80)
    print("TEST 1: ITU-T P.56 Active Speech Level Estimation & Activity Factor")
    print("=" * 80)
    fs = 8000
    estimator = ITUTP56SpeechLevelEstimator(sample_rate=fs, margin_db=15.9)

    # Generate 1.0s of active speech (sine tone at -12 dBov) followed by 1.0s silence
    t = np.linspace(0, 1.0, fs, endpoint=False)
    speech = np.sin(2 * np.pi * 200.0 * t) * 0.25  # ~ -12 dBov
    silence = np.random.normal(0, 0.001, fs)       # low ambient noise

    combined = np.concatenate([speech, silence]).astype(np.float32)

    level_db, activity_factor = estimator.process_samples(combined)

    print(f"[Measured] ITU-T P.56 Active Speech Level: {level_db:.2f} dBov")
    print(f"[Measured] Speech Activity Factor (p): {activity_factor:.3f}")

    assert -20.0 <= level_db <= -6.0, f"Expected active level in [-20, -6] dBov, got {level_db}"
    assert 0.35 <= activity_factor <= 0.65, f"Expected activity factor ~0.50, got {activity_factor}"
    print("[PASS] ITU-T P.56 speech level and activity factor verified.")


def test_2_falling_pitch_declination():
    print("\n" + "=" * 80)
    print("TEST 2: Pitch Declination & Parabolic NACF Tracking (Terminal Statement)")
    print("=" * 80)
    fs = 8000
    tracker = PitchDeclinationTracker(sample_rate=fs)

    # 400ms speech ending with falling pitch: 220 Hz down to 140 Hz (approx -4.5 semitones)
    duration = 0.40
    sweep_bytes = generate_sweep_pcm(220.0, 140.0, duration, sample_rate=fs, amplitude=14000.0)
    frame_len = int(fs * 0.02) * 2  # 20ms frames in bytes (320 bytes)

    final_trend = PitchTrend.UNVOICED
    final_delta_st = 0.0
    f0_measured = 0.0

    for i in range(0, len(sweep_bytes), frame_len):
        chunk = sweep_bytes[i : i + frame_len]
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        ts = i / (fs * 2)
        f0_measured, final_trend, final_delta_st = tracker.update(samples, ts)

    print(f"[Terminal Cadence] Measured Trailing F0: {f0_measured:.1f} Hz")
    print(f"[Terminal Cadence] Pitch Trend: {final_trend.value}")
    print(f"[Terminal Cadence] Total Pitch Delta: {final_delta_st:.2f} semitones")

    assert final_trend == PitchTrend.FALLING_DECLINATION, f"Expected FALLING_DECLINATION, got {final_trend}"
    assert final_delta_st <= -1.8, f"Expected pitch drop <= -1.8 st, got {final_delta_st}"
    print("[PASS] Falling pitch declination successfully categorized.")


def test_3_rising_pitch_continuation():
    print("\n" + "=" * 80)
    print("TEST 3: Rising Pitch Intonation (Query / Mid-Clause Floor Holding)")
    print("=" * 80)
    fs = 8000
    tracker = PitchDeclinationTracker(sample_rate=fs)

    # 400ms speech ending with rising pitch: 150 Hz up to 240 Hz (approx +4.2 semitones)
    duration = 0.40
    sweep_bytes = generate_sweep_pcm(150.0, 240.0, duration, sample_rate=fs, amplitude=14000.0)
    frame_len = int(fs * 0.02) * 2

    final_trend = PitchTrend.UNVOICED
    final_delta_st = 0.0

    for i in range(0, len(sweep_bytes), frame_len):
        chunk = sweep_bytes[i : i + frame_len]
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        ts = i / (fs * 2)
        _, final_trend, final_delta_st = tracker.update(samples, ts)

    print(f"[Continuation] Pitch Trend: {final_trend.value}")
    print(f"[Continuation] Total Pitch Delta: {final_delta_st:+.2f} semitones")

    assert final_trend == PitchTrend.RISING_CONTINUATION, f"Expected RISING_CONTINUATION, got {final_trend}"
    assert final_delta_st >= 1.4, f"Expected pitch rise >= +1.4 st, got {final_delta_st}"
    print("[PASS] Rising pitch continuation intonation detected.")


def test_4_short_time_energy_decay():
    print("\n" + "=" * 80)
    print("TEST 4: Short-Time Energy Decay Rate (Terminal Consonant Cutoff)")
    print("=" * 80)
    fs = 8000
    tracker = EnergyAndFluxTracker(sample_rate=fs, frame_duration_ms=20.0)

    # Simulate 3 frames of loud voice followed by sudden terminal decay (stop consonant)
    t = np.linspace(0, 0.02, 160, endpoint=False)
    frame_loud = (np.sin(2 * np.pi * 200.0 * t) * 0.8).astype(np.float32)
    frame_mid = (np.sin(2 * np.pi * 200.0 * t) * 0.2).astype(np.float32)
    frame_soft = (np.sin(2 * np.pi * 200.0 * t) * 0.02).astype(np.float32)

    tracker.process(frame_loud)
    tracker.process(frame_mid)
    energy_db, decay_rate, _, _ = tracker.process(frame_soft)

    print(f"[Decay Rate] Final Energy: {energy_db:.2f} dB")
    print(f"[Decay Rate] Energy Decay Rate: {decay_rate:.1f} dB/second")

    assert decay_rate <= -40.0, f"Expected steep terminal decay <= -40.0 dB/s, got {decay_rate}"
    print("[PASS] Rapid energy decay rate verified.")


def test_5_spectral_flux_and_inhalation():
    print("\n" + "=" * 80)
    print("TEST 5: Spectral Flux & Inhalation Breath Intake Detection")
    print("=" * 80)
    fs = 8000
    tracker = EnergyAndFluxTracker(sample_rate=fs, frame_duration_ms=20.0)

    # Baseline quiet frame
    quiet = np.random.normal(0, 0.001, 160).astype(np.float32)
    tracker.process(quiet)

    # Inhalation frame: mid-frequency noise (1.5kHz to 3.5kHz) at moderate level
    t = np.linspace(0, 0.02, 160, endpoint=False)
    # Bandpass white noise simulator
    breath = (
        np.sin(2 * np.pi * 2000.0 * t) * 0.015
        + np.sin(2 * np.pi * 2800.0 * t) * 0.015
        + np.sin(2 * np.pi * 3200.0 * t) * 0.012
    ).astype(np.float32)

    energy_db, _, flux, inhalation = tracker.process(breath)

    print(f"[Breath Intake] Frame Energy: {energy_db:.2f} dB")
    print(f"[Breath Intake] Spectral Flux: {flux:.4f}")
    print(f"[Breath Intake] Inhalation Detected: {inhalation}")

    assert inhalation is True, "Expected breath intake signature to be detected!"
    print("[PASS] Breath inhalation detection successfully identified.")


def test_6_dynamic_adaptive_silence_window():
    print("\n" + "=" * 80)
    print("TEST 6: Dynamic Variable Silence Window (Sub-120ms Fast vs 650ms Hold)")
    print("=" * 80)
    fs = 8000
    predictor = VoiceBoundaryPredictor(
        sample_rate=fs,
        fast_eot_threshold_ms=120.0,
        standard_eot_threshold_ms=350.0,
        hesitation_hold_threshold_ms=650.0,
    )

    # Scenario A: Completed declarative sentence (falling pitch + sudden energy stop)
    print("\n--- Sub-test 6A: Declarative Terminal Turn (Target: 120ms EoT) ---")
    speech_pcm = generate_sweep_pcm(240.0, 130.0, 0.30, sample_rate=fs, amplitude=14000.0)
    frame_bytes = int(fs * 0.02 * 2)

    # Feed speech frames
    for i in range(0, len(speech_pcm), frame_bytes):
        chunk = speech_pcm[i : i + frame_bytes]
        predictor.process_frame(chunk)

    # Now feed silence frames and measure exactly when EoT is triggered
    silence_frame = (np.random.normal(0, 10, 160)).astype(np.int16).tobytes()
    eot_triggered_at_ms = None

    for f_idx in range(1, 20):  # up to 380ms silence
        res = predictor.process_frame(silence_frame)
        if res.eot_triggered and eot_triggered_at_ms is None:
            eot_triggered_at_ms = res.trailing_silence_ms
            break

    print(f"[Declarative Terminal] EoT Probability: {res.eot_probability:.2f}")
    print(f"[Declarative Terminal] Adaptive Threshold: {res.adaptive_silence_threshold_ms:.1f} ms")
    print(f"[Declarative Terminal] EoT Triggered At: {eot_triggered_at_ms} ms trailing silence")
    print(f"[Declarative Terminal] Decision: {res.decision.value}")

    assert eot_triggered_at_ms is not None, "EoT should have triggered!"
    assert eot_triggered_at_ms <= 140.0, f"Expected fast EoT <= 140ms, triggered at {eot_triggered_at_ms}ms"
    assert res.decision == TurnBoundaryDecision.FAST_TERMINAL, f"Expected FAST_TERMINAL, got {res.decision}"
    print("[PASS] Sub-120ms fast turn boundary successfully committed.")

    # Scenario B: Mid-clause continuation with rising pitch (Target: 650ms hold)
    print("\n--- Sub-test 6B: Rising Pitch Continuation (Target: 650ms Floor Hold) ---")
    predictor.reset()
    continuation_pcm = generate_sweep_pcm(140.0, 240.0, 0.30, sample_rate=fs, amplitude=14000.0)

    for i in range(0, len(continuation_pcm), frame_bytes):
        chunk = continuation_pcm[i : i + frame_bytes]
        predictor.process_frame(chunk)

    # Feed 200ms silence - should NOT trigger EoT
    for _ in range(10):  # 200ms
        res_pause = predictor.process_frame(silence_frame)

    print(f"[Continuation Pause] Trailing Silence: {res_pause.trailing_silence_ms:.1f} ms")
    print(f"[Continuation Pause] EoT Probability: {res_pause.eot_probability:.2f}")
    print(f"[Continuation Pause] Adaptive Threshold: {res_pause.adaptive_silence_threshold_ms:.1f} ms")
    print(f"[Continuation Pause] EoT Triggered: {res_pause.eot_triggered}")
    print(f"[Continuation Pause] Decision: {res_pause.decision.value}")

    assert res_pause.eot_triggered is False, "EoT must NOT trigger at 200ms for rising pitch!"
    assert res_pause.adaptive_silence_threshold_ms >= 600.0, "Floor protection threshold should be >= 600ms!"
    assert res_pause.decision == TurnBoundaryDecision.HOLDING_FLOOR, f"Expected HOLDING_FLOOR, got {res_pause.decision}"
    print("[PASS] Floor protection verified; speaker held floor without interruption.")


def test_7_fastapi_endpoints():
    print("\n" + "=" * 80)
    print("TEST 7: FastAPI Endpoints (/health, /telephony/eot/analyze, /telephony/eot/benchmark)")
    print("=" * 80)
    from fastapi.testclient import TestClient
    from verbalyze.telephony.server import create_app

    app = create_app()
    client = TestClient(app)

    # 1. Test /health
    h_resp = client.get("/health")
    assert h_resp.status_code == 200, f"Health check failed: {h_resp.status_code}"
    h_data = h_resp.json()
    assert h_data.get("voice_boundary_predictor_status") == "ready", "Predictor not ready in health check!"
    print("[PASS] /health reports voice_boundary_predictor_status: ready")

    # 2. Test /telephony/eot/analyze
    test_pcm = generate_sweep_pcm(200.0, 140.0, 0.10, sample_rate=8000)
    b64_audio = base64.b64encode(test_pcm).decode("ascii")

    a_resp = client.post("/telephony/eot/analyze", json={"audio_base64": b64_audio, "sample_rate": 8000})
    assert a_resp.status_code == 200, f"Analyze failed: {a_resp.status_code} - {a_resp.text}"
    a_data = a_resp.json()
    assert a_data["status"] == "ok"
    assert a_data["total_frames"] == 5
    assert len(a_data["telemetry"]) == 5
    print(f"[PASS] /telephony/eot/analyze processed {a_data['total_frames']} frames successfully.")

    # 3. Test /telephony/eot/benchmark
    b_resp = client.post("/telephony/eot/benchmark")
    assert b_resp.status_code == 200, f"Benchmark failed: {b_resp.status_code}"
    b_data = b_resp.json()
    assert b_data["status"] == "ok"
    assert b_data["meets_eot_sla"] is True
    print(f"[PASS] /telephony/eot/benchmark: avg={b_data['avg_eot_time_ms']}ms, p95={b_data['p95_eot_time_ms']}ms, headroom={b_data['real_time_headroom_factor']}x")


def test_8_throughput_sub_0_1ms_benchmark():
    print("\n" + "=" * 80)
    print("TEST 8: Sub-0.1ms Pure-Math Execution Throughput Benchmark")
    print("=" * 80)
    fs = 8000
    predictor = VoiceBoundaryPredictor(sample_rate=fs)

    # 160-sample (20ms) PCM frame
    t = np.linspace(0, 0.02, 160, endpoint=False)
    frame_pcm = (np.sin(2 * np.pi * 200.0 * t) * 12000.0).astype(np.int16).tobytes()

    # Warm-up
    for _ in range(10):
        predictor.process_frame(frame_pcm)

    iterations = 200
    durations_ms = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        res = predictor.process_frame(frame_pcm)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(durations_ms))
    p95_ms = float(np.percentile(durations_ms, 95))
    max_ms = float(np.max(durations_ms))
    headroom = 20.0 / max(mean_ms, 1e-4)

    print(f"[Benchmark] Iterations: {iterations} frames (20ms each)")
    print(f"[Benchmark] Mean Processing Time: {mean_ms:.4f} ms")
    print(f"[Benchmark] 95th Percentile:      {p95_ms:.4f} ms")
    print(f"[Benchmark] Maximum Observed:     {max_ms:.4f} ms")
    print(f"[Benchmark] Real-Time Headroom:   {headroom:.1f}x real-time speed")
    print(f"[Benchmark] Target SLA (<0.10ms): {'PASS' if mean_ms < 0.10 else 'FAIL'}")

    assert mean_ms < 0.10, f"Expected mean execution time < 0.10ms, got {mean_ms:.4f}ms"
    assert p95_ms < 0.20, f"Expected p95 execution time < 0.20ms, got {p95_ms:.4f}ms"
    print("[PASS] Sub-0.1ms pure-math throughput verified.")


def run_all_eot_tests():
    print("\n" + "=" * 80)
    print("VERBALYZE: ACOUSTIC END-OF-TURN (EoT) PREDICTOR TEST SUITE")
    print("=" * 80)
    t0 = time.perf_counter()

    test_1_itut_p56_speech_level()
    test_2_falling_pitch_declination()
    test_3_rising_pitch_continuation()
    test_4_short_time_energy_decay()
    test_5_spectral_flux_and_inhalation()
    test_6_dynamic_adaptive_silence_window()
    test_7_fastapi_endpoints()
    test_8_throughput_sub_0_1ms_benchmark()

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"ALL 8 END-OF-TURN VOICE BOUNDARY PREDICTOR TESTS PASSED IN {total_time:.2f}s!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_eot_tests()
