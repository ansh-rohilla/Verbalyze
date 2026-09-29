#!/usr/bin/env python3
"""
scripts/test_backchannel_injector.py

Comprehensive Verification Suite for Verbalyze Sub-Conscious Acoustic Backchannel Injector:
Intra-Turn Micro-Pause Detection, Pitch-Gating, Instant Soft-Ducking, and Synthetic Murmurs.

Tests:
1. Speech Burst Accumulation & Silence Suppression.
2. Intra-Turn Micro-Pause Opportunity Trigger (140ms-380ms gap).
3. Suppression on Insufficient Speech Burst (< 1600ms).
4. Suppression on Extended Silence (Pass-Through to Main VAD).
5. Instant Soft-Ducking When Caller Resumes Speaking (<5ms).
6. Cadence Refractory Cooldown Rate Limiting (3500ms).
7. Synthetic Murmur Soundbank Generation (HMM, HAAN_HAAN, JI, ACHHA).
8. Multilingual Linguistic Persona Selection (Hindi, Tamil, English).
9. FastAPI Endpoints (/health, /telephony/backchannel/process, /telephony/backchannel/benchmark).
10. Sub-0.05ms Pure-Math Execution SLA Benchmark.

Zero-emoji compliant.
"""

import sys
import time
import math
import base64
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.telephony.backchannel_injector import (
    SubconsciousBackchannelInjector,
    BackchannelType,
    BackchannelState,
    BackchannelTelemetry,
)


def generate_speech_frame_pcm(sample_rate: int = 8000, duration_sec: float = 0.02, freq_hz: float = 160.0, amplitude: float = 10000.0) -> bytes:
    """Generates synthetic 20ms speech audio frame."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    samples = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.int16)
    return samples.tobytes()


def generate_silence_frame_pcm(sample_rate: int = 8000, duration_sec: float = 0.02) -> bytes:
    """Generates synthetic 20ms silence frame."""
    n_samples = int(sample_rate * duration_sec)
    return b"\x00" * (n_samples * 2)


def test_1_speech_burst_accumulation():
    print("\n" + "=" * 80)
    print("TEST 1: Speech Burst Accumulation & Active Speech Tracking")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000)

    speech_frame = generate_speech_frame_pcm(amplitude=12000.0)

    # Feed 80 frames = 1600ms of active speech
    for _ in range(80):
        out_pcm, telem = injector.process_frame(speech_frame)
        assert telem.caller_speaking is True
        assert len(out_pcm) == 320
        # No backchannel should ever be outputted while caller is actively speaking
        assert telem.backchannel_active is False

    print(f"[Speech Accumulation] Accumulated Duration: {injector.speech_burst_ms:.1f} ms")
    print(f"[Speech Accumulation] Silence Gap:          {injector.silence_gap_ms:.1f} ms")
    print(f"[Speech Accumulation] State:                {injector.state.value}")

    assert injector.speech_burst_ms >= 1600.0
    assert injector.silence_gap_ms == 0.0
    assert injector.state == BackchannelState.MONITORING
    print("[PASS] Speech burst accumulation and non-interference during active speech verified.")


def test_2_intra_turn_pause_opportunity_trigger():
    print("\n" + "=" * 80)
    print("TEST 2: Intra-Turn Micro-Pause Opportunity Trigger (140ms - 380ms gap)")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000, min_speech_burst_ms=1600.0)

    speech_frame = generate_speech_frame_pcm(amplitude=12000.0)
    silence_frame = generate_silence_frame_pcm()

    # 1. Feed 100 frames (2000ms) of continuous speech
    for _ in range(100):
        injector.process_frame(speech_frame)

    assert injector.speech_burst_ms >= 2000.0

    # 2. Feed silence frames (simulating breath pause)
    # Pause starts: 20ms, 40ms, 60ms, 80ms, 100ms, 120ms (no trigger yet, threshold is 140ms)
    for _ in range(6):
        _, telem = injector.process_frame(silence_frame)
        assert telem.opportunity_detected is False

    # 3. Frame at 140ms-160ms should trigger opportunity
    _, telem_trigger = injector.process_frame(silence_frame)
    _, telem_active = injector.process_frame(silence_frame)

    print(f"[Pause Trigger] Opportunity Detected: {telem_trigger.opportunity_detected}")
    print(f"[Pause Trigger] State:                {telem_trigger.state.value}")
    print(f"[Pause Trigger] Backchannel Type:     {telem_active.backchannel_type.value if telem_active.backchannel_type else None}")
    print(f"[Pause Trigger] Audio Active:         {telem_active.backchannel_active}")
    print(f"[Pause Trigger] Injected Audio RMS:   {telem_active.injected_audio_rms_db:.2f} dBov")

    assert telem_trigger.opportunity_detected is True
    assert telem_trigger.state == BackchannelState.INJECTING
    assert telem_active.backchannel_active is True
    assert telem_active.injected_audio_rms_db > -40.0
    print("[PASS] Intra-turn micro-pause detected and backchannel injection triggered.")


def test_3_suppression_on_insufficient_speech():
    print("\n" + "=" * 80)
    print("TEST 3: Suppression on Insufficient Speech Burst (< 1600ms)")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000, min_speech_burst_ms=1600.0)

    speech_frame = generate_speech_frame_pcm()
    silence_frame = generate_silence_frame_pcm()

    # Feed only 20 frames = 400ms of speech (e.g. short 1-word answer)
    for _ in range(20):
        injector.process_frame(speech_frame)

    # Feed 10 frames = 200ms of pause
    triggered = False
    for _ in range(10):
        _, telem = injector.process_frame(silence_frame)
        if telem.opportunity_detected:
            triggered = True

    print(f"[Insufficient Burst] Burst Duration: {injector.speech_burst_ms} ms (Min: 1600 ms)")
    print(f"[Insufficient Burst] Triggered:      {triggered} (Expected: False)")

    assert triggered is False
    assert injector.state == BackchannelState.MONITORING
    print("[PASS] Backchannel correctly suppressed for short non-narrative utterances.")


def test_4_suppression_on_extended_silence():
    print("\n" + "=" * 80)
    print("TEST 4: Suppression on Extended Silence (Pass-Through to Main VAD)")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000, max_pause_window_ms=380.0)
    silence_frame = generate_silence_frame_pcm()

    # Pre-seed speech burst
    injector.speech_burst_ms = 2500.0
    # Long pause of 500ms (> max pause window of 380ms)
    injector.silence_gap_ms = 500.0

    _, telem = injector.process_frame(silence_frame)
    print(f"[Extended Silence] Silence Gap:          {telem.silence_gap_duration_ms} ms")
    print(f"[Extended Silence] Opportunity Detected: {telem.opportunity_detected} (Expected: False)")

    assert telem.opportunity_detected is False
    print("[PASS] Backchannel correctly suppressed on extended silence to avoid turn-taking conflict.")


def test_5_instant_soft_ducking_when_caller_resumes():
    print("\n" + "=" * 80)
    print("TEST 5: Instant Soft-Ducking When Caller Resumes Speaking (<5ms)")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000)

    # 1. Trigger backchannel manually
    injector.trigger_backchannel(BackchannelType.HAAN_HAAN)
    assert injector.state == BackchannelState.INJECTING

    # 2. Render first frame while caller is still silent
    silence_frame = generate_silence_frame_pcm()
    out_silent, telem_pre = injector.process_frame(silence_frame)
    rms_pre = telem_pre.injected_audio_rms_db
    print(f"[Ducking Pre]  Injected Audio RMS: {rms_pre:.2f} dBov")

    # 3. Caller suddenly resumes speaking loudly!
    loud_speech_frame = generate_speech_frame_pcm(amplitude=14000.0)
    out_ducked, telem_duck = injector.process_frame(loud_speech_frame)
    rms_ducked = telem_duck.injected_audio_rms_db

    print(f"[Ducking Post] State:              {telem_duck.state.value}")
    print(f"[Ducking Post] Ducking Applied:    {telem_duck.ducking_applied}")
    print(f"[Ducking Post] Injected Audio RMS: {rms_ducked:.2f} dBov")
    print(f"[Ducking Post] Attenuation Drop:   {rms_pre - rms_ducked:.2f} dB")

    assert telem_duck.state == BackchannelState.DUCKING
    assert telem_duck.ducking_applied is True
    # Injected murmur must be ducked by at least 8 dB immediately
    assert rms_ducked < rms_pre - 8.0
    print("[PASS] Instant soft-ducking successfully eliminated potential speaker collision.")


def test_6_cadence_refractory_cooldown():
    print("\n" + "=" * 80)
    print("TEST 6: Cadence Refractory Cooldown Rate Limiting (3500ms)")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000, cooldown_duration_ms=3500.0)

    # Trigger first affirmation
    injector.trigger_backchannel(BackchannelType.HMM)
    print(f"[Cooldown] Initial Cooldown Set: {injector.cooldown_ms:.1f} ms")
    assert injector.cooldown_ms == 3500.0

    # Advance 50 frames = 1000ms while pretending caller continues narrative
    speech_frame = generate_speech_frame_pcm()
    for _ in range(50):
        injector.process_frame(speech_frame)

    print(f"[Cooldown] Cooldown After 1s:    {injector.cooldown_ms:.1f} ms")
    assert injector.cooldown_ms <= 2600.0

    # Simulate another pause during cooldown period
    injector.silence_gap_ms = 200.0
    injector.speech_burst_ms = 2000.0
    _, telem = injector.process_frame(generate_silence_frame_pcm())

    print(f"[Cooldown] Opportunity Triggered in Cooldown: {telem.opportunity_detected} (Expected: False)")
    assert telem.opportunity_detected is False
    print("[PASS] Cadence refractory cooldown prevented unnatural over-affirmation.")


def test_7_synthetic_murmur_soundbank_generation():
    print("\n" + "=" * 80)
    print("TEST 7: Synthetic Murmur Soundbank Generation (HMM, HAAN_HAAN, JI, ACHHA)")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000)

    for b_type in [BackchannelType.HMM, BackchannelType.HAAN_HAAN, BackchannelType.JI, BackchannelType.ACHHA, BackchannelType.RIGHT]:
        murmur = injector._murmur_bank[b_type]
        duration_ms = (len(murmur) / 8000.0) * 1000.0
        peak = float(np.max(np.abs(murmur)))
        rms_db = 20.0 * math.log10(max(1e-5, float(np.sqrt(np.mean(murmur ** 2)))))

        print(f"[Soundbank] {b_type.value:12s}: Samples={len(murmur):4d}, Duration={duration_ms:5.1f}ms, Peak={peak:.3f}, RMS={rms_db:.2f}dBov")

        assert len(murmur) > 800
        assert peak <= 1.0
        assert -30.0 <= rms_db <= -10.0
        assert not np.isnan(murmur).any()

    print("[PASS] Pure-math acoustic murmur soundbank generation verified.")


def test_8_multilingual_persona_selection():
    print("\n" + "=" * 80)
    print("TEST 8: Multilingual Linguistic Persona Selection (Hindi, Tamil, English)")
    print("=" * 80)
    inj_hi = SubconsciousBackchannelInjector(language="hi")
    inj_ta = SubconsciousBackchannelInjector(language="ta")
    inj_en = SubconsciousBackchannelInjector(language="en")

    type_hi = inj_hi._select_affirmation_type()
    type_ta = inj_ta._select_affirmation_type()
    type_en = inj_en._select_affirmation_type()

    print(f"[Linguistics] Hindi Type:   {type_hi.value}")
    print(f"[Linguistics] Tamil Type:   {type_ta.value}")
    print(f"[Linguistics] English Type: {type_en.value}")

    assert type_hi in (BackchannelType.HAAN_HAAN, BackchannelType.HMM, BackchannelType.JI, BackchannelType.ACHHA)
    assert type_ta in (BackchannelType.HMM, BackchannelType.JI, BackchannelType.HAAN_HAAN)
    assert type_en in (BackchannelType.HMM, BackchannelType.RIGHT, BackchannelType.HAAN_HAAN)
    print("[PASS] Multilingual linguistic persona selection verified.")


def test_9_fastapi_backchannel_endpoints():
    print("\n" + "=" * 80)
    print("TEST 9: FastAPI Telephony Backchannel Endpoints (/health, /process, /benchmark)")
    print("=" * 80)
    from fastapi.testclient import TestClient
    from verbalyze.telephony.server import create_app

    app = create_app()
    client = TestClient(app)

    # 1. Health check verification
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data.get("backchannel_injector_status") == "ready", "Backchannel injector not ready in health check!"
    print("[PASS] /health reports backchannel_injector_status: ready")

    # 2. Process audio frame endpoint with simulated pause
    silence_pcm = generate_silence_frame_pcm()
    p_resp = client.post("/telephony/backchannel/process", json={
        "caller_pcm_base64": base64.b64encode(silence_pcm).decode("ascii"),
        "language": "hi",
        "simulate_speech_burst_ms": 2200.0,
        "simulate_silence_gap_ms": 160.0,
    })
    assert p_resp.status_code == 200
    p_data = p_resp.json()
    assert p_data["status"] == "ok"
    assert "backchannel_pcm_base64" in p_data
    assert p_data["telemetry"]["opportunity_detected"] is True
    print("[PASS] /telephony/backchannel/process successfully triggered backchannel opportunity.")

    # 3. Benchmark endpoint
    b_resp = client.post("/telephony/backchannel/benchmark", json={})
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert b_data["status"] == "ok"
    assert b_data["meets_sla"] is True
    print(f"[PASS] /telephony/backchannel/benchmark: avg={b_data['avg_backchannel_time_ms']}ms, SLA={b_data['target_sla_ms']}ms, Headroom={b_data['real_time_headroom_factor']}x")


def test_10_sub_005ms_execution_latency_benchmark():
    print("\n" + "=" * 80)
    print("TEST 10: Sub-0.05ms Pure-Math Execution SLA Benchmark")
    print("=" * 80)
    injector = SubconsciousBackchannelInjector(sample_rate=8000, language="hi")
    speech_frame = generate_speech_frame_pcm()

    # Warm-up
    injector.process_frame(speech_frame)

    n_iterations = 200
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        injector.process_frame(speech_frame)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))
    headroom = 20.0 / max(avg_ms, 1e-4)

    print(f"[Benchmark] Iterations:          {n_iterations} frames (20ms each)")
    print(f"[Benchmark] Mean Execution Time: {avg_ms:.4f} ms")
    print(f"[Benchmark] 95th Percentile:     {p95_ms:.4f} ms")
    print(f"[Benchmark] Maximum Observed:    {max_ms:.4f} ms")
    print(f"[Benchmark] Real-Time Headroom:  {headroom:.1f}x real-time speed")
    print(f"[Benchmark] Target SLA (<0.05ms): {'PASS' if avg_ms < 0.05 else 'FAIL'}")

    assert avg_ms < 0.05, f"Mean execution time {avg_ms:.4f}ms exceeded 0.05ms SLA"
    print("[PASS] Sub-0.05ms backchannel execution throughput verified.")


def run_all_backchannel_tests():
    t_start = time.perf_counter()
    print("\n" + "=" * 80)
    print("VERBALYZE: SUBCONSCIOUS ACOUSTIC BACKCHANNEL INJECTOR TEST SUITE")
    print("=" * 80)

    test_1_speech_burst_accumulation()
    test_2_intra_turn_pause_opportunity_trigger()
    test_3_suppression_on_insufficient_speech()
    test_4_suppression_on_extended_silence()
    test_5_instant_soft_ducking_when_caller_resumes()
    test_6_cadence_refractory_cooldown()
    test_7_synthetic_murmur_soundbank_generation()
    test_8_multilingual_persona_selection()
    test_9_fastapi_backchannel_endpoints()
    test_10_sub_005ms_execution_latency_benchmark()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL 10 BACKCHANNEL INJECTOR TESTS PASSED IN {total_time:.2f}s!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_backchannel_tests()
