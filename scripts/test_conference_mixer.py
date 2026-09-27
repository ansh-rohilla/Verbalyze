#!/usr/bin/env python3
"""
scripts/test_conference_mixer.py

Comprehensive Test Suite for 3-Way Soft-Switch Telephony Audio Mixer (ITU-T G.115):
1. Silent Monitor Mode (Supervisor listens to both; uplink muted to both).
2. Whisper Coach Mode (Supervisor coaches Agent only; Customer hears nothing).
3. Hard Takeover Mode (Supervisor takes over call; Agent uplink muted to Customer).
4. Three-Way Conference Bridge (All 3 parties mixed with attenuation).
5. Click-Free Gain Crossfading across mode transitions.
6. Soft Saturation Peak Limiting under high-volume multi-speaker talk-over.
7. Custom Gain Routing Matrix configuration.
8. FastAPI REST Endpoints (/health, /route, /mix, /benchmark).
9. Sub-0.1ms Execution Throughput & Headroom Benchmark.

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

from verbalyze.telephony.conference_mixer import (
    ConferenceAudioMixer,
    ConferenceMode,
    ChannelMixResult,
)


def generate_tone_pcm(freq_hz: float, duration_sec: float = 0.02, sample_rate: int = 8000, amplitude: float = 10000.0) -> bytes:
    """Generates pure sine tone linear 16-bit PCM bytes."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    signal = (np.sin(2.0 * np.pi * freq_hz * t) * amplitude).astype(np.int16)
    return signal.tobytes()


def pcm_energy(pcm_bytes: bytes) -> float:
    """Calculates RMS energy of PCM bytes."""
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def test_1_silent_monitor_mode():
    print("=" * 80)
    print("TEST 1: Silent Monitor Mode Routing (Supervisor Listens Only)")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000, initial_mode=ConferenceMode.SILENT_MONITOR)

    # Audio sources: Customer = 300Hz tone, Agent = 500Hz tone, Supervisor = 800Hz tone
    pcm_c = generate_tone_pcm(300.0, amplitude=12000.0)
    pcm_a = generate_tone_pcm(500.0, amplitude=12000.0)
    pcm_s = generate_tone_pcm(800.0, amplitude=12000.0)

    res = mixer.process_frame(customer_pcm=pcm_c, agent_pcm=pcm_a, supervisor_pcm=pcm_s)

    energy_c_out = pcm_energy(res.customer_out_pcm)
    energy_a_out = pcm_energy(res.agent_out_pcm)
    energy_s_out = pcm_energy(res.supervisor_out_pcm)

    print(f"[Silent Monitor] Customer Output Energy:   {energy_c_out:.1f} (hears Agent only)")
    print(f"[Silent Monitor] Agent Output Energy:      {energy_a_out:.1f} (hears Customer only)")
    print(f"[Silent Monitor] Supervisor Output Energy: {energy_s_out:.1f} (hears Customer + Agent)")

    # 1. Customer must hear Agent (non-zero energy), but NO supervisor audio
    assert energy_c_out > 5000.0, "Customer should hear Agent audio!"
    
    # 2. Agent must hear Customer (non-zero energy), but NO supervisor audio
    assert energy_a_out > 5000.0, "Agent should hear Customer audio!"

    # 3. Supervisor must hear both (combined energy)
    assert energy_s_out > 7000.0, "Supervisor should hear mixed Customer + Agent!"

    # 4. Verify supervisor is completely silent to Customer when Agent is silent
    res_silent_agent = mixer.process_frame(customer_pcm=None, agent_pcm=None, supervisor_pcm=pcm_s)
    energy_c_when_supervisor_talks = pcm_energy(res_silent_agent.customer_out_pcm)
    energy_a_when_supervisor_talks = pcm_energy(res_silent_agent.agent_out_pcm)

    print(f"[Silent Monitor] Customer hears Supervisor: {energy_c_when_supervisor_talks:.2f} (Expected: 0.0)")
    print(f"[Silent Monitor] Agent hears Supervisor:    {energy_a_when_supervisor_talks:.2f} (Expected: 0.0)")

    assert energy_c_when_supervisor_talks == 0.0, "Customer must hear zero audio from supervisor in silent monitor!"
    assert energy_a_when_supervisor_talks == 0.0, "Agent must hear zero audio from supervisor in silent monitor!"
    print("[PASS] Silent monitor mode routing verified.")


def test_2_whisper_coach_mode():
    print("\n" + "=" * 80)
    print("TEST 2: Whisper Coach Mode Routing (Supervisor Coaches Agent Only)")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000, initial_mode=ConferenceMode.WHISPER_COACH)

    pcm_c = generate_tone_pcm(300.0, amplitude=10000.0)
    pcm_a = generate_tone_pcm(500.0, amplitude=10000.0)
    pcm_s = generate_tone_pcm(800.0, amplitude=10000.0)

    # Scenario: Supervisor gives a whispered coaching prompt while Customer is speaking
    res = mixer.process_frame(customer_pcm=pcm_c, agent_pcm=None, supervisor_pcm=pcm_s)

    energy_c_out = pcm_energy(res.customer_out_pcm)
    energy_a_out = pcm_energy(res.agent_out_pcm)
    energy_s_out = pcm_energy(res.supervisor_out_pcm)

    print(f"[Whisper Coach] Customer Output Energy:   {energy_c_out:.2f} (Expected: 0.0)")
    print(f"[Whisper Coach] Agent Output Energy:      {energy_a_out:.1f} (hears Customer + Supervisor)")
    print(f"[Whisper Coach] Supervisor Output Energy: {energy_s_out:.1f} (hears Customer)")

    # 1. Customer must hear absolutely NOTHING from the supervisor whisper
    assert energy_c_out == 0.0, "Customer must NEVER hear supervisor coaching prompts!"

    # 2. Agent must hear both Customer and the whispered instruction
    assert energy_a_out > 7000.0, "Agent must hear supervisor coaching audio!"

    # 3. Supervisor hears the customer
    assert energy_s_out > 5000.0, "Supervisor should hear customer audio!"
    print("[PASS] Whisper coach mode verified; privacy isolation between supervisor and customer confirmed.")


def test_3_hard_takeover_mode():
    print("\n" + "=" * 80)
    print("TEST 3: Hard Takeover Mode Routing (Supervisor Intervenes Live)")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000, initial_mode=ConferenceMode.HARD_TAKEOVER)

    pcm_c = generate_tone_pcm(300.0, amplitude=10000.0)
    pcm_a = generate_tone_pcm(500.0, amplitude=10000.0)
    pcm_s = generate_tone_pcm(800.0, amplitude=10000.0)

    res = mixer.process_frame(customer_pcm=pcm_c, agent_pcm=pcm_a, supervisor_pcm=pcm_s)

    # Customer should hear Supervisor, and Agent must be muted to Customer
    res_agent_only = mixer.process_frame(customer_pcm=None, agent_pcm=pcm_a, supervisor_pcm=None)
    energy_c_from_agent = pcm_energy(res_agent_only.customer_out_pcm)

    res_sup_only = mixer.process_frame(customer_pcm=None, agent_pcm=None, supervisor_pcm=pcm_s)
    energy_c_from_sup = pcm_energy(res_sup_only.customer_out_pcm)

    print(f"[Hard Takeover] Customer hears Agent:      {energy_c_from_agent:.2f} (Expected: 0.0 - Muted)")
    print(f"[Hard Takeover] Customer hears Supervisor: {energy_c_from_sup:.1f} (Expected: Direct Speech)")

    assert energy_c_from_agent == 0.0, "Agent audio must be muted to Customer in hard takeover!"
    assert energy_c_from_sup > 5000.0, "Customer must hear Supervisor directly in hard takeover!"
    print("[PASS] Hard takeover routing verified.")


def test_4_three_way_conference():
    print("\n" + "=" * 80)
    print("TEST 4: Three-Way Conference Bridge & Multi-Party Talk-Over")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000, initial_mode=ConferenceMode.THREE_WAY_CONFERENCE)

    pcm_c = generate_tone_pcm(300.0, amplitude=9000.0)
    pcm_a = generate_tone_pcm(500.0, amplitude=9000.0)
    pcm_s = generate_tone_pcm(700.0, amplitude=9000.0)

    res = mixer.process_frame(customer_pcm=pcm_c, agent_pcm=pcm_a, supervisor_pcm=pcm_s)

    print(f"[3-Way Bridge] Customer Speaking:   {res.customer_speaking}")
    print(f"[3-Way Bridge] Agent Speaking:      {res.agent_speaking}")
    print(f"[3-Way Bridge] Supervisor Speaking: {res.supervisor_speaking}")
    print(f"[3-Way Bridge] Cross-Talk Detected: {res.cross_talk_detected}")

    assert res.customer_speaking is True, "Customer should be detected as speaking"
    assert res.agent_speaking is True, "Agent should be detected as speaking"
    assert res.supervisor_speaking is True, "Supervisor should be detected as speaking"
    assert res.cross_talk_detected is True, "Cross-talk must be detected when multiple parties speak"

    # All parties receive non-zero mixed audio
    assert pcm_energy(res.customer_out_pcm) > 4000.0
    assert pcm_energy(res.agent_out_pcm) > 4000.0
    assert pcm_energy(res.supervisor_out_pcm) > 4000.0
    print("[PASS] Three-way conference bridging and cross-talk detection verified.")


def test_5_smooth_crossfade():
    print("\n" + "=" * 80)
    print("TEST 5: Smooth Click-Free Gain Crossfading across Mode Transitions")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000, initial_mode=ConferenceMode.SILENT_MONITOR)

    # Initial frame in silent monitor
    pcm_s = generate_tone_pcm(800.0, amplitude=12000.0)
    res1 = mixer.process_frame(supervisor_pcm=pcm_s)
    assert res1.crossfade_active is False

    # Trigger transition to Hard Takeover
    mixer.set_mode(ConferenceMode.HARD_TAKEOVER)

    # First frame after mode change must have active crossfade
    res2 = mixer.process_frame(supervisor_pcm=pcm_s)
    print(f"[Crossfade] Transition Frame Crossfade Active: {res2.crossfade_active}")
    assert res2.crossfade_active is True, "Crossfade must be active during gain transition!"

    # Subsequent frame should have completed crossfade
    res3 = mixer.process_frame(supervisor_pcm=pcm_s)
    print(f"[Crossfade] Steady-State Frame Crossfade Active: {res3.crossfade_active}")
    assert res3.crossfade_active is False, "Crossfade should settle after transition frame"
    print("[PASS] Smooth gain crossfade verified.")


def test_6_soft_peak_limiter():
    print("\n" + "=" * 80)
    print("TEST 6: Soft Saturation Peak Limiting under Multi-Speaker Talk-Over")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000, initial_mode=ConferenceMode.THREE_WAY_CONFERENCE)

    # Extremely loud tones on Agent and Supervisor summing to > 1.8 full scale
    pcm_c = generate_tone_pcm(300.0, amplitude=28000.0)
    pcm_a = generate_tone_pcm(500.0, amplitude=28000.0)
    pcm_s = generate_tone_pcm(700.0, amplitude=28000.0)

    res = mixer.process_frame(customer_pcm=pcm_c, agent_pcm=pcm_a, supervisor_pcm=pcm_s)

    samples_c = np.frombuffer(res.customer_out_pcm, dtype=np.int16).astype(np.float32) / 32767.0
    max_peak = float(np.max(np.abs(samples_c)))

    print(f"[Peak Limiter] Maximum Peak Output: {max_peak:.4f} (Must be <= 1.0)")
    print(f"[Peak Limiter] Clipping Prevented Flag: {res.clipping_prevented}")

    assert max_peak <= 1.0, f"Peak exceeded full scale [-1.0, 1.0]: {max_peak}"
    assert res.clipping_prevented is True, "Soft limiter should flag clipping_prevented when summing loud speakers"
    print("[PASS] Soft-saturation peak limiter successfully prevented digital clipping.")


def test_7_custom_gain_matrix():
    print("\n" + "=" * 80)
    print("TEST 7: Custom Gain Routing Matrix Configuration")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000)

    # Custom gains: g_CA=0.5, g_CS=0.2, g_AC=0.8, g_AS=0.3, g_SC=0.9, g_SA=0.4
    custom = (0.5, 0.2, 0.8, 0.3, 0.9, 0.4)
    mixer.set_mode(ConferenceMode.CUSTOM_MATRIX, custom_gains=custom)

    gains = mixer.get_gains()
    print(f"[Custom Matrix] Configured Gains: {gains}")

    assert gains["g_CA"] == 0.5
    assert gains["g_CS"] == 0.2
    assert gains["g_AC"] == 0.8
    assert gains["g_AS"] == 0.3
    assert gains["g_SC"] == 0.9
    assert gains["g_SA"] == 0.4
    print("[PASS] Custom gain matrix configuration verified.")


def test_8_fastapi_conference_endpoints():
    print("\n" + "=" * 80)
    print("TEST 8: FastAPI Telephony Conference Endpoints (/health, /route, /mix, /benchmark)")
    print("=" * 80)
    from fastapi.testclient import TestClient
    from verbalyze.telephony.server import create_app

    app = create_app()
    client = TestClient(app)

    # 1. Health check verification
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data.get("conference_mixer_status") == "ready", "Mixer not ready in health check!"
    print("[PASS] /health reports conference_mixer_status: ready")

    # 2. Route configuration endpoint
    r_resp = client.post("/telephony/conference/route", json={"mode": "whisper_coach"})
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert r_data["mode"] == "WHISPER_COACH"
    assert r_data["gains"]["g_AS"] == 1.0
    assert r_data["gains"]["g_CS"] == 0.0
    print("[PASS] /telephony/conference/route successfully configured whisper coach mode.")

    # 3. Audio frame mixing endpoint
    pcm_c = generate_tone_pcm(300.0, amplitude=8000.0)
    pcm_a = generate_tone_pcm(500.0, amplitude=8000.0)
    pcm_s = generate_tone_pcm(800.0, amplitude=8000.0)

    mix_resp = client.post("/telephony/conference/mix", json={
        "mode": "three_way_conference",
        "sample_rate": 8000,
        "customer_pcm_base64": base64.b64encode(pcm_c).decode("ascii"),
        "agent_pcm_base64": base64.b64encode(pcm_a).decode("ascii"),
        "supervisor_pcm_base64": base64.b64encode(pcm_s).decode("ascii"),
    })
    assert mix_resp.status_code == 200
    m_data = mix_resp.json()
    assert m_data["status"] == "ok"
    assert "customer_out_base64" in m_data
    assert m_data["telemetry"]["mode"] == "THREE_WAY_CONFERENCE"
    print("[PASS] /telephony/conference/mix successfully mixed 3-channel frame.")

    # 4. Benchmark endpoint
    b_resp = client.post("/telephony/conference/benchmark")
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert b_data["status"] == "ok"
    assert b_data["meets_mixer_sla"] is True
    print(f"[PASS] /telephony/conference/benchmark: avg={b_data['avg_mixer_time_ms']}ms, headroom={b_data['real_time_headroom_factor']}x")


def test_9_sub_0_1ms_performance_benchmark():
    print("\n" + "=" * 80)
    print("TEST 9: Sub-0.1ms Pure-Math Execution Throughput Benchmark")
    print("=" * 80)
    mixer = ConferenceAudioMixer(sample_rate=8000, initial_mode=ConferenceMode.THREE_WAY_CONFERENCE)

    pcm_c = generate_tone_pcm(300.0, amplitude=10000.0)
    pcm_a = generate_tone_pcm(500.0, amplitude=10000.0)
    pcm_s = generate_tone_pcm(700.0, amplitude=10000.0)

    # Warm-up
    for _ in range(10):
        mixer.process_frame(pcm_c, pcm_a, pcm_s)

    iterations = 200
    durations_ms = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        mixer.process_frame(pcm_c, pcm_a, pcm_s)
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
    print("[PASS] Sub-0.1ms pure-math execution throughput verified.")


def run_all_conference_tests():
    print("\n" + "=" * 80)
    print("VERBALYZE: 3-WAY TELEPHONY CONFERENCE MIXER TEST SUITE")
    print("=" * 80)
    t0 = time.perf_counter()

    test_1_silent_monitor_mode()
    test_2_whisper_coach_mode()
    test_3_hard_takeover_mode()
    test_4_three_way_conference()
    test_5_smooth_crossfade()
    test_6_soft_peak_limiter()
    test_7_custom_gain_matrix()
    test_8_fastapi_conference_endpoints()
    test_9_sub_0_1ms_performance_benchmark()

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"ALL 9 CONFERENCE MIXER TESTS PASSED IN {total_time:.2f}s!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_conference_tests()
