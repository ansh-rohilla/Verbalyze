#!/usr/bin/env python3
"""
scripts/test_in_band_disconnect_gate.py

Test Suite for In-Band Disconnect & Busy-Cadence Call Termination Gate.
Validates ITU-T E.180 / Q.35 telecom disconnect detection, fast busy cadences,
continuous howler tones, speech immunity, and sub-second zombie call teardown:
1. Pure 400Hz / 425Hz Disconnect Tone DTFT Basis Projection.
2. Indian Standard 375ms Busy / Disconnect Cadence Detection.
3. Slow Busy 750ms Cadence Detection.
4. Fast Network Congestion 200ms Reorder Tone Detection.
5. Continuous Off-Hook Howler Tone (> 1200ms) Detection.
6. Special Information Tone (SIT) Frequency Recognition.
7. Conversational Speech Immunity (Zero False Hangups).
8. Stateful SIP Session Orchestrator Auto-Teardown on Disconnect.
9. FastAPI REST Telephony Endpoints (/analyze, /benchmark, /health).
10. Ultra-Low Latency Benchmark (< 0.030 ms SLA per 20ms frame, > 650x headroom).

Target SLA: Latency < 0.030 ms per 20 ms frame | Headroom > 650x
Zero-emoji compliant.
"""

import base64
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.telephony.disconnect_gate import (
    InBandDisconnectGate,
    DisconnectPattern,
    DisconnectState,
    DisconnectTelemetry,
    DisconnectReport,
)
from verbalyze.telephony.sip_orchestrator import (
    SIPSession,
    SIPCallState,
    CallLegRole,
)
from verbalyze.telephony.server import create_app


def generate_tone_frame(
    freq: float,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
    amp: float = 16000.0,
) -> bytes:
    """Generates pure sine tone linear PCM frame."""
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    sig = np.sin(2.0 * np.pi * freq * t) * amp
    return sig.astype(np.int16).tobytes()


def generate_speech_frame(
    f0: float = 140.0,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
    amp: float = 16000.0,
) -> bytes:
    """Generates harmonic voiced speech frame."""
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    speech = (
        0.55 * np.sin(2.0 * np.pi * f0 * t)
        + 0.30 * np.sin(2.0 * np.pi * (2 * f0) * t)
        + 0.15 * np.sin(2.0 * np.pi * (3 * f0) * t)
    ) * amp
    return speech.astype(np.int16).tobytes()


def test_1_pure_400hz_disconnect_tone_detection():
    print("\n--- Test 1: Pure 400Hz / 425Hz Disconnect Tone DTFT Basis Projection ---")
    gate = InBandDisconnectGate(sample_rate=8000)

    # 400 Hz Indian standard tone frame
    tone_400 = generate_tone_frame(400.0, duration_ms=20.0)
    telem_400 = gate.process_frame(tone_400)

    print(f"[400Hz Frame] Is Tone: {telem_400.is_tone} | State: {telem_400.state.value}")
    print(f"[400Hz Power] 400Hz: {telem_400.power_400hz_dbfs:.2f} dBFS | RMS: {telem_400.broadband_rms_dbfs:.2f} dBFS")
    print(f"[Tone Ratio] {telem_400.tone_energy_ratio:.4f} (Threshold: 0.75)")

    assert telem_400.is_tone is True
    assert telem_400.tone_frequency_hz == 400.0
    assert telem_400.tone_energy_ratio >= 0.85
    assert telem_400.power_400hz_dbfs > -30.0

    # 425 Hz alternate tone frame
    tone_425 = generate_tone_frame(425.0, duration_ms=20.0)
    telem_425 = gate.process_frame(tone_425)

    print(f"[425Hz Frame] Is Tone: {telem_425.is_tone} | Freq: {telem_425.tone_frequency_hz:.1f} Hz")
    assert telem_425.is_tone is True
    assert telem_425.tone_frequency_hz == 425.0
    assert telem_425.tone_energy_ratio >= 0.85
    print("[OK] DTFT basis vectors accurately projected 400Hz and 425Hz disconnect tones")


def test_2_indian_standard_375ms_busy_cadence_disconnect():
    print("\n--- Test 2: Indian Standard 375ms Busy / Disconnect Cadence Detection ---")
    gate = InBandDisconnectGate(sample_rate=8000)

    tone_chunk = generate_tone_frame(400.0, duration_ms=20.0)
    silence_chunk = (np.zeros(160, dtype=np.int16)).tobytes()

    stream_frames = []
    # 2 completed cycles of 380ms ON (19 frames) and 380ms OFF (19 frames)
    # Cycle 1: 380ms ON, 380ms OFF
    stream_frames.extend([tone_chunk] * 19)
    stream_frames.extend([silence_chunk] * 19)
    # Cycle 2: 380ms ON
    stream_frames.extend([tone_chunk] * 19)

    pcm_stream = b"".join(stream_frames)
    telemetries, report = gate.process_stream(pcm_stream)

    print(f"[Cadence Stream] Frames: {report.total_frames} ({report.total_duration_ms:.1f} ms)")
    print(f"[Report] Disconnect Triggered: {report.disconnect_triggered}")
    print(f"[Report] Detected Pattern: {report.detected_pattern.value}")
    print(f"[Report] Hangup Latency: {report.hangup_latency_ms:.1f} ms")

    assert report.disconnect_triggered is True
    assert report.detected_pattern == DisconnectPattern.INDIAN_BUSY_375MS
    assert report.hangup_latency_ms is not None
    assert report.hangup_latency_ms <= 1200.0
    print("[OK] Correctly triggered call teardown on Indian standard 375ms busy cadence")


def test_3_slow_busy_750ms_cadence_disconnect():
    print("\n--- Test 3: Slow Busy 750ms Cadence Detection ---")
    gate = InBandDisconnectGate(sample_rate=8000)

    tone_chunk = generate_tone_frame(400.0, duration_ms=20.0)
    silence_chunk = (np.zeros(160, dtype=np.int16)).tobytes()

    stream_frames = []
    # 1 cycle of 760ms ON (38 frames) and 760ms OFF (38 frames)
    stream_frames.extend([tone_chunk] * 38)
    stream_frames.extend([silence_chunk] * 38)

    telemetries, report = gate.process_stream(b"".join(stream_frames))

    print(f"[Slow Busy] Disconnect Triggered: {report.disconnect_triggered}")
    print(f"[Slow Busy] Pattern: {report.detected_pattern.value}")

    assert report.disconnect_triggered is True
    assert report.detected_pattern == DisconnectPattern.INDIAN_BUSY_750MS
    print("[OK] Accurately detected slow 750ms busy / congestion cadence")


def test_4_fast_congestion_200ms_reorder_tone():
    print("\n--- Test 4: Fast Network Congestion 200ms Reorder Tone Detection ---")
    gate = InBandDisconnectGate(sample_rate=8000)

    tone_chunk = generate_tone_frame(400.0, duration_ms=20.0)
    silence_chunk = (np.zeros(160, dtype=np.int16)).tobytes()

    stream_frames = []
    # 2 cycles of 200ms ON (10 frames) and 200ms OFF (10 frames)
    stream_frames.extend([tone_chunk] * 10)
    stream_frames.extend([silence_chunk] * 10)
    stream_frames.extend([tone_chunk] * 10)

    telemetries, report = gate.process_stream(b"".join(stream_frames))

    print(f"[Fast Congestion] Disconnect Triggered: {report.disconnect_triggered}")
    print(f"[Fast Congestion] Pattern: {report.detected_pattern.value}")

    assert report.disconnect_triggered is True
    assert report.detected_pattern == DisconnectPattern.CONGESTION_200MS
    print("[OK] Correctly flagged fast 200ms reorder congestion tone")


def test_5_continuous_off_hook_howler_tone():
    print("\n--- Test 5: Continuous Off-Hook Howler Tone (> 1200ms) Detection ---")
    gate = InBandDisconnectGate(sample_rate=8000)

    # 1300ms continuous 400Hz tone (65 frames)
    tone_chunk = generate_tone_frame(400.0, duration_ms=20.0)
    howler_frames = [tone_chunk] * 65

    telemetries, report = gate.process_stream(b"".join(howler_frames))

    print(f"[Howler Tone] Disconnect Triggered: {report.disconnect_triggered}")
    print(f"[Howler Tone] Pattern: {report.detected_pattern.value}")
    print(f"[Howler Tone] Timestamp: {report.disconnect_timestamp_ms:.1f} ms")

    assert report.disconnect_triggered is True
    assert report.detected_pattern == DisconnectPattern.CONTINUOUS_HOWLER
    assert report.disconnect_timestamp_ms is not None
    assert report.disconnect_timestamp_ms <= 1300.0
    print("[OK] Successfully detected off-hook howler tone and commanded teardown")


def test_6_special_information_tones_sit():
    print("\n--- Test 6: Special Information Tone (SIT) Frequency Recognition ---")
    gate = InBandDisconnectGate(sample_rate=8000)

    # SIT tone 1400 Hz frame
    sit_frame = generate_tone_frame(1400.0, duration_ms=20.0)
    telem = gate.process_frame(sit_frame)

    print(f"[SIT Frame] Disconnect Triggered: {telem.disconnect_triggered}")
    print(f"[SIT Frame] Pattern: {telem.detected_pattern.value}")
    print(f"[SIT Frame] Frequency: {telem.tone_frequency_hz:.1f} Hz")

    assert telem.disconnect_triggered is True
    assert telem.detected_pattern == DisconnectPattern.SIT_TONES
    assert telem.tone_frequency_hz == 1400.0
    print("[OK] Special Information Tone (SIT) accurately recognized")


def test_7_conversational_speech_immunity_zero_false_hangups():
    print("\n--- Test 7: Conversational Speech Immunity (Zero False Hangups) ---")
    gate = InBandDisconnectGate(sample_rate=8000)

    # 1.5 seconds (75 frames) of voiced human speech with pitch modulations
    speech_frames = []
    for i in range(75):
        f0 = 130.0 + 25.0 * math.sin(i * 0.25)
        speech_frames.append(generate_speech_frame(f0=f0, duration_ms=20.0))

    telemetries, report = gate.process_stream(b"".join(speech_frames))

    print(f"[Speech Stream] Duration: {report.total_duration_ms:.1f} ms")
    print(f"[Speech Stream] Disconnect Triggered: {report.disconnect_triggered}")
    print(f"[Speech Stream] Pattern: {report.detected_pattern.value}")
    print(f"[Speech Stream] Avg Tone Ratio: {report.average_tone_ratio:.4f}")

    assert report.disconnect_triggered is False
    assert report.detected_pattern == DisconnectPattern.NONE
    assert report.average_tone_ratio < 0.35
    print("[OK] Conversational speech cleanly rejected with zero false disconnect triggers")


def test_8_sip_session_orchestrator_auto_teardown():
    print("\n--- Test 8: Stateful SIP Session Orchestrator Auto-Teardown ---")
    session = SIPSession(
        session_id="call-disconnect-test-01",
        caller_uri="sip:customer@carrier.airtel.in",
        agent_uri="sip:agent@verbalyze.ai",
        sample_rate=8000,
    )

    # Call is actively connected
    session.leg_a.state = SIPCallState.CONNECTED
    session.leg_b.state = SIPCallState.CONNECTED

    print(f"[Initial SIP State] Leg A: {session.leg_a.state.value} | Leg B: {session.leg_b.state.value}")
    assert session.leg_a.state == SIPCallState.CONNECTED

    tone_chunk = generate_tone_frame(400.0, duration_ms=20.0)
    silence_chunk = (np.zeros(160, dtype=np.int16)).tobytes()

    # Stream busy tone until disconnect triggers
    for _ in range(19):
        session.process_disconnect_frame(tone_chunk)
    for _ in range(19):
        session.process_disconnect_frame(silence_chunk)
    telem = None
    for _ in range(19):
        telem = session.process_disconnect_frame(tone_chunk)

    print(f"[Teardown Telemetry] Disconnect Triggered: {telem.disconnect_triggered}")
    print(f"[Final SIP State] Leg A: {session.leg_a.state.value} | Leg B: {session.leg_b.state.value}")
    print(f"[Terminated At] Leg A: {session.leg_a.terminated_at is not None}")

    assert telem.disconnect_triggered is True
    assert session.leg_a.state == SIPCallState.TERMINATED
    assert session.leg_b.state == SIPCallState.TERMINATED
    assert session.leg_a.terminated_at is not None
    print("[OK] SIP Session Orchestrator automatically terminated dialog upon in-band disconnect")


def test_9_fastapi_rest_endpoints_verification():
    print("\n--- Test 9: FastAPI Telephony REST Endpoints Verification ---")
    app = create_app(auth_token="test-secret-disconnect")
    from starlette.testclient import TestClient
    client = TestClient(app, headers={"Authorization": "Bearer test-secret-disconnect"})

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    print(f"[GET /health] disconnect_gate_status: {h_data.get('disconnect_gate_status')}")
    assert h_data.get("disconnect_gate_status") == "ready"

    # 2. POST /telephony/disconnect/analyze
    tone_chunk = generate_tone_frame(400.0, duration_ms=20.0)
    silence_chunk = (np.zeros(160, dtype=np.int16)).tobytes()
    busy_stream = tone_chunk * 19 + silence_chunk * 19 + tone_chunk * 19
    b64_audio = base64.b64encode(busy_stream).decode("ascii")

    ana_resp = client.post(
        "/telephony/disconnect/analyze",
        json={"audio_base64": b64_audio, "sample_rate": 8000},
    )
    assert ana_resp.status_code == 200
    a_data = ana_resp.json()
    print(f"[POST /telephony/disconnect/analyze] Triggered: {a_data.get('disconnect_triggered')} | Pattern: {a_data.get('detected_pattern')}")
    assert a_data.get("status") == "ok"
    assert a_data.get("disconnect_triggered") is True
    assert a_data.get("detected_pattern") == DisconnectPattern.INDIAN_BUSY_375MS.value

    # 3. POST /telephony/disconnect/benchmark
    bench_resp = client.post("/telephony/disconnect/benchmark")
    assert bench_resp.status_code == 200
    b_data = bench_resp.json()
    print(f"[POST /telephony/disconnect/benchmark] Avg Latency: {b_data.get('avg_disconnect_processing_time_ms')} ms")
    print(f"[POST /telephony/disconnect/benchmark] Meets SLA: {b_data.get('meets_sla')} | Headroom: {b_data.get('real_time_headroom_factor')}x")
    assert b_data.get("meets_sla") is True
    assert b_data.get("real_time_headroom_factor") >= 650.0
    print("[OK] All FastAPI disconnect endpoints verified cleanly")


def test_10_sub_0_03ms_latency_sla_benchmark():
    print("\n--- Test 10: Pure-Math Sub-0.03ms SLA Latency Benchmark ---")
    gate = InBandDisconnectGate(sample_rate=8000)
    frame = generate_tone_frame(400.0, duration_ms=20.0)

    # Warm-up
    for _ in range(50):
        gate.process_frame(frame)

    n_iterations = 500
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        gate.process_frame(frame)
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
    print("[OK] Pure-math In-Band Disconnect Gate easily meets sub-0.03ms latency SLA with massive headroom")


def main():
    print("=" * 80)
    print("Verbalyze In-Band Disconnect & Busy-Cadence Call Termination Gate Suite")
    print("ITU-T E.180 / Q.35 Pure-Math Tone & Fast Disconnect Watchdog / Zombie Guard")
    print("Target SLA: Latency < 0.030 ms per 20 ms frame | Headroom > 650x")
    print("=" * 80)

    t0 = time.perf_counter()
    test_1_pure_400hz_disconnect_tone_detection()
    test_2_indian_standard_375ms_busy_cadence_disconnect()
    test_3_slow_busy_750ms_cadence_disconnect()
    test_4_fast_congestion_200ms_reorder_tone()
    test_5_continuous_off_hook_howler_tone()
    test_6_special_information_tones_sit()
    test_7_conversational_speech_immunity_zero_false_hangups()
    test_8_sip_session_orchestrator_auto_teardown()
    test_9_fastapi_rest_endpoints_verification()
    test_10_sub_0_03ms_latency_sla_benchmark()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"ALL 10 IN-BAND DISCONNECT GATE TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | Pure Math & SLA Compliant")
    print("=" * 80)


if __name__ == "__main__":
    main()
