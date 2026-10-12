#!/usr/bin/env python3
"""
scripts/test_dtmf_talk_off_gate.py

Test Suite for ITU-T Q.24 & Bellcore GR-506-CORE DTMF Talk-Off Immunity Gate.
Validates pure-math harmonic discrimination, speech guard band enforcement,
twist compliance, and zero false keypad triggers during human conversation:
1. Pure DTMF Keypad Dual-Tone Detection.
2. Speech Talk-Off Rejection via Second-Harmonic (2f) Ratio.
3. Speech Talk-Off Rejection via Third-Harmonic (3f) Ratio.
4. Speech Guard Band Energy Ratio Violation Rejection.
5. Forward and Reverse Twist Compliance Testing (-8 dB to +4 dB).
6. Continuous Conversational Speech Stream (Zero False Keypad Triggers).
7. Debounced Inter-Digit Silence and Multi-Digit Sequence Collection.
8. DTMFPad Debouncer Integration with Talk-Off Protection.
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

from verbalyze.telephony.talk_off_gate import (
    DTMFTalkOffImmunityGate,
    TalkOffState,
    TalkOffRejectionReason,
    TalkOffTelemetry,
    TalkOffReport,
    DIGIT_TO_FREQUENCIES,
)
from verbalyze.telephony.dtmf_engine import DTMFPad
from verbalyze.telephony.server import create_app


def generate_dtmf_frame(
    digit: str,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
    amp_row: float = 8000.0,
    amp_col: float = 8000.0,
) -> bytes:
    """Generates pure DTMF dual-sine wave linear PCM frame."""
    f_row, f_col = DIGIT_TO_FREQUENCIES[digit]
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    sig = amp_row * np.sin(2.0 * np.pi * f_row * t) + amp_col * np.sin(2.0 * np.pi * f_col * t)
    return sig.astype(np.int16).tobytes()


def generate_speech_talk_off_frame(
    f_row: float = 770.0,
    f_col: float = 1336.0,
    harmonic_ratio_2f: float = 0.25,
    harmonic_ratio_3f: float = 0.20,
    guard_noise_amp: float = 1000.0,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
) -> bytes:
    """
    Generates synthetic speech frame where vocal tract formants align with DTMF frequencies,
    but glottal excitation introduces 2nd and 3rd harmonics along with vocal skirts.
    """
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    # Fundamental formants
    sig = 8000.0 * np.sin(2.0 * np.pi * f_row * t) + 8000.0 * np.sin(2.0 * np.pi * f_col * t)
    # 2nd harmonics
    sig += (8000.0 * harmonic_ratio_2f) * np.sin(2.0 * np.pi * (2 * f_row) * t)
    sig += (8000.0 * harmonic_ratio_2f) * np.sin(2.0 * np.pi * (2 * f_col) * t)
    # 3rd harmonics
    sig += (8000.0 * harmonic_ratio_3f) * np.sin(2.0 * np.pi * (3 * f_row) * t)
    # Guard band noise / extra formant energy
    sig += guard_noise_amp * np.sin(2.0 * np.pi * 500.0 * t)
    return sig.astype(np.int16).tobytes()


def generate_natural_speech_frame(
    f0: float = 140.0,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
) -> bytes:
    """Generates natural vowel voiced frame."""
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    speech = (
        0.55 * np.sin(2.0 * np.pi * f0 * t)
        + 0.30 * np.sin(2.0 * np.pi * (2 * f0) * t)
        + 0.15 * np.sin(2.0 * np.pi * (3 * f0) * t)
    ) * 16000.0
    return speech.astype(np.int16).tobytes()


def test_1_pure_dtmf_keypad_tone_detection():
    print("\n--- Test 1: Pure DTMF Keypad Dual-Tone Detection ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)

    for digit in ["1", "5", "9", "#"]:
        frame = generate_dtmf_frame(digit)
        telem = gate.process_frame(frame)
        print(f"[Digit '{digit}'] State: {telem.state.value} | Reason: {telem.rejection_reason.value}")
        print(f"2f Row Ratio: {telem.row_2nd_harmonic_ratio:.4f} | 3f Row Ratio: {telem.row_3rd_harmonic_ratio:.4f}")
        print(f"Speech Guard Ratio: {telem.speech_guard_ratio:.4f} | Twist: {telem.twist_db:.2f} dB")

        assert telem.is_genuine_tone is True
        assert telem.candidate_digit == digit
        assert telem.rejection_reason == TalkOffRejectionReason.NONE
        assert telem.row_2nd_harmonic_ratio <= 0.05
        assert telem.row_3rd_harmonic_ratio <= 0.05
        assert telem.speech_guard_ratio <= 0.20
    print("[OK] All tested pure touch-tones detected with near-zero harmonics and low guard energy")


def test_2_speech_talk_off_rejection_second_harmonic():
    print("\n--- Test 2: Speech Talk-Off Rejection via Second-Harmonic (2f) Ratio ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)

    # Candidate frequencies match '5' (770 Hz / 1336 Hz), but glottal 2nd harmonics are 50% amplitude of fundamental (~25% power)
    frame = generate_speech_talk_off_frame(
        f_row=770.0,
        f_col=1336.0,
        harmonic_ratio_2f=0.50,
        harmonic_ratio_3f=0.01,
        guard_noise_amp=0.0,
    )
    telem = gate.process_frame(frame)

    print(f"[2f Violation] State: {telem.state.value} | Rejection Reason: {telem.rejection_reason.value}")
    print(f"Row 2f Ratio: {telem.row_2nd_harmonic_ratio:.4f} (Max: 0.10) | Col 2f Ratio: {telem.col_2nd_harmonic_ratio:.4f}")

    assert telem.is_genuine_tone is False
    assert telem.rejection_reason == TalkOffRejectionReason.SECOND_HARMONIC_VIOLATION
    assert telem.is_talk_off_speech is True
    print("[OK] Successfully detected and rejected vocal 2nd harmonics, preventing talk-off")


def test_3_speech_talk_off_rejection_third_harmonic():
    print("\n--- Test 3: Speech Talk-Off Rejection via Third-Harmonic (3f) Ratio ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)

    # 2f is clean, but glottal 3rd harmonics are elevated (50% amplitude of fundamental = ~25% power)
    frame = generate_speech_talk_off_frame(
        f_row=770.0,
        f_col=1336.0,
        harmonic_ratio_2f=0.02,
        harmonic_ratio_3f=0.50,
        guard_noise_amp=0.0,
    )
    telem = gate.process_frame(frame)

    print(f"[3f Violation] State: {telem.state.value} | Rejection Reason: {telem.rejection_reason.value}")
    print(f"Row 3f Ratio: {telem.row_3rd_harmonic_ratio:.4f} (Max: 0.10)")

    assert telem.is_genuine_tone is False
    assert telem.rejection_reason == TalkOffRejectionReason.THIRD_HARMONIC_VIOLATION
    print("[OK] Correctly caught vocal glottal 3rd harmonics and blocked false trigger")


def test_4_speech_guard_band_energy_violation():
    print("\n--- Test 4: Speech Guard Band Energy Ratio Violation Rejection ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)

    # Low harmonics, but heavy energy scattered across other voice bands (guard ratio > 0.35)
    frame = generate_speech_talk_off_frame(
        f_row=770.0,
        f_col=1336.0,
        harmonic_ratio_2f=0.02,
        harmonic_ratio_3f=0.02,
        guard_noise_amp=12000.0,
    )
    telem = gate.process_frame(frame)

    print(f"[Guard Violation] State: {telem.state.value} | Rejection Reason: {telem.rejection_reason.value}")
    print(f"Speech Guard Ratio: {telem.speech_guard_ratio:.4f} (Max: 0.35)")

    assert telem.is_genuine_tone is False
    assert telem.rejection_reason == TalkOffRejectionReason.SPEECH_GUARD_BAND_VIOLATION
    print("[OK] Out-of-band speech skirts correctly triggered guard band violation")


def test_5_twist_tolerance_bounds():
    print("\n--- Test 5: Forward and Reverse Twist Compliance Testing (-8 dB to +4 dB) ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)

    # 1. Acceptable forward twist (+3 dB: col amp = 1.41 * row amp)
    frame_fwd = generate_dtmf_frame("9", amp_row=7000.0, amp_col=10000.0)
    telem_fwd = gate.process_frame(frame_fwd)
    print(f"[Forward Twist] Twist: {telem_fwd.twist_db:.2f} dB | Genuine: {telem_fwd.is_genuine_tone}")
    assert telem_fwd.is_genuine_tone is True

    # 2. Acceptable reverse twist (-6 dB: row amp = 2.0 * col amp)
    frame_rev = generate_dtmf_frame("9", amp_row=12000.0, amp_col=6000.0)
    telem_rev = gate.process_frame(frame_rev)
    print(f"[Reverse Twist] Twist: {telem_rev.twist_db:.2f} dB | Genuine: {telem_rev.is_genuine_tone}")
    assert telem_rev.is_genuine_tone is True

    # 3. Excessive forward twist (+8 dB > +4 dB)
    frame_bad = generate_dtmf_frame("9", amp_row=3000.0, amp_col=12000.0)
    telem_bad = gate.process_frame(frame_bad)
    print(f"[Bad Twist] Twist: {telem_bad.twist_db:.2f} dB | Reason: {telem_bad.rejection_reason.value}")
    assert telem_bad.is_genuine_tone is False
    assert telem_bad.rejection_reason == TalkOffRejectionReason.TWIST_OUT_OF_BOUNDS
    print("[OK] Forward and reverse twist tolerances strictly verified")


def test_6_continuous_human_speech_stream_zero_false_triggers():
    print("\n--- Test 6: Continuous Conversational Speech Stream (Zero False Keypad Triggers) ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)

    # 2.0 seconds (100 frames) of simulated natural speech with pitch modulations
    speech_frames = []
    for i in range(100):
        f0 = 130.0 + 30.0 * math.sin(i * 0.2)
        speech_frames.append(generate_natural_speech_frame(f0=f0, duration_ms=20.0))

    telemetries, report = gate.process_stream(b"".join(speech_frames))

    print(f"[Speech Stream] Duration: {report.total_duration_ms:.1f} ms | Frames: {report.total_frames}")
    print(f"[Speech Stream] Detected Digits: {report.detected_digits}")
    print(f"[Speech Stream] Genuine Tone Frames: {report.genuine_tone_frame_count}")
    print(f"[Speech Stream] Average Guard Ratio: {report.average_guard_ratio:.4f}")

    assert len(report.detected_digits) == 0
    assert report.genuine_tone_frame_count == 0
    print("[OK] Conversational speech yielded ZERO false keypad keypresses")


def test_7_inter_digit_debouncing_and_sequence_collection():
    print("\n--- Test 7: Debounced Inter-Digit Silence and Multi-Digit Sequence Collection ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)

    silence = (np.zeros(160, dtype=np.int16)).tobytes()
    sequence_frames = []

    # Sequence: "5" -> pause -> "8" -> pause -> "2"
    # Digit "5": 60ms (3 frames)
    sequence_frames.extend([generate_dtmf_frame("5")] * 3)
    # Pause: 60ms (3 frames)
    sequence_frames.extend([silence] * 3)
    # Digit "8": 60ms (3 frames)
    sequence_frames.extend([generate_dtmf_frame("8")] * 3)
    # Pause: 60ms (3 frames)
    sequence_frames.extend([silence] * 3)
    # Digit "2": 60ms (3 frames)
    sequence_frames.extend([generate_dtmf_frame("2")] * 3)

    telemetries, report = gate.process_stream(b"".join(sequence_frames))

    print(f"[Collected Digits] {report.detected_digits}")
    assert report.detected_digits == ["5", "8", "2"]
    print("[OK] Successfully registered clean multi-digit DTMF sequence '582' with debouncing")


def test_8_dtmf_pad_integration():
    print("\n--- Test 8: DTMFPad Debouncer Integration with Talk-Off Protection ---")
    # Instantiate DTMFPad with enable_talk_off_gate=True
    pad = DTMFPad(sample_rate=8000, enable_talk_off_gate=True)

    # 1. Feed speech chunk that has vocal energy
    speech_frame = generate_speech_talk_off_frame(770.0, 1336.0, harmonic_ratio_2f=0.50) * 5
    digits_speech = pad.process_pcm_chunk(speech_frame)
    print(f"[Speech Input to Pad] Registered Digits: {digits_speech}")
    assert len(digits_speech) == 0

    # 2. Feed genuine DTMF digit '7' for 80ms (4 frames)
    tone_frame = generate_dtmf_frame("7") * 4
    digits_tone = pad.process_pcm_chunk(tone_frame)
    print(f"[Tone Input to Pad] Registered Digits: {digits_tone}")
    assert digits_tone == ["7"]
    print("[OK] DTMFPad cleanly integrated with DTMFTalkOffImmunityGate")


def test_9_fastapi_rest_endpoints_verification():
    print("\n--- Test 9: FastAPI Telephony REST Endpoints Verification ---")
    app = create_app(auth_token="test-secret-talkoff")
    from starlette.testclient import TestClient
    client = TestClient(app, headers={"Authorization": "Bearer test-secret-talkoff"})

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    print(f"[GET /health] talk_off_gate_status: {h_data.get('talk_off_gate_status')}")
    assert h_data.get("talk_off_gate_status") == "ready"

    # 2. POST /telephony/dtmf/talk-off/analyze
    tone_stream = generate_dtmf_frame("4") * 5
    b64_audio = base64.b64encode(tone_stream).decode("ascii")

    ana_resp = client.post(
        "/telephony/dtmf/talk-off/analyze",
        json={"audio_base64": b64_audio, "sample_rate": 8000},
    )
    assert ana_resp.status_code == 200
    a_data = ana_resp.json()
    print(f"[POST /telephony/dtmf/talk-off/analyze] Detected Digits: {a_data.get('detected_digits')}")
    assert a_data.get("status") == "ok"
    assert a_data.get("detected_digits") == ["4"]

    # 3. POST /telephony/dtmf/talk-off/benchmark
    bench_resp = client.post("/telephony/dtmf/talk-off/benchmark")
    assert bench_resp.status_code == 200
    b_data = bench_resp.json()
    print(f"[POST /telephony/dtmf/talk-off/benchmark] Avg Latency: {b_data.get('avg_talk_off_processing_time_ms')} ms")
    print(f"[POST /telephony/dtmf/talk-off/benchmark] Meets SLA: {b_data.get('meets_sla')} | Headroom: {b_data.get('real_time_headroom_factor')}x")
    assert b_data.get("meets_sla") is True
    assert b_data.get("real_time_headroom_factor") >= 650.0
    print("[OK] All FastAPI talk-off endpoints verified cleanly")


def test_10_sub_0_03ms_latency_sla_benchmark():
    print("\n--- Test 10: Pure-Math Sub-0.03ms SLA Latency Benchmark ---")
    gate = DTMFTalkOffImmunityGate(sample_rate=8000)
    frame = generate_dtmf_frame("5")

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
    print("[OK] Pure-math Talk-Off Immunity Gate easily meets sub-0.03ms latency SLA with massive headroom")


def main():
    print("=" * 80)
    print("Verbalyze ITU-T Q.24 In-Band DTMF Talk-Off Immunity Gate Suite")
    print("Pure-Math Harmonic Discriminator & Speech Guard Gate")
    print("Target SLA: Latency < 0.030 ms per 20 ms frame | Headroom > 650x")
    print("=" * 80)

    t0 = time.perf_counter()
    test_1_pure_dtmf_keypad_tone_detection()
    test_2_speech_talk_off_rejection_second_harmonic()
    test_3_speech_talk_off_rejection_third_harmonic()
    test_4_speech_guard_band_energy_violation()
    test_5_twist_tolerance_bounds()
    test_6_continuous_human_speech_stream_zero_false_triggers()
    test_7_inter_digit_debouncing_and_sequence_collection()
    test_8_dtmf_pad_integration()
    test_9_fastapi_rest_endpoints_verification()
    test_10_sub_0_03ms_latency_sla_benchmark()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"ALL 10 DTMF TALK-OFF IMMUNITY GATE TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | Pure Math & SLA Compliant")
    print("=" * 80)


if __name__ == "__main__":
    main()
