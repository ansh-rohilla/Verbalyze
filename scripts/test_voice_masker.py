#!/usr/bin/env python3
"""
scripts/test_voice_masker.py

Comprehensive Verification Suite for Verbalyze Pure-Math TD-PSOLA Voice Masker
and Collector Anonymization Engine.

Tests:
1. Voicing Detection & Fundamental Frequency (F0) Estimation.
2. Deep Authoritative Mode (Scale 0.84 / ~3 Semitones Drop).
3. High Neutral Mode (Scale 1.18 / ~2.8 Semitones Rise).
4. Gender Register Shifts (Feminine Shift 1.25 / Masculine Shift 0.80).
5. Unvoiced Speech & Consonant Transient Bypass.
6. Formant Envelope Compensation & Loudness Invariance.
7. Deterministic Pseudorandom Session Anonymization (Call-ID Hash).
8. Soft-Saturation Peak Limiter under High Input Amplitudes.
9. FastAPI Endpoints (/health, /telephony/masker/process, /telephony/masker/benchmark).
10. Sub-0.15ms Pure-Math Execution SLA Benchmark.

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

from verbalyze.telephony.voice_masker import (
    PSOLAVoiceMasker,
    MaskingMode,
    MaskerTelemetry,
)


def generate_tone_pcm(freq_hz: float, sample_rate: int = 8000, duration_sec: float = 0.02, amplitude: float = 12000.0) -> bytes:
    """Generates synthetic 16-bit linear PCM sinusoidal audio."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    samples = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.int16)
    return samples.tobytes()


def generate_unvoiced_noise_pcm(sample_rate: int = 8000, duration_sec: float = 0.02, amplitude: float = 3000.0) -> bytes:
    """Generates synthetic unvoiced white noise audio."""
    n_samples = int(sample_rate * duration_sec)
    rng = np.random.default_rng(42)
    samples = (amplitude * rng.uniform(-1.0, 1.0, n_samples)).astype(np.int16)
    return samples.tobytes()


def pcm_energy(pcm_bytes: bytes) -> float:
    """Computes RMS energy of PCM audio."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def test_1_voicing_detection_and_f0_estimation():
    print("\n" + "=" * 80)
    print("TEST 1: Voicing Detection & Fundamental Frequency (F0) Estimation")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000)

    # 1. Periodic 150 Hz tone (voiced speech simulation)
    pcm_voiced = generate_tone_pcm(150.0, amplitude=14000.0)
    _, telem_voiced = masker.process_frame(pcm_voiced)

    print(f"[Voiced] Is Voiced:  {telem_voiced.is_voiced} (Expected: True)")
    print(f"[Voiced] Pitch (Hz):  {telem_voiced.pitch_hz:.1f} Hz (Expected ~150 Hz)")
    assert telem_voiced.is_voiced is True
    assert 140.0 <= telem_voiced.pitch_hz <= 160.0

    # 2. Silence / extremely low amplitude
    pcm_silence = b"\x00" * 320
    _, telem_silence = masker.process_frame(pcm_silence)

    print(f"[Silence] Is Voiced: {telem_silence.is_voiced} (Expected: False)")
    assert telem_silence.is_voiced is False
    print("[PASS] Voicing detection and F0 estimation verified.")


def test_2_deep_authoritative_mode():
    print("\n" + "=" * 80)
    print("TEST 2: Deep Authoritative Mode (Scale 0.84 / ~3 Semitones Drop)")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000, default_mode=MaskingMode.DEEP_AUTHORITATIVE)
    assert masker.get_pitch_scale() == 0.84

    pcm_in = generate_tone_pcm(160.0, amplitude=12000.0)
    out_pcm, telem = masker.process_frame(pcm_in)

    print(f"[Deep Mode] Configured Pitch Scale: {telem.pitch_scale}")
    print(f"[Deep Mode] Original RMS:           {telem.original_rms_db:.2f} dBov")
    print(f"[Deep Mode] Masked RMS:             {telem.masked_rms_db:.2f} dBov")
    print(f"[Deep Mode] Output Bytes Length:    {len(out_pcm)} (Must match 320 bytes)")

    assert len(out_pcm) == 320
    assert telem.pitch_scale == 0.84
    assert telem.is_voiced is True
    # Output energy should remain consistent with original speech
    assert abs(telem.original_rms_db - telem.masked_rms_db) < 6.0
    print("[PASS] Deep authoritative voice masking verified.")


def test_3_high_neutral_mode():
    print("\n" + "=" * 80)
    print("TEST 3: High Neutral Mode (Scale 1.18 / ~2.8 Semitones Rise)")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000, default_mode=MaskingMode.HIGH_NEUTRAL)
    assert masker.get_pitch_scale() == 1.18

    pcm_in = generate_tone_pcm(180.0, amplitude=12000.0)
    out_pcm, telem = masker.process_frame(pcm_in)

    print(f"[High Neutral] Configured Pitch Scale: {telem.pitch_scale}")
    print(f"[High Neutral] Pitch Marks Detected:   {telem.num_pitch_marks}")
    print(f"[High Neutral] Masked RMS:             {telem.masked_rms_db:.2f} dBov")

    assert len(out_pcm) == 320
    assert telem.pitch_scale == 1.18
    assert telem.num_pitch_marks >= 2
    assert abs(telem.original_rms_db - telem.masked_rms_db) < 6.0
    print("[PASS] High neutral voice masking verified.")


def test_4_gender_register_shifts():
    print("\n" + "=" * 80)
    print("TEST 4: Gender Register Shifts (Feminine Shift 1.25 / Masculine Shift 0.80)")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000)

    # 1. Feminine shift
    masker.set_mode(MaskingMode.FEMININE_SHIFT)
    assert masker.get_pitch_scale() == 1.25
    _, telem_fem = masker.process_frame(generate_tone_pcm(120.0))
    print(f"[Register] Feminine Shift Scale:  {telem_fem.pitch_scale}")
    assert telem_fem.pitch_scale == 1.25

    # 2. Masculine shift
    masker.set_mode(MaskingMode.MASCULINE_SHIFT)
    assert masker.get_pitch_scale() == 0.80
    _, telem_masc = masker.process_frame(generate_tone_pcm(220.0))
    print(f"[Register] Masculine Shift Scale: {telem_masc.pitch_scale}")
    assert telem_masc.pitch_scale == 0.80
    print("[PASS] Gender register shifts verified.")


def test_5_unvoiced_speech_bypass():
    print("\n" + "=" * 80)
    print("TEST 5: Unvoiced Speech & Consonant Transient Bypass")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000, default_mode=MaskingMode.DEEP_AUTHORITATIVE)

    # Unvoiced random noise (simulating fricatives 'स', 'श' or unvoiced stops)
    pcm_unvoiced = generate_unvoiced_noise_pcm(amplitude=4000.0)
    out_pcm, telem = masker.process_frame(pcm_unvoiced)

    print(f"[Unvoiced] Is Voiced Flag: {telem.is_voiced} (Expected: False)")
    print(f"[Unvoiced] Pitch Marks:    {telem.num_pitch_marks} (Expected: 0)")

    assert telem.is_voiced is False
    assert telem.num_pitch_marks == 0
    # Transparent bypass preserves energy exactly
    assert abs(telem.original_rms_db - telem.masked_rms_db) < 0.5
    print("[PASS] Unvoiced speech transient preservation verified.")


def test_6_formant_envelope_compensation():
    print("\n" + "=" * 80)
    print("TEST 6: Formant Envelope Compensation & Loudness Invariance")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000, default_mode=MaskingMode.DEEP_AUTHORITATIVE)

    # Multi-harmonic synthetic vowel (fundamental 130 Hz + 2nd harmonic 260 Hz + 3rd harmonic 390 Hz)
    t = np.linspace(0, 0.02, 160, endpoint=False)
    vowel = (8000.0 * np.sin(2 * np.pi * 130.0 * t) +
             4000.0 * np.sin(2 * np.pi * 260.0 * t) +
             2000.0 * np.sin(2 * np.pi * 390.0 * t)).astype(np.int16).tobytes()

    out_pcm, telem = masker.process_frame(vowel)
    samples_out = np.frombuffer(out_pcm, dtype=np.int16).astype(np.float32)

    print(f"[Formant] Input RMS:  {telem.original_rms_db:.2f} dBov")
    print(f"[Formant] Output RMS: {telem.masked_rms_db:.2f} dBov")
    print(f"[Formant] Peak Amp:   {np.max(np.abs(samples_out)):.1f}")

    assert abs(telem.original_rms_db - telem.masked_rms_db) < 4.0
    assert not np.isnan(samples_out).any()
    print("[PASS] Formant envelope compensation and loudness preservation verified.")


def test_7_random_session_anonymization():
    print("\n" + "=" * 80)
    print("TEST 7: Deterministic Pseudorandom Session Anonymization (Call-ID Hash)")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000, default_mode=MaskingMode.RANDOM_SESSION)

    # Session 1
    masker.set_mode(MaskingMode.RANDOM_SESSION, session_id="call-sbi-recovery-98765")
    scale_1a = masker.get_pitch_scale()
    scale_1b = masker.get_pitch_scale()
    print(f"[Session 1] Call-ID 'call-sbi-recovery-98765' Scale: {scale_1a:.4f}")

    # Session 1 must be deterministic
    assert scale_1a == scale_1b
    assert scale_1a != 1.0

    # Session 2
    masker.set_mode(MaskingMode.RANDOM_SESSION, session_id="call-hdfc-dispute-11223")
    scale_2 = masker.get_pitch_scale()
    print(f"[Session 2] Call-ID 'call-hdfc-dispute-11223' Scale: {scale_2:.4f}")

    assert scale_2 != 1.0
    # Different Call-IDs must produce different scales
    assert abs(scale_1a - scale_2) > 0.02
    print("[PASS] Deterministic pseudorandom session anonymization verified.")


def test_8_soft_peak_limiter_saturation():
    print("\n" + "=" * 80)
    print("TEST 8: Soft-Saturation Peak Limiter under High Input Amplitudes")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000, default_mode=MaskingMode.DEEP_AUTHORITATIVE)

    # Excessively loud signal that would digitally clip under linear scaling
    pcm_loud = generate_tone_pcm(150.0, amplitude=32000.0)
    out_pcm, telem = masker.process_frame(pcm_loud)

    samples = np.frombuffer(out_pcm, dtype=np.int16).astype(np.float32) / 32767.0
    max_peak = float(np.max(np.abs(samples)))

    print(f"[Limiter] Maximum Output Peak:    {max_peak:.4f} (Must be <= 1.0)")
    print(f"[Limiter] Clipping Prevented Flag: {telem.clipping_prevented}")

    assert max_peak <= 1.0, f"Peak exceeded full scale [-1.0, 1.0]: {max_peak}"
    assert telem.clipping_prevented is True
    print("[PASS] Soft-saturation peak limiter successfully prevented digital clipping.")


def test_9_fastapi_masker_endpoints():
    print("\n" + "=" * 80)
    print("TEST 9: FastAPI Telephony Voice Masker Endpoints (/health, /process, /benchmark)")
    print("=" * 80)
    from fastapi.testclient import TestClient
    from verbalyze.telephony.server import create_app

    app = create_app()
    client = TestClient(app)

    # 1. Health check verification
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data.get("voice_masker_status") == "ready", "Voice masker not ready in health check!"
    print("[PASS] /health reports voice_masker_status: ready")

    # 2. Process audio frame endpoint
    pcm_voiced = generate_tone_pcm(150.0, amplitude=10000.0)
    pcm_b64 = base64.b64encode(pcm_voiced).decode("ascii")

    p_resp = client.post("/telephony/masker/process", json={
        "pcm_base64": pcm_b64,
        "mode": "deep_authoritative",
        "sample_rate": 8000,
    })
    assert p_resp.status_code == 200
    p_data = p_resp.json()
    assert p_data["status"] == "ok"
    assert "masked_pcm_base64" in p_data
    assert p_data["telemetry"]["pitch_scale"] == 0.84
    assert p_data["telemetry"]["is_voiced"] is True
    print("[PASS] /telephony/masker/process successfully processed 20ms frame.")

    # 3. Benchmark endpoint
    b_resp = client.post("/telephony/masker/benchmark", json={})
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert b_data["status"] == "ok"
    assert b_data["meets_sla"] is True
    print(f"[PASS] /telephony/masker/benchmark: avg={b_data['avg_masker_time_ms']}ms, SLA={b_data['target_sla_ms']}ms, Headroom={b_data['real_time_headroom_factor']}x")


def test_10_sub_015ms_pure_math_throughput_benchmark():
    print("\n" + "=" * 80)
    print("TEST 10: Sub-0.15ms Pure-Math Execution SLA Benchmark")
    print("=" * 80)
    masker = PSOLAVoiceMasker(sample_rate=8000, default_mode=MaskingMode.DEEP_AUTHORITATIVE)

    pcm_voiced = generate_tone_pcm(150.0, amplitude=12000.0)

    # Warm-up
    masker.process_frame(pcm_voiced)

    n_iterations = 200
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        masker.process_frame(pcm_voiced)
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
    print(f"[Benchmark] Target SLA (<0.15ms): {'PASS' if avg_ms < 0.15 else 'FAIL'}")

    assert avg_ms < 0.15, f"Mean execution time {avg_ms:.4f}ms exceeded 0.15ms SLA"
    print("[PASS] Sub-0.15ms TD-PSOLA execution throughput verified.")


def run_all_voice_masker_tests():
    t_start = time.perf_counter()
    print("\n" + "=" * 80)
    print("VERBALYZE: PURE-MATH TD-PSOLA VOICE MASKER TEST SUITE")
    print("=" * 80)

    test_1_voicing_detection_and_f0_estimation()
    test_2_deep_authoritative_mode()
    test_3_high_neutral_mode()
    test_4_gender_register_shifts()
    test_5_unvoiced_speech_bypass()
    test_6_formant_envelope_compensation()
    test_7_random_session_anonymization()
    test_8_soft_peak_limiter_saturation()
    test_9_fastapi_masker_endpoints()
    test_10_sub_015ms_pure_math_throughput_benchmark()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL 10 VOICE MASKER TESTS PASSED IN {total_time:.2f}s!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_voice_masker_tests()
