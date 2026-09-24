#!/usr/bin/env python3
"""
scripts/test_indic_formant_equalizer.py

Comprehensive 8-Part Automated Test Suite for Dynamic Multi-Band Acoustic Equalizer
(Indic Telecom Formant Enhancer).
Validates 5-band parametric biquads in Direct Form II Transposed structure,
selective retroflex Formant 3 (F3) amplification, upper telecom rolloff compensation,
frame-boundary state preservation, dynamic speech gating, soft-saturation peak limiting,
Indic phonetic spectral contrast improvement, and FastAPI REST endpoints.

Target SLA: Frame Processing Latency < 0.5 ms per 20 ms frame | Zero Audio Clicks
Zero-Emoji Compliant.
"""

import sys
import os
import time
import base64
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.telephony.equalizer import (
    BiquadFilter,
    EQBandConfig,
    EQTelemetry,
    IndicFormantEqualizer,
)
from verbalyze.telephony.server import create_app
from starlette.testclient import TestClient


def generate_tone(freq_hz: float, duration_s: float, sample_rate: int = 8000, amplitude: float = 6000.0) -> np.ndarray:
    """Generates a clean synthetic sinusoidal audio waveform."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.float32)


def test_1_biquad_coefficients_and_frequency_response():
    print("\n--- Test 1: Biquad Filter Coefficients & Frequency Response Across 5 Bands ---")
    sample_rate = 8000
    eq = IndicFormantEqualizer(sample_rate=sample_rate, preset_name="INDIC_RETROFLEX_ENHANCE")

    # Center frequencies for the 5 bands
    target_bands = [
        ("Band 1 (Nasal Warmth)", 300.0, 2.0),
        ("Band 2 (Vowel Body F1)", 750.0, 1.0),
        ("Band 3 (Palatal F2)", 1600.0, 2.5),
        ("Band 4 (Retroflex F3)", 2400.0, 4.5),
        ("Band 5 (Sibilant Burst)", 3200.0, 5.0),
    ]

    for idx, (name, center_f, target_gain) in enumerate(target_bands):
        filter_stage = eq.bands[idx]
        actual_gain = filter_stage.frequency_response(center_f)
        print(f"[{name}] Center: {center_f} Hz | Target Gain: +{target_gain:.1f} dB | Filter Response: +{actual_gain:.2f} dB")
        assert abs(actual_gain - target_gain) <= 0.35, f"{name} response {actual_gain:.2f} dB deviates from target {target_gain:.1f} dB"

    # Verify composite response across all 5 bands
    freqs = [300.0, 750.0, 1600.0, 2400.0, 3200.0]
    comp_resp = eq.get_frequency_response(freqs)
    print(f"Composite Cascaded Response (dB): {comp_resp}")
    assert comp_resp[3] >= 4.5, "Retroflex F3 composite gain must be >= +4.5 dB"
    assert comp_resp[4] >= 5.0, "Sibilant burst composite gain must be >= +5.0 dB"
    print("[OK] All 5 biquad filter stages calibrated accurately to Audio EQ Cookbook specifications")


def test_2_selective_retroflex_f3_amplification():
    print("\n--- Test 2: Selective Retroflex Formant 3 (F3) Amplification ---")
    eq = IndicFormantEqualizer(sample_rate=8000, preset_name="INDIC_RETROFLEX_ENHANCE")

    # Generate 20ms sine at 2400 Hz (Retroflex F3)
    tone_2400 = generate_tone(2400.0, 0.02, sample_rate=8000, amplitude=5000.0).astype(np.int16).tobytes()
    out_2400, telem_2400 = eq.process_frame(tone_2400)

    print(f"[Retroflex F3 2400Hz] Input RMS: {telem_2400.rms_in:.1f} -> Output RMS: {telem_2400.rms_out:.1f} | Gain: +{telem_2400.gain_applied_db:.2f} dB")
    assert telem_2400.gain_applied_db >= 4.5, f"Measured gain {telem_2400.gain_applied_db:.2f} dB should be >= +4.5 dB"

    # Compare with Neutral Bypass
    eq_bypass = IndicFormantEqualizer(sample_rate=8000, preset_name="NEUTRAL_BYPASS")
    out_byp, telem_byp = eq_bypass.process_frame(tone_2400)
    print(f"[Neutral Bypass] Input RMS: {telem_byp.rms_in:.1f} -> Output RMS: {telem_byp.rms_out:.1f} | Gain: {telem_byp.gain_applied_db:.2f} dB")
    assert abs(telem_byp.gain_applied_db) <= 0.1, "Neutral bypass gain must be ~ 0 dB"

    print("[OK] Retroflex F3 band selectively boosted by >+4.5 dB without uncalibrated attenuation")


def test_3_upper_edge_telecom_rolloff_compensation():
    print("\n--- Test 3: Upper-Edge Telecom Rolloff Compensation (3200 Hz) ---")
    eq = IndicFormantEqualizer(sample_rate=8000, preset_name="INDIC_RETROFLEX_ENHANCE")

    # Generate 20ms sine at 3200 Hz (High-frequency fricative sibilant near ITU-T G.712 cutoff)
    tone_3200 = generate_tone(3200.0, 0.02, sample_rate=8000, amplitude=4000.0).astype(np.int16).tobytes()
    out_3200, telem_3200 = eq.process_frame(tone_3200)

    print(f"[Sibilant Burst 3200Hz] Input RMS: {telem_3200.rms_in:.1f} -> Output RMS: {telem_3200.rms_out:.1f} | Gain: +{telem_3200.gain_applied_db:.2f} dB")
    assert telem_3200.gain_applied_db >= 4.5, f"Measured gain {telem_3200.gain_applied_db:.2f} dB should be >= +4.5 dB"
    print("[OK] Upper-edge telecom rolloff successfully counteracted with +5.0 dB sibilant boost")


def test_4_direct_form_ii_transposed_state_continuity():
    print("\n--- Test 4: Direct Form II Transposed Frame Boundary State Continuity ---")
    eq_chunked = IndicFormantEqualizer(sample_rate=8000, preset_name="INDIC_RETROFLEX_ENHANCE")
    eq_continuous = IndicFormantEqualizer(sample_rate=8000, preset_name="INDIC_RETROFLEX_ENHANCE")

    # Generate 60ms of continuous 1600 Hz tone (3 frames of 160 samples)
    total_audio = generate_tone(1600.0, 0.06, sample_rate=8000, amplitude=6000.0)
    frame_len = 160

    # 1. Process as 3 separate 20ms chunks sequentially
    f1 = total_audio[0:frame_len].astype(np.int16).tobytes()
    f2 = total_audio[frame_len:2*frame_len].astype(np.int16).tobytes()
    f3 = total_audio[2*frame_len:3*frame_len].astype(np.int16).tobytes()

    o1, _ = eq_chunked.process_frame(f1)
    o2, _ = eq_chunked.process_frame(f2)
    o3, _ = eq_chunked.process_frame(f3)
    chunked_output = np.concatenate([
        np.frombuffer(o1, dtype=np.int16),
        np.frombuffer(o2, dtype=np.int16),
        np.frombuffer(o3, dtype=np.int16),
    ]).astype(np.float32)

    # 2. Process all 480 samples continuously
    continuous_input = total_audio.astype(np.int16).tobytes()
    continuous_output_bytes, _ = eq_continuous.process_frame(continuous_input)
    continuous_output = np.frombuffer(continuous_output_bytes, dtype=np.int16).astype(np.float32)

    # Measure maximum deviation between chunked and continuous filtering
    max_dev = float(np.max(np.abs(chunked_output - continuous_output)))
    print(f"Max Deviation between Chunked (3x20ms) and Continuous (1x60ms): {max_dev:.2f} quantization levels")

    # Check continuity at frame boundary indices (sample 159 -> 160 and 319 -> 320)
    step_at_b1 = abs(chunked_output[160] - chunked_output[159])
    step_at_b2 = abs(chunked_output[320] - chunked_output[319])
    expected_step = abs(continuous_output[160] - continuous_output[159])
    print(f"Boundary 1 Step Jump: {step_at_b1:.1f} (Continuous Ground Truth: {expected_step:.1f})")

    assert max_dev <= 2.0, f"State discontinuity detected! Max deviation {max_dev} exceeds 2 LSB"
    assert abs(step_at_b1 - expected_step) <= 1.0, "Boundary step deviates from smooth continuous waveform"
    print("[OK] Direct Form II Transposed filter states cleanly preserved across 20ms boundaries (zero clicks)")


def test_5_dynamic_speech_adaptive_gating():
    print("\n--- Test 5: Dynamic Speech-Adaptive Gating (Silence & Noise Protection) ---")
    eq = IndicFormantEqualizer(sample_rate=8000, preset_name="INDIC_RETROFLEX_ENHANCE", dynamic_gating=True, speech_rms_threshold=80.0)

    # 1. Background Noise / Pause Frame (low amplitude, RMS ≈ 25.0)
    noise_frame = (np.random.normal(0, 25.0, 160)).astype(np.int16).tobytes()
    _, telem_noise = eq.process_frame(noise_frame)

    print(f"[Low-Energy Noise] RMS In: {telem_noise.rms_in:.1f} | Speech Active: {telem_noise.speech_active} | Gain Applied: +{telem_noise.gain_applied_db:.2f} dB")
    assert telem_noise.speech_active is False, "Noise frame should be detected as non-speech"
    assert telem_noise.gain_applied_db <= 0.60, f"Gated gain {telem_noise.gain_applied_db:.2f} dB should not amplify noise floor"

    # 2. Active Speech Frame (RMS ≈ 3500.0)
    speech_frame = generate_tone(2400.0, 0.02, sample_rate=8000, amplitude=5000.0).astype(np.int16).tobytes()
    _, telem_speech = eq.process_frame(speech_frame)

    print(f"[Active Speech] RMS In: {telem_speech.rms_in:.1f} | Speech Active: {telem_speech.speech_active} | Gain Applied: +{telem_speech.gain_applied_db:.2f} dB")
    assert telem_speech.speech_active is True, "Speech frame must be detected as active"
    assert telem_speech.gain_applied_db >= 4.5, "Full equalization gain must be applied during active speech"
    print("[OK] Dynamic speech gating prevents background noise amplification during pauses")


def test_6_soft_saturation_limiter():
    print("\n--- Test 6: Soft-Saturation Limiter (Zero Digital Clipping on 0 dBFS Input) ---")
    eq = IndicFormantEqualizer(sample_rate=8000, preset_name="INDIC_RETROFLEX_ENHANCE", soft_clip_threshold=30000.0)

    # High-amplitude tone (peak = 31,500) at 2400 Hz (+5 dB gain would exceed 55,000 without limiter)
    loud_tone = generate_tone(2400.0, 0.02, sample_rate=8000, amplitude=31500.0).astype(np.int16).tobytes()
    out_pcm, telem = eq.process_frame(loud_tone)
    samples = np.frombuffer(out_pcm, dtype=np.int16)

    max_peak = int(np.max(np.abs(samples)))
    print(f"[Loud 0 dBFS Input] Input Peak: 31500 -> Output Peak with Soft Saturation: {max_peak}")

    assert max_peak <= 32767, "Output peak must not exceed 16-bit integer maximum (32767)"
    assert max_peak > 30000, "Soft limiter should allow controlled peak compression into headroom"
    print("[OK] Soft saturation peak limiter smoothly compressed audio without harsh digital clipping")


def test_7_indic_phonetic_spectral_contrast_gain():
    print("\n--- Test 7: Indic Phonetic Spectral Contrast Gain Simulation ---")
    eq = IndicFormantEqualizer(sample_rate=8000, preset_name="INDIC_RETROFLEX_ENHANCE")

    # Simulate Indic retroflex phoneme (combining 250Hz nasal, 750Hz F1, 2400Hz F3 retroflex locus, and 3200Hz sibilant burst)
    t = np.linspace(0, 0.02, 160, endpoint=False)
    synthetic_retroflex = (
        np.sin(2 * np.pi * 250.0 * t) * 4000.0 +   # Nasal pole
        np.sin(2 * np.pi * 750.0 * t) * 5000.0 +   # Vowel body
        np.sin(2 * np.pi * 1000.0 * t) * 3000.0 +  # Mid valley
        np.sin(2 * np.pi * 2400.0 * t) * 2000.0 +  # Retroflex F3
        np.sin(2 * np.pi * 3200.0 * t) * 1500.0    # Sibilant burst
    ).astype(np.int16).tobytes()

    out_pcm, telem = eq.process_frame(synthetic_retroflex)
    in_arr = np.frombuffer(synthetic_retroflex, dtype=np.int16).astype(np.float32)
    out_arr = np.frombuffer(out_pcm, dtype=np.int16).astype(np.float32)

    # Compute FFT power spectra
    in_fft = np.abs(np.fft.rfft(in_arr))
    out_fft = np.abs(np.fft.rfft(out_arr))
    freq_bins = np.fft.rfftfreq(160, 1.0 / 8000.0)

    # Find bin indices for 1000Hz (neutral valley) and 2400Hz (retroflex F3)
    idx_1000 = int(np.argmin(np.abs(freq_bins - 1000.0)))
    idx_2400 = int(np.argmin(np.abs(freq_bins - 2400.0)))

    in_contrast = float(in_fft[idx_2400] / (in_fft[idx_1000] + 1e-4))
    out_contrast = float(out_fft[idx_2400] / (out_fft[idx_1000] + 1e-4))
    contrast_gain_db = 20.0 * np.log10(out_contrast / in_contrast)

    print(f"Retroflex F3 (2400Hz) vs Neutral Valley (1000Hz) Contrast: Input={in_contrast:.3f} -> Output={out_contrast:.3f}")
    print(f"Phonetic Spectral Contrast Gain: +{contrast_gain_db:.2f} dB")

    assert contrast_gain_db >= 3.0, f"Contrast gain +{contrast_gain_db:.2f} dB should be >= +3.0 dB"
    print("[OK] Indic retroflex phonetic contrast boosted by >+3.0 dB, enhancing telephone intelligibility")


def test_8_fastapi_telephony_equalizer_rest_endpoints():
    print("\n--- Test 8: FastAPI Telephony Equalizer REST Endpoints ---")
    app = create_app()
    client = TestClient(app)

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_json = health_resp.json()
    print(f"[GET /health] Status: {health_json.get('status')}, Equalizer: {health_json.get('equalizer_status')}")
    assert health_json.get("equalizer_status") == "ready"

    # 2. POST /telephony/eq/process
    test_tone = generate_tone(2400.0, 0.04, sample_rate=8000, amplitude=6000.0).astype(np.int16).tobytes()
    b64_audio = base64.b64encode(test_tone).decode("ascii")

    eq_payload = {
        "audio_base64": b64_audio,
        "preset": "INDIC_RETROFLEX_ENHANCE",
        "sample_rate": 8000,
    }
    proc_resp = client.post("/telephony/eq/process", json=eq_payload)
    assert proc_resp.status_code == 200
    proc_json = proc_resp.json()

    print(f"[POST /telephony/eq/process] Status: {proc_json.get('status')}, Preset: {proc_json.get('preset')}")
    assert proc_json.get("status") == "ok"
    assert proc_json.get("preset") == "INDIC_RETROFLEX_ENHANCE"
    assert "audio_base64" in proc_json
    assert proc_json.get("frequency_response_db", {}).get("2400", 0) >= 4.5

    # 3. POST /telephony/eq/benchmark
    bench_resp = client.post("/telephony/eq/benchmark")
    assert bench_resp.status_code == 200
    bench_json = bench_resp.json()

    avg_time = bench_json.get("avg_eq_time_ms")
    p95_time = bench_json.get("p95_eq_time_ms")
    headroom = bench_json.get("real_time_headroom_factor")
    meets_sla = bench_json.get("meets_eq_sla")

    print(f"[POST /telephony/eq/benchmark] Avg: {avg_time} ms | P95: {p95_time} ms | Meets SLA (<0.5ms): {meets_sla} | Headroom: {headroom}x")
    assert meets_sla is True, f"Equalizer execution time {avg_time} ms exceeded 0.5 ms SLA target"
    assert headroom >= 30.0, f"Headroom {headroom}x should be >= 30x"

    print("[OK] All FastAPI Telephony Equalizer REST endpoints verified successfully")


def main():
    print("=" * 80)
    print("Verbalyze Dynamic Multi-Band Acoustic Equalizer Test Suite")
    print("Indic Telecom Formant Enhancer | Target Latency SLA < 0.5 ms per 20 ms frame")
    print("=" * 80)

    t0 = time.time()
    test_1_biquad_coefficients_and_frequency_response()
    test_2_selective_retroflex_f3_amplification()
    test_3_upper_edge_telecom_rolloff_compensation()
    test_4_direct_form_ii_transposed_state_continuity()
    test_5_dynamic_speech_adaptive_gating()
    test_6_soft_saturation_limiter()
    test_7_indic_phonetic_spectral_contrast_gain()
    test_8_fastapi_telephony_equalizer_rest_endpoints()

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ALL 8 INDIC FORMANT EQUALIZER TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | Direct Form II Transposed | 5-Band Biquad")
    print("=" * 80)


if __name__ == "__main__":
    main()
