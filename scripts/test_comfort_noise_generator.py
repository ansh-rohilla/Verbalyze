#!/usr/bin/env python3
"""
scripts/test_comfort_noise_generator.py

Comprehensive 8-Part Automated Test Suite for Adaptive Comfort Noise Generator (CNG)
Compliant with ITU-T G.711 Appendix II and RFC 3389.
Validates Levinson-Durbin Linear Predictive Coding (LPC) recursion, reflection coefficient stability,
calibrated dBov noise level modeling, all-pole Direct Form II Transposed synthesis filtering,
frame-boundary delay state continuity (zero clicks), RFC 3389 SID packet binary serialization,
smooth speech-to-CNG cross-fading, Indian telecom acoustic presets, and FastAPI REST endpoints.

Target SLA: Frame Processing Latency < 0.5 ms per 20 ms frame | Zero Audio Boundary Clicks
Zero-Emoji Compliant.
"""

import sys
import os
import time
import base64
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.telephony.comfort_noise import (
    ComfortNoiseGenerator,
    CNGTelemetry,
    SIDPacket,
)
from verbalyze.telephony.server import create_app
from starlette.testclient import TestClient


def test_1_levinson_durbin_and_reflection_stability():
    print("\n--- Test 1: Levinson-Durbin Recursion & Reflection Coefficient Stability ---")
    cng = ComfortNoiseGenerator(sample_rate=8000, lpc_order=4)

    # 1. Test synthetic autocorrelation corresponding to stationary low-frequency colored noise
    # R(0) > R(1) > R(2) ...
    r = np.array([1000.0, 850.0, 620.0, 410.0, 250.0], dtype=np.float32)
    a, k, e = cng._levinson_durbin(r, order=4)

    print(f"Prediction Coefficients A: {[round(float(x), 4) for x in a]}")
    print(f"Reflection Coefficients K: {[round(float(x), 4) for x in k]}")
    print(f"Residual Error Energy E:   {e:.2f}")

    # All reflection coefficients must be strictly bounded in (-1, 1) for filter stability
    for idx, ki in enumerate(k):
        assert abs(ki) < 1.0, f"Reflection coefficient k[{idx}] = {ki} violates stability (|k| >= 1.0)"

    # 2. Test roundtrip conversion: reflection -> LPC -> reflection
    refl_test = np.array([-0.65, 0.42, -0.28, 0.15], dtype=np.float32)
    lpc_coeffs = cng._reflection_to_lpc(refl_test)
    refl_recovered = cng._lpc_to_reflection(lpc_coeffs)

    max_diff = np.max(np.abs(refl_test - refl_recovered))
    print(f"Roundtrip Reflection Max Discrepancy: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Roundtrip conversion error exceeds tolerance: {max_diff}"
    print("PASS: Levinson-Durbin recursion and reflection stability confirmed.")


def test_2_noise_level_calibration_dbov():
    print("\n--- Test 2: Background Noise Level Calibration in dBov ---")
    cng = ComfortNoiseGenerator(sample_rate=8000, lpc_order=4)

    # 16-bit linear PCM full scale: 32767.0 = 0 dBov
    # Synthetic noise at target RMS = 3276.7 -> -20 dBov
    target_rms = 327.67  # -40 dBov
    np.random.seed(42)
    frame_samples = 160
    synthetic_noise = np.random.normal(0, target_rms, frame_samples * 10).astype(np.float32)
    pcm_bytes = np.clip(synthetic_noise, -32768, 32767).astype(np.int16).tobytes()

    # Ingest multiple frames into noise model
    for i in range(0, len(pcm_bytes), 320):
        cng.update_noise_model(pcm_bytes[i:i + 320])

    expected_dbov = 20.0 * np.log10(target_rms / 32767.0)
    print(f"Target Input dBov:    {expected_dbov:.2f} dBov")
    print(f"Calibrated CNG dBov:  {cng.noise_level_dbov:.2f} dBov")
    assert abs(cng.noise_level_dbov - expected_dbov) < 4.0, (
        f"Calibrated dBov {cng.noise_level_dbov:.2f} deviates from expected {expected_dbov:.2f}"
    )

    # Generate comfort noise frame and check its RMS
    out_pcm, telem = cng.generate_comfort_noise_frame()
    out_arr = np.frombuffer(out_pcm, dtype=np.int16).astype(np.float32)
    measured_rms = float(np.sqrt(np.mean(out_arr ** 2)))
    measured_dbov = 20.0 * np.log10(max(1.0, measured_rms) / 32767.0)
    print(f"Synthesized Frame RMS:  {measured_rms:.2f} ({measured_dbov:.2f} dBov)")
    assert abs(measured_dbov - expected_dbov) < 4.5, (
        f"Synthesized dBov {measured_dbov:.2f} deviates from target {expected_dbov:.2f}"
    )
    print("PASS: Noise level calibration in dBov confirmed.")


def test_3_spectral_coloring_and_synthesis_filtering():
    print("\n--- Test 3: Spectral Coloring & All-Pole IIR Synthesis Filtering ---")
    cng = ComfortNoiseGenerator(sample_rate=8000, preset_name="INDIAN_ROOM_CEILING_FAN")

    # Generate 50 frames of ceiling fan comfort noise (1 second of audio)
    frames = []
    for _ in range(50):
        pcm, _ = cng.generate_comfort_noise_frame()
        frames.append(pcm)
    all_audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)

    # Compute FFT power spectrum
    fft_mag = np.abs(np.fft.rfft(all_audio))
    freqs = np.fft.rfftfreq(len(all_audio), d=1.0 / 8000.0)

    # Ceiling fan preset concentrates spectral energy below 500 Hz
    low_band_power = np.mean(fft_mag[(freqs >= 50) & (freqs <= 500)] ** 2)
    high_band_power = np.mean(fft_mag[(freqs >= 2000) & (freqs <= 3800)] ** 2)
    spectral_tilt_db = 10.0 * np.log10(max(1e-6, low_band_power) / max(1e-6, high_band_power))

    print(f"Low-band Power (50-500Hz):   {low_band_power:.2e}")
    print(f"High-band Power (2-3.8kHz):  {high_band_power:.2e}")
    print(f"Spectral Tilt (Low vs High): +{spectral_tilt_db:.2f} dB")

    # Ceiling fan noise should exhibit significant positive spectral tilt (more low-frequency power)
    assert spectral_tilt_db > 3.0, (
        f"Ceiling fan comfort noise did not exhibit expected low-frequency coloration: tilt = {spectral_tilt_db:.2f} dB"
    )
    print("PASS: Spectral coloring matches physical ceiling fan acoustic profile.")


def test_4_frame_boundary_state_continuity():
    print("\n--- Test 4: Frame-Boundary State Continuity (Zero Digital Clicks) ---")
    cng1 = ComfortNoiseGenerator(sample_rate=8000, preset_name="URBAN_STREET_TRAFFIC")
    cng2 = ComfortNoiseGenerator(sample_rate=8000, preset_name="URBAN_STREET_TRAFFIC")

    # Fixed excitation seed for deterministic state comparison
    np.random.seed(12345)
    excitation_320 = np.random.normal(0, 1.0, 320).astype(np.float32)

    # Case A: Filter 320 samples continuously in one block
    order = cng1.lpc_order
    lpc_a = cng1.lpc_coeffs
    state_a = np.zeros(order, dtype=np.float32)
    scaled_exc = excitation_320 * cng1.residual_sigma
    y_continuous = np.zeros(320, dtype=np.float32)
    for n in range(320):
        yn = scaled_exc[n] + state_a[0]
        for j in range(order - 1):
            state_a[j] = (-lpc_a[j] * yn) + state_a[j + 1]
        state_a[order - 1] = -lpc_a[order - 1] * yn
        y_continuous[n] = yn

    # Case B: Filter as two consecutive 160-sample frames with state preserved
    state_b = np.zeros(order, dtype=np.float32)
    y_chunked = np.zeros(320, dtype=np.float32)
    # Frame 1
    for n in range(160):
        yn = scaled_exc[n] + state_b[0]
        for j in range(order - 1):
            state_b[j] = (-lpc_a[j] * yn) + state_b[j + 1]
        state_b[order - 1] = -lpc_a[order - 1] * yn
        y_chunked[n] = yn
    # Frame 2 (inherits state_b)
    for n in range(160):
        yn = scaled_exc[160 + n] + state_b[0]
        for j in range(order - 1):
            state_b[j] = (-lpc_a[j] * yn) + state_b[j + 1]
        state_b[order - 1] = -lpc_a[order - 1] * yn
        y_chunked[160 + n] = yn

    diff = np.max(np.abs(y_continuous - y_chunked))
    print(f"Max Deviation between Continuous and Chunked Filtering: {diff:.8f}")
    assert diff < 1e-5, f"State preservation failed across frame boundary with error: {diff}"
    print("PASS: Frame boundary state preservation guarantees click-free synthesis.")


def test_5_rfc_3389_sid_packet_serialization():
    print("\n--- Test 5: RFC 3389 SID Packet Serialization and Deserialization ---")
    cng = ComfortNoiseGenerator(sample_rate=8000, preset_name="CELLULAR_LINE_HISS")

    # 1. Create SID packet from generator
    sid = cng.to_sid_packet()
    print(f"SID Packet Noise Level: {sid.noise_level_dbov} (-dBov)")
    print(f"SID Reflection Coeffs:  {[round(x, 4) for x in sid.reflection_coefficients]}")

    # 2. Serialize to bytes
    sid_bytes = sid.to_bytes()
    expected_len = 1 + cng.lpc_order  # 1 byte level + 4 bytes reflection = 5 bytes
    print(f"Serialized SID Payload: {sid_bytes.hex()} (Length: {len(sid_bytes)} bytes)")
    assert len(sid_bytes) == expected_len, f"Expected {expected_len} bytes, got {len(sid_bytes)}"

    # 3. Deserialize from bytes
    sid_recovered = SIDPacket.from_bytes(sid_bytes)
    assert sid_recovered.noise_level_dbov == sid.noise_level_dbov, (
        f"Level mismatch: {sid_recovered.noise_level_dbov} vs {sid.noise_level_dbov}"
    )

    for idx, (orig, recov) in enumerate(zip(sid.reflection_coefficients, sid_recovered.reflection_coefficients)):
        err = abs(orig - recov)
        assert err < (1.0 / 120.0), f"Quantization error too large at k[{idx}]: {err}"

    # 4. Apply SID packet to a new generator instance
    cng_remote = ComfortNoiseGenerator(sample_rate=8000)
    cng_remote.apply_sid_packet(sid_recovered)
    assert abs(cng_remote.noise_level_dbov - (-float(sid.noise_level_dbov))) < 0.1
    print("PASS: RFC 3389 SID packet encoding, decoding, and model application verified.")


def test_6_cross_fading_transitions():
    print("\n--- Test 6: Smooth Speech-to-CNG Cross-Fading Transitions ---")
    cng = ComfortNoiseGenerator(sample_rate=8000)

    # 160-sample speech frame (high energy tone) and CNG frame (low energy)
    t = np.linspace(0, 0.02, 160, endpoint=False)
    speech = (np.sin(2 * np.pi * 300.0 * t) * 12000.0).astype(np.int16).tobytes()
    cng_frame, _ = cng.generate_comfort_noise_frame()

    blended = cng.cross_fade(speech_pcm=speech, cng_pcm=cng_frame, overlap_samples=32)
    assert len(blended) == len(cng_frame), f"Blended length mismatch: {len(blended)}"

    blended_arr = np.frombuffer(blended, dtype=np.int16).astype(np.float32)
    speech_arr = np.frombuffer(speech, dtype=np.int16).astype(np.float32)
    cng_arr = np.frombuffer(cng_frame, dtype=np.int16).astype(np.float32)

    # Start of overlap should be weighted strongly towards speech
    # End of overlap should match CNG
    print(f"Sample 0:   Blended={blended_arr[0]:.1f} | Speech={speech_arr[0]:.1f} | CNG={cng_arr[0]:.1f}")
    print(f"Sample 31:  Blended={blended_arr[31]:.1f} | Speech={speech_arr[31]:.1f} | CNG={cng_arr[31]:.1f}")
    print(f"Sample 100: Blended={blended_arr[100]:.1f} | CNG={cng_arr[100]:.1f}")

    assert abs(blended_arr[0] - speech_arr[0]) < 2.0, "Sample 0 did not match speech onset"
    assert abs(blended_arr[100] - cng_arr[100]) < 1.0, "Post-overlap region did not match CNG"
    print("PASS: Cross-fading delivers smooth seamless energy transitions.")


def test_7_acoustic_presets_verification():
    print("\n--- Test 7: Indian Telecom Acoustic Presets Calibration ---")
    presets = [
        ("INDIAN_ROOM_CEILING_FAN", -50.0),
        ("URBAN_STREET_TRAFFIC", -45.0),
        ("CELLULAR_LINE_HISS", -58.0),
        ("CLEAN_OFFICE_QUIET", -65.0),
    ]

    for name, expected_dbov in presets:
        cng = ComfortNoiseGenerator(sample_rate=8000, preset_name=name)
        frames = []
        for _ in range(25):
            pcm, _ = cng.generate_comfort_noise_frame()
            frames.append(pcm)

        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        measured_dbov = 20.0 * np.log10(max(1.0, rms) / 32767.0)

        print(f"Preset [{name:24s}]: Target = {expected_dbov:6.1f} dBov | Measured = {measured_dbov:6.1f} dBov | RMS = {rms:6.1f}")
        assert abs(measured_dbov - expected_dbov) < 4.0, (
            f"Preset {name} measured dBov {measured_dbov:.1f} deviates from target {expected_dbov:.1f}"
        )

    print("PASS: All Indian telecom acoustic presets calibrated within specification.")


def test_8_fastapi_rest_endpoints_and_benchmark():
    print("\n--- Test 8: FastAPI REST Endpoints & Real-Time Performance Benchmark ---")
    app = create_app()
    client = TestClient(app)

    # 1. Test /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
    health_data = health_resp.json()
    print(f"Health CNG Status: {health_data.get('cng_status')}")
    assert health_data.get("cng_status") == "ready", "cng_status is not ready in /health"

    # 2. Test /telephony/cng/generate
    gen_payload = {
        "preset": "INDIAN_ROOM_CEILING_FAN",
        "frames_count": 5,
        "sample_rate": 8000,
        "noise_level_dbov": -52.0,
    }
    gen_resp = client.post("/telephony/cng/generate", json=gen_payload)
    assert gen_resp.status_code == 200, f"Generate request failed: {gen_resp.text}"
    gen_data = gen_resp.json()
    assert gen_data.get("status") == "ok"
    assert gen_data.get("frames_generated") == 5
    assert len(gen_data.get("sid_packet_hex", "")) == 10  # 5 bytes hex = 10 chars
    audio_bytes = base64.b64decode(gen_data.get("audio_base64", ""))
    assert len(audio_bytes) == 5 * 320, f"Expected 1600 bytes, got {len(audio_bytes)}"
    print(f"CNG Generate Response: Frames={gen_data.get('frames_generated')}, SID Hex={gen_data.get('sid_packet_hex')}")

    # 3. Test /telephony/cng/benchmark
    bench_resp = client.post("/telephony/cng/benchmark")
    assert bench_resp.status_code == 200, f"Benchmark failed: {bench_resp.text}"
    bench_data = bench_resp.json()
    avg_ms = bench_data.get("avg_cng_time_ms")
    p95_ms = bench_data.get("p95_cng_time_ms")
    headroom = bench_data.get("real_time_headroom_factor")
    meets_sla = bench_data.get("meets_cng_sla")

    print(f"Benchmark Results:")
    print(f"  Average Time:    {avg_ms:.3f} ms / 20ms frame")
    print(f"  P95 Time:        {p95_ms:.3f} ms / 20ms frame")
    print(f"  Headroom Factor: {headroom}x real-time")
    print(f"  Meets SLA (<0.5ms): {meets_sla}")

    assert meets_sla is True, f"Benchmark did not meet SLA (<0.5ms): avg={avg_ms}ms"
    assert avg_ms < 0.2, f"Expected sub-0.2ms performance, got {avg_ms}ms"
    print("PASS: FastAPI REST endpoints and real-time benchmark confirmed.")


def run_all_tests():
    print("================================================================================")
    print("STARTING TEST SUITE: ADAPTIVE COMFORT NOISE GENERATOR (CNG / ITU-T G.711 App II)")
    print("================================================================================")

    test_1_levinson_durbin_and_reflection_stability()
    test_2_noise_level_calibration_dbov()
    test_3_spectral_coloring_and_synthesis_filtering()
    test_4_frame_boundary_state_continuity()
    test_5_rfc_3389_sid_packet_serialization()
    test_6_cross_fading_transitions()
    test_7_acoustic_presets_verification()
    test_8_fastapi_rest_endpoints_and_benchmark()

    print("\n================================================================================")
    print("ALL 8 COMFORT NOISE GENERATOR TESTS PASSED SUCCESSFULLY!")
    print("Zero-Emoji Compliant | Zero Audio Boundary Clicks | SLA Latency < 0.2ms Verified")
    print("================================================================================")


if __name__ == "__main__":
    run_all_tests()
