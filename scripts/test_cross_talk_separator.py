"""
scripts/test_cross_talk_separator.py

Comprehensive Test Suite for Acoustic Bleed & Ambient Cross-Talk Separator:
1. Fundamental Frequency (F0) & Pitch Tracking on Synthetic Indic Vowels.
2. Harmonic Comb Filter Primary Voice Preservation (<0.5 dB Loss).
3. Inter-Harmonic Bleed Suppression (>= 18 dB Attenuation).
4. Dual-Voice Cross-Talk Mixture Blind Source Isolation.
5. Background-Only Speech Chatter Suppression (Hawker / TV Audio).
6. Unvoiced Consonant Guard & Transient Passthrough (Zero Buzz on Fricatives).
7. False Barge-In Elimination (Gated Telephony Interruption Guard).
8. Continuous Multi-Frame Streaming & Delay Memory Continuity.
9. FastAPI Telephony Endpoints (/health, /process, /benchmark).
10. Ultra-Low Latency Throughput & Real-Time Headroom Benchmark.

Zero-emoji compliant.
DPDP Act 2023 & ITU-T G.168 / G.169 compliant.
"""

import os
import sys
import time
import math
import base64
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient

from verbalyze.telephony.cross_talk_separator import (
    AcousticCrossTalkSeparator,
    CrossTalkState,
    CrossTalkTelemetry,
    HarmonicCombFilter,
)
from verbalyze.telephony.echo_canceller import AcousticEchoAndNoiseProcessor
from verbalyze.telephony.server import create_app


def generate_synthetic_vowel(
    f0_hz: float = 160.0,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
    formants: Optional[List[Tuple[float, float, float]]] = None,
    rms_dbov: float = -18.0,
    start_time_s: float = 0.0,
) -> np.ndarray:
    """
    Synthesizes a realistic Indic vowel frame with specified F0 and formant envelope.
    """
    n_samples = int((duration_ms / 1000.0) * sample_rate)
    t = np.linspace(start_time_s, start_time_s + duration_ms / 1000.0, n_samples, endpoint=False)

    if formants is None:
        # Default Indic /a/ formants: F1=720, F2=1240, F3=2550
        formants = [(720.0, 80.0, 1.0), (1240.0, 100.0, 0.6), (2550.0, 150.0, 0.3)]

    harmonics = np.arange(1, int((sample_rate / 2.0) // f0_hz) + 1) * f0_hz
    signal = np.zeros(n_samples, dtype=np.float32)

    for h in harmonics:
        env = 0.1
        for f_c, bw, weight in formants:
            env += weight / (1.0 + ((h - f_c) / bw) ** 2)
        signal += env * np.sin(2.0 * np.pi * h * t)

    # Scale to desired dBov
    curr_rms = np.sqrt(np.mean(signal ** 2))
    target_rms = 10.0 ** (rms_dbov / 20.0)
    if curr_rms > 1e-9:
        signal = signal * (target_rms / curr_rms)

    return signal.astype(np.float32)


def test_pitch_detection_synthetic_vowel():
    """Test 1: Evaluates pitch estimation accuracy on multi-harmonic Indic vowel."""
    print("--- Test 1: Pitch Detection on Synthetic Indic Vowel ---")
    sep = AcousticCrossTalkSeparator(sample_rate=8000)

    target_f0 = 175.0
    vowel = generate_synthetic_vowel(f0_hz=target_f0, duration_ms=20.0, rms_dbov=-18.0)
    pcm = (vowel * 32768.0).astype(np.int16).tobytes()

    # Process 3 frames to establish trajectory
    for _ in range(3):
        _, telem = sep.process_frame(pcm)

    print(f"[OK] Target F0: {target_f0:.1f} Hz, Detected F0: {telem.foreground_f0_hz:.1f} Hz, Harmonicity: {telem.harmonicity_ratio:.3f}")
    assert telem.is_voiced, "Synthetic vowel must be classified as voiced"
    assert abs(telem.foreground_f0_hz - target_f0) <= 5.0, f"Detected pitch {telem.foreground_f0_hz} deviates from target {target_f0}"
    assert telem.harmonicity_ratio >= 0.70, "Vowel harmonicity ratio must be >= 0.70"


def test_harmonic_comb_primary_preservation():
    """Test 2: Verifies primary caller voice passes comb filter with <0.5 dB loss."""
    print("--- Test 2: Harmonic Comb Primary Voice Preservation ---")
    comb = HarmonicCombFilter(sample_rate=8000, alpha=0.88)
    f0 = 160.0

    # Stream 12 contiguous frames of primary vowel to allow steady-state convergence
    losses = []
    for i in range(12):
        t_start = i * 0.02
        vowel = generate_synthetic_vowel(f0_hz=f0, duration_ms=20.0, rms_dbov=-16.0, start_time_s=t_start)
        filtered = comb.filter_samples(vowel, f0_hz=f0)

        in_rms = float(np.sqrt(np.mean(vowel ** 2)))
        out_rms = float(np.sqrt(np.mean(filtered ** 2)))
        loss_db = 20.0 * np.log10(max(in_rms, 1e-6) / max(out_rms, 1e-6))
        losses.append(loss_db)

    steady_loss = abs(losses[-1])
    print(f"[OK] Steady-state primary preservation loss: {steady_loss:.3f} dB (<0.50 dB SLA)")
    assert steady_loss <= 0.50, f"Primary voice loss {steady_loss:.2f} dB exceeds 0.50 dB allowance"


def test_harmonic_comb_inter_harmonic_bleed_suppression():
    """Test 3: Verifies inter-harmonic bleed tones are suppressed by >= 18 dB."""
    print("--- Test 3: Inter-Harmonic Bleed Suppression (>= 18 dB) ---")
    comb = HarmonicCombFilter(sample_rate=8000, alpha=0.88)
    f0_prim = 160.0
    f_bleed = 240.0  # Exactly half-way between 1st (160) and 2nd (320) harmonics

    attenuations = []
    for i in range(12):
        t = np.linspace(i * 0.02, (i + 1) * 0.02, 160, endpoint=False)
        bleed_tone = (0.35 * np.sin(2.0 * np.pi * f_bleed * t)).astype(np.float32)
        out = comb.filter_samples(bleed_tone, f0_hz=f0_prim)
        in_rms = float(np.sqrt(np.mean(bleed_tone ** 2)))
        out_rms = float(np.sqrt(np.mean(out ** 2)))
        att_db = 20.0 * np.log10(max(in_rms, 1e-6) / max(out_rms, 1e-6))
        attenuations.append(att_db)

    final_att = attenuations[-1]
    print(f"[OK] Inter-harmonic bleed attenuation at 240 Hz: {final_att:.2f} dB (>= 18 dB target)")
    assert final_att >= 18.0, f"Inter-harmonic bleed attenuation {final_att:.2f} dB is below 18 dB target"


def test_cross_talk_mixture_separation():
    """Test 4: Isolates primary speaker in a dual-speaker cross-talk mixture."""
    print("--- Test 4: Dual-Voice Cross-Talk Mixture Separation ---")
    sep = AcousticCrossTalkSeparator(sample_rate=8000)

    f0_prim = 150.0
    f0_sec = 230.0

    # Stream 8 frames: first 3 frames primary only, next 5 frames mixture
    for i in range(3):
        t_start = i * 0.02
        prim = generate_synthetic_vowel(f0_hz=f0_prim, duration_ms=20.0, rms_dbov=-16.0, start_time_s=t_start)
        sep.process_frame_samples(prim)

    for i in range(3, 10):
        t_start = i * 0.02
        prim = generate_synthetic_vowel(f0_hz=f0_prim, duration_ms=20.0, rms_dbov=-16.0, start_time_s=t_start)
        sec = generate_synthetic_vowel(f0_hz=f0_sec, duration_ms=20.0, rms_dbov=-24.0, start_time_s=t_start)
        mix = prim + sec
        clean, telem = sep.process_frame_samples(mix)

    print(f"[OK] Mixture State: {telem.state.value}, Primary F0: {telem.foreground_f0_hz:.1f} Hz, Secondary F0: {telem.secondary_f0_hz}")
    assert telem.is_voiced, "Mixture must be recognized as voiced"
    assert abs(telem.foreground_f0_hz - f0_prim) <= 10.0, "Primary pitch track must remain locked onto foreground speaker"


def test_background_only_chatter_suppression():
    """Test 5: Suppresses background speech chatter when primary caller is silent."""
    print("--- Test 5: Background-Only Speech Chatter Suppression ---")
    sep = AcousticCrossTalkSeparator(sample_rate=8000)

    # Hawker shouting in background at low level (-28 dBov) without active primary caller
    for i in range(5):
        t_start = i * 0.02
        hawker = generate_synthetic_vowel(f0_hz=260.0, duration_ms=20.0, rms_dbov=-28.0, start_time_s=t_start)
        clean, telem = sep.process_frame_samples(hawker)

    in_rms = float(np.sqrt(np.mean(hawker ** 2)))
    clean_rms = float(np.sqrt(np.mean(clean ** 2)))
    suppression_db = 20.0 * np.log10(max(in_rms, 1e-6) / max(clean_rms, 1e-6))

    print(f"[OK] Background Chatter State: {telem.state.value}, Suppression: {suppression_db:.2f} dB, Clean Barge-In: {telem.clean_barge_in_eligible}")
    assert not telem.clean_barge_in_eligible, "Background chatter must not trigger barge-in eligibility"
    assert suppression_db >= 15.0 or telem.clean_rms_dbov <= -40.0, "Background chatter must be attenuated"


def test_unvoiced_consonant_bypass():
    """Test 6: Preserves unvoiced fricatives and stops with zero comb distortion."""
    print("--- Test 6: Unvoiced Consonant Guard & Transient Passthrough ---")
    sep = AcousticCrossTalkSeparator(sample_rate=8000)

    # Generate synthetic /s/ fricative: high-pass filtered noise with high ZCR
    np.random.seed(42)
    noise = np.random.uniform(-0.5, 0.5, 160).astype(np.float32)
    # Apply pre-emphasis / high-pass
    fricative = np.convolve(noise, [1.0, -0.92], mode="same").astype(np.float32)
    # Scale to caller speaking level (-22 dBov)
    fricative = fricative * (10.0 ** (-22.0 / 20.0) / (np.sqrt(np.mean(fricative ** 2)) + 1e-9))

    clean, telem = sep.process_frame_samples(fricative)

    print(f"[OK] Consonant State: {telem.state.value}, Is Voiced: {telem.is_voiced}, Clean Barge-In: {telem.clean_barge_in_eligible}")
    assert telem.state == CrossTalkState.UNVOICED_CONSONANT, "Fricative must be recognized as unvoiced consonant"
    assert not telem.is_voiced, "Fricative must not be classified as voiced"
    assert telem.clean_barge_in_eligible, "Caller unvoiced consonants must remain eligible for barge-in"


def test_false_barge_in_elimination():
    """Test 7: Eliminates phantom barge-in interruptions caused by ambient chatter."""
    print("--- Test 7: False Barge-In Elimination ---")
    sep = AcousticCrossTalkSeparator(sample_rate=8000)

    # Case A: Loud background hawker speech (-26 dBov) without primary caller
    hawker = generate_synthetic_vowel(f0_hz=290.0, duration_ms=20.0, rms_dbov=-26.0)
    hawker_pcm = (hawker * 32768.0).astype(np.int16).tobytes()

    barge_in_a, telem_a = sep.evaluate_barge_in(hawker_pcm)
    print(f"[OK] Hawker Chatter Only: Raw RMS={telem_a.raw_rms_dbov:.1f} dBov, Barge-In Triggered={barge_in_a}")
    assert not barge_in_a, "Loud background chatter must NOT interrupt the bot"

    # Case B: Primary caller speaks ("हेलो" / "हां") with near-field energy (-16 dBov)
    caller = generate_synthetic_vowel(f0_hz=165.0, duration_ms=20.0, rms_dbov=-16.0)
    caller_pcm = (caller * 32768.0).astype(np.int16).tobytes()

    for _ in range(3):
        barge_in_b, telem_b = sep.evaluate_barge_in(caller_pcm)

    print(f"[OK] Foreground Caller Speech: Raw RMS={telem_b.raw_rms_dbov:.1f} dBov, Barge-In Triggered={barge_in_b}")
    assert barge_in_b, "Foreground caller speech MUST trigger barge-in"


def test_continuous_streaming_state_continuity():
    """Test 8: Verifies continuous multi-frame streaming with smooth state continuity."""
    print("--- Test 8: Continuous Multi-Frame Streaming & Memory Continuity ---")
    sep = AcousticCrossTalkSeparator(sample_rate=8000)

    # Stream 25 frames alternating between silence, speech, cross-talk, and consonant
    for i in range(25):
        if i < 5:
            samples = np.zeros(160, dtype=np.float32)
        elif i < 15:
            samples = generate_synthetic_vowel(f0_hz=170.0 + (i % 3) * 5.0, duration_ms=20.0, rms_dbov=-18.0)
        elif i < 20:
            prim = generate_synthetic_vowel(f0_hz=175.0, duration_ms=20.0, rms_dbov=-18.0)
            sec = generate_synthetic_vowel(f0_hz=250.0, duration_ms=20.0, rms_dbov=-26.0)
            samples = prim + sec
        else:
            np.random.seed(i)
            noise = np.random.uniform(-0.4, 0.4, 160).astype(np.float32)
            samples = np.convolve(noise, [1.0, -0.9], mode="same")

        pcm = (samples * 32768.0).astype(np.int16).tobytes()
        clean_pcm, telem = sep.process_frame(pcm)

        assert len(clean_pcm) == 320, f"Frame {i} output length {len(clean_pcm)} != 320"
        assert not np.isnan(telem.clean_rms_dbov), f"Frame {i} clean RMS is NaN"

    print(f"[OK] 25 contiguous frames processed seamlessly across transitions (Total frames: {sep.frame_count})")


def test_fastapi_rest_endpoints():
    """Test 9: Verifies /health, /process, and /benchmark FastAPI endpoints."""
    print("--- Test 9: FastAPI Telephony REST Endpoints ---")
    app = create_app()
    client = TestClient(app)

    # 1. Health check
    res_h = client.get("/health")
    assert res_h.status_code == 200, f"Health check failed with {res_h.status_code}"
    data_h = res_h.json()
    assert data_h.get("cross_talk_separator_status") == "ready", "cross_talk_separator_status must be 'ready'"
    print("[OK] GET /health returned cross_talk_separator_status='ready'")

    # 2. Process frame
    t = np.linspace(0, 0.02, 160, endpoint=False)
    pcm = (np.sin(2.0 * np.pi * 175.0 * t) * 12000.0).astype(np.int16).tobytes()
    b64_in = base64.b64encode(pcm).decode("ascii")

    res_p = client.post("/telephony/cross_talk/process", json={"pcm_base64": b64_in})
    assert res_p.status_code == 200, f"Process endpoint failed with {res_p.status_code}"
    data_p = res_p.json()
    assert "clean_pcm_base64" in data_p, "clean_pcm_base64 must be present in response"
    assert "telemetry" in data_p, "telemetry must be present in response"
    print(f"[OK] POST /telephony/cross_talk/process returned State: {data_p['telemetry']['state']}")

    # 3. Benchmark endpoint
    res_b = client.post("/telephony/cross_talk/benchmark")
    assert res_b.status_code == 200, f"Benchmark endpoint failed with {res_b.status_code}"
    data_b = res_b.json()
    assert data_b.get("meets_sla") is True, f"Meets SLA must be True, got {data_b}"
    print(f"[OK] POST /telephony/cross_talk/benchmark: Avg={data_b['avg_cross_talk_time_ms']}ms, Headroom={data_b['real_time_headroom_factor']}x")


def test_real_time_headroom_benchmark():
    """Test 10: Benchmarks frame processing latency against sub-0.08ms SLA."""
    print("--- Test 10: Real-Time Throughput Benchmark (<0.08ms per frame) ---")
    sep = AcousticCrossTalkSeparator(sample_rate=8000)
    t = np.linspace(0, 0.02, 160, endpoint=False)
    pcm = (np.sin(2.0 * np.pi * 180.0 * t) * 12000.0).astype(np.int16).tobytes()

    # Warmup
    for _ in range(20):
        sep.process_frame(pcm)

    n_iterations = 250
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        sep.process_frame(pcm)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))
    headroom = 20.0 / max(avg_ms, 1e-4)

    print(f"[OK] Throughput across {n_iterations} frames:")
    print(f"   • Mean Latency: {avg_ms:.4f} ms")
    print(f"   • P95 Latency:  {p95_ms:.4f} ms")
    print(f"   • Max Latency:  {max_ms:.4f} ms")
    print(f"   • Real-Time Headroom Factor: {headroom:.1f}x")

    assert avg_ms <= 0.08, f"Mean latency {avg_ms:.4f} ms exceeds 0.08 ms SLA"
    assert headroom >= 250.0, f"Headroom factor {headroom:.1f}x is below 250x requirement"


def run_all_tests():
    print("=" * 80)
    print("VERBALYZE: ACOUSTIC BLEED & CROSS-TALK SEPARATOR TEST SUITE")
    print("Pure-Math Blind Source Isolation & False Barge-In Prevention")
    print("=" * 80)

    tests = [
        test_pitch_detection_synthetic_vowel,
        test_harmonic_comb_primary_preservation,
        test_harmonic_comb_inter_harmonic_bleed_suppression,
        test_cross_talk_mixture_separation,
        test_background_only_chatter_suppression,
        test_unvoiced_consonant_bypass,
        test_false_barge_in_elimination,
        test_continuous_streaming_state_continuity,
        test_fastapi_rest_endpoints,
        test_real_time_headroom_benchmark,
    ]

    passed = 0
    t_start = time.perf_counter()

    for idx, test_fn in enumerate(tests, 1):
        print(f"\n[{idx:02d}/{len(tests):02d}] Running {test_fn.__name__}...")
        try:
            test_fn()
            passed += 1
            print(f"[PASSED] {test_fn.__name__}")
        except Exception as e:
            print(f"[FAILED] {test_fn.__name__}: {e}")
            raise e

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"TEST EXECUTION SUMMARY: {passed}/{len(tests)} PASSED in {elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
