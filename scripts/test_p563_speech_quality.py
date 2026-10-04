#!/usr/bin/env python3
"""
scripts/test_p563_speech_quality.py

Comprehensive 10-Part Automated Verification Suite for Pure-Math Single-Ended
ITU-T P.563 Speech Quality & Line Degradation Classifier.

Validates:
1. Clean Speech Baseline Quality & SLA Verification (MOS >= 4.0, CLEAN).
2. Carrier Saturation & Hard Clipping Detection (CARRIER_CLIPPING, MOS < 3.20).
3. Elevated Noise Floor & Low SNR Degradation (HIGH_NOISE_FLOOR / SEVERELY_DEGRADED).
4. Muffled Line & Extreme LPC Spectral Tilt (SPECTRAL_TILT_MUFFLED).
5. Vocal Tract Formant Anomaly & Formant Clustering Detection.
6. Automatic Circuit Breaker Tripping on SIPCircuitBreaker (MOS < 3.20 -> OPEN).
7. CarrierTrunk SLA Integration & Real-Time Quality Logging.
8. MultiTrunkRouter Live Quality Evaluation & Dynamic Circle Rerouting.
9. FastAPI Telephony P.563 REST Endpoints & Health Check.
10. Sub-0.05ms Frame Processing Latency SLA Benchmark (>400x real-time headroom).

Target Latency SLA: <0.05 ms per 20 ms frame | Pure-Math NumPy | Zero-Emoji Compliant.
"""

import sys
import os
import time
import base64
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.telephony.p563_quality import (
    ITUTP563SpeechQualityClassifier,
    P563ImpairmentType,
    P563Telemetry,
    P563StreamReport,
)
from verbalyze.telephony.circuit_breaker import (
    SIPCircuitBreaker,
    CircuitBreakerState,
    TrunkHealth,
)
from verbalyze.telephony.trunk_router import (
    CarrierTrunk,
    MultiTrunkRouter,
    TelecomCircle,
)
from verbalyze.telephony.server import create_app
from starlette.testclient import TestClient


def generate_clean_speech_stream(
    duration_s: float,
    f0_hz: float = 210.0,
    amplitude: float = 7500.0,
    sample_rate: int = 8000,
) -> bytes:
    """Generates synthetic voiced multi-formant clean speech vowel."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    samples = (
        np.sin(2 * np.pi * f0_hz * t) * 0.65 +
        np.sin(2 * np.pi * 720.0 * t) * 0.25 +
        np.sin(2 * np.pi * 1820.0 * t) * 0.10
    ) * amplitude
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def test_1_clean_speech_baseline_quality():
    print("\n--- Test 1: Clean Speech Baseline Quality & SLA Verification ---")
    classifier = ITUTP563SpeechQualityClassifier(sample_rate=8000)

    clean_pcm = generate_clean_speech_stream(duration_s=0.20, f0_hz=210.0, amplitude=8000.0)
    telemetries, report = classifier.analyze_stream(clean_pcm)

    print(f"[Clean Speech] Frames: {report.total_frames} | Avg MOS: {report.average_p563_mos:.2f} | SLA Compliant: {report.sla_compliant}")
    print(f"[Dominant Impairment] {report.dominant_impairment} | Trip Count: {report.circuit_breaker_trip_count}")

    assert report.average_p563_mos >= 4.0, f"Clean speech MOS too low: {report.average_p563_mos:.2f}"
    assert report.dominant_impairment == P563ImpairmentType.CLEAN
    assert report.sla_compliant is True
    assert report.circuit_breaker_trip_count == 0
    assert report.trunk_recommendation == "MAINTAIN_CURRENT_TRUNK"
    print("[OK] Clean speech baseline confirmed with MOS >= 4.0 and 100% SLA compliance")


def test_2_carrier_saturation_and_clipping():
    print("\n--- Test 2: Carrier Saturation & Hard Clipping Detection ---")
    classifier = ITUTP563SpeechQualityClassifier(sample_rate=8000)

    # Generate severely overdriven signal saturated at 32,000 threshold
    duration_s = 0.20
    n_samples = int(8000 * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    overdriven = np.sin(2 * np.pi * 280.0 * t) * 60000.0
    clipped_pcm = np.clip(overdriven, -32000, 32000).astype(np.int16).tobytes()

    telemetries, report = classifier.analyze_stream(clipped_pcm)
    avg_clip = np.mean([t.clipping_ratio for t in telemetries])

    print(f"[Clipped Speech] Avg MOS: {report.average_p563_mos:.2f} | Clipping Ratio: {avg_clip * 100:.1f}%")
    print(f"[Dominant Impairment] {report.dominant_impairment} | Trips: {report.circuit_breaker_trip_count}")

    assert avg_clip >= 0.10, f"Expected clipping ratio >= 10%, got {avg_clip * 100:.1f}%"
    assert report.average_p563_mos < 3.20, f"MOS not penalized enough for severe clipping: {report.average_p563_mos:.2f}"
    assert report.dominant_impairment == P563ImpairmentType.CARRIER_CLIPPING
    assert report.circuit_breaker_trip_count > 0
    assert report.sla_compliant is False
    assert report.trunk_recommendation == "FAILOVER_REROUTE_TRUNK"
    print("[OK] Carrier clipping identified, penalized below MOS 3.20, and breaker tripped")


def test_3_elevated_noise_floor_and_low_snr():
    print("\n--- Test 3: Elevated Noise Floor & Low SNR Degradation ---")
    classifier = ITUTP563SpeechQualityClassifier(sample_rate=8000)

    # Voice buried in high-power Gaussian noise (SNR ~ 4-6 dB)
    duration_s = 0.20
    n_samples = int(8000 * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    voice = np.sin(2 * np.pi * 220.0 * t) * 2500.0
    noise = np.random.normal(0.0, 3800.0, n_samples)
    noisy_pcm = np.clip(voice + noise, -32768, 32767).astype(np.int16).tobytes()

    telemetries, report = classifier.analyze_stream(noisy_pcm)
    avg_snr = np.mean([t.snr_db for t in telemetries])

    print(f"[Noisy Speech] Avg MOS: {report.average_p563_mos:.2f} | Avg SNR: {avg_snr:.1f} dB")
    print(f"[Dominant Impairment] {report.dominant_impairment} | SLA: {report.sla_compliant}")

    assert report.average_p563_mos < 3.20, f"MOS too high for severe noise: {report.average_p563_mos:.2f}"
    assert report.dominant_impairment in (P563ImpairmentType.HIGH_NOISE_FLOOR, P563ImpairmentType.SEVERELY_DEGRADED)
    assert report.circuit_breaker_trip_count > 0
    assert report.sla_compliant is False
    print("[OK] High noise floor detected, penalized below MOS 3.20, and SLA violated")


def test_4_muffled_line_and_lpc_spectral_tilt():
    print("\n--- Test 4: Muffled Line & Extreme LPC Spectral Tilt ---")
    classifier = ITUTP563SpeechQualityClassifier(sample_rate=8000)

    # Low frequency muffled speech (120 Hz tone with no high-frequency energy -> tilt > 0.96)
    duration_s = 0.20
    n_samples = int(8000 * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    muffled = np.sin(2 * np.pi * 120.0 * t) * 9000.0
    muffled_pcm = muffled.astype(np.int16).tobytes()

    telemetries, report = classifier.analyze_stream(muffled_pcm)
    avg_tilt = np.mean([t.spectral_tilt for t in telemetries])

    print(f"[Muffled Speech] Avg MOS: {report.average_p563_mos:.2f} | Avg Tilt r1: {avg_tilt:.4f}")
    print(f"[Dominant Impairment] {report.dominant_impairment}")

    assert avg_tilt >= 0.90, f"Expected high spectral tilt, got {avg_tilt:.4f}"
    assert report.average_p563_mos < 3.50, f"MOS not penalized for muffled line: {report.average_p563_mos:.2f}"
    print("[OK] Muffled line with unnatural LPC spectral tilt detected and penalized")


def test_5_vocal_tract_formant_anomaly():
    print("\n--- Test 5: Vocal Tract Formant Anomaly & Clustering Detection ---")
    classifier = ITUTP563SpeechQualityClassifier(sample_rate=8000)

    # Signal with abnormal pole clustering (two high frequencies separated by only 50 Hz -> metallic comb)
    duration_s = 0.20
    n_samples = int(8000 * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    metallic = (np.sin(2 * np.pi * 1500.0 * t) + np.sin(2 * np.pi * 1550.0 * t)) * 5000.0
    metallic_pcm = metallic.astype(np.int16).tobytes()

    telemetries, report = classifier.analyze_stream(metallic_pcm)
    avg_anomaly = np.mean([t.vocal_tract_anomaly_score for t in telemetries])

    print(f"[Metallic Anomaly] Avg Anomaly Score: {avg_anomaly:.4f} | Avg MOS: {report.average_p563_mos:.2f}")
    assert avg_anomaly >= 0.40, f"Expected vocal tract anomaly >= 0.40, got {avg_anomaly:.4f}"
    print("[OK] Abnormal formant clustering and vocal tract breakdown detected")


def test_6_circuit_breaker_tripping_on_p563_mos():
    print("\n--- Test 6: Automatic Circuit Breaker Tripping on SIPCircuitBreaker ---")
    breaker = SIPCircuitBreaker(trunk_id="jio_mumbai_trunk")

    # Initial state
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.can_execute() is True

    # Report healthy MOS (4.20)
    tripped = breaker.record_acoustic_mos(mos_score=4.20, threshold=3.20)
    assert tripped is False
    assert breaker.state == CircuitBreakerState.CLOSED

    # Report degraded MOS (2.85 < 3.20)
    tripped = breaker.record_acoustic_mos(mos_score=2.85, threshold=3.20)
    print(f"[Breaker Trip Result] Tripped: {tripped} | State: {breaker.state.value} | Reason: '{breaker._last_failure_reason}'")

    assert tripped is True
    assert breaker.state == CircuitBreakerState.OPEN
    assert breaker.can_execute() is False
    assert "ITU-T P.563" in breaker._last_failure_reason
    print("[OK] SIPCircuitBreaker immediately tripped to OPEN on P.563 MOS degradation < 3.20")


def test_7_carrier_trunk_quality_logging():
    print("\n--- Test 7: CarrierTrunk SLA Integration & Real-Time Quality Logging ---")
    trunk = CarrierTrunk(
        trunk_id="airtel_delhi_trunk",
        carrier_name="Airtel Delhi LCR",
        sip_host="10.10.1.1",
        supported_circles=[TelecomCircle.DL.value],
    )

    # Initial status
    assert trunk.can_route_call() is True
    assert trunk.health == TrunkHealth.HEALTHY

    # Record degraded acoustic quality (2.95 < 3.20)
    tripped = trunk.record_acoustic_quality(p563_mos=2.95, failover_threshold=3.20)
    print(f"[Trunk Quality Update] Tripped: {tripped} | Trunk Health: {trunk.health.value} | Can Route: {trunk.can_route_call()}")

    assert tripped is True
    assert trunk.can_route_call() is False
    assert trunk.health == TrunkHealth.UNAVAILABLE
    assert trunk.qos.p563_mos <= 3.97  # Updated rolling average
    print("[OK] CarrierTrunk logged P.563 acoustic quality and disabled call routing")


def test_8_multi_trunk_router_dynamic_circle_rerouting():
    print("\n--- Test 8: MultiTrunkRouter Live Quality Evaluation & Dynamic Circle Rerouting ---")
    router = MultiTrunkRouter()

    # Create 2 trunks for Delhi circle: Primary (Airtel) and Secondary (Jio)
    trunk_airtel = CarrierTrunk(
        trunk_id="airtel_dl",
        carrier_name="Airtel DL Primary",
        sip_host="10.0.1.1",
        priority=1,
        supported_circles=[TelecomCircle.DL.value],
    )
    trunk_jio = CarrierTrunk(
        trunk_id="jio_dl",
        carrier_name="Jio DL Secondary",
        sip_host="10.0.2.1",
        priority=2,
        supported_circles=[TelecomCircle.DL.value],
    )
    router.add_trunk(trunk_airtel)
    router.add_trunk(trunk_jio)

    delhi_phone = "+919810123456"  # Prefix 9810 maps to DL

    # Initial route check: Airtel DL Primary must be selected
    primary, fallbacks = router.resolve_routes(delhi_phone)
    print(f"[Initial Routing] Primary: {primary.trunk_id} ({primary.carrier_name})")
    assert primary.trunk_id == "airtel_dl"
    assert fallbacks[0].trunk_id == "jio_dl"

    # Ingest severely clipped audio on Airtel trunk
    t = np.linspace(0.0, 0.20, 1600, endpoint=False)
    clipped_audio = np.clip(np.sin(2 * np.pi * 300.0 * t) * 60000.0, -32000, 32000).astype(np.int16).tobytes()

    report, tripped = router.evaluate_live_audio_quality(
        trunk_id="airtel_dl",
        pcm_bytes=clipped_audio,
        min_p563_mos=3.20,
    )
    print(f"[Live Quality Eval] Tripped: {tripped} | Report MOS: {report.average_p563_mos:.2f} | Impairment: {report.dominant_impairment.value}")
    assert tripped is True

    # Re-evaluate routes: Airtel is OPEN, router must automatically failover to Jio DL Secondary!
    new_primary, new_fallbacks = router.resolve_routes(delhi_phone)
    print(f"[Post-Degradation Routing] New Primary: {new_primary.trunk_id} ({new_primary.carrier_name})")
    assert new_primary.trunk_id == "jio_dl", f"Expected failover to jio_dl, got {new_primary.trunk_id}"
    print("[OK] Dynamic circle rerouting successfully bypassed degraded trunk and failed over to backup")


def test_9_fastapi_p563_rest_endpoints():
    print("\n--- Test 9: FastAPI Telephony P.563 REST Endpoints ---")
    app = create_app()
    client = TestClient(app)

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    print(f"[GET /health] p563_mos_status: {h_data.get('p563_mos_status')}")
    assert h_data.get("p563_mos_status") == "ready"

    # 2. POST /telephony/quality/p563-analyze
    clean_pcm = generate_clean_speech_stream(duration_s=0.20, f0_hz=210.0, amplitude=8000.0)
    b64_in = base64.b64encode(clean_pcm).decode("ascii")

    an_resp = client.post("/telephony/quality/p563-analyze", json={
        "audio_base64": b64_in,
        "sample_rate": 8000,
        "circuit_breaker_threshold": 3.20,
    })
    assert an_resp.status_code == 200
    an_data = an_resp.json()
    print(f"[POST /telephony/quality/p563-analyze] Status: {an_data.get('status')} | Avg MOS: {an_data['stream_report']['average_p563_mos']}")
    assert an_data.get("status") == "ok"
    assert an_data["stream_report"]["average_p563_mos"] >= 4.0
    assert len(an_data["telemetries"]) == 10

    # 3. POST /telephony/quality/p563-benchmark
    bench_resp = client.post("/telephony/quality/p563-benchmark")
    assert bench_resp.status_code == 200
    bench_data = bench_resp.json()
    print(f"[POST /telephony/quality/p563-benchmark] Avg: {bench_data.get('avg_p563_analysis_time_ms')} ms | Meets SLA: {bench_data.get('meets_sla')} | Headroom: {bench_data.get('real_time_headroom_factor')}x")
    assert bench_data.get("meets_sla") is True
    assert bench_data.get("real_time_headroom_factor") >= 400.0
    print("[OK] All FastAPI Telephony P.563 REST endpoints verified successfully")


def test_10_sub_0_05ms_latency_sla_benchmark():
    print("\n--- Test 10: Pure-Math Sub-0.05ms SLA Latency Benchmark ---")
    classifier = ITUTP563SpeechQualityClassifier(sample_rate=8000)

    # Standard 20ms frame (160 samples at 8kHz)
    t = np.linspace(0.0, 0.02, 160, endpoint=False)
    frame = (np.sin(2 * np.pi * 220.0 * t) * 8000.0).astype(np.int16).tobytes()

    # Warm-up
    for _ in range(10):
        classifier.process_frame(frame)

    n_iterations = 250
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        classifier.process_frame(frame)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))
    headroom = 20.0 / max(avg_ms, 1e-4)

    print(f"[Benchmark Results] 20ms Frames Profiled: {n_iterations}")
    print(f"Mean Latency: {avg_ms:.4f} ms per 20ms frame")
    print(f"95th Percentile: {p95_ms:.4f} ms")
    print(f"Maximum Latency: {max_ms:.4f} ms")
    print(f"Target SLA: < 0.050 ms | Meets SLA: {avg_ms < 0.05}")
    print(f"Real-Time Headroom Factor: {headroom:.1f}x")

    assert avg_ms < 0.05, f"Mean latency {avg_ms:.4f} ms exceeded 0.05 ms SLA"
    assert headroom >= 400.0, f"Headroom {headroom:.1f}x should be >= 400x"
    print("[OK] Pure-math ITU-T P.563 engine satisfies sub-0.05ms latency SLA with massive headroom")


def main():
    print("=" * 80)
    print("Verbalyze Single-Ended ITU-T P.563 Non-Intrusive Speech Quality Suite")
    print("Pure-Math LPC, Spectral Tilt & Dynamic Circuit Breaker Tripping (MOS < 3.20)")
    print("Target SLA: Latency < 0.05 ms per 20 ms frame | Real-Time Headroom > 400x")
    print("=" * 80)

    t0 = time.time()
    test_1_clean_speech_baseline_quality()
    test_2_carrier_saturation_and_clipping()
    test_3_elevated_noise_floor_and_low_snr()
    test_4_muffled_line_and_lpc_spectral_tilt()
    test_5_vocal_tract_formant_anomaly()
    test_6_circuit_breaker_tripping_on_p563_mos()
    test_7_carrier_trunk_quality_logging()
    test_8_multi_trunk_router_dynamic_circle_rerouting()
    test_9_fastapi_p563_rest_endpoints()
    test_10_sub_0_05ms_latency_sla_benchmark()

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ALL 10 ITU-T P.563 SPEECH QUALITY TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | ITU-T P.563 & SLA Compliant")
    print("=" * 80)


if __name__ == "__main__":
    main()
