"""
scripts/test_voice_stress_and_sarcasm.py

Comprehensive Test Suite for Acoustic Sarcasm, Distress & Coercion Detector:
1. Lippold Physiological Micro-Tremor Baseline in Calm Speech (8-14 Hz).
2. Acute Vocal Distress, Sympathetic Tremor Suppression & Pitch Jitter.
3. Sarcastic Pitch Velocity & Syllabic Vowel Elongation Index.
4. Sarcastic Assent Polarity Discrepancy Gate (Overriding False Positive PTPs).
5. Coercion & Panic Alert Circuit Breaker (RBI Fair Practices Escalation).
6. Multi-Lingual Indic Sarcasm Prosodic Contours (Hindi, Hinglish, Marathi).
7. UnifiedSentimentEngine Acoustic-Lexical Integration.
8. SupervisorManager Real-Time Observability & Takeover Alert.
9. FastAPI Telephony Endpoints (/health, /analyze, /benchmark).
10. Ultra-Low Frame Latency SLA (<0.05ms) & Real-Time Headroom (>400x).

Zero-emoji compliant.
RBI Fair Practices Code for NBFCs/Lenders & ITU-T P.59 compliant.
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

from verbalyze.telephony.voice_stress import (
    VoiceStressAndSarcasmDetector,
    StressCategory,
    ComplianceAction,
    VoiceStressTelemetry,
    LippoldMicroTremorFilter,
)
from verbalyze.agent.sentiment import (
    UnifiedSentimentEngine,
    SentimentCategory,
    DisputeType,
)
from verbalyze.telephony.supervisor import SupervisorManager
from verbalyze.telephony.server import create_app


def synthesize_tone_frame(
    f0_hz: float,
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
    amplitude: float = 0.5,
    tremor_hz: float = 10.0,
    tremor_depth: float = 0.0,
) -> np.ndarray:
    """Synthesizes a 20ms voiced speech frame with optional micro-tremor envelope modulation."""
    n_samples = int((duration_ms / 1000.0) * sample_rate)
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    # Fundamental plus harmonics
    harmonics = (
        0.6 * np.sin(2.0 * np.pi * f0_hz * t)
        + 0.3 * np.sin(2.0 * np.pi * 2.0 * f0_hz * t)
        + 0.1 * np.sin(2.0 * np.pi * 3.0 * f0_hz * t)
    )
    mod = 1.0 + tremor_depth * np.sin(2.0 * np.pi * tremor_hz * t)
    sig = amplitude * harmonics * mod
    return sig.astype(np.float32)


def test_lippold_micro_tremor_calm_speech():
    """
    Validates that calm, relaxed speech maintains a natural 8-14 Hz Lippold micro-tremor
    modulation baseline without triggering distress or sarcasm alerts.
    """
    detector = VoiceStressAndSarcasmDetector(sample_rate=8000)

    # Stream 30 consecutive frames (~600ms) of calm speech with 10 Hz micro-tremor
    telems = []
    for i in range(30):
        # 10 Hz subtle tremor modulation (depth 0.08)
        frame = synthesize_tone_frame(
            f0_hz=140.0,
            duration_ms=20.0,
            sample_rate=8000,
            amplitude=0.35,
            tremor_hz=10.0,
            tremor_depth=0.08,
        )
        pcm_bytes = (frame * 32767.0).astype(np.int16).tobytes()
        t = detector.process_frame(pcm_bytes)
        telems.append(t)

    last = telems[-1]
    print("[OK] Calm Speech Micro-Tremor Baseline:")
    print(f"   • Distress Score:           {last.distress_score:.3f} (Required: < 0.40)")
    print(f"   • Sarcasm Score:            {last.sarcasm_score:.3f} (Required: < 0.40)")
    print(f"   • Stress Category:          {last.stress_category.value} (Expected: CALM)")
    print(f"   • Compliance Action:        {last.recommended_action.value} (Expected: PROCEED_NORMAL)")

    assert last.distress_score < 0.40, f"Calm speech distress score {last.distress_score} was too high"
    assert last.sarcasm_score < 0.40, f"Calm speech sarcasm score {last.sarcasm_score} was too high"
    assert last.stress_category == StressCategory.CALM, f"Unexpected category {last.stress_category}"
    assert last.recommended_action == ComplianceAction.PROCEED_NORMAL


def test_acute_stress_tremor_suppression_and_jitter():
    """
    Validates that acute caller distress (shouting loudness + jitter + rigid tremor suppression)
    drives elevated distress scores and trips cooling-off / supervisor alert actions.
    """
    detector = VoiceStressAndSarcasmDetector(sample_rate=8000)

    # Stream 25 frames of high loudness with erratic jumping pitch (pitch jitter) and rigid tremor
    telems = []
    for i in range(25):
        # Jittery frequency jumping between 260 Hz and 340 Hz (shouting screech)
        jitter_f0 = 280.0 + 50.0 * (1.0 if (i % 2 == 0) else -1.0)
        # High amplitude (loudness ~ -6 dBov) with zero natural micro-tremor
        frame = synthesize_tone_frame(
            f0_hz=jitter_f0,
            duration_ms=20.0,
            sample_rate=8000,
            amplitude=0.85,
            tremor_hz=0.0,
            tremor_depth=0.0,
        )
        pcm_bytes = (frame * 32767.0).astype(np.int16).tobytes()
        t = detector.process_frame(pcm_bytes)
        telems.append(t)

    peak_distress = max(t.distress_score for t in telems)
    last = telems[-1]

    print("[OK] Acute Distress & Jitter Detection:")
    print(f"   • Peak Distress Score:      {peak_distress:.3f} (Required: >= 0.65)")
    print(f"   • Pitch Jitter:             {last.pitch_jitter:.4f}")
    print(f"   • Raw Loudness:             {last.raw_rms_dbov:.2f} dBov")
    print(f"   • Final Stress Category:    {last.stress_category.value}")

    assert peak_distress >= 0.65, f"Peak distress {peak_distress} failed to meet 0.65 threshold"
    assert last.stress_category in (StressCategory.ACUTE_DISTRESS, StressCategory.COERCION_PANIC)
    assert last.recommended_action in (ComplianceAction.COOLING_OFF_PAUSE, ComplianceAction.IMMEDIATE_SUPERVISOR_TAKEOVER)


def test_sarcastic_pitch_velocity_and_vowel_elongation():
    """
    Validates that prolonged vowel elongation coupled with melodic pitch glide
    yields high sarcasm scores and triggers sarcastic dispute flagging.
    """
    detector = VoiceStressAndSarcasmDetector(sample_rate=8000)

    # Simulate elongated mocking vowel ("Haaan...") across 20 frames (~400ms)
    # Slow descending pitch glide: 220 Hz down to 140 Hz (glide span = 80 Hz)
    telems = []
    for i in range(20):
        glide_f0 = 220.0 - (float(i) / 20.0) * 80.0
        frame = synthesize_tone_frame(
            f0_hz=glide_f0,
            duration_ms=20.0,
            sample_rate=8000,
            amplitude=0.45,
        )
        pcm_bytes = (frame * 32767.0).astype(np.int16).tobytes()
        t = detector.process_frame(pcm_bytes)
        telems.append(t)

    peak_sarcasm = max(t.sarcasm_score for t in telems)
    last = telems[-1]

    print("[OK] Sarcastic Pitch Glide & Vowel Elongation:")
    print(f"   • Peak Sarcasm Score:       {peak_sarcasm:.3f} (Required: >= 0.55)")
    print(f"   • Vowel Elongation Index:   {last.vowel_elongation_index:.3f}")
    print(f"   • Stress Category:          {last.stress_category.value}")
    print(f"   • Recommended Action:       {last.recommended_action.value}")

    assert peak_sarcasm >= 0.55, f"Peak sarcasm score {peak_sarcasm} below 0.55"
    assert last.stress_category == StressCategory.CONTROLLED_SARCASTIC
    assert last.recommended_action == ComplianceAction.FLAG_SARCASTIC_DISPUTE


def test_sarcastic_assent_discrepancy_flip():
    """
    Validates that when a borrower enters an affirmative transcript ("Haan haan zaroor de dunga... kal aana")
    with sarcastic prosody, the detector flags the polarity discrepancy as SARCASTIC_ASSENT.
    """
    detector = VoiceStressAndSarcasmDetector(sample_rate=8000)

    # 15 frames of elongated pitch glide with sarcastic text
    frames = []
    for i in range(15):
        glide_f0 = 200.0 - (float(i) / 15.0) * 60.0
        frame = synthesize_tone_frame(f0_hz=glide_f0, duration_ms=20.0, sample_rate=8000, amplitude=0.40)
        frames.append(frame)

    pcm_utterance = (np.concatenate(frames) * 32767.0).astype(np.int16).tobytes()
    transcript = "Haan haan zaroor de dunga... kal aana"

    _, aggregate = detector.process_utterance(pcm_utterance, lexical_transcript=transcript)

    print("[OK] Sarcastic Assent Polarity Discrepancy:")
    print(f"   • Transcript:               '{transcript}'")
    print(f"   • Sarcasm Score:            {aggregate.sarcasm_score:.3f}")
    print(f"   • Is Sarcastic Assent:      {aggregate.is_sarcastic_assent}")
    print(f"   • Category:                 {aggregate.stress_category.value}")
    print(f"   • Recommended Action:       {aggregate.recommended_action.value}")

    assert aggregate.is_sarcastic_assent is True, "Failed to flag sarcastic assent discrepancy"
    assert aggregate.sarcasm_score >= 0.70, f"Expected elevated sarcasm score >= 0.70, got {aggregate.sarcasm_score}"
    assert aggregate.recommended_action == ComplianceAction.FLAG_SARCASTIC_DISPUTE


def test_coercion_panic_circuit_breaker():
    """
    Validates that sustained extreme distress (>0.75 across >=3 consecutive frames)
    trips the RBI Fair Practices COERCION_PANIC circuit breaker.
    """
    detector = VoiceStressAndSarcasmDetector(sample_rate=8000)

    # Stream 6 consecutive frames of severe vocal panic (loudness + extreme jitter)
    coercion_triggered = False
    for i in range(6):
        jitter_f0 = 320.0 + 80.0 * (1.0 if (i % 2 == 0) else -1.0)
        frame = synthesize_tone_frame(f0_hz=jitter_f0, duration_ms=20.0, sample_rate=8000, amplitude=0.95)
        pcm_bytes = (frame * 32767.0).astype(np.int16).tobytes()
        t = detector.process_frame(pcm_bytes)
        if t.coercion_alert:
            coercion_triggered = True

    print("[OK] Coercion & Panic Alert Circuit Breaker:")
    print(f"   • Coercion Triggered:       {coercion_triggered}")
    print(f"   • Coercion Streak:          {detector.coercion_streak}")
    print(f"   • Recommended Action:       {t.recommended_action.value}")

    assert coercion_triggered is True, "Coercion alert circuit breaker was not triggered"
    assert t.stress_category == StressCategory.COERCION_PANIC
    assert t.recommended_action == ComplianceAction.IMMEDIATE_SUPERVISOR_TAKEOVER


def test_multilingual_indic_sarcasm_contours():
    """
    Tests detector handling of diverse Indic sarcastic expressions:
    - Hindi: 'Haan bilkul le jao sab'
    - Hinglish: 'Sure sure why not kal aana'
    - Marathi / Regional: 'Arre waah sab de dunga'
    """
    detector = VoiceStressAndSarcasmDetector(sample_rate=8000)

    test_cases = [
        ("Haan bilkul le jao sab", 180.0, 70.0),
        ("Sure sure why not kal aana", 210.0, 80.0),
        ("Arre waah sab de dunga", 195.0, 65.0),
    ]

    for text, start_f0, glide_span in test_cases:
        detector.reset()
        frames = []
        for i in range(15):
            f0 = start_f0 - (float(i) / 15.0) * glide_span
            frame = synthesize_tone_frame(f0_hz=f0, duration_ms=20.0, sample_rate=8000, amplitude=0.45)
            frames.append(frame)

        pcm = (np.concatenate(frames) * 32767.0).astype(np.int16).tobytes()
        _, agg = detector.process_utterance(pcm, lexical_transcript=text)

        assert agg.is_sarcastic_assent is True, f"Failed on text: '{text}'"
        assert agg.sarcasm_score >= 0.70

    print(f"[OK] Multi-Lingual Indic Sarcasm verified across {len(test_cases)} expressions.")


def test_unified_sentiment_engine_integration():
    """
    Validates end-to-end integration with UnifiedSentimentEngine:
    - Acoustic sarcasm flips false positive calm commitments into disputes.
    - Acute vocal distress elevates sentiment to CRITICAL with transfer recommended.
    """
    engine = UnifiedSentimentEngine(sample_rate=8000)

    # 1. Sarcastic assent: affirmative text + sarcastic audio
    frames = []
    for i in range(15):
        f0 = 210.0 - (float(i) / 15.0) * 75.0
        frame = synthesize_tone_frame(f0_hz=f0, duration_ms=20.0, sample_rate=8000, amplitude=0.45)
        frames.append(frame)
    sarcastic_pcm = (np.concatenate(frames) * 32767.0).astype(np.int16).tobytes()

    res_sarcastic = engine.analyze(
        transcript="Haan haan zaroor kal aana",
        pcm_bytes=sarcastic_pcm,
    )

    assert res_sarcastic.is_sarcastic is True
    assert "sarcastic_acoustic_prosody" in res_sarcastic.detected_cues
    assert res_sarcastic.category in (SentimentCategory.ELEVATED, SentimentCategory.AGITATED)
    assert res_sarcastic.dispute_type == DisputeType.PAYMENT_DISPUTE

    # 2. Acute panic: screaming audio
    panic_frames = []
    for i in range(12):
        f0 = 340.0 + 80.0 * (1.0 if (i % 2 == 0) else -1.0)
        frame = synthesize_tone_frame(f0_hz=f0, duration_ms=20.0, sample_rate=8000, amplitude=0.95)
        panic_frames.append(frame)
    panic_pcm = (np.concatenate(panic_frames) * 32767.0).astype(np.int16).tobytes()

    res_panic = engine.analyze(
        transcript="Please stop calling me",
        pcm_bytes=panic_pcm,
    )

    assert res_panic.category == SentimentCategory.CRITICAL
    assert res_panic.transfer_recommended is True
    assert res_panic.coercion_detected is True
    assert "coercion_distress_alert" in res_panic.detected_cues

    print("[OK] UnifiedSentimentEngine Integration:")
    print(f"   • Sarcastic Cue:            {res_sarcastic.detected_cues}")
    print(f"   • Panic Category:           {res_panic.category.value}")
    print(f"   • Panic Transfer Rec:       {res_panic.transfer_recommended}")


def test_supervisor_manager_coercion_alert():
    """
    Validates that SupervisorManager receives voice stress alerts and updates call state
    to SUPERVISOR_TAKEOVER_REQUIRED.
    """
    mgr = SupervisorManager()
    record = mgr.register_call(call_id="call_stress_999", caller_phone="9876543210")

    # Update with coercion alert
    mgr.update_voice_stress(
        call_id="call_stress_999",
        distress_score=0.88,
        sarcasm_score=0.10,
        stress_category="COERCION_PANIC",
        coercion_alert=True,
    )

    updated = mgr.active_calls.get("call_stress_999")
    assert updated is not None
    assert updated.distress_score == 0.88
    assert updated.coercion_alert is True
    assert updated.stress_category == "COERCION_PANIC"
    assert updated.status == "SUPERVISOR_TAKEOVER_REQUIRED"

    print("[OK] SupervisorManager Coercion Takeover:")
    print(f"   • Status:                   {updated.status}")
    print(f"   • Distress Score:           {updated.distress_score}")
    print(f"   • Coercion Alert:           {updated.coercion_alert}")


def test_fastapi_stress_endpoints():
    """
    Validates FastAPI REST endpoints (/health, /telephony/stress/analyze, /telephony/stress/benchmark).
    """
    app = create_app()
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data.get("voice_stress_detector_status") == "ready"

    # 2. Analyze endpoint
    frame = synthesize_tone_frame(f0_hz=160.0, duration_ms=40.0, sample_rate=8000, amplitude=0.4)
    pcm_bytes = (frame * 32767.0).astype(np.int16).tobytes()
    pcm_b64 = base64.b64encode(pcm_bytes).decode("ascii")

    res_analyze = client.post(
        "/telephony/stress/analyze",
        json={"pcm_base64": pcm_b64, "text": "Haan zaroor"},
    )
    assert res_analyze.status_code == 200, f"Analyze failed: {res_analyze.text}"
    analyze_data = res_analyze.json()
    assert analyze_data["status"] == "ok"
    assert "distress_score" in analyze_data
    assert "sarcasm_score" in analyze_data
    assert "stress_category" in analyze_data

    # 3. Benchmark endpoint
    res_bench = client.post("/telephony/stress/benchmark")
    assert res_bench.status_code == 200, f"Benchmark failed: {res_bench.text}"
    bench_data = res_bench.json()
    assert bench_data["status"] == "ok"

    avg_ms = bench_data["avg_stress_analysis_time_ms"]
    p95_ms = bench_data["p95_stress_analysis_time_ms"]
    headroom = bench_data["real_time_headroom_factor"]

    print("[OK] FastAPI Stress Endpoints & Benchmark:")
    print(f"   • Mean Latency:             {avg_ms:.4f} ms (Target: < 0.0500 ms)")
    print(f"   • P95 Latency:              {p95_ms:.4f} ms")
    print(f"   • Real-Time Headroom:       {headroom:.1f}x (Target: >= 400x)")

    assert avg_ms <= 0.05, f"Mean latency {avg_ms:.4f} ms exceeds 0.05 ms SLA"
    assert headroom >= 400.0, f"Headroom factor {headroom:.1f}x below 400x threshold"


def test_sub_005ms_throughput_and_headroom():
    """
    Validates raw DSP throughput across 250 contiguous frames.
    """
    detector = VoiceStressAndSarcasmDetector(sample_rate=8000)
    frame = synthesize_tone_frame(f0_hz=160.0, duration_ms=20.0, sample_rate=8000, amplitude=0.4)
    pcm_bytes = (frame * 32767.0).astype(np.int16).tobytes()

    # Warmup
    for _ in range(10):
        detector.process_frame(pcm_bytes)

    n_iterations = 250
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        detector.process_frame(pcm_bytes)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))
    headroom = 20.0 / max(avg_ms, 1e-4)

    print(f"[OK] Pure-Math DSP Throughput across {n_iterations} frames:")
    print(f"   • Mean Frame Latency:       {avg_ms:.4f} ms")
    print(f"   • P95 Latency:              {p95_ms:.4f} ms")
    print(f"   • Max Latency:              {max_ms:.4f} ms")
    print(f"   • Real-Time Headroom:       {headroom:.1f}x")

    assert avg_ms <= 0.05, f"Mean latency {avg_ms:.4f} ms exceeds 0.05 ms requirement"
    assert headroom >= 400.0, f"Headroom factor {headroom:.1f}x is below 400x requirement"


def run_all_tests():
    print("=" * 80)
    print("VERBALYZE: ACOUSTIC SARCASM, DISTRESS & COERCION DETECTOR TEST SUITE")
    print("ITU-T P.59 Voice Stress Analysis & RBI Fair Practices Compliance")
    print("=" * 80)

    tests = [
        test_lippold_micro_tremor_calm_speech,
        test_acute_stress_tremor_suppression_and_jitter,
        test_sarcastic_pitch_velocity_and_vowel_elongation,
        test_sarcastic_assent_discrepancy_flip,
        test_coercion_panic_circuit_breaker,
        test_multilingual_indic_sarcasm_contours,
        test_unified_sentiment_engine_integration,
        test_supervisor_manager_coercion_alert,
        test_fastapi_stress_endpoints,
        test_sub_005ms_throughput_and_headroom,
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
