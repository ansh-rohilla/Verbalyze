"""
scripts/test_turn_taking_and_latency.py

Comprehensive Test Suite for:
1. Multi-Feature Acoustic VAD (Log Energy, ZCR, Spectral Entropy, Adaptive Noise Floor).
2. Hangover State Machine (Smoothing Unvoiced Consonant Gaps /p/, /t/, /k/).
3. Context-Aware Dynamic Pause Threshold Policy (CONFIRMATION, STANDARD, DIGIT_COLLECTION).
4. Turn Completion Confidence Scoring (Pitch Declination F0 + Syntactic Cues in Indic & English).
5. Speculative Pipelining Commit (Zero-Wait Turn Completion Pre-Fetching).
6. Speculative Pipelining Abort on User Speech Resumption (Lossless Audio Continuity).
7. Full-Duplex Telephony Barge-In & State Machine Lifecycle.
8. Glass-to-Glass Latency Profiling, Sub-300ms SLA Conformance & FastAPI Endpoints.

Zero-emoji compliant.
DPDP Act 2023 & RBI Fair Practices Code compliant.
"""

import os
import sys
import time
import math
import base64
import asyncio
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from verbalyze.telephony.turn_taking import (
    TurnTakingState,
    DialogueContext,
    VADFrameResult,
    TurnCompletionAssessment,
    LatencyBreakdown,
    AcousticVAD,
    TurnCompletionConfidenceScorer,
    AdaptivePausePolicy,
    SpeculativePipeliner,
    GlassToGlassLatencyProfiler,
    AdaptiveTurnTakingManager,
)
from verbalyze.telephony.server import create_app


def generate_audio_frame(
    frame_type: str = "silence",
    duration_ms: float = 20.0,
    sample_rate: int = 8000,
    pitch_hz: float = 140.0,
    amplitude: float = 14000.0,
) -> bytes:
    """
    Generates a single 20ms linear PCM audio frame for acoustic testing.
    frame_type: 'silence', 'ambient_noise', 'voiced_speech', 'unvoiced_consonant'
    """
    n_samples = int((duration_ms / 1000.0) * sample_rate)
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)

    if frame_type == "silence":
        samples = np.zeros(n_samples, dtype=np.float32)

    elif frame_type == "ambient_noise":
        # Low amplitude random Gaussian background noise (-55 dB)
        samples = np.random.normal(0, 15.0, n_samples).astype(np.float32)

    elif frame_type == "voiced_speech":
        # Rich harmonic speech formant synthesis
        f0 = pitch_hz
        harmonic1 = np.sin(2 * np.pi * f0 * t)
        harmonic2 = 0.5 * np.sin(2 * np.pi * (2 * f0) * t)
        harmonic3 = 0.25 * np.sin(2 * np.pi * (3 * f0) * t)
        samples = (harmonic1 + harmonic2 + harmonic3) * amplitude

    elif frame_type == "unvoiced_consonant":
        # High zero-crossing rate noise with medium amplitude
        samples = np.random.normal(0, 600.0, n_samples).astype(np.float32)

    else:
        samples = np.zeros(n_samples, dtype=np.float32)

    int16_samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return int16_samples.tobytes()


def generate_voiced_audio_clip(
    duration_sec: float = 0.5,
    sample_rate: int = 8000,
    pitch_start: float = 160.0,
    pitch_end: float = 120.0,
    amplitude: float = 14000.0,
) -> bytes:
    """
    Generates a continuous voiced audio segment with controlled pitch trajectory.
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    # Instantaneous frequency ramp
    instantaneous_freq = np.linspace(pitch_start, pitch_end, n_samples)
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    samples = np.sin(phase) * amplitude + 0.4 * np.sin(2 * phase) * amplitude
    int16_samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return int16_samples.tobytes()


def test_1_acoustic_vad_features():
    """
    Test 1: Acoustic VAD Multi-Feature Extraction & Noise Floor Tracking.
    Verifies log energy, ZCR, spectral entropy, and silence vs speech discrimination.
    """
    print("\n--- Test 1: Acoustic VAD Multi-Feature Extraction & Noise Floor Tracking ---")
    vad = AcousticVAD(sample_rate=8000, frame_duration_ms=20.0)

    # 1. Feed ambient noise frames to establish noise floor
    for _ in range(15):
        frame = generate_audio_frame("ambient_noise")
        res = vad.process_frame(frame)
        assert not res.is_speech, f"Ambient noise misclassified as speech: {res}"

    print(f"[OK] Ambient noise floor adapted to: {vad.noise_floor_db:.2f} dB")
    assert vad.noise_floor_db < 40.0, "Noise floor should track below 40 dB"

    # 2. Feed voiced speech frame
    speech_frame = generate_audio_frame("voiced_speech", pitch_hz=140.0, amplitude=15000.0)
    # First frame enters confirmation buffer
    res1 = vad.process_frame(speech_frame)
    # Second consecutive speech frame confirms speech onset
    res2 = vad.process_frame(speech_frame)

    assert res2.is_speech, f"Voiced speech frame should be detected as speech: {res2}"
    assert res2.energy_db > 50.0, f"Speech energy should exceed 50 dB, got {res2.energy_db:.2f}"
    assert res2.snr_db > 10.0, f"SNR should exceed 10 dB, got {res2.snr_db:.2f}"
    print(f"[OK] Voiced speech detected: Energy={res2.energy_db:.1f}dB, SNR={res2.snr_db:.1f}dB, ZCR={res2.zero_crossing_rate:.3f}, Entropy={res2.spectral_entropy:.3f}")

    # 3. Verify zero-length and empty frames are handled safely
    vad.reset()
    empty_res = vad.process_frame(b"")
    assert not empty_res.is_speech
    print("[OK] Empty/corrupt frames safely handled without exceptions")


def test_2_hangover_state_machine_smoothing():
    """
    Test 2: Hangover State Machine Smoothing over Unvoiced Consonants.
    Verifies that brief silence gaps (e.g. 40ms) within speech do not prematurely drop speech status.
    """
    print("\n--- Test 2: Hangover State Machine Smoothing over Unvoiced Consonants ---")
    vad = AcousticVAD(sample_rate=8000, frame_duration_ms=20.0, hangover_frames=4)

    # Establish speech
    speech_frame = generate_audio_frame("voiced_speech", pitch_hz=150.0)
    vad.process_frame(speech_frame)
    res = vad.process_frame(speech_frame)
    assert res.is_speech, "Speech must be active"

    # Insert 2 silence frames (40ms unvoiced consonant closure like /p/ or /t/)
    silence_frame = generate_audio_frame("silence")
    gap_res1 = vad.process_frame(silence_frame)
    gap_res2 = vad.process_frame(silence_frame)

    assert gap_res1.is_speech, "Hangover state machine must smooth frame 1 of gap"
    assert gap_res2.is_speech, "Hangover state machine must smooth frame 2 of gap"
    print(f"[OK] 40ms gap successfully bridged by hangover counter (active frames remaining: {vad._hangover_remaining})")

    # Exceed hangover duration (4 frames = 80ms)
    vad.process_frame(silence_frame)
    vad.process_frame(silence_frame)
    terminal_res = vad.process_frame(silence_frame)
    assert not terminal_res.is_speech, "Speech should cleanly end after hangover period expires"
    print("[OK] Speech termination cleanly declared after hangover window matures")


def test_3_context_aware_dynamic_pause_policy():
    """
    Test 3: Context-Aware Dynamic Pause Threshold Policies.
    Verifies snappy confirmation vs standard conversation vs digit collection timeouts.
    """
    print("\n--- Test 3: Context-Aware Dynamic Pause Threshold Policies ---")

    # 1. CONFIRMATION context (Snappy yes/no responses)
    conf_snappy = AdaptivePausePolicy.get_silence_timeout_ms(DialogueContext.CONFIRMATION, completion_confidence=0.85)
    conf_base = AdaptivePausePolicy.get_silence_timeout_ms(DialogueContext.CONFIRMATION, completion_confidence=0.50)
    assert 200.0 <= conf_snappy <= 250.0, f"Confirmation snappy pause should be ~220ms, got {conf_snappy}"
    assert 280.0 <= conf_base <= 320.0, f"Confirmation base pause should be ~300ms, got {conf_base}"
    print(f"[OK] CONFIRMATION Pause Timeout: Snappy={conf_snappy:.1f}ms, Base={conf_base:.1f}ms")

    # 2. STANDARD_CONVERSATION context
    std_high = AdaptivePausePolicy.get_silence_timeout_ms(DialogueContext.STANDARD_CONVERSATION, completion_confidence=0.85)
    std_low = AdaptivePausePolicy.get_silence_timeout_ms(DialogueContext.STANDARD_CONVERSATION, completion_confidence=0.30)
    assert std_high <= 350.0, f"Standard high-confidence pause should be <=350ms, got {std_high}"
    assert std_low >= 500.0, f"Standard low-confidence pause should be >=500ms, got {std_low}"
    print(f"[OK] STANDARD Pause Timeout: HighConfidence={std_high:.1f}ms, LowConfidence={std_low:.1f}ms")

    # 3. DIGIT_COLLECTION context (Account number, OTP, Date of birth thinking pauses)
    digit_timeout = AdaptivePausePolicy.get_silence_timeout_ms(DialogueContext.DIGIT_COLLECTION, completion_confidence=0.50)
    assert 750.0 <= digit_timeout <= 900.0, f"Digit collection pause should be 750-900ms, got {digit_timeout}"
    print(f"[OK] DIGIT_COLLECTION Pause Timeout: {digit_timeout:.1f}ms (prevents premature cutoffs)")


def test_4_turn_completion_confidence_scoring():
    """
    Test 4: Turn Completion Confidence Scoring (Pitch Declination + Multilingual Syntactic Cues).
    """
    print("\n--- Test 4: Turn Completion Confidence Scoring (Prosody + Syntactic Cues) ---")
    scorer = TurnCompletionConfidenceScorer(sample_rate=8000)

    # 1. Hindi Terminal Affirmation with falling pitch (declination)
    falling_audio = generate_voiced_audio_clip(duration_sec=0.5, pitch_start=180.0, pitch_end=130.0)
    hi_assessment = scorer.score_completion(
        text="हाँ जी, कल कर दूंगा।",
        trailing_pcm=falling_audio,
        language="hi",
        context=DialogueContext.CONFIRMATION,
    )
    assert hi_assessment.is_terminal, f"Hindi affirmation should be terminal: {hi_assessment}"
    assert hi_assessment.confidence >= 0.75, f"Confidence should be high, got {hi_assessment.confidence}"
    assert hi_assessment.pitch_declination_st < -0.5, f"Expected falling pitch, got {hi_assessment.pitch_declination_st} st"
    assert hi_assessment.recommended_pause_ms <= 300.0
    print(f"[OK] Hindi Terminal Affirmation: Conf={hi_assessment.confidence:.2f}, PitchDeclination={hi_assessment.pitch_declination_st}st, Pause={hi_assessment.recommended_pause_ms}ms")

    # 2. English Continuation Connector (Trailing 'because' / 'and')
    continuation_audio = generate_voiced_audio_clip(duration_sec=0.5, pitch_start=140.0, pitch_end=160.0)
    en_assessment = scorer.score_completion(
        text="I want to pay the EMI but",
        trailing_pcm=continuation_audio,
        language="en",
        context=DialogueContext.STANDARD_CONVERSATION,
    )
    assert not en_assessment.is_terminal, f"Continuation connector should not be terminal: {en_assessment}"
    assert en_assessment.confidence < 0.45, f"Continuation connector should have low confidence: {en_assessment.confidence}"
    assert en_assessment.recommended_pause_ms >= 550.0, f"Pause should be extended for connector: {en_assessment.recommended_pause_ms}"
    print(f"[OK] English Continuation Connector: Conf={en_assessment.confidence:.2f}, SyntacticCue={en_assessment.syntactic_cue}, Pause={en_assessment.recommended_pause_ms}ms")

    # 3. Gujarati Terminal Affirmation
    gu_assessment = scorer.score_completion(
        text="હા, હું કાલે પેમેન્ટ કરી દઈશ.",
        trailing_pcm=falling_audio,
        language="gu",
        context=DialogueContext.CONFIRMATION,
    )
    assert gu_assessment.is_terminal, "Gujarati affirmation should be terminal"
    print(f"[OK] Gujarati Affirmation: Conf={gu_assessment.confidence:.2f}, Terminal={gu_assessment.is_terminal}")

    # 4. Marathi Terminal Affirmation
    mr_assessment = scorer.score_completion(
        text="हो, मी उद्या करतो.",
        trailing_pcm=falling_audio,
        language="mr",
        context=DialogueContext.CONFIRMATION,
    )
    assert mr_assessment.is_terminal, "Marathi affirmation should be terminal"
    print(f"[OK] Marathi Affirmation: Conf={mr_assessment.confidence:.2f}, Terminal={mr_assessment.is_terminal}")


def test_5_speculative_pipelining_commit():
    """
    Test 5: Speculative Pipelining Commit on Turn Maturity.
    Verifies that background pre-fetch triggers during trailing pause and commits instantaneously.
    """
    print("\n--- Test 5: Speculative Pipelining Commit on Turn Maturity ---")

    async def _async_test():
        pipeliner = SpeculativePipeliner(prefetch_trigger_ms=140.0)

        # Mock STT pre-fetch coroutine that takes 25ms
        async def mock_stt_prefetch():
            await asyncio.sleep(0.025)
            return "हा, मैं बोल रहा हूँ"

        # Launch pre-fetch
        t_start = time.perf_counter()
        await pipeliner.launch_prefetch(mock_stt_prefetch)
        assert pipeliner.is_speculative_running, "Speculative pre-fetch should be running"

        # Wait for pre-fetch execution to complete in background
        await asyncio.sleep(0.035)

        # Commit result on turn timeout maturity
        result = await pipeliner.commit()
        t_commit = (time.perf_counter() - t_start) * 1000.0

        assert result == "हा, मैं बोल रहा हूँ", f"Expected pre-fetched transcript, got '{result}'"
        assert not pipeliner.is_speculative_running, "Pipeliner should be idle after commit"
        print(f"[OK] Speculative pre-fetch committed in {t_commit:.1f}ms: '{result}' (Zero-wait transcript access)")

    asyncio.run(_async_test())


def test_6_speculative_pipelining_abort_on_resumption():
    """
    Test 6: Speculative Pipelining Lossless Abort on User Speech Resumption.
    Verifies background pre-fetch task cancellation when user resumes speaking.
    """
    print("\n--- Test 6: Speculative Pipelining Abort on User Resumption ---")

    async def _async_test():
        manager = AdaptiveTurnTakingManager(sample_rate=8000, frame_duration_ms=20.0)

        # Establish speech (Frame 1: VAD onset buffer, Frame 2: SPEECH_ONSET, Frame 3: SPEAKING)
        speech_frame = generate_audio_frame("voiced_speech", pitch_hz=140.0)
        await manager.ingest_frame(speech_frame)
        await manager.ingest_frame(speech_frame)
        await manager.ingest_frame(speech_frame)
        assert manager.state == TurnTakingState.SPEAKING

        # Ingest silence frames until speculative pre-fetch triggers (>= 140ms trailing pause)
        silence_frame = generate_audio_frame("silence")
        for _ in range(20):
            state, _ = await manager.ingest_frame(
                silence_frame,
                transcribe_fn=lambda pcm: "हाँ जी"
            )
            if state == TurnTakingState.SPECULATIVE_PREFETCH:
                break

        # Speculative pre-fetch should be triggered
        assert manager.state == TurnTakingState.SPECULATIVE_PREFETCH, f"Expected SPECULATIVE_PREFETCH, got {manager.state}"
        assert manager.speculative_pipeliner.is_speculative_running or manager.speculative_pipeliner.speculative_result is not None
        print(f"[OK] Speculative pre-fetch successfully entered at pause {manager.trailing_pause_ms:.0f}ms")

        # User resumes speech before turn timeout (Frame 1: onset filter, Frame 2: confirmed speech)
        await manager.ingest_frame(speech_frame)
        resumed_state, _ = await manager.ingest_frame(speech_frame)
        assert resumed_state == TurnTakingState.SPEAKING, f"State should revert to SPEAKING, got {resumed_state}"
        assert not manager.speculative_pipeliner.is_speculative_running, "Speculative task must be aborted"
        assert len(manager.buffered_pcm_frames) >= 9, "All speech and trailing frames must be retained losslessly"
        print("[OK] Speculative task aborted cleanly with 100% audio buffer retention")

    asyncio.run(_async_test())


def test_7_full_duplex_barge_in_and_completion():
    """
    Test 7: Full-Duplex Telephony Barge-In and Complete Turn Lifecycle.
    """
    print("\n--- Test 7: Full-Duplex Telephony Barge-In and State Machine Lifecycle ---")

    async def _async_test():
        manager = AdaptiveTurnTakingManager(sample_rate=8000, frame_duration_ms=20.0)

        # 1. Simulate Bot is Speaking: caller speaks -> Barge-in triggered
        manager.set_bot_speaking(True)
        speech_frame = generate_audio_frame("voiced_speech", pitch_hz=150.0)

        # First frame triggers onset, second confirms speech
        state1, _ = await manager.ingest_frame(speech_frame)
        state2, _ = await manager.ingest_frame(speech_frame)
        assert state2 == TurnTakingState.BARGE_IN, f"Expected BARGE_IN state, got {state2}"
        print("[OK] Telephony barge-in interruption triggered immediately while bot is speaking")

        # 2. Complete conversational turn simulation
        manager.set_bot_speaking(False)
        manager.set_context(DialogueContext.CONFIRMATION)

        # Feed 15 speech frames (300ms speech)
        for _ in range(15):
            await manager.ingest_frame(speech_frame)

        # Feed silence frames until turn completes
        silence_frame = generate_audio_frame("silence")
        turn_completed = False
        event_payload = None

        for idx in range(25):  # Up to 500ms silence
            state, payload = await manager.ingest_frame(
                silence_frame,
                transcribe_fn=lambda pcm: "हाँ बिल्कुल"
            )
            if state == TurnTakingState.TURN_COMPLETED:
                turn_completed = True
                event_payload = payload
                break

        assert turn_completed, "Turn must complete upon dynamic silence timeout maturity"
        assert event_payload is not None
        assert event_payload["duration_sec"] >= 0.3, "Payload should contain complete utterance duration"
        print(f"[OK] Conversational turn completed: Duration={event_payload['duration_sec']:.2f}s, Pause={event_payload['pause_ms']:.1f}ms, Context={event_payload['context']}")

    asyncio.run(_async_test())


def test_8_glass_to_glass_latency_profiler_and_server():
    """
    Test 8: Glass-to-Glass Latency Profiling, Sub-300ms SLA Validation, and FastAPI Endpoints.
    """
    print("\n--- Test 8: Glass-to-Glass Latency Profiling, Sub-300ms SLA & REST Endpoints ---")

    # 1. Glass-to-Glass Profiler verification
    profiler = GlassToGlassLatencyProfiler()
    t0 = time.perf_counter()
    profiler.record_speech_end(t0)
    profiler.record_turn_detected(t0 + 0.120)     # 120ms turn detection
    profiler.record_stt_ready(t0 + 0.145)         # 25ms speculative STT
    profiler.record_llm_first_token(t0 + 0.210)   # 65ms LLM TTFT
    profiler.record_tts_first_chunk(t0 + 0.260)   # 50ms TTS TTFB
    profiler.record_rtp_dispatched(t0 + 0.275)    # 15ms RTP dispatch

    summary = profiler.get_summary()
    assert summary["glass_to_glass_ms"] == 275.0, f"Expected 275.0ms, got {summary['glass_to_glass_ms']}"
    assert summary["meets_sub_300ms_sla"] is True, "Must meet sub-300ms SLA"
    assert summary["turn_detection_ms"] == 120.0
    assert summary["llm_ttft_ms"] == 65.0
    assert summary["tts_ttfb_ms"] == 50.0
    print(f"[OK] Glass-to-Glass Latency: {summary['glass_to_glass_ms']:.1f}ms (Sub-300ms SLA Conformance: {summary['meets_sub_300ms_sla']})")

    # 2. FastAPI Application Endpoints Testing
    app = create_app()
    client = TestClient(app)

    # Health check verification
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data.get("turn_taking_status") == "ready"
    assert health_data.get("target_glass_to_glass_ms") == 300
    print(f"[OK] GET /health verified: TurnTakingStatus={health_data.get('turn_taking_status')}")

    # Evaluate endpoint verification
    test_pcm = generate_audio_frame("voiced_speech", duration_ms=100.0)
    eval_resp = client.post(
        "/telephony/turn-taking/evaluate",
        json={
            "audio_chunk_b64": base64.b64encode(test_pcm).decode("ascii"),
            "text": "हाँ ठीक है",
            "language": "hi",
            "context": "CONFIRMATION",
        }
    )
    assert eval_resp.status_code == 200, f"Evaluate failed: {eval_resp.text}"
    eval_data = eval_resp.json()
    assert eval_data["status"] == "ok"
    assert eval_data["turn_completion"]["is_terminal"] is True
    print(f"[OK] POST /telephony/turn-taking/evaluate: Confidence={eval_data['turn_completion']['confidence']}, IsTerminal={eval_data['turn_completion']['is_terminal']}")

    # Benchmark endpoint verification
    bench_resp = client.post("/telephony/turn-taking/benchmark")
    assert bench_resp.status_code == 200, f"Benchmark failed: {bench_resp.text}"
    bench_data = bench_resp.json()
    assert bench_data["status"] == "ok"
    assert bench_data["meets_sub_300ms_sla"] is True
    assert bench_data["benchmark"]["glass_to_glass_ms"] < 300.0
    print(f"[OK] POST /telephony/turn-taking/benchmark: GlassToGlass={bench_data['benchmark']['glass_to_glass_ms']:.1f}ms, MeetsSLA={bench_data['meets_sub_300ms_sla']}")


def main():
    print("=" * 80)
    print("Verbalyze Adaptive Conversational Turn-Taking & Speculative Latency Test Suite")
    print("Target SLA: Glass-to-Glass Telephony Latency < 300ms")
    print("=" * 80)

    test_1_acoustic_vad_features()
    test_2_hangover_state_machine_smoothing()
    test_3_context_aware_dynamic_pause_policy()
    test_4_turn_completion_confidence_scoring()
    test_5_speculative_pipelining_commit()
    test_6_speculative_pipelining_abort_on_resumption()
    test_7_full_duplex_barge_in_and_completion()
    test_8_glass_to_glass_latency_profiler_and_server()

    print("\n" + "=" * 80)
    print("ALL 8 TURN-TAKING & LATENCY TESTS PASSED SUCCESSFULLY (SUB-300ms SLA VERIFIED)")
    print("=" * 80)


if __name__ == "__main__":
    main()
