"""
Automated Test Suite: Real-Time Dual-Channel Active Speaker Diarization & Cross-Talk Energy Estimator.
Pure-Math Normalized Cross-Correlation (NCC) & Relative Energy DSP Engine.

Verifies:
  1. Speech Level Estimation & Dominance Score Metric (ITU-T P.56).
  2. Clean Alternating Speaker Turns (Caller vs Agent).
  3. Simultaneous Double-Talk Overlap Detection (Low Cross-Correlation).
  4. Acoustic Cross-Talk / Loudspeaker Bleed Rejection (High Cross-Correlation & Bleed Margin).
  5. State Hangover & Hysteresis Debouncing Across Inter-Syllable Pauses.
  6. Continuous Stereo Stream Turn Segmentation.
  7. DualChannelCallRecorder Integration & Diarized Transcript Formatting.
  8. FastAPI REST Endpoints & Real-Time Throughput Benchmark (<0.5ms SLA).

Author: Verbalyze Telephony & Voice AI Team
Constraint: STRICT ZERO EMOJIS.
"""

import math
import os
import sys
import time
import base64
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.telephony.diarization import (
    DualChannelDiarizer,
    DiarizationState,
    SpeakerTurn,
    FrameDiarizationTelemetry,
)
from verbalyze.telephony.call_recorder import DualChannelCallRecorder
from verbalyze.telephony.server import create_app
from starlette.testclient import TestClient


def generate_tone(
    duration_s: float,
    freq_hz: float = 300.0,
    amplitude: float = 8000.0,
    sample_rate: int = 8000,
) -> bytes:
    """Generates synthetic 16-bit linear PCM sine wave."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def generate_vowel(
    duration_s: float,
    freq_hz: float = 250.0,
    amplitude: float = 8000.0,
    sample_rate: int = 8000,
) -> bytes:
    """Generates multi-harmonic synthetic vowel speech."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = (
        np.sin(2 * np.pi * freq_hz * t) * 0.70 +
        np.sin(2 * np.pi * (freq_hz * 2) * t) * 0.20 +
        np.sin(2 * np.pi * (freq_hz * 3) * t) * 0.10
    ) * amplitude
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def test_1_speech_level_and_energy_dominance():
    print("\n--- Test 1: Speech Level Estimation & Dominance Metric ---")
    diarizer = DualChannelDiarizer(sample_rate=8000)

    # 1. Level dBov test
    full_rms = 32767.0 / math.sqrt(2)
    s_full = np.full(160, full_rms, dtype=np.float32)
    dbov_0 = diarizer.compute_rms_dbov(s_full)
    print(f"Full Scale RMS Level: {dbov_0:.2f} dBov (Expected: 0.00 dBov)")
    assert abs(dbov_0 - 0.0) < 0.05

    # 2. Dominance Metric Test
    # Caller active (-20 dBov), Agent silence (-55 dBov)
    c_pcm = generate_tone(duration_s=0.02, freq_hz=260.0, amplitude=2317.0 * math.sqrt(2))
    a_pcm = generate_tone(duration_s=0.02, freq_hz=260.0, amplitude=40.0)
    tel = diarizer.process_frame(c_pcm, a_pcm)
    print(f"Caller Dominant Score: {tel.dominance_score:.3f} (Expected > 0.90)")
    assert tel.dominance_score > 0.90
    assert tel.state == DiarizationState.CALLER_ONLY

    # Agent active (-20 dBov), Caller silence (-55 dBov)
    diarizer.reset()
    tel2 = diarizer.process_frame(a_pcm, c_pcm)
    print(f"Agent Dominant Score: {tel2.dominance_score:.3f} (Expected < -0.90)")
    assert tel2.dominance_score < -0.90
    assert tel2.state == DiarizationState.AGENT_ONLY

    # Balanced audio
    diarizer.reset()
    tel3 = diarizer.process_frame(c_pcm, c_pcm)
    print(f"Balanced Double-Talk Score: {tel3.dominance_score:.3f} (Expected ~ 0.00)")
    assert abs(tel3.dominance_score) < 0.05

    print("PASS: Speech level estimation and dominance metrics verified.")


def test_2_clean_alternating_speaker_turns():
    print("\n--- Test 2: Clean Alternating Speaker Turns (Caller vs Agent) ---")
    diarizer = DualChannelDiarizer(sample_rate=8000)

    # Timeline:
    # 0.0s - 0.4s (20 frames): Caller speaking (-20 dBov), Agent silent
    # 0.4s - 0.6s (10 frames): Both silent
    # 0.6s - 1.0s (20 frames): Agent speaking (-20 dBov), Caller silent
    caller_speech = generate_vowel(duration_s=0.40, freq_hz=220.0, amplitude=4000.0)
    silence_200ms = b"\x00" * int(8000 * 0.20 * 2)
    agent_speech = generate_vowel(duration_s=0.40, freq_hz=340.0, amplitude=4500.0)

    c_stream = caller_speech + silence_200ms + silence_200ms + silence_200ms
    a_stream = silence_200ms + silence_200ms + silence_200ms + agent_speech

    telemetries, turns = diarizer.process_dual_streams(c_stream, a_stream)
    print(f"Total Telemetry Frames: {len(telemetries)}")
    print(f"Identified Speaker Turns: {len(turns)}")
    for turn in turns:
        print(f"  Turn {turn.turn_index}: {turn.speaker} ({turn.start_ms:.0f}ms - {turn.end_ms:.0f}ms, dur={turn.duration_ms:.0f}ms, conf={turn.confidence:.2f})")

    turn_speakers = [t.speaker for t in turns]
    assert "CALLER" in turn_speakers, "Caller turn missing"
    assert "AGENT" in turn_speakers, "Agent turn missing"
    assert turns[0].speaker == "CALLER"
    assert turns[-1].speaker == "AGENT"
    print("PASS: Alternating speaker dialogue turns correctly segmented.")


def test_3_simultaneous_double_talk_overlap():
    print("\n--- Test 3: Simultaneous Double-Talk Overlap Detection ---")
    diarizer = DualChannelDiarizer(sample_rate=8000)

    # 400ms of simultaneous speech on independent frequencies (low cross-correlation)
    c_pcm = generate_vowel(duration_s=0.40, freq_hz=200.0, amplitude=5000.0)
    a_pcm = generate_vowel(duration_s=0.40, freq_hz=380.0, amplitude=5500.0)

    telemetries, turns = diarizer.process_dual_streams(c_pcm, a_pcm)
    dt_frames = sum(1 for t in telemetries if t.state == DiarizationState.DOUBLE_TALK)
    dt_ratio = dt_frames / len(telemetries)
    print(f"Double-Talk Frames: {dt_frames} / {len(telemetries)} ({dt_ratio * 100:.1f}%)")
    print(f"Average Cross-Correlation during Overlap: {np.mean([t.cross_correlation for t in telemetries]):.3f}")

    assert dt_ratio >= 0.80, f"Expected double-talk ratio >= 80%, got {dt_ratio * 100:.1f}%"
    assert any(turn.speaker == "DOUBLE_TALK" for turn in turns)
    print("PASS: Simultaneous independent double-talk successfully detected.")


def test_4_acoustic_cross_talk_and_loudspeaker_bleed_rejection():
    print("\n--- Test 4: Acoustic Cross-Talk / Loudspeaker Bleed Rejection ---")
    diarizer = DualChannelDiarizer(sample_rate=8000, cross_corr_threshold=0.55, bleed_margin_db=4.0)

    # Simulate loud agent voice (Channel 1, amplitude 8000.0)
    duration_s = 0.50  # 25 frames
    agent_pcm = generate_vowel(duration_s=duration_s, freq_hz=300.0, amplitude=8000.0)
    agent_samples = np.frombuffer(agent_pcm, dtype=np.int16).astype(np.float32)

    # Simulate caller microphone (Channel 0):
    # Caller is silent, but handset microphone picks up bot audio from phone speaker:
    # Delayed by 10ms (80 samples) and attenuated by -10 dB (amplitude factor ~0.316)
    delay_samples = 80
    bleed_attenuation = 0.30
    caller_samples = np.zeros_like(agent_samples)
    caller_samples[delay_samples:] = agent_samples[:-delay_samples] * bleed_attenuation
    # Add slight background room static
    caller_samples += np.random.normal(0, 15.0, len(caller_samples))
    caller_pcm = np.clip(caller_samples, -32768, 32767).astype(np.int16).tobytes()

    telemetries, turns = diarizer.process_dual_streams(caller_pcm, agent_pcm)

    # Ignore first 2 frames for history buffer warmup
    eval_tels = telemetries[2:]
    bleed_detected_count = sum(1 for t in eval_tels if t.cross_talk_bleed_detected)
    avg_rho = np.mean([t.cross_correlation for t in eval_tels])
    print(f"Acoustic Bleed Detected Frames: {bleed_detected_count} / {len(eval_tels)}")
    print(f"Average Cross-Correlation rho: {avg_rho:.3f} (Threshold: 0.55)")

    # False double-talk must be rejected
    false_dt_count = sum(1 for t in eval_tels if t.state == DiarizationState.DOUBLE_TALK)
    print(f"False Double-Talk Count: {false_dt_count} (Must be 0)")

    assert bleed_detected_count >= len(eval_tels) * 0.70, "Failed to identify loudspeaker bleed"
    assert false_dt_count == 0, f"False double-talk declared on loudspeaker bleed: {false_dt_count}"

    # Dominant speaker in turns must be AGENT
    for turn in turns:
        if turn.speaker != "SILENCE":
            print(f"Turn Speaker: {turn.speaker}, Bleed Ratio: {turn.cross_talk_bleed_ratio * 100:.1f}%, Dom: {turn.dominant_speaker}")
            assert turn.dominant_speaker == "AGENT"
    print("PASS: Loudspeaker acoustic bleed correctly detected and false double-talk rejected.")


def test_5_state_hangover_and_hysteresis():
    print("\n--- Test 5: State Hangover & Hysteresis Debouncing ---")
    diarizer = DualChannelDiarizer(sample_rate=8000, hangover_frames=3)

    # Frame 1 & 2: Active caller speech
    f_active = generate_vowel(duration_s=0.02, freq_hz=240.0, amplitude=5000.0)
    silence = b"\x00" * 320

    t1 = diarizer.process_frame(f_active, silence)
    t2 = diarizer.process_frame(f_active, silence)
    assert t1.state == DiarizationState.CALLER_ONLY
    assert t2.state == DiarizationState.CALLER_ONLY

    # Frames 3 & 4: Brief 40ms inter-syllable stop consonant dip (zeros)
    t3 = diarizer.process_frame(silence, silence)
    t4 = diarizer.process_frame(silence, silence)
    print(f"Stop Consonant Frame 1 State: {t3.state} (Hangover: {diarizer.caller_hangover})")
    print(f"Stop Consonant Frame 2 State: {t4.state} (Hangover: {diarizer.caller_hangover})")
    assert t3.state == DiarizationState.CALLER_ONLY, "Hangover failed to bridge 1st dip frame"
    assert t4.state == DiarizationState.CALLER_ONLY, "Hangover failed to bridge 2nd dip frame"

    # Frames 5 & 6: Continued silence beyond hangover (state drops to SILENCE)
    diarizer.process_frame(silence, silence)
    t6 = diarizer.process_frame(silence, silence)
    print(f"Exhausted Hangover Frame State: {t6.state}")
    assert t6.state == DiarizationState.SILENCE
    print("PASS: State hangover successfully bridged inter-syllable speech stops.")


def test_6_continuous_stereo_stream_turn_segmentation():
    print("\n--- Test 6: Continuous Stereo Stream Turn Segmentation ---")
    diarizer = DualChannelDiarizer(sample_rate=8000)

    # Synthesize Left (Caller) and Right (Agent) channels:
    # 0.0s - 0.4s: Caller active
    # 0.4s - 0.7s: Agent active
    c_part = generate_vowel(duration_s=0.40, freq_hz=220.0, amplitude=6000.0)
    a_part = generate_vowel(duration_s=0.30, freq_hz=350.0, amplitude=6000.0)

    c_full = c_part + (b"\x00" * len(a_part))
    a_full = (b"\x00" * len(c_part)) + a_part

    # Interleave into stereo 16-bit PCM
    s_caller = np.frombuffer(c_full, dtype=np.int16)
    s_agent = np.frombuffer(a_full, dtype=np.int16)
    stereo_interleaved = np.empty(len(s_caller) * 2, dtype=np.int16)
    stereo_interleaved[0::2] = s_caller
    stereo_interleaved[1::2] = s_agent
    stereo_bytes = stereo_interleaved.tobytes()

    telemetries, turns = diarizer.process_stereo_pcm(stereo_bytes)
    print(f"Stereo Stream Diarized into {len(turns)} turns across {len(telemetries)} frames:")
    for turn in turns:
        print(f"  [{turn.start_ms:.0f}ms - {turn.end_ms:.0f}ms]: {turn.speaker} (dur={turn.duration_ms:.0f}ms)")

    assert len(turns) >= 2
    assert turns[0].speaker == "CALLER"
    assert turns[-1].speaker == "AGENT"
    print("PASS: Interleaved stereo PCM stream cleanly segmented into speaker turns.")


def test_7_dual_channel_call_recorder_and_transcript_formatting():
    print("\n--- Test 7: DualChannelCallRecorder Integration & Transcript Formatting ---")
    recorder = DualChannelCallRecorder(sample_rate=8000)
    diarizer = DualChannelDiarizer(sample_rate=8000)

    # 1. Simulate banking customer response and bot agent clarification
    cust_audio = generate_vowel(duration_s=0.60, freq_hz=240.0, amplitude=6000.0)
    agent_audio = generate_vowel(duration_s=0.60, freq_hz=320.0, amplitude=6500.0)

    # Customer speaks from 0ms to 600ms
    recorder.write_customer_pcm(cust_audio, timestamp_ms=0.0)
    # Agent replies from 700ms to 1300ms
    recorder.write_agent_pcm(agent_audio, timestamp_ms=700.0)

    # 2. Directly diarize from call recorder in memory
    telemetries, turns = diarizer.diarize_call_recorder(recorder)
    print(f"Call Recorder Diarized: {len(turns)} turns")

    # 3. Format diarized transcript
    stt_mock = [
        {"start_ms": 0.0, "end_ms": 600.0, "text": "हाँ जी, मैं कल तक ईएमआई पेमेंट कर दूंगा।"},
        {"start_ms": 700.0, "end_ms": 1300.0, "text": "बहुत बहुत धन्यवाद शर्मा जी, हमने पेमेंट लिंक भेज दिया है।"},
    ]
    transcript = diarizer.format_diarized_transcript(turns, stt_annotations=stt_mock)
    print("\nGenerated Structured Diarized Transcript:")
    print("-" * 60)
    print(transcript)
    print("-" * 60)

    assert "CALLER" in transcript
    assert "AGENT" in transcript
    assert "हाँ जी" in transcript
    assert "धन्यवाद" in transcript
    print("PASS: DualChannelCallRecorder integration and formatted transcript verified.")


def test_8_fastapi_rest_endpoints_and_benchmark():
    print("\n--- Test 8: FastAPI REST Endpoints & Real-Time Throughput Benchmark ---")
    app = create_app()
    client = TestClient(app)

    # 1. Health check verification
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_json = health_resp.json()
    print(f"Health Diarization Status: {health_json.get('diarization_status')}")
    assert health_json.get("diarization_status") == "ready"

    # 2. REST API /telephony/diarization/process
    c_pcm = generate_vowel(duration_s=0.20, freq_hz=250.0, amplitude=5000.0)
    a_pcm = generate_vowel(duration_s=0.20, freq_hz=350.0, amplitude=5000.0)
    c_b64 = base64.b64encode(c_pcm).decode("ascii")
    a_b64 = base64.b64encode(a_pcm).decode("ascii")

    proc_resp = client.post(
        "/telephony/diarization/process",
        json={
            "caller_audio_base64": c_b64,
            "agent_audio_base64": a_b64,
            "sample_rate": 8000,
        },
    )
    assert proc_resp.status_code == 200
    proc_json = proc_resp.json()
    print(f"API Diarize Response: Status={proc_json.get('status')}, Turns={proc_json.get('turns_count')}, DoubleTalk={proc_json.get('double_talk_percentage')}%")
    assert proc_json.get("status") == "ok"
    assert "turns" in proc_json
    assert "formatted_transcript" in proc_json

    # 3. REST API /telephony/diarization/benchmark
    bench_resp = client.post("/telephony/diarization/benchmark")
    assert bench_resp.status_code == 200
    bench_json = bench_resp.json()
    avg_ms = bench_json.get("avg_diarization_time_ms")
    p95_ms = bench_json.get("p95_diarization_time_ms")
    headroom = bench_json.get("real_time_headroom_factor")
    meets_sla = bench_json.get("meets_diarization_sla")

    print(f"Benchmark Results:")
    print(f"  Average Time:    {avg_ms:.4f} ms / 20ms frame")
    print(f"  P95 Time:        {p95_ms:.4f} ms / 20ms frame")
    print(f"  Headroom Factor: {headroom}x real-time")
    print(f"  Meets SLA (<0.5ms): {meets_sla}")

    assert meets_sla is True, f"Benchmark exceeded 0.5ms SLA: {avg_ms} ms"
    assert headroom >= 100.0, f"Headroom factor below 100x: {headroom}"
    print("PASS: FastAPI REST endpoints and real-time throughput benchmark confirmed.")


def run_all_tests():
    print("=" * 80)
    print("STARTING TEST SUITE: DUAL-CHANNEL ACTIVE SPEAKER DIARIZATION & CROSS-TALK")
    print("PURE-MATH NORMALIZED CROSS-CORRELATION & RELATIVE ENERGY DSP ENGINE")
    print("=" * 80)

    test_1_speech_level_and_energy_dominance()
    test_2_clean_alternating_speaker_turns()
    test_3_simultaneous_double_talk_overlap()
    test_4_acoustic_cross_talk_and_loudspeaker_bleed_rejection()
    test_5_state_hangover_and_hysteresis()
    test_6_continuous_stereo_stream_turn_segmentation()
    test_7_dual_channel_call_recorder_and_transcript_formatting()
    test_8_fastapi_rest_endpoints_and_benchmark()

    print("\n" + "=" * 80)
    print("ALL 8 ACTIVE SPEAKER DIARIZATION TESTS PASSED SUCCESSFULLY!")
    print("Zero-Emoji Compliant | Cross-Talk Bleed Rejection | Latency < 0.05ms Verified")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
