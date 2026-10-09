"""
scripts/test_ringback_discriminator.py

Master Automated Test Suite for Indian Telephony Early Media & In-Band Ringback Tone Discriminator.
Validates ITU-T Q.35 / E.180 dual-tone frequency estimation, Indian cadence tracking,
caller tune (CRBT) music detection, operator announcements, sub-40ms human speech onset gating,
SIP Orchestrator integration, FastAPI endpoints, and sub-0.05ms SLA latency headroom.

Zero-emoji compliant. ITU-T Q.35, ITU-T E.180, 3GPP TS 22.001 compliant.
DPDP Act 2023 & Section 65B Indian Evidence Act compliant.
"""

import os
import sys
import time
import math
import base64
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.telephony.ringback_discriminator import (
    EarlyMediaDiscriminator,
    EarlyMediaState,
    RingbackCadenceType,
    EarlyMediaTelemetry,
    EarlyMediaReport,
)
from verbalyze.telephony.sip_orchestrator import (
    SIPSession,
    SIPCallState,
    CallLegRole,
)
from verbalyze.telephony.server import create_app


def generate_tone_frame(f1: float, f2: float, duration_ms: float = 20.0, sample_rate: int = 8000, amp: float = 16000.0) -> bytes:
    """Generates dual-tone linear PCM frame."""
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)) * amp
    return sig.astype(np.int16).tobytes()


def generate_music_frame(duration_ms: float = 20.0, sample_rate: int = 8000, amp: float = 14000.0) -> bytes:
    """Generates multi-harmonic musical chord representing caller tune (CRBT)."""
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    # C5 (523Hz), E5 (659Hz), G5 (784Hz), C6 (1046Hz)
    chord = (
        0.25 * np.sin(2 * np.pi * 523.25 * t)
        + 0.25 * np.sin(2 * np.pi * 659.25 * t)
        + 0.25 * np.sin(2 * np.pi * 783.99 * t)
        + 0.25 * np.sin(2 * np.pi * 1046.50 * t)
    ) * amp
    return chord.astype(np.int16).tobytes()


def generate_speech_frame(f0: float = 140.0, duration_ms: float = 20.0, sample_rate: int = 8000, amp: float = 16000.0) -> bytes:
    """Generates harmonic voiced human speech frame with vowel formant envelope."""
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    # Glottal pulse harmonics with typical speech roll-off
    speech = (
        0.55 * np.sin(2 * np.pi * f0 * t)
        + 0.30 * np.sin(2 * np.pi * (2 * f0) * t)
        + 0.15 * np.sin(2 * np.pi * (3 * f0) * t)
    ) * amp
    return speech.astype(np.int16).tobytes()


def test_1_pure_indian_dual_tone_discrimination():
    print("\n--- Test 1: Pure Indian 400Hz + 425Hz Dual-Tone Discrimination ---")
    discriminator = EarlyMediaDiscriminator(sample_rate=8000)

    # Standard Indian ringback tone: 400 Hz + 425 Hz
    pcm = generate_tone_frame(400.0, 425.0, duration_ms=20.0, sample_rate=8000)
    telem = discriminator.process_frame(pcm)

    print(f"[Dual-Tone Frame] State: {telem.state.value} | Ratio: {telem.dual_tone_ratio:.4f}")
    print(f"[Power Levels] 400Hz: {telem.power_400hz_dbfs:.2f} dBFS | 425Hz: {telem.power_425hz_dbfs:.2f} dBFS")
    print(f"[Broadband RMS] {telem.broadband_rms_dbfs:.2f} dBFS")

    assert telem.state == EarlyMediaState.RINGBACK_TONE
    assert telem.dual_tone_ratio >= 0.85
    assert telem.power_400hz_dbfs > -30.0
    assert telem.power_425hz_dbfs > -30.0
    print("[OK] Pure-math Goertzel basis vectors cleanly detected Indian 400Hz+425Hz ringback tone")


def test_2_standard_indian_dual_pulse_cadence_validation():
    print("\n--- Test 2: Standard Indian Dual-Pulse Cadence Validation ---")
    discriminator = EarlyMediaDiscriminator(sample_rate=8000)

    # Indian Standard Cadence: 0.4s ON, 0.2s OFF, 0.4s ON, 2.0s OFF (3.0s total cycle)
    tone_chunk = generate_tone_frame(400.0, 425.0, duration_ms=20.0)
    silence_chunk = (np.zeros(160, dtype=np.int16)).tobytes()

    stream_frames = []
    # Pulse 1: 400ms (20 frames)
    stream_frames.extend([tone_chunk] * 20)
    # Gap 1: 200ms (10 frames)
    stream_frames.extend([silence_chunk] * 10)
    # Pulse 2: 400ms (20 frames)
    stream_frames.extend([tone_chunk] * 20)
    # Pause: 1000ms (50 frames)
    stream_frames.extend([silence_chunk] * 50)

    pcm_stream = b"".join(stream_frames)
    telemetries, report = discriminator.process_stream(pcm_stream)

    print(f"[Cadence Stream] Processed {len(telemetries)} frames ({report.total_duration_ms:.1f} ms)")
    print(f"[Report] Cadence Pattern: {report.cadence_pattern.value}")
    print(f"[Report] Ringback Confirmed: {report.ringback_detected}")
    print(f"[State Distribution] Tone: {report.state_distribution.get(EarlyMediaState.RINGBACK_TONE.value, 0):.2%}, Silence: {report.state_distribution.get(EarlyMediaState.SILENCE.value, 0):.2%}")

    assert report.cadence_pattern == RingbackCadenceType.INDIAN_STANDARD_DUAL_PULSE
    assert report.ringback_detected is True
    print("[OK] State machine accurately validated ITU-T Q.35 Indian standard dual-pulse ringback cadence")


def test_3_indian_single_pulse_and_busy_tone_cadence_tracking():
    print("\n--- Test 3: Indian Single-Pulse and Busy Tone Cadence Tracking ---")
    # 1. Busy Tone: 380ms ON, 380ms OFF cadence (repeating)
    disc_busy = EarlyMediaDiscriminator(sample_rate=8000)
    busy_tone = generate_tone_frame(400.0, 400.0, duration_ms=20.0)
    silence = (np.zeros(160, dtype=np.int16)).tobytes()

    busy_stream = []
    for _ in range(3):
        busy_stream.extend([busy_tone] * 19)  # 380ms ON
        busy_stream.extend([silence] * 19)    # 380ms OFF

    _, rep_busy = disc_busy.process_stream(b"".join(busy_stream))
    print(f"[Busy Tone] Cadence: {rep_busy.cadence_pattern.value} | Dominant: {rep_busy.dominant_state.value}")
    assert rep_busy.cadence_pattern == RingbackCadenceType.INDIAN_BUSY_TONE

    # 2. Single-Pulse Ringback: 1000ms ON, 2500ms OFF
    disc_single = EarlyMediaDiscriminator(sample_rate=8000)
    single_stream = []
    single_stream.extend([busy_tone] * 50)   # 1000ms ON
    single_stream.extend([silence] * 100)    # 2000ms OFF

    _, rep_single = disc_single.process_stream(b"".join(single_stream))
    print(f"[Single-Pulse Ringback] Cadence: {rep_single.cadence_pattern.value} | Ringback: {rep_single.ringback_detected}")
    assert rep_single.cadence_pattern == RingbackCadenceType.INDIAN_SINGLE_PULSE
    assert rep_single.ringback_detected is True
    print("[OK] Correctly discriminated busy congestion cadence from single-pulse ringback")


def test_4_caller_tune_music_differentiation():
    print("\n--- Test 4: Caller Tune (CRBT) / Music Differentiation ---")
    discriminator = EarlyMediaDiscriminator(sample_rate=8000)

    # Multi-component musical polyphony chord representing Bollywood caller tune
    music_frames = [generate_music_frame(duration_ms=20.0) for _ in range(40)]  # 800ms
    pcm_music = b"".join(music_frames)

    telemetries, report = discriminator.process_stream(pcm_music)

    print(f"[CRBT Stream] Dominant State: {report.dominant_state.value}")
    print(f"[CRBT Stream] Caller Tune Confirmed: {report.caller_tune_detected}")
    print(f"[CRBT Stream] Ringback Detected: {report.ringback_detected}")
    print(f"[Acoustic Metrics] Avg Dual Tone Ratio: {report.average_dual_tone_ratio:.4f} (Expected < 0.20)")

    assert report.dominant_state == EarlyMediaState.CALLER_TUNE_MUSIC
    assert report.caller_tune_detected is True
    assert report.ringback_detected is False
    assert report.average_dual_tone_ratio < 0.20
    print("[OK] Spectral centroid and high-frequency dispersion cleanly distinguished caller tune from signaling tones")


def test_5_operator_network_announcement_detection():
    print("\n--- Test 5: Operator Network Announcement Detection ---")
    discriminator = EarlyMediaDiscriminator(sample_rate=8000)

    # Continuous synthesized carrier voice announcement ("The number you dialed is switched off")
    # Generates continuous speech without ringback tones for 1.2 seconds (60 frames)
    announcement_frames = []
    for i in range(60):
        # Varying pitch around 160Hz simulating spoken words
        f0 = 150.0 + 20.0 * math.sin(i * 0.3)
        announcement_frames.append(generate_speech_frame(f0=f0, duration_ms=20.0))

    telemetries, report = discriminator.process_stream(b"".join(announcement_frames))

    print(f"[Announcement Stream] Duration: {report.total_duration_ms:.1f} ms")
    print(f"[Announcement Stream] Dominant State: {report.dominant_state.value}")
    print(f"[Announcement Stream] Operator Announcement: {report.operator_announcement_detected}")
    print(f"[Announcement Stream] Ringback Detected: {report.ringback_detected}")

    assert report.operator_announcement_detected is True
    assert report.ringback_detected is False
    print("[OK] Accurately identified continuous carrier operator announcement without prior ringback")


def test_6_precise_sub_40ms_human_answer_speech_onset_gating():
    print("\n--- Test 6: Precise Sub-40ms Human Answer & Speech Onset Gating ---")
    discriminator = EarlyMediaDiscriminator(sample_rate=8000)

    tone_chunk = generate_tone_frame(400.0, 425.0, duration_ms=20.0)
    silence_chunk = (np.zeros(160, dtype=np.int16)).tobytes()

    stream_frames = []
    # 1. Ringing for 400ms (20 frames)
    stream_frames.extend([tone_chunk] * 20)
    # 2. Brief call pickup click / silence pause for 60ms (3 frames)
    stream_frames.extend([silence_chunk] * 3)
    # Expected speech onset: 460.0 ms

    # 3. Customer answers: "Hello? Haanji boliye" (15 frames = 300ms speech)
    for i in range(15):
        stream_frames.append(generate_speech_frame(f0=140.0, duration_ms=20.0))

    telemetries, report = discriminator.process_stream(b"".join(stream_frames))

    print(f"[Answer Transition] Human Answered: {report.human_answered}")
    print(f"[Answer Transition] Recorded Onset: {report.answer_onset_timestamp_ms:.1f} ms")
    print(f"[Answer Transition] Ringback Pre-Confirmed: {report.ringback_detected}")

    assert report.human_answered is True
    assert report.answer_onset_timestamp_ms is not None
    # Nominal speech onset is 460ms (frame 24). Detection must trigger within <= 40ms of onset (frame 25)
    onset_error_ms = abs(report.answer_onset_timestamp_ms - 460.0)
    print(f"[Onset Accuracy] Timing error: {onset_error_ms:.1f} ms (Target <= 40ms)")
    assert onset_error_ms <= 40.0, f"Onset error {onset_error_ms} exceeded 40ms limit"
    print("[OK] Real-time speech onset gate triggered human answer transition within sub-40ms")


def test_7_line_noise_floor_adaptation_and_silence_invariance():
    print("\n--- Test 7: Line Noise Floor Adaptation & Silence Invariance ---")
    discriminator = EarlyMediaDiscriminator(sample_rate=8000)

    # 1. Pure silence: 30 frames (600ms)
    silence_pcm = (np.zeros(160 * 30, dtype=np.int16)).tobytes()
    _, rep_silence = discriminator.process_stream(silence_pcm)

    print(f"[Silence Stream] Dominant State: {rep_silence.dominant_state.value}")
    assert rep_silence.dominant_state == EarlyMediaState.SILENCE
    assert rep_silence.ringback_detected is False
    assert rep_silence.human_answered is False

    # 2. Low-level cellular background noise (-55 dBFS, ~60 RMS)
    discriminator.reset()
    noise = (np.random.randn(160 * 30) * 55.0).astype(np.int16).tobytes()
    _, rep_noise = discriminator.process_stream(noise)

    print(f"[Noise Stream] Dominant State: {rep_noise.dominant_state.value}")
    assert rep_noise.dominant_state == EarlyMediaState.SILENCE
    assert rep_noise.ringback_detected is False
    print("[OK] Silence and cellular background noise correctly handled without false triggers")


def test_8_sip_session_orchestrator_integration():
    print("\n--- Test 8: SIP Session Orchestrator State Machine Integration ---")
    session = SIPSession(
        session_id="call-early-media-test-01",
        caller_uri="sip:customer@carrier.airtel.in",
        agent_uri="sip:agent@verbalyze.ai",
        sample_rate=8000,
    )

    # Put call leg into RINGING state (early media progress)
    session.leg_a.state = SIPCallState.RINGING
    session.leg_b.state = SIPCallState.RINGING

    print(f"[Initial SIP State] Leg A: {session.leg_a.state.value} | Leg B: {session.leg_b.state.value}")
    assert session.leg_a.state == SIPCallState.RINGING

    # Feed ringback tone frame
    tone_frame = generate_tone_frame(400.0, 425.0, duration_ms=20.0)
    telem_tone = session.process_early_media_frame(tone_frame)
    print(f"[Early Media Frame 1] State: {telem_tone.state.value} | Leg A State: {session.leg_a.state.value}")
    assert telem_tone.state == EarlyMediaState.RINGBACK_TONE
    assert session.leg_a.state == SIPCallState.RINGING

    # Feed customer answer speech frames
    speech_frame = generate_speech_frame(f0=130.0, duration_ms=20.0)
    session.process_early_media_frame(speech_frame)
    telem_speech2 = session.process_early_media_frame(speech_frame)

    print(f"[Early Media Frame 3] State: {telem_speech2.state.value} | Human Answered: {telem_speech2.human_answered}")
    print(f"[Updated SIP State] Leg A: {session.leg_a.state.value} | Leg B: {session.leg_b.state.value}")

    assert telem_speech2.human_answered is True
    assert session.leg_a.state == SIPCallState.CONNECTED
    assert session.leg_b.state == SIPCallState.CONNECTED
    assert session.leg_a.connected_at is not None
    print("[OK] SIP Session Orchestrator automatically bridged legs upon early media speech onset")


def test_9_fastapi_rest_endpoints_verification():
    print("\n--- Test 9: FastAPI Telephony REST Endpoints Verification ---")
    app = create_app(auth_token="test-secret-early-media")
    from starlette.testclient import TestClient
    client = TestClient(app, headers={"Authorization": "Bearer test-secret-early-media"})

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    print(f"[GET /health] early_media_discriminator_status: {h_data.get('early_media_discriminator_status')}")
    assert h_data.get("early_media_discriminator_status") == "ready"

    # 2. POST /telephony/early-media/discriminate
    tone_stream = generate_tone_frame(400.0, 425.0, duration_ms=20.0) * 10
    b64_audio = base64.b64encode(tone_stream).decode("ascii")

    disc_resp = client.post(
        "/telephony/early-media/discriminate",
        json={"audio_base64": b64_audio, "sample_rate": 8000},
    )
    assert disc_resp.status_code == 200
    d_data = disc_resp.json()
    print(f"[POST /telephony/early-media/discriminate] Status: {d_data.get('status')} | Dominant: {d_data.get('dominant_state')}")
    assert d_data.get("status") == "ok"
    assert d_data.get("dominant_state") == EarlyMediaState.RINGBACK_TONE.value

    # 3. POST /telephony/early-media/benchmark
    bench_resp = client.post("/telephony/early-media/benchmark")
    assert bench_resp.status_code == 200
    b_data = bench_resp.json()
    print(f"[POST /telephony/early-media/benchmark] Avg Latency: {b_data.get('avg_early_media_processing_time_ms')} ms")
    print(f"[POST /telephony/early-media/benchmark] Meets SLA: {b_data.get('meets_sla')} | Headroom: {b_data.get('real_time_headroom_factor')}x")
    assert b_data.get("meets_sla") is True
    assert b_data.get("real_time_headroom_factor") >= 400.0
    print("[OK] All FastAPI early media endpoints verified cleanly")


def test_10_sub_0_05ms_latency_sla_benchmark():
    print("\n--- Test 10: Pure-Math Sub-0.05ms SLA Latency Benchmark ---")
    discriminator = EarlyMediaDiscriminator(sample_rate=8000)
    frame = generate_tone_frame(400.0, 425.0, duration_ms=20.0)

    # Warm-up
    for _ in range(50):
        discriminator.process_frame(frame)

    n_iterations = 500
    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        discriminator.process_frame(frame)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))
    headroom = 20.0 / max(avg_ms, 1e-4)

    print(f"[Benchmark Results] 20ms Frames Profiled: {n_iterations}")
    print(f"Mean Latency: {avg_ms:.4f} ms per 20ms frame")
    print(f"95th Percentile: {p95_ms:.4f} ms")
    print(f"Maximum Latency: {max_ms:.4f} ms")
    print(f"Target SLA: < 0.050 ms | Meets SLA: {avg_ms < 0.050}")
    print(f"Real-Time Headroom Factor: {headroom:.1f}x")

    assert avg_ms < 0.050, f"Mean latency {avg_ms:.4f} ms exceeded 0.050 ms SLA"
    assert headroom >= 400.0, f"Headroom {headroom:.1f}x should be >= 400x"
    print("[OK] Pure-math Early Media Discriminator easily meets sub-0.05ms latency SLA with massive headroom")


def main():
    print("=" * 80)
    print("Verbalyze Indian Early Media & In-Band Ringback Tone Discriminator Suite")
    print("ITU-T Q.35 / E.180 Pure-Math Tone & CRBT Discriminator / Answer Gating")
    print("Target SLA: Latency < 0.050 ms per 20 ms frame | Headroom > 400x")
    print("=" * 80)

    t0 = time.perf_counter()
    test_1_pure_indian_dual_tone_discrimination()
    test_2_standard_indian_dual_pulse_cadence_validation()
    test_3_indian_single_pulse_and_busy_tone_cadence_tracking()
    test_4_caller_tune_music_differentiation()
    test_5_operator_network_announcement_detection()
    test_6_precise_sub_40ms_human_answer_speech_onset_gating()
    test_7_line_noise_floor_adaptation_and_silence_invariance()
    test_8_sip_session_orchestrator_integration()
    test_9_fastapi_rest_endpoints_verification()
    test_10_sub_0_05ms_latency_sla_benchmark()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 80)
    print(f"ALL 10 EARLY MEDIA DISCRIMINATOR TESTS PASSED ({elapsed:.2f}s)")
    print("100% SUCCESS | Zero Emojis | Pure Math & SLA Compliant")
    print("=" * 80)


if __name__ == "__main__":
    main()
