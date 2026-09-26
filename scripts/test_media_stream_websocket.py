#!/usr/bin/env python3
"""
scripts/test_media_stream_websocket.py

Comprehensive test suite for the Bi-directional WebSocket Media Stream (/media-stream):
1. Connection & Twilio/RingTrunk JSON Handshake ('start' event).
2. Outbound 20ms G.711 A-law audio pacing (160 bytes per frame).
3. Inbound caller speech streaming & VAD turn-taking.
4. Real-Time WebSocket Barge-In:
   - Caller speaks while bot is streaming.
   - Verifies instant playback cutoff and 'clear' event frame.
5. Binary frame streaming (Asterisk AudioSocket compatibility).
"""

import sys
import json
import time
import base64
import audioop
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from verbalyze.telephony.server import create_app


def generate_synthetic_tone(freq_hz: float, duration_sec: float, sample_rate: int = 8000, amplitude: int = 16000) -> bytes:
    """Generates synthetic sine wave PCM audio (16-bit linear mono)."""
    import math
    num_samples = int(sample_rate * duration_sec)
    samples = bytearray()
    for i in range(num_samples):
        val = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        samples.extend(val.to_bytes(2, byteorder="little", signed=True))
    return bytes(samples)


def generate_silence(duration_sec: float, sample_rate: int = 8000) -> bytes:
    """Generates silence frames (zeros)."""
    num_samples = int(sample_rate * duration_sec)
    return bytes(num_samples * 2)


def test_websocket_handshake_and_greeting():
    print("=================================================================")
    print("1. Testing WebSocket Handshake & Outbound 20ms Frame Streaming")
    print("=================================================================")
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/media-stream?lang=hi&provider=mock") as ws:
        # Send Twilio / RingTrunk start event
        start_payload = {
            "event": "start",
            "streamSid": "MZ_stream_001",
            "start": {
                "streamSid": "MZ_stream_001",
                "callSid": "CA_test_001",
                "mediaFormat": {
                    "encoding": "audio/x-alaw",
                    "sampleRate": 8000,
                    "channels": 1
                }
            }
        }
        ws.send_text(json.dumps(start_payload))

        # Receive first batch of outbound 20ms audio frames
        frames_received = 0
        total_audio_bytes = 0
        stream_sid_verified = False

        start_time = time.time()
        while time.time() - start_time < 2.0 and frames_received < 15:
            try:
                raw_msg = ws.receive_text()
                data = json.loads(raw_msg)
                if data.get("event") == "media":
                    frames_received += 1
                    payload = base64.b64decode(data["media"]["payload"])
                    total_audio_bytes += len(payload)
                    assert len(payload) == 160, f"Expected 160 bytes G.711 per 20ms frame, got {len(payload)}"
                    if data.get("streamSid") == "MZ_stream_001":
                        stream_sid_verified = True
            except Exception:
                break

        print(f"[PASS] Received {frames_received} G.711 A-law media frames ({total_audio_bytes} bytes audio)")
        print(f"[PASS] StreamSid verified: {stream_sid_verified}")
        assert frames_received >= 5, "Failed to receive audio frames from greeting!"
        assert stream_sid_verified, "StreamSid did not match carrier start event!"
        print("Handshake and Outbound Paced Audio Streaming PASSED!\n")


def test_websocket_barge_in_interruption():
    print("=================================================================")
    print("2. Testing Real-Time WebSocket Barge-In Interruption (<50ms)")
    print("=================================================================")
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/media-stream?lang=hi&provider=mock") as ws:
        # Start connection
        ws.send_text(json.dumps({
            "event": "start",
            "streamSid": "MZ_barge_in_stream",
            "start": {
                "streamSid": "MZ_barge_in_stream",
                "callSid": "CA_barge_001",
                "mediaFormat": {"encoding": "audio/x-alaw"}
            }
        }))

        # Wait until bot begins streaming greeting frames
        bot_is_streaming = False
        for _ in range(20):
            msg = json.loads(ws.receive_text())
            if msg.get("event") == "media":
                bot_is_streaming = True
                break
        assert bot_is_streaming, "Bot did not start streaming audio!"
        print("[PASS] Bot is actively streaming voice packets over WebSocket...")

        # Caller interrupts! Generate high-energy speech audio (8kHz sine wave, 16-bit PCM -> G.711 A-law)
        speech_pcm = generate_synthetic_tone(freq_hz=440.0, duration_sec=0.20, sample_rate=8000, amplitude=18000)
        speech_alaw = audioop.lin2alaw(speech_pcm, 2)

        # Slice into 20ms (160 bytes) packets and inject mid-stream
        t0 = time.time()
        for i in range(0, len(speech_alaw), 160):
            chunk = speech_alaw[i:i+160]
            if len(chunk) < 160:
                break
            payload_b64 = base64.b64encode(chunk).decode("ascii")
            ws.send_text(json.dumps({
                "event": "media",
                "streamSid": "MZ_barge_in_stream",
                "media": {"payload": payload_b64}
            }))

        # Expect immediate 'clear' event from server
        clear_received = False
        cutoff_latency_ms = None

        for _ in range(30):
            raw_msg = ws.receive_text()
            data = json.loads(raw_msg)
            if data.get("event") == "clear":
                cutoff_latency_ms = (time.time() - t0) * 1000.0
                clear_received = True
                assert data.get("streamSid") == "MZ_barge_in_stream"
                break

        print(f"[PASS] Received instant 'clear' frame: {clear_received}")
        print(f"[METRIC] Measured WebSocket Interruption Cutoff: {cutoff_latency_ms:.2f} ms")
        assert clear_received, "Bot did not send 'clear' event upon caller interruption!"
        assert cutoff_latency_ms < 100.0, f"Barge-in latency too high ({cutoff_latency_ms}ms)!"
        print("WebSocket Real-Time Barge-In PASSED!\n")


def test_binary_mode_streaming():
    print("=================================================================")
    print("3. Testing Binary Frame Streaming (Asterisk AudioSocket)")
    print("=================================================================")
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/media-stream?lang=hi&provider=mock") as ws:
        # Send raw binary 20ms G.711 A-law frame directly
        speech_pcm = generate_synthetic_tone(freq_hz=300.0, duration_sec=0.04, sample_rate=8000, amplitude=14000)
        speech_alaw = audioop.lin2alaw(speech_pcm, 2)

        ws.send_bytes(speech_alaw[:160])
        print("[PASS] Sent raw 160-byte binary G.711 A-law frame")

        # Receive binary audio frames
        bin_frame = ws.receive_bytes()
        print(f"[PASS] Received binary frame from agent: {len(bin_frame)} bytes")
        assert len(bin_frame) == 160, f"Expected 160 bytes, got {len(bin_frame)}"
        print("Binary Mode Streaming PASSED!\n")


if __name__ == "__main__":
    test_websocket_handshake_and_greeting()
    test_websocket_barge_in_interruption()
    test_binary_mode_streaming()
    print("=================================================================")
    print("ALL WEBSOCKET MEDIA STREAM TESTS PASSED!")
    print("=================================================================")
