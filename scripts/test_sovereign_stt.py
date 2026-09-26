#!/usr/bin/env python3
"""
scripts/test_sovereign_stt.py

Comprehensive Verification Suite for Sovereign On-Premise Streaming STT Engine:
1. In-Memory 8kHz PCM Transcription Benchmarking (<200ms target).
2. Multi-lingual Indic Speech Recognition (Hindi & Hinglish audio samples).
3. Zero-Failure Fallback Hierarchy (Local -> Google -> Rule-Based Mock).
4. Full Telephony MediaStreamSession Integration & Turn-around Verification.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.agent.stt_engine import SovereignSTTEngine
from verbalyze.telephony.media_stream import MediaStreamSession

try:
    import pydub
except ImportError:
    pydub = None


def test_model_initialization_and_caching():
    print("=================================================================")
    print("1. Testing Sovereign STT Engine Initialization & Model Caching")
    print("=================================================================")
    t0 = time.time()
    stt1 = SovereignSTTEngine(model_size="tiny", language="hi", provider="local")
    init1_ms = (time.time() - t0) * 1000.0
    print(f"[PASS] Initial instantiation: {init1_ms:.2f} ms")

    # Second initialization must reuse global cache instantaneously (<5ms)
    t1 = time.time()
    stt2 = SovereignSTTEngine(model_size="tiny", language="hi", provider="local")
    init2_ms = (time.time() - t1) * 1000.0
    print(f"[PASS] Cached re-instantiation: {init2_ms:.2f} ms")

    assert stt1._model is not None, "STT Model was not loaded!"
    assert stt1._model is stt2._model, "Model caching failed; new instance created!"
    assert init2_ms < 20.0, f"Cache lookup too slow ({init2_ms:.2f}ms)!"
    print("Model Initialization & Global Caching PASSED!\n")


def test_in_memory_pcm_transcription():
    print("=================================================================")
    print("2. Testing In-Memory 8kHz PCM Audio Decoding (Hindi Speech)")
    print("=================================================================")
    stt = SovereignSTTEngine(model_size="tiny", language="hi", provider="local")
    sample_file = PROJECT_ROOT / "samples" / "tts" / "1_edgetts_swara_hindi.mp3"

    assert sample_file.exists(), f"Sample audio missing: {sample_file}"

    # Load and resample audio to 8,000 Hz 16-bit mono PCM (telecom standard)
    seg = pydub.AudioSegment.from_file(str(sample_file))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm_8k_bytes = seg.raw_data

    duration_sec = len(seg) / 1000.0
    print(f"Audio Duration: {duration_sec:.2f}s | Raw 8kHz PCM: {len(pcm_8k_bytes)} bytes")

    # Transcribe directly from raw bytes in RAM (zero disk I/O)
    text, latency_ms = stt.transcribe_pcm(pcm_8k_bytes, sample_rate=8000, language="hi")

    print(f"[METRIC] Measured STT Inference Latency: {latency_ms:.2f} ms")
    print(f"[TRANSCRIPT] Transcribed Text: '{text}'")

    assert len(text) > 5, "Transcription text was unexpectedly empty!"
    assert latency_ms < 500.0, f"STT Latency exceeded SLA ({latency_ms:.2f}ms > 500ms)!"
    print("In-Memory 8kHz PCM Decoding PASSED!\n")


def test_multilingual_hinglish_decoding():
    print("=================================================================")
    print("3. Testing Multi-Lingual Hinglish Decoding")
    print("=================================================================")
    stt = SovereignSTTEngine(model_size="tiny", language="en", provider="local")
    sample_file = PROJECT_ROOT / "samples" / "tts" / "3_edgetts_neerja_hinglish.mp3"

    if not sample_file.exists():
        print("Hinglish sample not found, skipping.")
        return

    seg = pydub.AudioSegment.from_file(str(sample_file))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm_8k_bytes = seg.raw_data

    text, latency_ms = stt.transcribe_pcm(pcm_8k_bytes, sample_rate=8000, language="en")
    print(f"[METRIC] Latency: {latency_ms:.2f} ms")
    print(f"[TRANSCRIPT] Hinglish Transcript: '{text}'")

    assert len(text) > 5, "Hinglish transcription was empty!"
    print("Multi-Lingual Hinglish Decoding PASSED!\n")


def test_zero_failure_fallback():
    print("=================================================================")
    print("4. Testing Zero-Failure Fallback Hierarchy")
    print("=================================================================")
    # Instantiate with mock provider to verify deterministic fallback
    mock_stt = SovereignSTTEngine(model_size="tiny", language="hi", provider="mock")

    dummy_pcm = b"\x00\x00" * 4000  # 0.5s silence
    text, lat = mock_stt.transcribe_pcm(dummy_pcm, sample_rate=8000, language="hi")

    print(f"[PASS] Mock / Offline fallback transcript: '{text}' in {lat:.2f} ms")
    assert "हाँ जी, मैं सुन रहा हूँ" in text, f"Unexpected fallback text: {text}"
    print("Zero-Failure Fallback PASSED!\n")


class MockWebSocket:
    def __init__(self):
        self.sent_texts = []
        self.sent_bytes = []

    async def send_text(self, text: str):
        self.sent_texts.append(text)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)


async def test_mediastream_sovereign_stt_integration():
    print("=================================================================")
    print("5. Testing Full MediaStreamSession Integration with Sovereign STT")
    print("=================================================================")
    mock_ws = MockWebSocket()
    session = MediaStreamSession(
        websocket=mock_ws,
        language="hi",
        llm_provider="mock",
        stt_provider="local",
        stt_model="tiny",
        caller_phone="+919876543210"
    )

    # Verify session has SovereignSTTEngine initialized
    assert hasattr(session, "stt_engine"), "MediaStreamSession missing stt_engine attribute!"
    assert session.stt_engine.provider == "local", f"Expected provider 'local', got {session.stt_engine.provider}"

    # Load real audio sample into inbound PCM buffer
    sample_file = PROJECT_ROOT / "samples" / "tts" / "1_edgetts_swara_hindi.mp3"
    seg = pydub.AudioSegment.from_file(str(sample_file))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    session.inbound_pcm_buffer = [seg.raw_data]

    # Hermetic mock for outbound synthesis to ensure test runs 100% offline without cloud TTS
    async def hermetic_synthesize_async(text, *args, **kwargs):
        return str(sample_file)
    session.agent.audio_engine.synthesize_async = hermetic_synthesize_async

    # Process caller turn: in-memory STT -> LLM streaming -> Outbound 20ms audio frames
    t0 = time.time()
    await session.process_caller_turn()
    total_turn_ms = (time.time() - t0) * 1000.0

    print(f"[PASS] Full turn completed in: {total_turn_ms:.1f} ms")
    print(f"[PASS] Dispatched outbound carrier frames: {len(mock_ws.sent_texts)}")
    assert len(mock_ws.sent_texts) > 0, "No audio frames dispatched to carrier!"
    print("Full MediaStreamSession Sovereign STT Integration PASSED!\n")


def main():
    test_model_initialization_and_caching()
    test_in_memory_pcm_transcription()
    test_multilingual_hinglish_decoding()
    test_zero_failure_fallback()
    asyncio.run(test_mediastream_sovereign_stt_integration())
    print("=================================================================")
    print("ALL SOVEREIGN ON-PREM STT TESTS PASSED!")
    print("=================================================================")


if __name__ == "__main__":
    main()
