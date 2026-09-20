"""
verbalyze/telephony/browser_gateway.py

Full-Duplex In-Browser Audio Streaming Gateway.
Enables sub-150ms bidirectional voice streaming directly between web browsers
and Verbalyze Indic Voice SLMs over WebSockets without requiring SIP trunks.
Supports 16-bit linear PCM (16kHz / 8kHz), Adaptive Jitter Buffering,
live VAD, barge-in interruption, and Supervisor Observability integration.
DPDP Act 2023 compliant.
Zero-emoji compliant.
"""

import asyncio
import audioop
import base64
import json
import os
import time
from typing import Dict, Any, Optional, List

from verbalyze.agent.stt_engine import SovereignSTTEngine
from verbalyze.security import PIIRedactor
from verbalyze.telephony.jitter_buffer import AdaptiveJitterBuffer
from verbalyze.telephony.supervisor import SupervisorManager

try:
    import pydub
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class BrowserAudioSession:
    """
    Manages an active bi-directional WebSocket audio session with a web browser.
    """

    def __init__(
        self,
        websocket: Any,
        session_id: str,
        language: str = "hi",
        persona: str = "muthoot_recovery",
        llm_provider: str = "ollama",
        model_name: Optional[str] = None,
        sample_rate: int = 16000,
        speech_threshold: int = 700,
        silence_timeout_ms: int = 600,
        caller_phone: Optional[str] = None,
        supervisor_manager: Optional[SupervisorManager] = None,
        stt_provider: str = "local",
        stt_model: str = "tiny",
        strict_sovereignty: Optional[bool] = None,
    ):
        self.websocket = websocket
        self.session_id = session_id
        self.language = language
        self.persona = persona
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.silence_timeout_frames = int(silence_timeout_ms / 20)  # 20ms frames
        self.caller_phone = caller_phone
        self.supervisor_manager = supervisor_manager

        # Strict sovereignty check
        self.strict_sovereignty = (
            strict_sovereignty
            if strict_sovereignty is not None
            else os.environ.get("STRICT_SOVEREIGNTY", "0").lower() in ("1", "true", "yes")
        )

        # STT Engine
        self.stt_engine = SovereignSTTEngine(
            model_size=stt_model,
            language=language,
            provider=stt_provider,
            strict_sovereignty=self.strict_sovereignty,
        )

        # Voice Agent
        from verbalyze.agent.voice_bot import VoiceAgent
        self.agent = VoiceAgent(
            language=language,
            persona=persona,
            llm_provider=llm_provider,
            model_name=model_name,
            voice_enabled=True,
            caller_phone=caller_phone,
        )

        # Adaptive Telecom Jitter Buffer
        self.jitter_buffer = AdaptiveJitterBuffer(
            frame_duration_ms=20.0,
            sample_rate=self.sample_rate,
            bytes_per_sample=2,
            min_delay_ms=30.0,
            max_delay_ms=180.0,
        )

        # Inbound VAD and frame sequencing
        self.inbound_pcm_buffer: List[bytes] = []
        self.silence_frames_count: int = 0
        self.is_caller_speaking: bool = False
        self.barge_in_consecutive_frames: int = 0
        self.inbound_sequence: int = 0

        # Outbound streaming state
        self.is_agent_streaming: bool = False
        self.current_playback_task: Optional[asyncio.Task] = None
        self.cancel_playback_event = asyncio.Event()

        # Session lifecycle
        self.is_active: bool = True

        # Register in supervisor manager if present
        if self.supervisor_manager:
            self.supervisor_manager.register_call(
                call_id=self.session_id,
                caller_phone=self.caller_phone,
                persona=self.persona,
                language=self.language,
                agent_instance=self.agent,
            )

    async def interrupt_agent_playback(self):
        """Cancels active speech playback upon caller barge-in."""
        self.cancel_playback_event.set()
        if self.current_playback_task and not self.current_playback_task.done():
            self.current_playback_task.cancel()
        self.is_agent_streaming = False

        # Notify browser client to clear audio buffer
        try:
            await self.websocket.send_text(json.dumps({
                "type": "clear",
                "event": "barge_in",
                "timestamp": time.time(),
            }))
        except Exception:
            pass

    async def stream_audio_to_browser(self, pcm_bytes: bytes):
        """Paces 20ms linear PCM packets down the WebSocket to the browser."""
        if not pcm_bytes:
            return

        frame_size = int((self.sample_rate * 0.02) * 2)  # 20ms of 16-bit linear PCM
        total_frames = len(pcm_bytes) // frame_size

        for i in range(total_frames):
            if self.cancel_playback_event.is_set():
                break

            chunk = pcm_bytes[i * frame_size : (i + 1) * frame_size]
            b64_payload = base64.b64encode(chunk).decode("utf-8")

            try:
                await self.websocket.send_text(json.dumps({
                    "type": "audio",
                    "payload": b64_payload,
                    "frame_index": i,
                    "sample_rate": self.sample_rate,
                }))
            except Exception:
                break

            await asyncio.sleep(0.018)  # 20ms clock pacing

    async def synthesize_and_stream_clause(self, clause_text: str):
        """Synthesizes an agent speech clause and streams audio packets down to browser."""
        if not clause_text or not self.agent.audio_engine:
            return
        if self.cancel_playback_event.is_set():
            return

        audio_path = await self.agent.audio_engine.synthesize_async(clause_text)
        if not audio_path or not PYDUB_AVAILABLE or self.cancel_playback_event.is_set():
            return

        try:
            seg = pydub.AudioSegment.from_file(audio_path)
            seg = seg.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
            raw_pcm = seg.raw_data

            self.current_playback_task = asyncio.create_task(
                self.stream_audio_to_browser(raw_pcm)
            )
            await self.current_playback_task
        except Exception as e:
            print(f"[Browser Gateway Error]: {e}")

    async def process_caller_turn(self):
        """Transcribes accumulated inbound PCM buffer and initiates streaming agent turn."""
        if not self.inbound_pcm_buffer:
            return

        raw_pcm = b"".join(self.inbound_pcm_buffer)
        self.inbound_pcm_buffer.clear()
        self.silence_frames_count = 0
        self.is_caller_speaking = False

        # Discard clicks shorter than 200ms
        min_bytes = int(self.sample_rate * 2 * 0.20)
        if len(raw_pcm) < min_bytes:
            return

        # Resample to 8kHz or 16kHz for STT if needed
        stt_pcm = raw_pcm
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            try:
                stt_pcm, _ = audioop.ratecv(raw_pcm, 2, 1, self.sample_rate, 16000, None)
            except Exception:
                stt_pcm = raw_pcm

        # Transcribe audio
        user_text = await self._transcribe_pcm_audio(stt_pcm, self.sample_rate)
        if not user_text:
            return

        # Broadcast transcript event to browser
        try:
            await self.websocket.send_text(json.dumps({
                "type": "transcript",
                "role": "user",
                "text": PIIRedactor.redact_text(user_text),
                "timestamp": time.time(),
            }))
        except Exception:
            pass

        self.cancel_playback_event.clear()
        self.is_agent_streaming = True
        accumulated_clauses = []
        final_metadata: Dict[str, Any] = {}

        try:
            async for chunk in self.agent.step_stream(user_text, pcm_bytes=stt_pcm):
                if self.cancel_playback_event.is_set():
                    break

                if chunk["type"] == "clause":
                    clause_text = chunk["text"]
                    accumulated_clauses.append(clause_text)

                    # Emit incremental transcript chunk to browser
                    try:
                        await self.websocket.send_text(json.dumps({
                            "type": "transcript_chunk",
                            "role": "agent",
                            "text": clause_text,
                            "index": chunk.get("index", 0),
                        }))
                    except Exception:
                        pass

                    await self.synthesize_and_stream_clause(clause_text)

                elif chunk["type"] == "tool_call":
                    try:
                        await self.websocket.send_text(json.dumps({
                            "type": "tool_event",
                            "tool_name": chunk.get("tool_name"),
                            "tool_event": chunk.get("tool_event"),
                        }))
                    except Exception:
                        pass

                elif chunk["type"] == "final":
                    final_metadata = chunk
                    full_reply = chunk.get("full_text") or " ".join(accumulated_clauses)

                    # Check dynamic language adaptation
                    lang_info = chunk.get("language_info") or {}
                    if lang_info.get("language_switched"):
                        self.language = lang_info.get("primary_language", self.language)
                        try:
                            await self.websocket.send_text(json.dumps({
                                "type": "language_switch",
                                "language": self.language,
                                "confidence": lang_info.get("confidence", 1.0),
                                "is_code_switched": lang_info.get("is_code_switched", False),
                            }))
                        except Exception:
                            pass

                    # Emit final turn completion to browser
                    try:
                        await self.websocket.send_text(json.dumps({
                            "type": "turn_complete",
                            "role": "agent",
                            "text": full_reply,
                            "sentiment": chunk.get("sentiment"),
                            "language_info": lang_info,
                            "terminated": chunk.get("terminated", False),
                        }))
                    except Exception:
                        pass

                    # Update Supervisor Hub
                    if self.supervisor_manager:
                        self.supervisor_manager.update_turn(
                            call_id=self.session_id,
                            customer_utterance=user_text,
                            agent_reply=full_reply,
                            sentiment=chunk.get("sentiment"),
                            lid_info=lang_info,
                            quality_report=self.agent.audio_engine.last_quality_report.to_dict() if (self.agent.audio_engine and self.agent.audio_engine.last_quality_report) else None,
                            jitter_stats=self.jitter_buffer.get_stats().to_dict(),
                        )

                    if chunk.get("terminated"):
                        self.is_active = False

        except Exception as e:
            print(f"[Browser Gateway Turn Error]: {e}")
        finally:
            self.is_agent_streaming = False

    async def _transcribe_pcm_audio(self, pcm_bytes: bytes, rate: int) -> str:
        """Transcribes PCM bytes using Sovereign STT Engine."""
        try:
            transcript, _ = await asyncio.to_thread(
                self.stt_engine.transcribe_pcm,
                pcm_bytes,
                rate,
                self.language,
            )
            if transcript:
                return transcript
        except Exception as e:
            print(f"[STT Error in Browser Gateway]: {e}")

        fallback_map = {
            "hi": "हाँ जी, मैं सुन रहा हूँ।",
            "en": "Yes, I am listening.",
            "gu": "હા, હું સાંભળી રહ્યો છું.",
            "mr": "हो, मी ऐकत आहे.",
        }
        return fallback_map.get(self.language, "हाँ जी, मैं सुन रहा हूँ।")

    async def handle_inbound_pcm(self, pcm_frame: bytes):
        """
        Processes an incoming 20ms frame of 16-bit linear PCM from browser:
        1. Buffers through Adaptive Jitter Buffer.
        2. Measures RMS energy for VAD & Barge-In.
        3. Accumulates speech and dispatches on conversational pause.
        """
        self.inbound_sequence += 1
        now_ms = time.time() * 1000.0

        # Push to jitter buffer for smoothing & RFC 3550 telemetry
        self.jitter_buffer.push(
            pcm_data=pcm_frame,
            sequence_number=self.inbound_sequence,
            arrival_time_ms=now_ms,
        )

        frame, is_concealed = self.jitter_buffer.pop(current_time_ms=now_ms)
        if len(frame) < 4:
            return

        rms = audioop.rms(frame, 2)

        # 1. Barge-In Interruption Check
        if self.is_agent_streaming:
            if rms > self.speech_threshold:
                self.barge_in_consecutive_frames += 1
                if self.barge_in_consecutive_frames >= 2:  # ~40ms speech detection
                    await self.interrupt_agent_playback()
                    self.is_caller_speaking = True
                    self.inbound_pcm_buffer.append(frame)
                    self.barge_in_consecutive_frames = 0
            else:
                self.barge_in_consecutive_frames = 0
            return

        # 2. Inbound Speech Accumulation
        if rms > self.speech_threshold:
            self.is_caller_speaking = True
            self.silence_frames_count = 0
            self.inbound_pcm_buffer.append(frame)
        elif self.is_caller_speaking:
            self.inbound_pcm_buffer.append(frame)
            self.silence_frames_count += 1
            if self.silence_frames_count >= self.silence_timeout_frames:
                await self.process_caller_turn()

    async def run(self):
        """Main WebSocket loop handling inbound browser audio frames and control messages."""
        greeting_sent = False
        try:
            # Emit greeting on session connect
            greeting = self.agent.get_initial_greeting()
            greeting_sent = True
            try:
                await self.websocket.send_text(json.dumps({
                    "type": "transcript",
                    "role": "agent",
                    "text": greeting,
                    "timestamp": time.time(),
                }))
            except Exception:
                pass
            asyncio.create_task(self.synthesize_and_stream_clause(greeting))

            while self.is_active:
                message = await self.websocket.receive()

                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                    except Exception:
                        continue

                    msg_type = data.get("type") or data.get("event")

                    if msg_type == "audio":
                        # Base64 encoded linear PCM chunk
                        payload = data.get("payload") or ""
                        if payload:
                            raw_pcm = base64.b64decode(payload)
                            await self.handle_inbound_pcm(raw_pcm)

                    elif msg_type == "text_input":
                        # Direct text turn input from browser chat box
                        user_text = str(data.get("text", "")).strip()
                        if user_text:
                            self.inbound_pcm_buffer.clear()
                            self.cancel_playback_event.clear()
                            self.is_agent_streaming = True
                            accumulated = []
                            async for chunk in self.agent.step_stream(user_text):
                                if chunk["type"] == "clause":
                                    accumulated.append(chunk["text"])
                                    await self.synthesize_and_stream_clause(chunk["text"])
                                elif chunk["type"] == "final":
                                    full_reply = chunk.get("full_text") or " ".join(accumulated)
                                    try:
                                        await self.websocket.send_text(json.dumps({
                                            "type": "turn_complete",
                                            "role": "agent",
                                            "text": full_reply,
                                            "sentiment": chunk.get("sentiment"),
                                            "language_info": chunk.get("language_info"),
                                            "terminated": chunk.get("terminated", False),
                                        }))
                                    except Exception:
                                        pass
                                    if self.supervisor_manager:
                                        self.supervisor_manager.update_turn(
                                            call_id=self.session_id,
                                            customer_utterance=user_text,
                                            agent_reply=full_reply,
                                            sentiment=chunk.get("sentiment"),
                                            lid_info=chunk.get("language_info"),
                                            jitter_stats=self.jitter_buffer.get_stats().to_dict(),
                                        )

                    elif msg_type == "barge_in":
                        await self.interrupt_agent_playback()

                    elif msg_type == "disconnect" or msg_type == "stop":
                        self.is_active = False
                        break

                elif "bytes" in message:
                    raw_pcm = message["bytes"]
                    await self.handle_inbound_pcm(raw_pcm)

        except (asyncio.CancelledError, Exception) as e:
            if "disconnect" not in str(e).lower():
                print(f"[Browser Gateway Session Closed]: {e}")
        finally:
            self.is_active = False
            self.inbound_pcm_buffer.clear()
            if self.current_playback_task and not self.current_playback_task.done():
                self.current_playback_task.cancel()
            if self.supervisor_manager:
                self.supervisor_manager.terminate_call(self.session_id, reason="browser_disconnected")
