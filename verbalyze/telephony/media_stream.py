"""
verbalyze/telephony/media_stream.py

Bi-Directional WebSocket Media Stream Processor for Live Telephony Trunks:
- Supports Twilio Media Streams, RingTrunk, Asterisk AudioSocket, & FreeSWITCH.
- Formats: ITU-T G.711 A-law (PCMA), G.711 μ-law (PCMU), and Linear 16-bit PCM (8,000 Hz).
- Concurrent 20ms Frame VAD with automatic turn boundary detection.
- Sub-50ms Real-Time WebSocket Barge-In Interruption with 'clear' event frame dispatch.
- 20ms paced outbound audio streaming matching carrier RTP clock.
"""

import asyncio
import audioop
import base64
import io
import json
import os
import time
import wave
from typing import Dict, Any, Optional, List, Callable

from verbalyze.agent.voice_bot import VoiceAgent
from verbalyze.agent.stt_engine import SovereignSTTEngine
from verbalyze.security import PIIRedactor
from verbalyze.telephony.jitter_buffer import AdaptiveJitterBuffer
from verbalyze.telephony.supervisor import SupervisorManager
from verbalyze.telephony.dtmf_engine import DTMFPad, RFC4733EventDecoder
from verbalyze.telephony.ivr_tree import IVRStateMachine, IVRTransitionResult
from verbalyze.telephony.turn_taking import (
    AdaptiveTurnTakingManager,
    TurnTakingState,
    DialogueContext,
    AdaptivePausePolicy,
    TurnCompletionConfidenceScorer,
    GlassToGlassLatencyProfiler,
)
from verbalyze.telephony.echo_canceller import (
    AcousticEchoAndNoiseProcessor,
    DSPTelemetry,
)
from verbalyze.telephony.equalizer import (
    IndicFormantEqualizer,
    EQTelemetry,
)
from verbalyze.telephony.comfort_noise import (
    ComfortNoiseGenerator,
    CNGTelemetry,
)

try:
    import pydub
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class MediaStreamSession:
    """
    Manages an active bi-directional WebSocket audio session for a telephony call.
    """

    def __init__(
        self,
        websocket: Any,
        language: str = "hi",
        persona: str = "muthoot_recovery",
        llm_provider: str = "ollama",
        model_name: Optional[str] = None,
        codec: str = "audio/x-alaw",  # "audio/x-alaw", "audio/x-mulaw", "audio/l16"
        speech_threshold: int = 650,  # 16-bit linear PCM RMS energy threshold
        silence_timeout_ms: int = 600, # Trailing silence before turn completion
        caller_phone: Optional[str] = None,
        supervisor_manager: Optional[SupervisorManager] = None,
        stt_provider: str = "local",
        stt_model: str = "tiny",
        strict_sovereignty: Optional[bool] = None,
        ivr_machine: Optional[IVRStateMachine] = None,
    ):
        self.websocket = websocket
        self.language = language
        self.persona = persona
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.codec = codec.lower()
        self.speech_threshold = speech_threshold
        self.silence_timeout_frames = int(silence_timeout_ms / 20)  # 20ms per frame
        self.caller_phone = caller_phone
        self.supervisor_manager = supervisor_manager
        self.stt_provider = stt_provider
        self.stt_model = stt_model
        self.strict_sovereignty = (
            strict_sovereignty
            if strict_sovereignty is not None
            else os.environ.get("STRICT_SOVEREIGNTY", "0").lower() in ("1", "true", "yes")
        )
        self.ivr_machine = ivr_machine

        # DTMF Acoustic & RFC 4733 Decoders
        self.dtmf_pad = DTMFPad(sample_rate=8000)
        self.rfc4733_decoder = RFC4733EventDecoder()

        # Adaptive Telecom Jitter Buffer for carrier RTP streams
        self.jitter_buffer = AdaptiveJitterBuffer(
            frame_duration_ms=20.0,
            sample_rate=8000,
            bytes_per_sample=2,
            min_delay_ms=40.0,
            max_delay_ms=200.0,
        )
        self.inbound_seq: int = 0

        # Sovereign STT Engine with strict data residency enforcement
        self.stt_engine = SovereignSTTEngine(
            model_size=stt_model,
            language=language,
            provider=stt_provider,
            strict_sovereignty=self.strict_sovereignty
        )

        # Call identifiers
        self.stream_sid: str = "stream_default"
        self.call_sid: str = "call_default"

        # Voice Agent
        self.agent = VoiceAgent(
            language=language,
            persona=persona,
            llm_provider=llm_provider,
            model_name=model_name,
            voice_enabled=True,
            caller_phone=caller_phone
        )

        # Adaptive Conversational Turn-Taking & Speculative Early-Pipelining Manager
        self.turn_manager = AdaptiveTurnTakingManager(
            sample_rate=8000,
            frame_duration_ms=20.0,
            default_context=DialogueContext.STANDARD_CONVERSATION,
            barge_in_enabled=True,
        )

        # Real-Time Acoustic Echo Cancellation (AEC) & Spectral Noise Suppression Engine
        self.dsp_processor = AcousticEchoAndNoiseProcessor(
            sample_rate=8000,
            filter_length=256,
            aec_enabled=True,
            noise_suppression_enabled=True,
        )

        # Dynamic Multi-Band Acoustic Equalizer (Indic Telecom Formant Enhancer)
        self.equalizer = IndicFormantEqualizer(
            sample_rate=8000,
            preset_name="INDIC_RETROFLEX_ENHANCE",
            dynamic_gating=True,
        )

        # Adaptive Comfort Noise Generator (ITU-T G.711 App II & RFC 3389)
        self.cng = ComfortNoiseGenerator(
            sample_rate=8000,
            preset_name="INDIAN_ROOM_CEILING_FAN",
        )

        # Inbound VAD state
        self.inbound_pcm_buffer: List[bytes] = []
        self.silence_frames_count: int = 0
        self.is_caller_speaking: bool = False
        self.barge_in_consecutive_frames: int = 0

        # Outbound streaming state
        self.is_agent_streaming: bool = False
        self.current_playback_task: Optional[asyncio.Task] = None
        self.cancel_playback_event = asyncio.Event()

        # Session lifecycle
        self.is_active: bool = True
        self.is_binary_mode: bool = False

    def decode_inbound_audio(self, raw_bytes: bytes) -> bytes:
        """Decodes raw carrier bytes to 8kHz 16-bit linear mono PCM."""
        if not raw_bytes:
            return b""
        try:
            if "alaw" in self.codec or self.codec == "pcma":
                return audioop.alaw2lin(raw_bytes, 2)
            elif "mulaw" in self.codec or "ulaw" in self.codec or self.codec == "pcmu":
                return audioop.ulaw2lin(raw_bytes, 2)
            else:
                return raw_bytes
        except Exception:
            return b""

    def encode_outbound_audio(self, pcm_16_bytes: bytes) -> bytes:
        """Encodes 8kHz 16-bit linear mono PCM to target carrier codec."""
        if not pcm_16_bytes:
            return b""
        try:
            if "alaw" in self.codec or self.codec == "pcma":
                return audioop.lin2alaw(pcm_16_bytes, 2)
            elif "mulaw" in self.codec or "ulaw" in self.codec or self.codec == "pcmu":
                return audioop.lin2ulaw(pcm_16_bytes, 2)
            else:
                return pcm_16_bytes
        except Exception:
            return pcm_16_bytes

    async def send_clear_event(self):
        """Sends instant buffer flush event to carrier for sub-50ms barge-in."""
        if not self.is_binary_mode:
            clear_msg = json.dumps({
                "event": "clear",
                "streamSid": self.stream_sid
            })
            try:
                await self.websocket.send_text(clear_msg)
            except Exception:
                pass

    async def interrupt_agent_playback(self):
        """Atomically aborts active agent speech streaming upon caller interruption."""
        if self.is_agent_streaming:
            self.turn_manager.set_bot_speaking(False)
            self.dsp_processor.set_bot_speaking(False)
            self.cancel_playback_event.set()
            if self.current_playback_task and not self.current_playback_task.done():
                self.current_playback_task.cancel()
            self.is_agent_streaming = False
            await self.send_clear_event()
            print("[WebSocket Barge-In] Interrupted bot playback! Sent clear event to carrier.")

    async def handle_dtmf_digit(self, digit: str):
        """
        Processes a DTMF keypress event:
        1. Triggers instantaneous playback abort (Barge-In) with carrier buffer clear event.
        2. Routes digit to active IVR state machine if bound.
        """
        print(f"[DTMF Keypad Event] Digit pressed: '{digit}'")
        await self.interrupt_agent_playback()

        if self.ivr_machine and not self.ivr_machine.is_completed:
            res = self.ivr_machine.handle_digit(digit)
            print(f"[IVR State] Transitioned to '{res.node_id}' status='{res.status}'")
            if res.prompt:
                self.current_playback_task = asyncio.create_task(
                    self.synthesize_and_play_reply(res.prompt)
                )
            if res.is_terminal and res.action == "hangup":
                self.is_active = False

    async def stream_audio_to_carrier(self, pcm_8k_bytes: bytes):
        """
        Paces audio down the WebSocket in 20ms frames (160 samples = 160 bytes G.711 / 320 bytes PCM).
        Matches carrier RTP clock (20ms intervals) with instant cancellation support.
        """
        self.cancel_playback_event.clear()
        self.is_agent_streaming = True
        self.turn_manager.set_bot_speaking(True)
        self.dsp_processor.set_bot_speaking(True)

        # Frame size at 8,000 Hz mono 16-bit: 20ms = 160 samples = 320 bytes
        frame_size_pcm = 320
        total_frames = len(pcm_8k_bytes) // frame_size_pcm

        try:
            for idx in range(total_frames):
                if self.cancel_playback_event.is_set():
                    break

                chunk_pcm = pcm_8k_bytes[idx * frame_size_pcm : (idx + 1) * frame_size_pcm]
                # Register outbound reference frame in AEC adaptive filter
                self.dsp_processor.register_reference_frame(chunk_pcm)
                encoded_chunk = self.encode_outbound_audio(chunk_pcm)

                if self.is_binary_mode:
                    await self.websocket.send_bytes(encoded_chunk)
                else:
                    payload_b64 = base64.b64decode(encoded_chunk).decode("ascii") if False else base64.b64encode(encoded_chunk).decode("ascii")
                    media_msg = json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": payload_b64
                        }
                    })
                    await self.websocket.send_text(media_msg)

                # Record first outbound RTP packet for glass-to-glass latency SLA
                if idx == 0 and not self.turn_manager.latency_profiler.breakdown.rtp_dispatched_ts:
                    self.turn_manager.latency_profiler.record_rtp_dispatched()

                # 20ms telephony pacing
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            pass
        finally:
            self.is_agent_streaming = False
            self.turn_manager.set_bot_speaking(False)
            self.dsp_processor.set_bot_speaking(False)

    async def synthesize_and_play_reply(self, text: str):
        """Synthesizes text via AudioEngine and streams 20ms packets down the socket."""
        if not text or not self.agent.audio_engine:
            return

        # 1. Synthesize audio file (verified by 80% MOS Quality Gate)
        audio_path = await self.agent.audio_engine.synthesize_async(text)
        if not audio_path or not PYDUB_AVAILABLE:
            return

        try:
            # 2. Resample synthesized audio to 8,000 Hz mono 16-bit linear PCM
            seg = pydub.AudioSegment.from_file(audio_path)
            seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            raw_pcm = seg.raw_data

            # 3. Launch paced streaming task
            self.current_playback_task = asyncio.create_task(
                self.stream_audio_to_carrier(raw_pcm)
            )
            await self.current_playback_task
        except Exception as e:
            print(f"Error streaming audio to carrier: {e}")

    async def synthesize_and_stream_clause(self, clause_text: str):
        """Synthesizes a single clause and streams 20ms packets down the socket."""
        if not clause_text or not self.agent.audio_engine:
            return
        if self.cancel_playback_event.is_set():
            return

        audio_path = await self.agent.audio_engine.synthesize_async(clause_text)
        if not audio_path or not PYDUB_AVAILABLE:
            return
        if self.cancel_playback_event.is_set():
            return

        # Record first TTS audio chunk generated for latency SLA tracking
        if not self.turn_manager.latency_profiler.breakdown.tts_first_chunk_ts:
            self.turn_manager.latency_profiler.record_tts_first_chunk()

        try:
            seg = pydub.AudioSegment.from_file(audio_path)
            seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            raw_pcm = seg.raw_data

            self.current_playback_task = asyncio.create_task(
                self.stream_audio_to_carrier(raw_pcm)
            )
            await self.current_playback_task
        except Exception as e:
            print(f"Error streaming clause audio to carrier: {e}")

    async def process_caller_turn(
        self,
        raw_pcm_override: Optional[bytes] = None,
        speculative_result: Optional[str] = None,
    ):
        """Transcribes accumulated inbound PCM buffer and triggers streaming pipelined agent step."""
        if raw_pcm_override is not None:
            raw_pcm = raw_pcm_override
        else:
            if not self.inbound_pcm_buffer:
                return
            raw_pcm = b"".join(self.inbound_pcm_buffer)
            self.inbound_pcm_buffer.clear()
            self.silence_frames_count = 0

        self.is_caller_speaking = False

        # If audio is shorter than 250ms (8k * 2 bytes * 0.25s = 4000 bytes), discard as click/pop
        if len(raw_pcm) < 4000:
            return

        # 1. Transcribe audio or use pre-fetched speculative STT result
        if speculative_result:
            user_text = speculative_result
            self.turn_manager.latency_profiler.record_stt_ready()
            print(f"[Turn Taking] Using pre-fetched speculative STT transcript: '{user_text}'")
        else:
            user_text = await self._transcribe_pcm_audio(raw_pcm)
            self.turn_manager.latency_profiler.record_stt_ready()

        if not user_text:
            return

        caller_tag = PIIRedactor.mask_phone(self.caller_phone) if self.caller_phone else "Unknown"
        redacted_user_text = PIIRedactor.redact_text(user_text)
        print(f"[Caller {caller_tag} Turn Transcribed]: '{redacted_user_text}'")

        # If an IVR tree is active, evaluate spoken utterance against IVR tree options
        if self.ivr_machine and not self.ivr_machine.is_completed:
            ivr_res = self.ivr_machine.handle_speech(user_text)
            if ivr_res.status == "OK":
                print(f"[IVR Voice Traversal] Navigated to '{ivr_res.node_id}' via utterance")
                if ivr_res.prompt:
                    self.current_playback_task = asyncio.create_task(
                        self.synthesize_and_play_reply(ivr_res.prompt)
                    )
                if ivr_res.is_terminal:
                    if ivr_res.action == "hangup":
                        self.is_active = False
                    elif ivr_res.action == "handoff_bot":
                        pass
                return
            elif ivr_res.status in ("INVALID_INPUT", "TIMEOUT", "MAX_RETRIES_EXCEEDED"):
                if ivr_res.prompt:
                    self.current_playback_task = asyncio.create_task(
                        self.synthesize_and_play_reply(ivr_res.prompt)
                    )
                if ivr_res.is_terminal and ivr_res.action == "hangup":
                    self.is_active = False
                return

        print("[Streaming Pipeline] Beginning token-to-TTS pipeline for low-latency response...")

        self.cancel_playback_event.clear()
        self.is_agent_streaming = True
        accumulated_reply = []
        first_clause = True

        try:
            async for chunk in self.agent.step_stream(user_text, pcm_bytes=raw_pcm):
                if self.cancel_playback_event.is_set():
                    print("[Streaming Pipeline] Discarded remaining clauses due to caller barge-in.")
                    break

                if chunk["type"] == "clause":
                    clause_text = chunk["text"]
                    accumulated_reply.append(clause_text)
                    if first_clause:
                        self.turn_manager.latency_profiler.record_llm_first_token()
                        first_clause = False
                    print(f"[Agent Clause {chunk.get('index', 0)}]: '{clause_text}'")
                    await self.synthesize_and_stream_clause(clause_text)

                elif chunk["type"] == "tool_call":
                    print(f"[Telephony Tool Triggered]: {chunk.get('tool_event')}")

                elif chunk["type"] == "final":
                    full_text = chunk.get("full_text") or " ".join(accumulated_reply)
                    print(f"[Agent Full Turn Completed]: '{full_text}'")
                    
                    # Process dynamic language identification event
                    lang_info = chunk.get("language_info")
                    if lang_info and lang_info.get("language_switched"):
                        new_lang = lang_info.get("primary_language", self.language)
                        self.language = new_lang
                        if not self.is_binary_mode:
                            switch_msg = json.dumps({
                                "event": "language_switch",
                                "streamSid": self.stream_sid,
                                "language": new_lang,
                                "confidence": lang_info.get("confidence", 1.0),
                                "is_code_switched": lang_info.get("is_code_switched", False),
                                "recommended_voice": lang_info.get("recommended_voice", "")
                            })
                            try:
                                await self.websocket.send_text(switch_msg)
                            except Exception:
                                pass
                        print(f"[LID Gate] Dynamic language switch detected: {new_lang} (confidence {lang_info.get('confidence', 1.0):.2f}, code_switched={lang_info.get('is_code_switched', False)})")

                    if chunk.get("terminated"):
                        print("[Call Terminated]: Agent hung up.")
                        self.is_active = False

                    # Update Supervisor Hub
                    if self.supervisor_manager:
                        self.supervisor_manager.update_turn(
                            call_id=self.call_sid,
                            customer_utterance=user_text,
                            agent_reply=full_text,
                            sentiment=chunk.get("sentiment"),
                            lid_info=lang_info,
                            quality_report=self.agent.audio_engine.last_quality_report.to_dict() if (self.agent.audio_engine and self.agent.audio_engine.last_quality_report) else None,
                            jitter_stats=self.jitter_buffer.get_stats().to_dict(),
                        )

            # Profile glass-to-glass latency at completion of turn
            latency_summary = self.turn_manager.latency_profiler.get_summary()
            if latency_summary.get("glass_to_glass_ms"):
                print(
                    f"[Glass-to-Glass Profiler] Latency Breakdown: "
                    f"Turn={latency_summary.get('turn_detection_ms', 0):.1f}ms, "
                    f"STT={latency_summary.get('stt_latency_ms', 0):.1f}ms, "
                    f"LLM_TTFT={latency_summary.get('llm_ttft_ms', 0):.1f}ms, "
                    f"TTS_TTFB={latency_summary.get('tts_ttfb_ms', 0):.1f}ms, "
                    f"Total={latency_summary['glass_to_glass_ms']:.1f}ms "
                    f"(Sub-300ms SLA: {latency_summary.get('meets_sub_300ms_sla', False)})"
                )
            self.turn_manager.latency_profiler = GlassToGlassLatencyProfiler()

        except Exception as e:
            print(f"[Streaming Turn Error]: {e}")
        finally:
            self.is_agent_streaming = False

    async def _transcribe_pcm_audio(self, pcm_8k_bytes: bytes) -> str:
        """Converts 8kHz PCM to speech transcript using Sovereign STT engine with fallback."""
        try:
            transcript, latency_ms = await asyncio.to_thread(
                self.stt_engine.transcribe_pcm,
                pcm_8k_bytes,
                8000,
                self.language
            )
            if transcript:
                print(f"[Sovereign STT Transcribed in {latency_ms:.1f}ms]: '{transcript}'")
                return transcript
        except Exception as e:
            print(f"[STT Error]: {e}")

        fallback_map = {
            "hi": "हाँ जी, मैं सुन रहा हूँ।",
            "en": "Yes, I am listening.",
            "gu": "હા, હું સાંભળી રહ્યો છું.",
            "mr": "हो, मी ऐकत आहे."
        }
        return fallback_map.get(self.language, "हाँ जी, मैं सुन रहा हूँ।")

    async def handle_inbound_frame(self, raw_frame_bytes: bytes):
        """
        Processes an incoming 20ms audio frame from the carrier:
        1. Decodes to 16-bit linear PCM.
        2. Evaluates acoustic DTMF keypad tones.
        3. Routes through AdaptiveTurnTakingManager for acoustic VAD, speculative early-pipelining, dynamic pause threshold, and barge-in.
        """
        pcm_16 = self.decode_inbound_audio(raw_frame_bytes)
        if len(pcm_16) < 4:
            return

        self.inbound_seq += 1
        now_ms = time.time() * 1000.0
        self.jitter_buffer.push(pcm_data=pcm_16, sequence_number=self.inbound_seq, arrival_time_ms=now_ms)
        frame, is_concealed = self.jitter_buffer.pop(current_time_ms=now_ms)
        if len(frame) < 4:
            return

        # 0. Clean inbound frame via AEC (subtracting bot echo) & Spectral Noise Suppression
        clean_frame, dsp_telemetry = self.dsp_processor.process_inbound_frame(frame)

        # 0.1 Update adaptive comfort noise model when caller is not speaking and bot is not streaming
        if not self.is_caller_speaking and not self.is_agent_streaming:
            self.cng.update_noise_model(clean_frame)

        # 0.2 Enhance Indic retroflex formants and nasal clarity via 5-band biquad equalizer
        enhanced_frame, eq_telemetry = self.equalizer.process_frame(clean_frame)

        # 1. Evaluate acoustic DTMF keypad tones on enhanced inbound PCM
        detected_dtmf_digits = self.dtmf_pad.process_pcm_chunk(enhanced_frame)
        for dtmf_digit in detected_dtmf_digits:
            await self.handle_dtmf_digit(dtmf_digit)

        # 2. Update turn manager dialogue context if agent has a context hint
        if hasattr(self.agent, "turn_context") and self.agent.turn_context:
            try:
                ctx_enum = DialogueContext(self.agent.turn_context)
                self.turn_manager.set_context(ctx_enum)
            except (ValueError, KeyError):
                pass

        # 3. Ingest enhanced frame into AdaptiveTurnTakingManager
        state, event_data = await self.turn_manager.ingest_frame(
            enhanced_frame,
            transcribe_fn=self._transcribe_pcm_audio,
        )

        if state == TurnTakingState.BARGE_IN:
            await self.interrupt_agent_playback()
            self.is_caller_speaking = True

        elif state in (TurnTakingState.SPEECH_ONSET, TurnTakingState.SPEAKING):
            self.is_caller_speaking = True

        elif state == TurnTakingState.TURN_COMPLETED and event_data:
            self.is_caller_speaking = False
            utterance_pcm = event_data.get("pcm_bytes", b"")
            speculative_res = event_data.get("speculative_result")
            await self.process_caller_turn(
                raw_pcm_override=utterance_pcm,
                speculative_result=speculative_res,
            )

    async def run(self):
        """Main event loop receiving frames from the WebSocket."""
        greeting_sent = False
        try:
            while self.is_active:
                message = await self.websocket.receive()

                if "text" in message:
                    text_data = message["text"]
                    try:
                        msg_json = json.loads(text_data)
                    except Exception:
                        continue

                    event = msg_json.get("event")

                    if event == "start":
                        self.stream_sid = msg_json.get("streamSid") or msg_json.get("start", {}).get("streamSid", "stream_001")
                        self.call_sid = msg_json.get("start", {}).get("callSid", "call_001")
                        media_fmt = msg_json.get("start", {}).get("mediaFormat", {})
                        if "encoding" in media_fmt:
                            self.codec = media_fmt["encoding"].lower()
                        print(f"[WebSocket Stream Connected] StreamSid: {self.stream_sid}, Codec: {self.codec}")

                        if not greeting_sent:
                            greeting = self.agent.get_initial_greeting()
                            asyncio.create_task(self.synthesize_and_play_reply(greeting))
                            greeting_sent = True

                        if self.supervisor_manager:
                            self.supervisor_manager.register_call(
                                call_id=self.call_sid,
                                caller_phone=self.caller_phone,
                                persona=self.persona,
                                language=self.language,
                                agent_instance=self.agent,
                            )

                    elif event == "media":
                        payload = msg_json.get("media", {}).get("payload", "")
                        if payload:
                            raw_bytes = base64.b64decode(payload)
                            await self.handle_inbound_frame(raw_bytes)

                    elif event == "dtmf":
                        digit = msg_json.get("dtmf", {}).get("digit") or msg_json.get("digit")
                        if digit:
                            await self.handle_dtmf_digit(str(digit))

                    elif event == "stop":
                        print(f"[WebSocket Stream Stopped] StreamSid: {self.stream_sid}")
                        self.is_active = False
                        break

                elif "bytes" in message:
                    self.is_binary_mode = True
                    if not greeting_sent:
                        greeting = self.agent.get_initial_greeting()
                        asyncio.create_task(self.synthesize_and_play_reply(greeting))
                        greeting_sent = True

                    raw_bytes = message["bytes"]
                    if len(raw_bytes) == 4:
                        rfc_digit = self.rfc4733_decoder.process_packet(raw_bytes)
                        if rfc_digit:
                            await self.handle_dtmf_digit(rfc_digit)
                            continue
                    await self.handle_inbound_frame(raw_bytes)

        except (asyncio.CancelledError, Exception) as e:
            # Normal socket closure on call hangup
            if "disconnect" not in str(e).lower():
                print(f"WebSocket session closed: {e}")
        finally:
            self.is_active = False
            self.inbound_pcm_buffer.clear()
            self.silence_frames_count = 0
            if self.current_playback_task and not self.current_playback_task.done():
                self.current_playback_task.cancel()
            if self.supervisor_manager:
                self.supervisor_manager.terminate_call(self.call_sid, reason="carrier_stream_closed")
