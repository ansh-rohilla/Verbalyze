"""
scripts/test_supervisor_and_browser_gateway.py

Comprehensive Test Suite for:
1. RFC 3550 Adaptive Telecom Jitter Buffer & Packet Reordering.
2. Packet Loss Concealment (PLC) & Underrun Mitigation.
3. Supervisor Observability Hub: Registration, Fleet Telemetry & DPDP Masking.
4. Supervisor Whisper Coaching: Private Injection & VoiceAgent Prompt Adaptation.
5. Supervisor Takeover & 1-Click Barge-In Escalation.
6. Real-Time High-Agitation Alert Generation & Pub/Sub Dispatch.
7. In-Browser Audio Gateway Frame Processing & VAD.
8. FastAPI Supervisor & Browser Gateway Endpoints Integration.

Zero-emoji compliant.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from verbalyze.agent.voice_bot import VoiceAgent
from verbalyze.telephony.jitter_buffer import AdaptiveJitterBuffer, JitterBufferPacket
from verbalyze.telephony.supervisor import SupervisorManager, CallSupervisorRecord
from verbalyze.telephony.browser_gateway import BrowserAudioSession
from verbalyze.telephony.server import create_app


def test_1_jitter_buffer_rfc3550_and_reordering():
    print("\n--- Test 1: Jitter Buffer RFC 3550 & Out-of-Order Reordering ---")
    jb = AdaptiveJitterBuffer(frame_duration_ms=20.0, sample_rate=8000, bytes_per_sample=2)
    pcm_dummy = b"\x05\x00" * 160  # 320 bytes = 20ms of 8kHz 16-bit PCM

    # Push packets out-of-order: seq 1, 3, 2
    jb.push(pcm_dummy, sequence_number=1, timestamp_ms=0.0, arrival_time_ms=0.0)
    jb.push(pcm_dummy, sequence_number=3, timestamp_ms=40.0, arrival_time_ms=65.0)  # +25ms transit delay
    jb.push(pcm_dummy, sequence_number=2, timestamp_ms=20.0, arrival_time_ms=70.0)  # delayed reordered packet

    stats = jb.get_stats()
    assert stats.packets_reordered >= 1, f"Expected packets_reordered >= 1, got {stats.packets_reordered}"
    assert stats.current_jitter_ms > 0.0, f"Expected current_jitter_ms > 0, got {stats.current_jitter_ms}"

    # Playout frames: seq 1, then seq 2, then seq 3
    f1, c1 = jb.pop()
    f2, c2 = jb.pop()
    f3, c3 = jb.pop()

    assert not c1, "Frame 1 should not be concealed"
    assert not c2, "Frame 2 should be reordered and played without concealment"
    assert not c3, "Frame 3 should not be concealed"
    print(f"[PASSED] Packets reordered successfully. Measured jitter: {stats.current_jitter_ms:.2f}ms, Reordered: {stats.packets_reordered}")


def test_2_packet_loss_concealment_and_underrun():
    print("\n--- Test 2: Packet Loss Concealment (PLC) & Underrun Mitigation ---")
    jb = AdaptiveJitterBuffer(frame_duration_ms=20.0, sample_rate=8000, bytes_per_sample=2)
    pcm_valid = b"\x10\x00" * 160

    # Push packet 1, then packet 3 (packet 2 is lost in telecom network)
    jb.push(pcm_valid, sequence_number=1, timestamp_ms=0.0, arrival_time_ms=0.0)
    jb.push(pcm_valid, sequence_number=3, timestamp_ms=40.0, arrival_time_ms=40.0)

    f1, c1 = jb.pop()
    assert not c1, "Frame 1 must be valid"

    # Frame 2 is missing, pop should trigger PLC
    f2, c2 = jb.pop()
    assert c2, "Frame 2 must be synthesized via PLC"
    assert len(f2) == 320, f"Expected 320 bytes concealment frame, got {len(f2)}"

    # Frame 3 is valid
    f3, c3 = jb.pop()
    assert not c3, "Frame 3 must be valid"

    # Queue is now empty, popping again should register underrun and produce comfort noise
    f4, c4 = jb.pop()
    assert c4, "Frame 4 must be synthesized on buffer underrun"
    stats = jb.get_stats()
    assert stats.concealed_frames_count >= 2, f"Expected >= 2 concealed frames, got {stats.concealed_frames_count}"
    assert stats.underrun_count >= 1, f"Expected >= 1 underrun, got {stats.underrun_count}"
    assert stats.packets_lost >= 1, f"Expected >= 1 packet lost, got {stats.packets_lost}"
    print(f"[PASSED] PLC smoothly synthesized missing frame and underrun. Concealed frames: {stats.concealed_frames_count}, Underruns: {stats.underrun_count}")


def test_3_supervisor_hub_registration_and_fleet_summary():
    print("\n--- Test 3: Supervisor Hub Registration, Turn Updating & Fleet Telemetry ---")
    sm = SupervisorManager()

    # Register two active calls
    r1 = sm.register_call("call_001", caller_phone="+919876543210", persona="muthoot_recovery", language="hi")
    r2 = sm.register_call("call_002", caller_phone="+919123456780", persona="bank_kyc", language="en")

    # Verify DPDP masking on caller phone
    assert r1.caller_phone_masked == "+91*****3210", f"Masking failed: {r1.caller_phone_masked}"
    assert r2.caller_phone_masked == "+91*****6780", f"Masking failed: {r2.caller_phone_masked}"
    assert len(sm.active_calls) == 2

    # Update dialogue turn with sentiment, LID and jitter telemetry
    sm.update_turn(
        call_id="call_001",
        customer_utterance="Maine payment kar diya hai",
        agent_reply="Dhanyavaad, main check karti hoon.",
        sentiment={"category": "CALM", "composite_agitation": 0.15, "dispute_type": "NONE"},
        lid_info={"primary_language": "hi", "is_code_switched": True, "language_switched": False},
        quality_report={"composite_mos": 4.35},
        jitter_stats={"current_jitter_ms": 14.5, "packet_loss_rate": 0.01},
    )

    call1 = sm.get_call("call_001")
    assert call1 is not None
    assert call1.turn_count == 1
    assert call1.agitation_score == 0.15
    assert call1.is_code_switched is True
    assert call1.mos_score == 4.35
    assert call1.jitter_ms == 14.5

    summary = sm.get_fleet_summary()
    assert summary["active_calls_count"] == 2
    assert summary["code_switched_calls"] == 1
    assert summary["high_agitation_calls"] == 0
    assert summary["status"] == "nominal"
    print(f"[PASSED] Call turns registered with DPDP masking. Fleet summary: {summary}")


def test_4_supervisor_whisper_coaching():
    print("\n--- Test 4: Supervisor Whisper Coaching Injection into VoiceAgent ---")
    sm = SupervisorManager()
    agent = VoiceAgent(language="hi", persona="muthoot_recovery", llm_provider="mock", voice_enabled=False)

    call_id = "call_whisper_test"
    sm.register_call(call_id=call_id, caller_phone="+919876543210", agent_instance=agent)

    # Supervisor injects private coaching instruction
    whisper_text = "Offer 100% late fee waiver if customer settles today"
    success = sm.inject_whisper(call_id=call_id, whisper_text=whisper_text, supervisor_id="sup_lead_01")
    assert success is True

    record = sm.get_call(call_id)
    assert record.status == "SUPERVISOR_COACHING"
    assert len(record.whisper_history) == 1
    assert record.whisper_history[0].text == whisper_text

    # Next conversational turn should incorporate supervisor coaching
    turn_res = agent.step("Main late fees nahi bharoonga, bohot zyada hai.")
    assert "whispers" in turn_res
    assert whisper_text in turn_res["whispers"]
    # Agent response should reflect the waiver
    assert "माफ" in turn_res["text"] or "waive" in turn_res["text"].lower() or "छूट" in turn_res["text"]
    print(f"[PASSED] Whisper directive injected and executed by agent. Response: '{turn_res['text']}'")


def test_5_supervisor_takeover_and_barge_in():
    print("\n--- Test 5: Supervisor Takeover & 1-Click Barge-In Escalation ---")
    sm = SupervisorManager()
    agent = VoiceAgent(language="hi", persona="muthoot_recovery", llm_provider="mock", voice_enabled=False)
    call_id = "call_takeover_test"
    sm.register_call(call_id=call_id, caller_phone="+919876543210", agent_instance=agent)

    # Trigger 1-click barge-in
    barge_res = sm.barge_in(call_id=call_id, supervisor_id="sup_lead_01")
    assert barge_res is True

    # Trigger supervisor takeover
    takeover_res = sm.takeover_call(
        call_id=call_id,
        supervisor_id="sup_lead_01",
        target_sip="sip:supervisor@telephony.internal",
        reason="Borrower demands human escalation"
    )
    assert takeover_res["success"] is True
    assert takeover_res["transfer_details"]["target_sip"] == "sip:supervisor@telephony.internal"

    record = sm.get_call(call_id)
    assert record.status == "SUPERVISOR_TAKEOVER"

    # Terminate call
    sm.terminate_call(call_id=call_id, reason="transferred_to_supervisor")
    assert sm.get_call(call_id) is None
    assert len(sm.call_history) == 1
    assert sm.call_history[0].status == "TERMINATED"
    print(f"[PASSED] Takeover initiated with SIP REFER and DPDP audit record.")


def test_6_high_agitation_alert_dispatch():
    print("\n--- Test 6: High-Agitation Alert Generation & Pub/Sub Dispatch ---")
    sm = SupervisorManager()
    call_id = "call_agitated_customer"
    sm.register_call(call_id=call_id, caller_phone="+919876543210")

    # Subscribe to WebSocket stream queue
    queue = sm.subscribe()

    # Simulate hostile customer turn
    sm.update_turn(
        call_id=call_id,
        customer_utterance="Main tum sab par FIR kar doonga, phone rakho!",
        agent_reply="Sir kripya shant rahein.",
        sentiment={"category": "HOSTILE", "composite_agitation": 0.88, "dispute_type": "LEGAL_THREAT"},
        jitter_stats={"current_jitter_ms": 18.0}
    )

    # Queue should receive turn_update event with alert
    assert not queue.empty(), "Expected event in supervisor queue"
    msg_str = queue.get_nowait()
    event = json.loads(msg_str)
    assert event["event"] == "turn_update"
    assert event["alert"] == "HIGH_AGITATION_ALERT", f"Expected HIGH_AGITATION_ALERT, got {event.get('alert')}"
    assert event["agitation_score"] >= 0.70

    summary = sm.get_fleet_summary()
    assert summary["high_agitation_calls"] == 1
    assert summary["status"] == "elevated_alert"

    sm.unsubscribe(queue)
    print(f"[PASSED] High-agitation anomaly triggered alert banner dispatch: {event['alert']}")


def test_7_browser_audio_session_packet_processing():
    print("\n--- Test 7: In-Browser Audio Gateway Frame Processing & VAD ---")
    sm = SupervisorManager()

    # Mock WebSocket
    class MockWebSocket:
        def __init__(self):
            self.sent_messages = []
            self.is_closed = False

        async def send_text(self, text: str):
            self.sent_messages.append(text)

        async def receive(self):
            await asyncio.sleep(10)
            return {"type": "stop"}

    mock_ws = MockWebSocket()
    session = BrowserAudioSession(
        websocket=mock_ws,
        session_id="browser_test_001",
        language="hi",
        persona="muthoot_recovery",
        llm_provider="mock",
        sample_rate=16000,
        caller_phone="+919876543210",
        supervisor_manager=sm,
    )

    # Ingest 20ms PCM audio frame (16kHz * 2 bytes * 0.02s = 640 bytes)
    # Loud speech frame (amplitude 3000 > threshold 700)
    loud_frame = (b"\xb8\x0b") * 320  # int16 value 3000
    asyncio.run(session.handle_inbound_pcm(loud_frame))

    assert session.is_caller_speaking is True
    assert len(session.inbound_pcm_buffer) >= 1

    # Ingest 35 trailing silence frames to trigger turn completion
    silence_frame = b"\x00" * 640
    for _ in range(35):
        asyncio.run(session.handle_inbound_pcm(silence_frame))

    # Turn should have executed and updated supervisor hub
    rec = sm.get_call("browser_test_001")
    assert rec is not None
    assert rec.turn_count >= 1
    print(f"[PASSED] In-browser frame buffered through jitter buffer and dispatched turn. Turns: {rec.turn_count}")


def test_8_fastapi_supervisor_and_browser_endpoints():
    print("\n--- Test 8: FastAPI Supervisor REST & Browser Gateway Endpoints ---")
    app = create_app()
    client = TestClient(app)

    # 1. Health check includes supervisor counts
    r = client.get("/health")
    assert r.status_code == 200
    assert "supervisor_active_calls" in r.json()

    # 2. Supervisor dashboard UI
    r = client.get("/telephony/supervisor/dashboard")
    assert r.status_code == 200
    assert "Verbalyze Telephony - Supervisor Live Console" in r.text

    # 3. Browser client UI
    r = client.get("/telephony/browser-client")
    assert r.status_code == 200
    assert "Verbalyze Telephony - In-Browser Voice Client" in r.text

    # 4. Supervisor active calls JSON
    r = client.get("/telephony/supervisor/calls")
    assert r.status_code == 200
    data = r.json()
    assert "calls" in data
    assert "summary" in data

    # 5. Supervisor fleet summary JSON
    r = client.get("/telephony/supervisor/summary")
    assert r.status_code == 200
    summary = r.json()
    assert "active_calls_count" in summary

    # 6. Supervisor whisper injection
    r = client.post("/telephony/supervisor/whisper", json={
        "call_id": "non_existent_call",
        "text": "Please waive the late fees."
    })
    assert r.status_code == 404  # correctly returns 404 for unknown call

    # 7. Supervisor barge-in
    r = client.post("/telephony/supervisor/barge-in", json={
        "call_id": "non_existent_call"
    })
    assert r.status_code == 200
    assert r.json()["success"] is False

    print("[PASSED] All supervisor REST and browser HTML gateway endpoints verified successfully.")


if __name__ == "__main__":
    test_1_jitter_buffer_rfc3550_and_reordering()
    test_2_packet_loss_concealment_and_underrun()
    test_3_supervisor_hub_registration_and_fleet_summary()
    test_4_supervisor_whisper_coaching()
    test_5_supervisor_takeover_and_barge_in()
    test_6_high_agitation_alert_dispatch()
    test_7_browser_audio_session_packet_processing()
    test_8_fastapi_supervisor_and_browser_endpoints()
    print("\n================================================================================")
    print("ALL 8 TELEPHONY SUPERVISOR & IN-BROWSER GATEWAY TESTS PASSED (100% SUCCESS)")
    print("================================================================================\n")
