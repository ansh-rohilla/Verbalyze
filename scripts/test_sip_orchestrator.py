#!/usr/bin/env python3
"""
scripts/test_sip_orchestrator.py

Comprehensive Verification Suite for Verbalyze Stateful SIP Soft-Switch Session Orchestrator
and RFC 3261 Call Forking / Blind & Attended Transfer Engine.

Tests:
1. Inbound SIP INVITE & 2-Leg Call Establishment (RFC 3261 Handshake).
2. Consultation Hold via re-INVITE (SDP Direction a=sendonly / MOH Mute).
3. Supervisor Call Leg Forking (Leg C Creation & Silent Monitor Bridging).
4. Supervisor Whisper Coaching Dynamic Escalation.
5. Supervisor Hard Takeover Dynamic Escalation.
6. Supervisor Leg Detachment without Dropping Borrower Call.
7. RFC 3515 Blind Transfer Execution & Clean Agent BYE.
8. RFC 3892 / RFC 3891 Attended Warm Transfer with Consultation Leg & Replaces.
9. FastAPI Telephony SIP Endpoints (/health, /create, /fork, /re-invite, /detach, /refer, /benchmark).
10. Sub-0.5ms Pure-Math & Signaling Latency SLA Benchmark.

Zero-emoji compliant.
"""

import sys
import time
import math
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.telephony.sip_orchestrator import (
    SIPSessionOrchestrator,
    SIPSession,
    SIPMessage,
    SIPMethod,
    SIPCallState,
    CallLegRole,
    TransferType,
    SDPDirection,
)
from verbalyze.telephony.conference_mixer import ConferenceMode


def generate_tone_pcm(freq_hz: float, sample_rate: int = 8000, duration_sec: float = 0.02, amplitude: float = 12000.0) -> bytes:
    """Generates synthetic 16-bit linear PCM sinusoidal audio."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    samples = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.int16)
    return samples.tobytes()


def pcm_energy(pcm_bytes: bytes) -> float:
    """Computes RMS energy of PCM audio."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def test_1_inbound_invite_and_dialog_establishment():
    print("\n" + "=" * 80)
    print("TEST 1: Inbound SIP INVITE & 2-Leg Call Establishment (RFC 3261)")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    call_id = "call-rfc3261-001"
    caller_uri = "sip:borrower9876@telecom.in"
    agent_uri = "sip:recovery_bot@verbalyze.ai"

    session = orchestrator.create_session(call_id=call_id, caller_uri=caller_uri, agent_uri=agent_uri)

    # Initial states
    assert session.leg_a.state == SIPCallState.IDLE
    assert session.leg_b.state == SIPCallState.IDLE

    # Receive inbound INVITE
    invite = SIPMessage(
        is_response=False,
        method=SIPMethod.INVITE,
        uri=agent_uri,
        headers={
            "Via": "SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK-inv01",
            "From": f"<{caller_uri}>;tag=tag-caller-a1",
            "To": f"<{agent_uri}>",
            "Call-ID": call_id,
            "CSeq": "1 INVITE",
            "Contact": f"<{caller_uri}>",
        },
        body=session._generate_sdp(SDPDirection.SENDRECV),
    )

    ringing, ok_resp = session.handle_inbound_invite(invite)

    print(f"[Handshake] 180 Ringing Code: {ringing.status_code} ({ringing.reason_phrase})")
    print(f"[Handshake] 200 OK Code:      {ok_resp.status_code} ({ok_resp.reason_phrase})")
    assert ringing.status_code == 180
    assert ok_resp.status_code == 200
    assert "application/sdp" in ok_resp.get_header("Content-Type")
    assert "a=sendrecv" in ok_resp.body

    assert session.leg_a.state == SIPCallState.RINGING
    assert session.leg_b.state == SIPCallState.RINGING

    # Client ACK
    ack = SIPMessage(
        is_response=False,
        method=SIPMethod.ACK,
        uri=agent_uri,
        headers={"Call-ID": call_id, "CSeq": "1 ACK"},
    )
    session.handle_ack(ack)

    print(f"[Handshake] Leg A State: {session.leg_a.state.value}")
    print(f"[Handshake] Leg B State: {session.leg_b.state.value}")
    assert session.leg_a.state == SIPCallState.CONNECTED
    assert session.leg_b.state == SIPCallState.CONNECTED
    assert session.mixer.current_mode == ConferenceMode.SILENT_MONITOR
    print("[PASS] Inbound SIP INVITE 3-way handshake established 2-leg session successfully.")


def test_2_consultation_hold_reinvite():
    print("\n" + "=" * 80)
    print("TEST 2: Consultation Hold via re-INVITE (SDP Direction a=sendonly / MOH Mute)")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    session = orchestrator.create_session("call-hold-002", "sip:borrower@telecom.in")

    # Connect initial call
    session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
    session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))

    # Place on Hold
    reinvite_hold = session.set_hold(hold=True, moh_enabled=True)
    print(f"[Hold] Leg A State: {session.leg_a.state.value} (hold_active={session.leg_a.hold_active})")
    print(f"[Hold] SDP Direction: {session.leg_a.sdp_direction.value}")
    print(f"[Hold] re-INVITE Header X-Verbalyze-Hold: {reinvite_hold.get_header('X-Verbalyze-Hold')}")

    assert session.leg_a.state == SIPCallState.ON_HOLD
    assert session.leg_a.hold_active is True
    assert session.leg_a.sdp_direction == SDPDirection.SENDONLY
    assert "a=sendonly" in reinvite_hold.body

    # Verify mixer gains muted during hold
    gains = session.mixer.get_gains()
    print(f"[Hold] Mixer Gains (Customer Uplink Muted): {gains}")
    assert gains["g_CA"] == 0.0
    assert gains["g_AC"] == 0.0

    # Resume from Hold (Unhold)
    reinvite_resume = session.set_hold(hold=False)
    print(f"[Resume] Leg A State: {session.leg_a.state.value} (hold_active={session.leg_a.hold_active})")
    assert session.leg_a.state == SIPCallState.CONNECTED
    assert session.leg_a.hold_active is False
    assert session.leg_a.sdp_direction == SDPDirection.SENDRECV
    assert "a=sendrecv" in reinvite_resume.body
    print("[PASS] Consultation hold & resume via re-INVITE and audio decoupling verified.")


def test_3_supervisor_call_leg_forking():
    print("\n" + "=" * 80)
    print("TEST 3: Supervisor Call Leg Forking (Leg C Creation & Silent Monitor Bridging)")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    session = orchestrator.create_session("call-fork-003", "sip:borrower@telecom.in")
    session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
    session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))

    supervisor_uri = "sip:supervisor_rajesh@sbi.co.in"
    leg_c, invite_msg = session.attach_supervisor(supervisor_uri, mode=ConferenceMode.SILENT_MONITOR)

    print(f"[Fork] Leg C ID:    {leg_c.leg_id}")
    print(f"[Fork] Leg C Role:  {leg_c.role.value}")
    print(f"[Fork] Leg C State: {leg_c.state.value}")
    print(f"[Fork] Leg C To:    {leg_c.to_uri}")
    print(f"[Fork] Mixer Mode:  {session.mixer.current_mode.value}")

    assert leg_c.role == CallLegRole.SUPERVISOR
    assert leg_c.state == SIPCallState.CONNECTED
    assert leg_c.to_uri == supervisor_uri
    assert session.mixer.current_mode == ConferenceMode.SILENT_MONITOR

    # Verify INVITE message headers
    assert invite_msg.method == SIPMethod.INVITE
    assert invite_msg.get_header("X-Verbalyze-Forked-Leg") == "supervisor"
    assert invite_msg.get_header("X-Verbalyze-Conference-Mode") == "SILENT_MONITOR"
    print("[PASS] Supervisor call leg successfully forked and bridged in silent monitor mode.")


def test_4_supervisor_whisper_coach_escalation():
    print("\n" + "=" * 80)
    print("TEST 4: Supervisor Whisper Coaching Dynamic Escalation")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    session = orchestrator.create_session("call-whisper-004", "sip:borrower@telecom.in")
    session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
    session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))
    session.attach_supervisor("sip:supervisor@bank.in", mode=ConferenceMode.SILENT_MONITOR)

    # Escalate to Whisper Coach
    session.set_supervisor_mode(ConferenceMode.WHISPER_COACH)
    print(f"[Whisper] Updated Mixer Mode: {session.mixer.current_mode.value}")
    assert session.mixer.current_mode == ConferenceMode.WHISPER_COACH

    # Process audio through coupled mixer to verify acoustics
    pcm_c = generate_tone_pcm(300.0, amplitude=6000.0)
    pcm_s = generate_tone_pcm(800.0, amplitude=10000.0)

    # Let crossfade settle
    session.mixer.process_frame(customer_pcm=pcm_c, supervisor_pcm=pcm_s)
    res = session.mixer.process_frame(customer_pcm=pcm_c, supervisor_pcm=pcm_s)

    energy_c_out = pcm_energy(res.customer_out_pcm)
    energy_a_out = pcm_energy(res.agent_out_pcm)

    print(f"[Whisper] Customer Output Energy: {energy_c_out:.2f} (Expected: 0.0)")
    print(f"[Whisper] Agent Output Energy:    {energy_a_out:.1f} (Must hear Supervisor)")

    assert energy_c_out == 0.0, "Customer must NEVER hear whisper coach!"
    assert energy_a_out > 7000.0, "Agent must hear supervisor whisper!"
    print("[PASS] Dynamic whisper coaching escalation verified with 100% customer isolation.")


def test_5_supervisor_hard_takeover_escalation():
    print("\n" + "=" * 80)
    print("TEST 5: Supervisor Hard Takeover Dynamic Escalation")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    session = orchestrator.create_session("call-takeover-005", "sip:borrower@telecom.in")
    session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
    session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))
    session.attach_supervisor("sip:supervisor@bank.in", mode=ConferenceMode.SILENT_MONITOR)

    # Escalate to Hard Takeover
    session.set_supervisor_mode(ConferenceMode.HARD_TAKEOVER)
    print(f"[Takeover] Updated Mixer Mode: {session.mixer.current_mode.value}")
    assert session.mixer.current_mode == ConferenceMode.HARD_TAKEOVER

    # Audio verification: Agent must be muted to Customer, Customer hears Supervisor
    pcm_a = generate_tone_pcm(500.0, amplitude=9000.0)
    pcm_s = generate_tone_pcm(800.0, amplitude=9000.0)

    session.mixer.process_frame(agent_pcm=pcm_a, supervisor_pcm=pcm_s)
    res_agent_only = session.mixer.process_frame(agent_pcm=pcm_a)
    res_sup_only = session.mixer.process_frame(supervisor_pcm=pcm_s)

    e_c_from_agent = pcm_energy(res_agent_only.customer_out_pcm)
    e_c_from_sup = pcm_energy(res_sup_only.customer_out_pcm)

    print(f"[Takeover] Customer hears Agent:      {e_c_from_agent:.2f} (Expected: 0.0 - Muted)")
    print(f"[Takeover] Customer hears Supervisor: {e_c_from_sup:.1f} (Expected: Direct Speech)")

    assert e_c_from_agent == 0.0
    assert e_c_from_sup > 6000.0
    print("[PASS] Hard takeover escalation verified: supervisor speaks directly, agent muted.")


def test_6_supervisor_leg_detachment_isolation():
    print("\n" + "=" * 80)
    print("TEST 6: Supervisor Leg Detachment without Dropping Borrower Call")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    session = orchestrator.create_session("call-detach-006", "sip:borrower@telecom.in")
    session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
    session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))
    session.attach_supervisor("sip:supervisor@bank.in", mode=ConferenceMode.WHISPER_COACH)

    assert session.leg_c is not None
    assert session.leg_c.state == SIPCallState.CONNECTED

    # Detach supervisor leg
    bye_msg = session.detach_supervisor(reason="Coaching session ended")

    print(f"[Detach] Leg C State: {session.leg_c.state.value}")
    print(f"[Detach] Leg A State: {session.leg_a.state.value} (Must be CONNECTED)")
    print(f"[Detach] Leg B State: {session.leg_b.state.value} (Must be CONNECTED)")
    print(f"[Detach] BYE Method:  {bye_msg.method.value if bye_msg else None}")

    assert bye_msg is not None
    assert bye_msg.method == SIPMethod.BYE
    assert session.leg_c.state == SIPCallState.TERMINATED
    assert session.leg_a.state == SIPCallState.CONNECTED
    assert session.leg_b.state == SIPCallState.CONNECTED
    assert session.mixer.current_mode == ConferenceMode.SILENT_MONITOR
    print("[PASS] Supervisor leg detached cleanly; borrower and agent dialog unaffected.")


def test_7_rfc3515_blind_transfer():
    print("\n" + "=" * 80)
    print("TEST 7: RFC 3515 Blind Transfer Execution & Clean Agent BYE")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    session = orchestrator.create_session("call-blind-007", "sip:borrower@telecom.in")
    session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
    session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))

    target_destination = "sip:recovery_head@bank.in"
    refer_msg, agent_bye = session.blind_transfer(target_destination)

    print(f"[Blind Transfer] REFER Method:   {refer_msg.method.value}")
    print(f"[Blind Transfer] Refer-To:       {refer_msg.get_header('Refer-To')}")
    print(f"[Blind Transfer] Leg A State:    {session.leg_a.state.value}")
    print(f"[Blind Transfer] Leg B State:    {session.leg_b.state.value}")
    print(f"[Blind Transfer] Agent BYE CSeq: {agent_bye.get_header('CSeq')}")

    assert refer_msg.method == SIPMethod.REFER
    assert refer_msg.get_header("Refer-To") == f"<{target_destination}>"
    assert session.leg_a.state == SIPCallState.TRANSFERRING
    assert session.leg_b.state == SIPCallState.TERMINATED
    assert agent_bye.method == SIPMethod.BYE
    print("[PASS] RFC 3515 Blind transfer successfully dispatched.")


def test_8_rfc3892_attended_warm_transfer():
    print("\n" + "=" * 80)
    print("TEST 8: RFC 3892 / RFC 3891 Attended Warm Transfer with Consultation & Replaces")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    session = orchestrator.create_session("call-attended-008", "sip:borrower@telecom.in")
    session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
    session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))

    target_destination = "sip:branch_manager@bank.in"

    # Step 1: Start Attended Transfer (puts borrower on hold, initiates consult leg)
    consult_leg, consult_invite = session.attended_transfer_start(target_destination)

    print(f"[Warm Transfer] Leg A State:          {session.leg_a.state.value} (Hold={session.leg_a.hold_active})")
    print(f"[Warm Transfer] Consult Leg Role:     {consult_leg.role.value}")
    print(f"[Warm Transfer] Consult Leg State:    {consult_leg.state.value}")
    print(f"[Warm Transfer] Consult Leg To:       {consult_leg.to_uri}")
    print(f"[Warm Transfer] Consult INVITE CSeq:  {consult_invite.get_header('CSeq')}")

    assert session.leg_a.state == SIPCallState.ON_HOLD
    assert session.leg_a.hold_active is True
    assert consult_leg.role == CallLegRole.TRANSFER_TARGET
    assert consult_leg.state == SIPCallState.CONNECTED

    # Step 2: Complete Attended Transfer (sends REFER with Replaces, drops agent)
    refer_msg, agent_bye = session.attended_transfer_complete()

    refer_to_header = refer_msg.get_header("Refer-To", "")
    print(f"[Warm Transfer] REFER Refer-To:       {refer_to_header}")
    print(f"[Warm Transfer] Leg B (Agent) State:  {session.leg_b.state.value}")

    assert "Replaces=" in refer_to_header
    assert consult_leg.call_id in refer_to_header
    assert session.leg_b.state == SIPCallState.TERMINATED
    assert agent_bye.method == SIPMethod.BYE
    print("[PASS] RFC 3892 Attended warm transfer with Replaces header verified.")


def test_9_fastapi_sip_endpoints():
    print("\n" + "=" * 80)
    print("TEST 9: FastAPI Telephony SIP Endpoints (/health, /create, /re-invite, /fork, /detach, /refer, /benchmark)")
    print("=" * 80)
    from fastapi.testclient import TestClient
    from verbalyze.telephony.server import create_app

    app = create_app()
    client = TestClient(app)

    # 1. Health check verification
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data.get("sip_orchestrator_status") == "ready", "SIP orchestrator not ready in health check!"
    print("[PASS] /health reports sip_orchestrator_status: ready")

    # 2. Create session endpoint
    c_resp = client.post("/telephony/sip/session/create", json={
        "call_id": "api-call-001",
        "caller_uri": "sip:borrower@telecom.in",
        "agent_uri": "sip:agent@verbalyze.ai",
    })
    assert c_resp.status_code == 200
    c_data = c_resp.json()
    assert c_data["status"] == "ok"
    assert c_data["session"]["session_id"] == "api-call-001"
    assert "SIP/2.0 180 Ringing" in c_data["sip_ringing"]
    assert "SIP/2.0 200 OK" in c_data["sip_ok"]
    print("[PASS] /telephony/sip/session/create verified.")

    # 3. Fork supervisor endpoint
    f_resp = client.post("/telephony/sip/session/fork", json={
        "call_id": "api-call-001",
        "supervisor_uri": "sip:sup@bank.in",
        "mode": "whisper_coach",
    })
    assert f_resp.status_code == 200
    f_data = f_resp.json()
    assert f_data["status"] == "ok"
    assert f_data["mixer_mode"] == "WHISPER_COACH"
    print("[PASS] /telephony/sip/session/fork verified.")

    # 4. Hold endpoint
    h_resp = client.post("/telephony/sip/session/re-invite", json={
        "call_id": "api-call-001",
        "hold": True,
        "moh": True,
    })
    assert h_resp.status_code == 200
    assert h_resp.json()["hold_active"] is True
    print("[PASS] /telephony/sip/session/re-invite (hold) verified.")

    # 5. Detach supervisor endpoint
    d_resp = client.post("/telephony/sip/session/detach", json={
        "call_id": "api-call-001",
        "reason": "Supervisor detached via console",
    })
    assert d_resp.status_code == 200
    assert d_resp.json()["detached"] is True
    print("[PASS] /telephony/sip/session/detach verified.")

    # 6. Refer (Blind Transfer) endpoint
    r_resp = client.post("/telephony/sip/session/refer", json={
        "call_id": "api-call-001",
        "transfer_type": "blind",
        "target_uri": "sip:manager@bank.in",
    })
    assert r_resp.status_code == 200
    assert r_resp.json()["type"] == "blind"
    print("[PASS] /telephony/sip/session/refer verified.")

    # 7. Session status endpoint
    s_resp = client.get("/telephony/sip/session/api-call-001")
    assert s_resp.status_code == 200
    assert s_resp.json()["session"]["session_id"] == "api-call-001"
    print("[PASS] GET /telephony/sip/session/{call_id} verified.")

    # 8. Benchmark endpoint
    b_resp = client.post("/telephony/sip/session/benchmark", json={})
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert b_data["meets_sla"] is True
    print(f"[PASS] /telephony/sip/session/benchmark: avg={b_data['avg_orchestration_ms']}ms, SLA={b_data['target_sla_ms']}ms")


def test_10_sub_05ms_orchestration_latency_benchmark():
    print("\n" + "=" * 80)
    print("TEST 10: Sub-0.5ms Pure-Math & Signaling Latency SLA Benchmark")
    print("=" * 80)
    orchestrator = SIPSessionOrchestrator()
    n_iterations = 200
    durations = []

    for i in range(n_iterations):
        t0 = time.perf_counter()
        call_id = f"bench-cycle-{i}"
        session = orchestrator.create_session(call_id, "sip:borrower@telecom.in")

        # 1. Inbound INVITE
        session.handle_inbound_invite(SIPMessage(is_response=False, method=SIPMethod.INVITE))
        # 2. Handshake ACK
        session.handle_ack(SIPMessage(is_response=False, method=SIPMethod.ACK))
        # 3. Fork Supervisor
        session.attach_supervisor("sip:sup@bank.in", mode=ConferenceMode.WHISPER_COACH)
        # 4. Hold
        session.set_hold(True)
        # 5. Unhold
        session.set_hold(False)
        # 6. Detach Supervisor
        session.detach_supervisor()
        # 7. Disconnect session
        session.terminate_session()
        orchestrator.remove_session(call_id)

        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(durations))
    p95_ms = float(np.percentile(durations, 95))
    max_ms = float(np.max(durations))

    print(f"[Benchmark] Iterations:              {n_iterations} complete dialog cycles")
    print(f"[Benchmark] Mean Cycle Latency:      {avg_ms:.4f} ms")
    print(f"[Benchmark] 95th Percentile Latency: {p95_ms:.4f} ms")
    print(f"[Benchmark] Maximum Observed:        {max_ms:.4f} ms")
    print(f"[Benchmark] Target SLA (<0.50ms):     {'PASS' if avg_ms < 0.5 else 'FAIL'}")

    assert avg_ms < 0.50, f"Mean latency {avg_ms:.4f}ms exceeded 0.50ms SLA"
    print("[PASS] Sub-0.5ms signaling state machine throughput verified.")


def run_all_sip_orchestrator_tests():
    t_start = time.perf_counter()
    print("\n" + "=" * 80)
    print("VERBALYZE: STATEFUL SIP SOFT-SWITCH SESSION ORCHESTRATOR TEST SUITE")
    print("=" * 80)

    test_1_inbound_invite_and_dialog_establishment()
    test_2_consultation_hold_reinvite()
    test_3_supervisor_call_leg_forking()
    test_4_supervisor_whisper_coach_escalation()
    test_5_supervisor_hard_takeover_escalation()
    test_6_supervisor_leg_detachment_isolation()
    test_7_rfc3515_blind_transfer()
    test_8_rfc3892_attended_warm_transfer()
    test_9_fastapi_sip_endpoints()
    test_10_sub_05ms_orchestration_latency_benchmark()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL 10 SIP ORCHESTRATOR TESTS PASSED IN {total_time:.2f}s!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_sip_orchestrator_tests()
