#!/usr/bin/env python3
"""
scripts/test_sentiment_and_transfer.py

Comprehensive Verification Suite for Verbalyze Real-Time Acoustic Sentiment Detection
and Human Warm Transfer Protocol (RFC 3515 SIP REFER):
1. Acoustic Agitation Scoring (RMS volume dynamics, energy variance, pitch jitter)
2. Lexical Dispute & Escalation Classification (Payment dispute, legal threats, harassment, human agent requests)
3. Unified Sentiment Engine & Composite Agitation Index Categorization (CALM, ELEVATED, AGITATED, CRITICAL)
4. Conversational De-escalation Posture in Indic VoiceAgent
5. Automatic Transfer Triggering upon Critical Agitation / Legal Threats
6. SIPTransferDispatcher (RFC 3515 SIP REFER, TwiML Dial, WebSocket Control Frames) & TransferContext
7. Campaign Dialing CDR Dispositions (TRANSFERRED_TO_SUPERVISOR, LEGAL_DISPUTE_ESCALATED)
8. FastAPI Telephony Endpoints (/telephony/sentiment, /telephony/transfer, /webhook/sip/turn)
"""

import os
import sys
import json
import base64
import asyncio
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timezone
from verbalyze.agent.sentiment import (
    AcousticSentimentAnalyzer,
    LexicalDisputeClassifier,
    UnifiedSentimentEngine,
    SentimentResult,
    SentimentCategory,
    DisputeType,
)
from verbalyze.telephony.transfer import (
    TransferContext,
    SIPTransferDispatcher,
)
from verbalyze.agent.voice_bot import VoiceAgent
from verbalyze.campaign.models import (
    Lead,
    CallDisposition,
    CampaignConfig,
    CallDetailRecord,
)
from verbalyze.campaign.dialer import CampaignDialer


def print_header(title: str):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def generate_synthetic_pcm(
    duration_sec: float = 1.0,
    sample_rate: int = 8000,
    amplitude: float = 1200.0,
    frequency: float = 200.0,
    noise_level: float = 50.0,
) -> bytes:
    """Generates synthetic 16-bit mono PCM audio for testing acoustic analyzer."""
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, noise_level, total_samples)
    signal = np.clip(sine_wave + noise, -32768, 32767).astype(np.int16)
    return signal.tobytes()


def generate_shouting_pcm(
    duration_sec: float = 1.0,
    sample_rate: int = 8000,
) -> bytes:
    """Generates loud, shouting audio with sudden energy bursts and high variance."""
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    # Fundamental frequency modulated with harsh harmonic spikes
    base = 9000.0 * np.sin(2 * np.pi * 380.0 * t)
    harmonic = 7000.0 * np.sin(2 * np.pi * 760.0 * t)
    # Add bursts of intense noise
    burst = np.random.normal(0, 3000.0, total_samples)
    signal = np.clip(base + harmonic + burst, -32768, 32767).astype(np.int16)
    return signal.tobytes()


# --------------------------------------------------------------------------
# Test 1: Acoustic Agitation Scoring
# --------------------------------------------------------------------------
def test_acoustic_scoring():
    print_header("1. Testing Acoustic Sentiment Analyzer (RMS Volume, Variance, Jitter)")

    analyzer = AcousticSentimentAnalyzer(sample_rate=8000)

    # 1. Calm speech audio (RMS ~1200)
    calm_pcm = generate_synthetic_pcm(duration_sec=1.2, amplitude=1200.0, frequency=180.0)
    calm_score = analyzer.score(calm_pcm)
    print(f"  - Calm Audio: Score = {calm_score:.3f}")
    assert calm_score < 0.40, f"Expected calm score < 0.40, got {calm_score}"

    # 2. Shouting / High-energy audio (RMS > 6000)
    shout_pcm = generate_shouting_pcm(duration_sec=1.2)
    shout_score = analyzer.score(shout_pcm)
    print(f"  - Shouting Audio: Score = {shout_score:.3f}")
    assert shout_score > 0.60, f"Expected shout score > 0.60, got {shout_score}"
    assert shout_score > calm_score, "Shouting score must exceed calm score"

    # 3. Empty or near-silent audio
    silent_pcm = bytes(1600)  # 100ms of zeros
    silence_score = analyzer.score(silent_pcm)
    print(f"  - Silent Audio: Score = {silence_score:.3f}")
    assert silence_score < 0.25, f"Expected silence score < 0.25, got {silence_score}"

    print("  [PASS] Acoustic sentiment analysis passed.")


# --------------------------------------------------------------------------
# Test 2: Lexical Dispute & Escalation Classification
# --------------------------------------------------------------------------
def test_lexical_classification():
    print_header("2. Testing Lexical Dispute & Escalation Classification")

    classifier = LexicalDisputeClassifier()

    cases = [
        (
            "Maine kal hi payment kar diya tha, receipt check karo",
            DisputeType.PAYMENT_DISPUTE,
            0.60,
        ),
        (
            "Main tumhare khilaf police FIR aur court case karunga",
            DisputeType.LEGAL_THREAT,
            0.85,
        ),
        (
            "Stop calling me repeatedly, you are harassing me constantly",
            DisputeType.HARASSMENT_COMPLAINT,
            0.80,
        ),
        (
            "Mujhe tumhari AI se baat nahi karni, kisi human supervisor se connect karo",
            DisputeType.HUMAN_REQUEST,
            0.65,
        ),
        (
            "Yeh galat number hai main Rahul Sharma nahi hoon",
            DisputeType.WRONG_PERSON,
            0.45,
        ),
        (
            "Haan main kal dopahar tak payment deposit kar dunga",
            DisputeType.NONE,
            0.00,
        ),
    ]

    for text, expected_dispute, min_score in cases:
        score, dispute_type, cues = classifier.classify(text)
        print(f"  - Text: '{text[:50]}...'")
        print(f"    Dispute: {dispute_type.value}, Score: {score:.2f}, Cues: {cues}")
        assert dispute_type == expected_dispute, f"Expected {expected_dispute}, got {dispute_type}"
        if expected_dispute != DisputeType.NONE:
            assert score >= min_score, f"Expected score >= {min_score}, got {score}"

    print("  [PASS] Lexical dispute classification passed.")


# --------------------------------------------------------------------------
# Test 3: Unified Sentiment Engine & Composite Categorization
# --------------------------------------------------------------------------
def test_unified_sentiment_engine():
    print_header("3. Testing Unified Sentiment Engine & Categorization")

    engine = UnifiedSentimentEngine()

    # Case A: Calm agreement
    res_calm = engine.analyze("Haan theek hai, main online payment kar raha hoon.")
    print(f"  - Calm Utterance: Index = {res_calm.composite_agitation:.2f}, Category = {res_calm.category.value}")
    assert res_calm.category == SentimentCategory.CALM
    assert not res_calm.transfer_recommended
    assert not res_calm.deescalation_recommended

    # Case B: Elevated customer (annoyance without direct threat)
    res_elev = engine.analyze("Aap bar bar kyu phone kar rahe ho, thoda time dijiye na please.")
    print(f"  - Elevated Utterance: Index = {res_elev.composite_agitation:.2f}, Category = {res_elev.category.value}")
    assert res_elev.category in (SentimentCategory.ELEVATED, SentimentCategory.AGITATED)
    assert res_elev.deescalation_recommended

    # Case C: Agitated with Payment Dispute
    res_dispute = engine.analyze("Main paise de chuka hoon, bank statement dekh lo, faltu call mat karo.")
    print(f"  - Dispute Utterance: Index = {res_dispute.composite_agitation:.2f}, Category = {res_dispute.category.value}, Dispute = {res_dispute.dispute_type.value}")
    assert res_dispute.dispute_type == DisputeType.PAYMENT_DISPUTE
    assert res_dispute.transfer_recommended

    # Case D: Critical with Legal Threat & High Shouting Audio
    shout_pcm = generate_shouting_pcm(duration_sec=1.0)
    res_crit = engine.analyze("Main court me le jaunga, police complaint karunga!", pcm_bytes=shout_pcm)
    print(f"  - Critical Utterance: Index = {res_crit.composite_agitation:.2f}, Category = {res_crit.category.value}, Dispute = {res_crit.dispute_type.value}")
    assert res_crit.category in (SentimentCategory.AGITATED, SentimentCategory.CRITICAL)
    assert res_crit.dispute_type == DisputeType.LEGAL_THREAT
    assert res_crit.transfer_recommended

    print("  [PASS] Unified sentiment engine categorization passed.")


# --------------------------------------------------------------------------
# Test 4: Conversational De-escalation Posture in VoiceAgent
# --------------------------------------------------------------------------
def test_agent_deescalation_posture():
    print_header("4. Testing Conversational De-escalation Posture in VoiceAgent")

    agent = VoiceAgent(language="hi", voice_enabled=False)

    # Mild annoyance triggers de-escalation posture
    user_input = "Aap log roz subah se pareshan karte ho, thoda time dijiye!"
    res = agent.step(user_input)

    print(f"  - User: '{user_input}'")
    print(f"  - Agent Reply: '{res['text']}'")
    print(f"  - Sentiment: {res['sentiment']['category']}")

    # Agent should prepend empathetic phrasing
    assert "चिंता समझ" in res["text"] or "स्थिति समझ" in res["text"] or "क्षमा" in res["text"], (
        f"Expected empathetic de-escalation prefix in response, got: {res['text']}"
    )
    assert not res["terminated"], "Mildly annoyed customer should not terminate immediately"

    print("  [PASS] Conversational de-escalation posture verified.")


# --------------------------------------------------------------------------
# Test 5: Automatic Transfer upon Critical Agitation or Legal Threat
# --------------------------------------------------------------------------
def test_agent_automatic_transfer():
    print_header("5. Testing Automatic Transfer upon Legal Threats & Human Requests")

    # 1. Legal Threat Transfer
    agent_legal = VoiceAgent(language="hi", voice_enabled=False, caller_phone="9876543210")
    threat_input = "Main abhi advocate ko call kar raha hoon aur consumer court me tumhare khilaf case karunga!"
    res_legal = agent_legal.step(threat_input)

    print(f"  - User: '{threat_input}'")
    print(f"  - Reply: '{res_legal['text']}'")
    print(f"  - Terminated: {res_legal['terminated']}")
    print(f"  - Tool Event: {res_legal['tool_event']}")
    print(f"  - Tool Data: {res_legal['tool_data']}")

    assert res_legal["terminated"], "Legal threat should terminate bot session for warm transfer"
    assert res_legal["tool_data"] is not None
    assert res_legal["tool_data"]["action"] == "transfer"
    assert "ट्रांसफर" in res_legal["text"] or "वरिष्ठ अधिकारी" in res_legal["text"]

    # 2. Direct Human Supervisor Request
    agent_human = VoiceAgent(language="en", voice_enabled=False, caller_phone="9876543210")
    human_req = "I do not want to talk to an automated machine. Connect me to a human manager right now."
    res_human = agent_human.step(human_req)

    print(f"  - User: '{human_req}'")
    print(f"  - Reply: '{res_human['text']}'")
    print(f"  - Terminated: {res_human['terminated']}")
    print(f"  - Tool Data: {res_human['tool_data']}")

    assert res_human["terminated"], "Direct human request should terminate bot session for transfer"
    assert res_human["tool_data"]["action"] == "transfer"

    print("  [PASS] Automatic transfer triggering verified.")


# --------------------------------------------------------------------------
# Test 6: SIP Transfer Dispatcher & TransferContext
# --------------------------------------------------------------------------
def test_sip_transfer_dispatcher():
    print_header("6. Testing SIPTransferDispatcher (RFC 3515 REFER, TwiML, WebSocket)")

    ctx = TransferContext(
        call_id="CALL_SIP_9912",
        caller_phone="+919876543210",
        loan_id="MUTH-8921",
        amount_due=5420.0,
        agitation_score=0.885,
        dispute_type="LEGAL_THREAT",
        briefing_summary="Customer threatened police complaint regarding EMI notice.",
        target_department="legal_disputes",
    )

    # 1. Base64 context header encoding & lossless roundtrip
    header_val = ctx.to_header_string()
    assert isinstance(header_val, str) and len(header_val) > 20
    restored_ctx = TransferContext.from_header_string(header_val)

    assert restored_ctx.call_id == ctx.call_id
    assert restored_ctx.caller_phone == ctx.caller_phone
    assert restored_ctx.loan_id == ctx.loan_id
    assert restored_ctx.amount_due == ctx.amount_due
    assert abs(restored_ctx.agitation_score - ctx.agitation_score) < 0.001
    assert restored_ctx.dispute_type == ctx.dispute_type
    assert restored_ctx.briefing_summary == ctx.briefing_summary
    assert restored_ctx.target_department == ctx.target_department
    print("  - Context Header Serialization & Deserialization: Lossless Roundtrip Verified")

    # 2. RFC 3515 SIP REFER Header generation
    refer_headers = SIPTransferDispatcher.build_sip_refer(
        target_uri="sip:legal_escalations@bank.internal",
        context=ctx,
    )
    print(f"  - SIP REFER Method: {refer_headers['Method']}")
    print(f"  - Refer-To: {refer_headers['Refer-To']}")
    print(f"  - X-Verbalyze-Context Header Length: {len(refer_headers['X-Verbalyze-Context'])} bytes")

    assert refer_headers["Method"] == "REFER"
    assert "sip:legal_escalations@bank.internal" in refer_headers["Refer-To"]
    assert refer_headers["X-Verbalyze-Context"] == header_val

    # 3. Twilio / Exotel TwiML Dial generation
    twiml_sip = SIPTransferDispatcher.build_twiml_dial_transfer(
        target="sip:supervisor@bank.internal",
        context=ctx,
        caller_id="+918000012345",
        language="hi",
    )
    assert "<Dial" in twiml_sip and "<Sip>" in twiml_sip
    assert "X-Verbalyze-Context=" in twiml_sip
    print("  - TwiML SIP Dial Transfer Payload: Verified")

    twiml_pstn = SIPTransferDispatcher.build_twiml_dial_transfer(
        target="+919811122233",
        context=ctx,
        language="hi",
    )
    assert "<Number>+919811122233</Number>" in twiml_pstn
    print("  - TwiML PSTN Number Dial Transfer Payload: Verified")

    # 4. WebSocket Transfer Control Event
    ws_event = SIPTransferDispatcher.build_websocket_transfer_event(
        target_uri="sip:supervisor_queue@bank.internal",
        context=ctx,
    )
    assert ws_event["event"] == "transfer"
    assert ws_event["action"] == "sip_refer"
    assert ws_event["context"]["caller_phone"] == "+919876543210"
    print("  - WebSocket Media Stream Transfer Event: Verified")

    print("  [PASS] SIPTransferDispatcher and TransferContext passed.")


# --------------------------------------------------------------------------
# Test 7: Campaign CDR Disposition Tracking for Transferred Calls
# --------------------------------------------------------------------------
def test_campaign_cdr_transfer_dispositions():
    print_header("7. Testing Campaign CDR Disposition Tracking (Warm Transfer)")

    config = CampaignConfig(
        campaign_id="CAMP_TRANSFER_TEST",
        campaign_name="Transfer Verification Campaign",
        max_concurrent_channels=2,
        enforce_trai_calling_hours=False,
        enforce_dnd_check=False,
    )
    dialer = CampaignDialer(config=config)

    # Ingest 2 leads
    leads = [
        {"customer_name": "Suresh Gupta", "phone_number": "9811122334", "loan_id": "MUTH-101", "amount_due": 12000.0},
        {"customer_name": "Ravi Kumar", "phone_number": "9822233445", "loan_id": "MUTH-102", "amount_due": 8500.0},
    ]
    accepted, rejected, errors = dialer.ingest_leads_from_list(leads)
    assert accepted == 2

    # Mock conversation with legal dispute for lead 1
    async def run_dialer_test():
        lead = dialer.leads[0]
        # Simulate call turn where customer threatens legal action
        agent = VoiceAgent(language="hi", voice_enabled=False, caller_phone=lead.phone_number)
        threat = "Main police case karunga, court me milenge!"
        step_res = agent.step(threat)

        # Update lead & record CDR as the dialer does
        tool_data = step_res.get("tool_data") or {}
        reason = tool_data.get("reason", "")
        if "legal" in reason.lower():
            disp = CallDisposition.LEGAL_DISPUTE_ESCALATED
        else:
            disp = CallDisposition.TRANSFERRED_TO_SUPERVISOR

        now = datetime.now(timezone.utc)
        cdr = CallDetailRecord(
            call_id="CALL_TEST_LEAD_01",
            campaign_id=config.campaign_id,
            lead_id=lead.lead_id,
            phone_number=lead.phone_number,
            loan_id=lead.loan_id,
            start_time=now,
            end_time=now,
            duration_seconds=14.5,
            amd_result=None,
            final_disposition=disp,
            agitation_score=step_res.get("sentiment", {}).get("composite_agitation", 0.9),
            dispute_type=str(tool_data.get("reason", "LEGAL_THREAT")),
            payment_link_sent=False,
            amount_recovered_or_promised=0.0,
            turns_count=1,
            transcript_turns=[{"speaker": "customer", "text": threat}],
            tool_events=[tool_data],
        )
        dialer.cdrs.append(cdr)
        dialer.summary.disposition_breakdown[disp.value] = 1
        dialer.summary.transferred_to_supervisor_count += 1

        print(f"  - Generated CDR: ID = {cdr.call_id}, Disposition = {cdr.final_disposition.value}")
        print(f"    Agitation Score: {cdr.agitation_score}, Dispute: {cdr.dispute_type}")
        print(f"    Masked Phone in Export: {cdr.to_dict(mask_pii=True)['phone_number']}")

        assert cdr.final_disposition == CallDisposition.LEGAL_DISPUTE_ESCALATED
        assert cdr.agitation_score > 0.70
        assert "legal" in cdr.dispute_type.lower()
        assert dialer.summary.transferred_to_supervisor_count == 1
        assert "*" in cdr.to_dict(mask_pii=True)["phone_number"]

    asyncio.run(run_dialer_test())
    print("  [PASS] Campaign CDR transfer disposition tracking passed.")


# --------------------------------------------------------------------------
# Test 8: FastAPI Telephony REST & SIP Turn Endpoints
# --------------------------------------------------------------------------
def test_fastapi_endpoints():
    print_header("8. Testing FastAPI Telephony Endpoints (/telephony/sentiment & /telephony/transfer)")

    from fastapi.testclient import TestClient
    from verbalyze.telephony.server import create_app

    app = create_app(auth_token="test_secret_token_123")
    client = TestClient(app)

    headers = {"Authorization": "Bearer test_secret_token_123"}

    # 1. Health Check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["engine"] == "Verbalyze Telephony v0.2.0" or "v0.3.0" in health_resp.json()["engine"]

    # 2. Unauthorized Access Blocked
    unauth_resp = client.post("/telephony/sentiment", json={"text": "Hello"})
    assert unauth_resp.status_code == 401
    print("  - Authentication Guard on /telephony/sentiment: Verified 401 Unauthorized")

    # 3. POST /telephony/sentiment
    sent_payload = {
        "text": "Main consumer court me tumhare khilaf case karunga, meri payment ho chuki hai",
        "audio_base64": base64.b64encode(generate_synthetic_pcm(0.5, amplitude=4000)).decode("ascii"),
    }
    sent_resp = client.post("/telephony/sentiment", json=sent_payload, headers=headers)
    assert sent_resp.status_code == 200
    sent_data = sent_resp.json()
    print(f"  - Sentiment Analysis Endpoint Result:")
    print(f"    Composite Agitation: {sent_data['composite_agitation']}")
    print(f"    Dispute Type: {sent_data['dispute_type']}")
    print(f"    Transfer Recommended: {sent_data['transfer_recommended']}")
    assert sent_data["dispute_type"] in ("LEGAL_THREAT", "PAYMENT_DISPUTE")
    assert sent_data["transfer_recommended"] is True

    # 4. POST /telephony/transfer (SIP REFER format)
    transfer_payload = {
        "call_id": "CALL_TEST_4401",
        "caller_phone": "+919876543210",
        "loan_id": "MUTH-8921",
        "amount_due": 5420.0,
        "agitation_score": 0.89,
        "dispute_type": "LEGAL_THREAT",
        "briefing_summary": "Customer threatened legal action regarding overdue EMI notice.",
        "target_department": "legal",
        "format": "sip_refer",
    }
    trans_resp = client.post("/telephony/transfer", json=transfer_payload, headers=headers)
    assert trans_resp.status_code == 200
    trans_data = trans_resp.json()
    assert trans_data["status"] == "transfer_initiated"
    assert trans_data["format"] == "sip_refer"
    assert "X-Verbalyze-Context" in trans_data["sip_refer_headers"]
    print("  - POST /telephony/transfer (sip_refer format): Verified")

    # 5. POST /telephony/transfer (TwiML format)
    transfer_payload["format"] = "twiml"
    twiml_resp = client.post("/telephony/transfer", json=transfer_payload, headers=headers)
    assert twiml_resp.status_code == 200
    assert "<Dial" in twiml_resp.text
    print("  - POST /telephony/transfer (twiml format): Verified")

    # 6. POST /webhook/sip/turn with customer transfer escalation
    turn_payload = {
        "call_id": "CALL_SIP_INTERACTION_1",
        "transcript": "Mujhe manager se baat karni hai, AI se nahi!",
    }
    sip_turn_resp = client.post("/webhook/sip/turn", json=turn_payload, headers=headers)
    assert sip_turn_resp.status_code == 200
    sip_turn_data = sip_turn_resp.json()
    print(f"  - SIP Turn Transfer Action: {sip_turn_data.get('action')}")
    assert sip_turn_data.get("action") == "transfer"
    assert "sip_refer" in sip_turn_data
    assert sip_turn_data["hangup"] is True
    print("  - POST /webhook/sip/turn Warm Transfer Handoff: Verified")

    print("  [PASS] FastAPI telephony endpoints passed.")


# --------------------------------------------------------------------------
# Main Test Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print_header("VERBALYZE SENTIMENT DETECTION & SIP REFER WARM TRANSFER SUITE")

    test_acoustic_scoring()
    test_lexical_classification()
    test_unified_sentiment_engine()
    test_agent_deescalation_posture()
    test_agent_automatic_transfer()
    test_sip_transfer_dispatcher()
    test_campaign_cdr_transfer_dispositions()
    test_fastapi_endpoints()

    print_header("ALL 8 SENTIMENT & WARM TRANSFER VERIFICATION TESTS PASSED")
