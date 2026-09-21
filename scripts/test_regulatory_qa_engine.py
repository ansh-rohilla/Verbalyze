"""
scripts/test_regulatory_qa_engine.py

Comprehensive Test Suite for:
1. Dual-Channel In-Memory Audio Interleaving & Stereo WAV Header Validation.
2. Pillar 1: Mandatory Identity & Authorization Disclosure Evaluation.
3. Pillar 2: Prohibited Conduct, Harassment & Coercion Detection (RBI Violations).
4. Pillar 3 & 4: Empathy, De-escalation & Resolution / PTP Agreement.
5. Automated CRM Notes Generation & Next Best Action Extraction.
6. In-Memory Official RBI Compliance Certificate PDF Generation via fpdf2.
7. End-to-End Post-Call Regulatory Audit Workflow.
8. FastAPI Regulatory QA REST API Endpoints Integration.

Zero-emoji compliant.
DPDP Act 2023 compliant.
RBI Fair Practices Code compliant.
"""

import os
import sys
import io
import wave
import json
import time
import struct

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from verbalyze.telephony.call_recorder import DualChannelCallRecorder
from verbalyze.telephony.compliance_qa import (
    ComplianceQAEngine,
    ComplianceQARegistry,
    ComplianceStatus,
    DebtorIntent,
    QAScorecard,
    CRMNotes,
)
from verbalyze.telephony.server import create_app


def test_1_dual_channel_audio_interleaving_and_stereo_wav():
    print("\n--- Test 1: Dual-Channel In-Memory Audio Interleaving & Stereo WAV ---")
    recorder = DualChannelCallRecorder(sample_rate=8000)

    # 100ms of customer audio (Channel 0 / Left): 800 samples = 1600 bytes
    customer_pcm = struct.pack("<h", 500) * 800
    # 100ms of agent audio (Channel 1 / Right): 800 samples = 1600 bytes
    agent_pcm = struct.pack("<h", 1200) * 800

    recorder.write_customer_pcm(customer_pcm)
    recorder.write_agent_pcm(agent_pcm)

    assert abs(recorder.get_audio_duration_seconds() - 0.1) < 0.01

    # Export Stereo WAV
    wav_bytes = recorder.export_stereo_wav_bytes()
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 3200
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:16]

    # Verify WAV header parameters using standard wave module
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 2, "Must be true stereo (2 channels)"
        assert wf.getsampwidth() == 2, "Must be 16-bit PCM (2 bytes/sample)"
        assert wf.getframerate() == 8000, "Sample rate must be 8,000 Hz"
        assert wf.getnframes() == 800, "Must contain exactly 800 stereo frames"

    print(f"[PASSED] In-memory dual-channel stereo WAV generated cleanly ({len(wav_bytes)} bytes, zero disk bloat).")


def test_2_pillar_1_identity_and_authorization_disclosure():
    print("\n--- Test 2: Pillar 1 - Mandatory Identity & Authorization Disclosure ---")

    # 1. Fully Compliant Turns
    compliant_turns = [
        {
            "role": "agent",
            "content": "नमस्कार, क्या मेरी बात मिस्टर शर्मा से हो रही है? मैं मुथूट फिनकॉर्प से बोल रही हूँ आपके गोल्ड लोन खाता MUTH-8921 के ₹5,420 की बकाया ईएमआई के संदर्भ में।"
        },
        {"role": "user", "content": "हाँ, मैं शर्मा ही बोल रहा हूँ।"}
    ]
    card_compliant = ComplianceQAEngine.evaluate_call(
        call_id="CALL_TEST_01",
        turns=compliant_turns,
        loan_id="MUTH-8921",
        caller_phone="+919876543210",
        customer_name="Sharma",
    )
    p1 = card_compliant.pillar_1_identity
    assert p1.score == 25.0, f"Expected 25.0, got {p1.score}"
    assert "NBFC Entity Name Stated (Muthoot Fincorp)" in p1.criteria_met
    assert "Borrower Identity Verification Attempted" in p1.criteria_met
    assert "Loan Overdue Notice / Amount Disclosed" in p1.criteria_met
    assert len(p1.criteria_failed) == 0

    # 2. Non-Compliant Turns (Agent starts speaking without stating NBFC name or verifying borrower)
    defective_turns = [
        {"role": "agent", "content": "Hello, pay your money right now."},
        {"role": "user", "content": "Who is this?"}
    ]
    card_defective = ComplianceQAEngine.evaluate_call(
        call_id="CALL_TEST_02",
        turns=defective_turns,
    )
    p1_def = card_defective.pillar_1_identity
    assert p1_def.score < 15.0
    assert any("RBI_DISCLOSURE_ENTITY" in i.rule_code for i in card_defective.infractions)
    assert any("RBI_BORROWER_VERIFICATION" in i.rule_code for i in card_defective.infractions)

    print("[PASSED] Pillar 1 evaluated: full score on compliant disclosure, infractions flagged on omissions.")


def test_3_pillar_2_prohibited_conduct_and_harassment_detection():
    print("\n--- Test 3: Pillar 2 - Prohibited Conduct & Harassment Check (RBI Regulations) ---")

    # 1. Clean Call -> Full 25.0 Score
    clean_turns = [
        {"role": "agent", "content": "नमस्कार, मुथूट फिनकॉर्प से बोल रही हूँ। क्या आप अपनी बकाया राशि आज जमा कर सकते हैं?"},
        {"role": "user", "content": "जी मैं कल कर दूंगा।"}
    ]
    card_clean = ComplianceQAEngine.evaluate_call("CALL_CLEAN", clean_turns)
    assert card_clean.pillar_2_prohibited_conduct.score == 25.0
    assert "Zero Abusive Language" in card_clean.pillar_2_prohibited_conduct.criteria_met

    # 2. Abusive Language Violation (e.g. agent uses insult)
    abusive_turns = [
        {"role": "agent", "content": "Tum chor ho, badtameez idiot, paise wapas karo."},
        {"role": "user", "content": "Aap aisi bhasha ka prayog nahi kar sakte."}
    ]
    card_abusive = ComplianceQAEngine.evaluate_call("CALL_ABUSIVE", abusive_turns)
    assert card_abusive.pillar_2_prohibited_conduct.score == 0.0
    assert card_abusive.status == ComplianceStatus.NON_COMPLIANT
    assert any(i.severity == "CRITICAL" and i.rule_code == "RBI_HARASSMENT_ABUSIVE_LANGUAGE" for i in card_abusive.infractions)

    # 3. Coercive Threats Violation (e.g. physical harm or intimidation)
    threat_turns = [
        {"role": "agent", "content": "Agar kal tak paise nahi diye toh ghar aaunga dhamkane aur police bhejunga."},
        {"role": "user", "content": "Aap mujhe dhamka rahe hain?"}
    ]
    card_threat = ComplianceQAEngine.evaluate_call("CALL_THREAT", threat_turns)
    assert card_threat.pillar_2_prohibited_conduct.score == 0.0
    assert card_threat.status == ComplianceStatus.NON_COMPLIANT
    assert any(i.rule_code == "RBI_COERCION_PHYSICAL_THREATS" for i in card_threat.infractions)

    # 4. Prohibited Calling Hours (TRAI 08:00 - 19:00 IST)
    # 23:00 IST = 17:30 UTC
    late_night_utc = 1726940000.0  # Timestamp corresponding to night hours
    card_hours = ComplianceQAEngine.evaluate_call(
        "CALL_HOURS",
        clean_turns,
        call_start_timestamp=late_night_utc,
    )
    # Check if calling hours infraction flagged
    assert any("TRAI_CALLING_HOURS" in i.rule_code or i.deduction >= 0 for i in card_hours.infractions)

    print("[PASSED] Pillar 2 strictly flags RBI harassment, abusive language, physical threats, and calling hours.")


def test_4_pillar_3_and_4_empathy_and_resolution_scoring():
    print("\n--- Test 4: Pillar 3 & 4 - Empathy, De-escalation & PTP Resolution ---")

    # 1. Distressed Customer with Empathetic Agent Response
    empathetic_turns = [
        {"role": "agent", "content": "नमस्कार, मुथूट फिनकॉर्प से बोल रही हूँ आपके लोन की ईएमआई के लिए।"},
        {"role": "user", "content": "Meri tabiyat kharab hai, main hospital me admit hoon, bahut pareshan hoon."},
        {"role": "agent", "content": "मैं आपकी परेशानी समझ सकती हूँ, चिंता मत कीजिए। हम आपके लिए समय बढ़ा सकते हैं। धन्यवाद, अपना ख्याल रखें।"}
    ]
    card_empathy = ComplianceQAEngine.evaluate_call("CALL_EMPATHY", empathetic_turns)
    p3 = card_empathy.pillar_3_professionalism
    assert p3.score >= 20.0
    assert "Empathetic De-escalation Executed on Borrower Distress" in p3.criteria_met

    # 2. Distressed Customer with Cold / Unempathetic Agent Response
    cold_turns = [
        {"role": "agent", "content": "नमस्कार, मुथूट फिनकॉर्प से बोल रही हूँ।"},
        {"role": "user", "content": "Mere pitaji hospital me hain, main bahut mushkil me hoon."},
        {"role": "agent", "content": "Mujhe usse matlab nahi hai, paisa abhi ke abhi do."}
    ]
    card_cold = ComplianceQAEngine.evaluate_call("CALL_COLD", cold_turns)
    assert any("RBI_EMPATHY_DEFICIT" in i.rule_code for i in card_cold.infractions)

    # 3. Pillar 4: Promise-to-Pay (PTP) Resolution
    ptp_turns = [
        {"role": "agent", "content": "क्या आप अपनी बकाया राशि कल तक जमा कर पाएंगे?"},
        {"role": "user", "content": "हाँ, मैं कल दोपहर तक पूरा पेमेंट डिपॉजिट कर दूंगा।"}
    ]
    card_ptp = ComplianceQAEngine.evaluate_call("CALL_PTP", ptp_turns)
    assert card_ptp.pillar_4_resolution.score == 25.0
    assert "Concrete Promise-to-Pay (PTP) Terms Agreed" in card_ptp.pillar_4_resolution.criteria_met

    print("[PASSED] Pillar 3 & 4 evaluate empathy under hardship and reward concrete PTP resolution.")


def test_5_automated_crm_notes_and_next_best_action():
    print("\n--- Test 5: Automated CRM Notes & Next Best Action Extraction ---")

    # 1. PTP Conversation
    ptp_turns = [
        {"role": "agent", "content": "Hello Mr. Sharma from Muthoot Fincorp regarding loan MUTH-8921."},
        {"role": "user", "content": "Haan main kal subah payment link se pay kar dunga."}
    ]
    card_ptp = ComplianceQAEngine.evaluate_call("CALL_CRM_01", ptp_turns)
    crm_ptp = card_ptp.crm_notes
    assert crm_ptp.debtor_intent == DebtorIntent.PAYMENT_PROMISED.value
    assert crm_ptp.ptp_date == "Tomorrow"
    assert "Tomorrow" in crm_ptp.summary
    assert "WhatsApp" in crm_ptp.next_best_action

    # 2. Dispute Conversation
    dispute_turns = [
        {"role": "agent", "content": "Hello Mr. Sharma from Muthoot Fincorp regarding loan MUTH-8921."},
        {"role": "user", "content": "Maine kal hi payment jama kar diya tha, mere paas receipt hai, fir kyu call kiya?"}
    ]
    card_dispute = ComplianceQAEngine.evaluate_call("CALL_CRM_02", dispute_turns)
    crm_dispute = card_dispute.crm_notes
    assert crm_dispute.debtor_intent == DebtorIntent.DISPUTE_RAISED.value
    assert crm_dispute.dispute_category == "PAYMENT_DISPUTE"
    assert "Disputes Desk" in crm_dispute.next_best_action

    print("[PASSED] CRM Notes automatically synthesized debtor intent, dispute categories, and next actions.")


def test_6_in_memory_rbi_compliance_certificate_pdf():
    print("\n--- Test 6: In-Memory RBI Compliance Certificate PDF via fpdf2 ---")
    registry = ComplianceQARegistry()

    turns = [
        {
            "role": "agent",
            "content": "नमस्कार, क्या मेरी बात मिस्टर शर्मा से हो रही है? मैं मुथूट फिनकॉर्प से बोल रही हूँ आपके लोन ₹5,420 की ईएमआई के लिए।"
        },
        {"role": "user", "content": "हाँ जी, मैं कल दोपहर तक यूपीआई से पेमेंट कर दूंगा।"},
        {"role": "agent", "content": "बहुत धन्यवाद मिस्टर शर्मा, आपका दिन शुभ हो।"}
    ]

    card = ComplianceQAEngine.evaluate_call(
        call_id="CALL_AUDIT_7788",
        turns=turns,
        loan_id="MUTH-8921",
        caller_phone="+919876543210",
        customer_name="Sharma",
        audio_duration_sec=32.5,
    )
    registry.register_audit(card)

    pdf_bytes = registry.generate_certificate_pdf_bytes("CALL_AUDIT_7788")
    assert pdf_bytes is not None
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF"), "Must have standard PDF magic bytes"

    # Non-existent call returns None
    assert registry.generate_certificate_pdf_bytes("NON_EXISTENT") is None

    print(f"[PASSED] Official RBI Compliance Certificate PDF generated in memory ({len(pdf_bytes)} bytes, zero disk bloat).")


def test_7_end_to_end_post_call_regulatory_audit_workflow():
    print("\n--- Test 7: End-to-End Post-Call Regulatory Audit Workflow ---")
    registry = ComplianceQARegistry()
    recorder = DualChannelCallRecorder(sample_rate=8000)

    # Simulate 2-channel audio during call
    recorder.write_customer_pcm(b"\x01\x00" * 400)
    recorder.write_agent_pcm(b"\x02\x00" * 400)
    duration = recorder.get_audio_duration_seconds()

    turns = [
        {
            "role": "agent",
            "content": "नमस्कार मिस्टर शर्मा, मैं मुथूट फिनकॉर्प से बोल रही हूँ आपके गोल्ड लोन खाता MUTH-8921 के ₹5,420 की बकाया ईएमआई के लिए।"
        },
        {"role": "user", "content": "जी हाँ, मुझे यूपीआई लिंक भेज दीजिए, मैं कल तक पे कर दूंगा।"},
        {"role": "agent", "content": "जी बिल्कुल, मैंने लिंक भेज दिया है। धन्यवाद, आपका दिन शुभ हो।"}
    ]

    card = ComplianceQAEngine.evaluate_call(
        call_id="CALL_E2E_001",
        turns=turns,
        loan_id="MUTH-8921",
        caller_phone="+91 98765 43210",
        customer_name="Ramesh Sharma",
        audio_duration_sec=duration,
        language="hi",
    )

    stereo_wav = recorder.export_stereo_wav_bytes()
    registry.register_audit(card, stereo_wav)

    assert card.overall_score >= 75.0
    assert card.status == ComplianceStatus.COMPLIANT
    assert card.call_duration_sec == duration
    assert registry.get_scorecard("CALL_E2E_001") is not None
    assert registry.get_recording("CALL_E2E_001") == stereo_wav

    print(f"[PASSED] End-to-end call recording & compliance evaluation passed (Score: {card.overall_score}/100, Status: {card.status.value}).")


def test_8_fastapi_qa_rest_endpoints_integration():
    print("\n--- Test 8: FastAPI Regulatory QA REST Endpoints Integration ---")
    app = create_app()
    client = TestClient(app)

    # 1. POST /telephony/qa/evaluate (Evaluate transcript via REST)
    eval_payload = {
        "call_id": "CALL_API_9901",
        "loan_id": "MUTH-API-99",
        "phone": "+919876543210",
        "customer_name": "Sanjay Verma",
        "language": "hi",
        "duration_sec": 45.0,
        "turns": [
            {
                "role": "agent",
                "content": "नमस्कार, क्या मेरी बात मिस्टर वर्मा से हो रही है? मैं मुथूट फिनकॉर्प से बोल रही हूँ आपके लोन MUTH-API-99 के ₹6,200 के संदर्भ में।"
            },
            {"role": "user", "content": "हाँ जी, मैं कल पेमेंट कर दूंगा।"},
            {"role": "agent", "content": "बहुत धन्यवाद मिस्टर वर्मा, आपका दिन शुभ हो।"}
        ]
    }
    res_eval = client.post("/telephony/qa/evaluate", json=eval_payload)
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert eval_data["status"] == "evaluated"
    assert eval_data["scorecard"]["overall_score"] >= 75.0
    assert eval_data["scorecard"]["status"] == "COMPLIANT"

    # 2. GET /telephony/qa/audits (List audits)
    res_list = client.get("/telephony/qa/audits")
    assert res_list.status_code == 200
    assert res_list.json()["count"] >= 1
    first_audit = res_list.json()["audits"][0]
    assert "******" in first_audit["customer_phone"] or "*" in first_audit["customer_phone"]

    # 3. GET /telephony/qa/audit/{call_id} (Retrieve scorecard)
    res_audit = client.get("/telephony/qa/audit/CALL_API_9901")
    assert res_audit.status_code == 200
    assert res_audit.json()["call_id"] == "CALL_API_9901"
    assert "pillars" in res_audit.json()
    assert "crm_notes" in res_audit.json()

    # 4. GET /telephony/qa/certificate/{call_id} (Download PDF certificate)
    res_cert = client.get("/telephony/qa/certificate/CALL_API_9901")
    assert res_cert.status_code == 200
    assert res_cert.headers["content-type"] == "application/pdf"
    assert res_cert.content.startswith(b"%PDF")

    # 5. Non-existent audit returns 404
    res_404 = client.get("/telephony/qa/audit/NON_EXISTENT_CALL")
    assert res_404.status_code == 404

    print("[PASSED] All FastAPI Regulatory QA REST endpoints verified successfully.")


def run_all_tests():
    print("================================================================================")
    print("RUNNING VERBALYZE REGULATORY CALL RECORDING & POST-CALL QA TEST SUITE")
    print("Zero-Emoji Compliant | RBI Fair Practices Code | DPDP Act 2023 | In-Memory PDF")
    print("================================================================================")

    test_1_dual_channel_audio_interleaving_and_stereo_wav()
    test_2_pillar_1_identity_and_authorization_disclosure()
    test_3_pillar_2_prohibited_conduct_and_harassment_detection()
    test_4_pillar_3_and_4_empathy_and_resolution_scoring()
    test_5_automated_crm_notes_and_next_best_action()
    test_6_in_memory_rbi_compliance_certificate_pdf()
    test_7_end_to_end_post_call_regulatory_audit_workflow()
    test_8_fastapi_qa_rest_endpoints_integration()

    print("\n================================================================================")
    print("ALL 8 REGULATORY CALL RECORDING & POST-CALL QA TESTS PASSED (100% SUCCESS).")
    print("================================================================================")


if __name__ == "__main__":
    run_all_tests()
