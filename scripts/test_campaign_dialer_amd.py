#!/usr/bin/env python3
"""
scripts/test_campaign_dialer_amd.py

Comprehensive Verification Suite for Verbalyze Outbound Campaign Batch Dialer
and Answering Machine Detection (AMD) Engine:
1. TRAI Calling Window Enforcement (09:00 - 19:00 IST)
2. TRAI DND Registry & Frequency Capping (Max 3 calls/day)
3. Multi-Modal Answering Machine Detection (Human vs Voicemail vs Operator Announcements vs Beep)
4. Lead Ingestion & DPDP Boundary Sanitization (CSV & JSON)
5. Concurrent Multi-Channel Worker Queue Execution (Semaphore concurrency control)
6. Automated Retry Logic with Exponential Backoff
7. DPDP-Sanitized Call Detail Record (CDR) Export (JSON & CSV)
8. FastAPI Campaign & AMD REST Endpoints
"""

import os
import sys
import json
import math
import asyncio
from datetime import datetime, timezone, timedelta, time as dtime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.campaign.models import (
    Lead,
    LeadStatus,
    CallDisposition,
    AMDDecision,
    AMDResult,
    CampaignConfig,
    CallDetailRecord,
    CampaignSummary,
)
from verbalyze.campaign.trai_compliance import (
    TRAIComplianceEngine,
    get_current_ist_time,
    IST_TIMEZONE,
)
from verbalyze.campaign.amd import AMDClassifier
from verbalyze.campaign.dialer import CampaignDialer


def print_header(title: str):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def test_traI_calling_window():
    print_header("1. Testing TRAI Calling Window Enforcement (09:00 - 19:00 IST)")

    engine = TRAIComplianceEngine()

    # 1. Allowed time: 14:30 IST (2:30 PM)
    allowed_dt = datetime(2026, 9, 17, 14, 30, 0, tzinfo=IST_TIMEZONE)
    is_allowed = engine.is_within_calling_window(allowed_dt)
    print(f"  - 14:30 IST Calling Allowed: {is_allowed} (Expected: True)")
    assert is_allowed is True, "Expected 14:30 IST to be within calling window"

    # 2. Blocked evening time: 20:15 IST (8:15 PM)
    night_dt = datetime(2026, 9, 17, 20, 15, 0, tzinfo=IST_TIMEZONE)
    is_night_allowed = engine.is_within_calling_window(night_dt)
    print(f"  - 20:15 IST Calling Allowed: {is_night_allowed} (Expected: False)")
    assert is_night_allowed is False, "Expected 20:15 IST to be blocked"

    # 3. Blocked morning time: 07:45 IST (7:45 AM)
    morning_dt = datetime(2026, 9, 17, 7, 45, 0, tzinfo=IST_TIMEZONE)
    is_morning_allowed = engine.is_within_calling_window(morning_dt)
    print(f"  - 07:45 IST Calling Allowed: {is_morning_allowed} (Expected: False)")
    assert is_morning_allowed is False, "Expected 07:45 IST to be blocked"

    # 4. Lead validation response
    dummy_lead = Lead(
        lead_id="TEST_01",
        phone_number="+919876543210",
        name="Rahul Sharma",
        loan_id="MUTH-101",
        amount_due=5000.0,
    )
    approved, reason = engine.validate_lead_for_dialing(dummy_lead, enforce_hours=True, now=night_dt)
    print(f"  - Validation outside hours: Approved={approved}, Reason='{reason}'")
    assert approved is False
    assert "HOURS_RESTRICTED" in reason

    print("[PASS] TRAI Calling Window enforcement verified.")


def test_trai_dnd_and_frequency_capping():
    print_header("2. Testing TRAI DND Registry & Frequency Capping")

    engine = TRAIComplianceEngine()
    engine.add_dnd_number("+919876543210")

    # 1. DND Registry Check
    is_dnd = engine.is_dnd_registered("9876543210")
    print(f"  - Phone +919876543210 in DND registry: {is_dnd} (Expected: True)")
    assert is_dnd is True

    non_dnd = engine.is_dnd_registered("+918765432109")
    print(f"  - Phone +918765432109 in DND registry: {non_dnd} (Expected: False)")
    assert non_dnd is False

    dnd_lead = Lead(
        lead_id="DND_LEAD",
        phone_number="+919876543210",
        name="Sunil Kumar",
        loan_id="MUTH-102",
        amount_due=4200.0,
    )
    approved, reason = engine.validate_lead_for_dialing(dnd_lead, enforce_hours=False)
    print(f"  - DND lead validation: Approved={approved}, Reason='{reason}'")
    assert approved is False
    assert "DND_BLOCKED" in reason

    # 2. Daily Frequency Capping (max 3 calls/day)
    clean_lead = Lead(
        lead_id="FREQ_LEAD",
        phone_number="+918765432109",
        name="Amit Patel",
        loan_id="MUTH-103",
        amount_due=3500.0,
    )
    now_ist = datetime(2026, 9, 17, 12, 0, 0, tzinfo=IST_TIMEZONE)

    # Simulate 3 prior calls today
    clean_lead.call_history = [
        {"timestamp": datetime(2026, 9, 17, 9, 30, 0, tzinfo=IST_TIMEZONE).isoformat()},
        {"timestamp": datetime(2026, 9, 17, 10, 45, 0, tzinfo=IST_TIMEZONE).isoformat()},
        {"timestamp": datetime(2026, 9, 17, 11, 30, 0, tzinfo=IST_TIMEZONE).isoformat()},
    ]

    freq_ok = engine.check_daily_frequency(clean_lead, max_per_day=3, now=now_ist)
    print(f"  - Frequency check after 3 calls today: {freq_ok} (Expected: False - capped)")
    assert freq_ok is False

    approved_freq, reason_freq = engine.validate_lead_for_dialing(
        clean_lead, enforce_hours=False, max_daily_calls=3, now=now_ist
    )
    assert approved_freq is False
    assert "FREQUENCY_EXCEEDED" in reason_freq
    print(f"  - 4th call attempt blocked: '{reason_freq}'")

    print("[PASS] TRAI DND Registry & Frequency Capping verified.")


def test_amd_engine():
    print_header("3. Testing Multi-Modal Answering Machine Detection (AMD)")

    amd = AMDClassifier()

    # Case A: Human Greeting (short speech + natural conversational pause)
    human_res = amd.classify(
        audio_duration_sec=0.9,
        transcript="Hello, Sharma bol raha hoon.",
        silence_after_burst_sec=1.0,
    )
    print(f"  - Case A (Human): Decision={human_res.decision.value}, Conf={human_res.confidence:.2f}, Reason='{human_res.reason}'")
    assert human_res.decision == AMDDecision.HUMAN_ANSWERED

    # Case B: Telecom Operator Announcement (Switched Off)
    op_res = amd.classify(
        audio_duration_sec=3.0,
        transcript="Aapka dial kiya gaya number abhi switched off hai. Kripya kuch samay baad prayas karein.",
    )
    print(f"  - Case B (Operator): Decision={op_res.decision.value}, Conf={op_res.confidence:.2f}, Detected={op_res.detected_phrases}")
    assert op_res.decision == AMDDecision.OPERATOR_ANNOUNCEMENT

    # Case C: Indian Telecom Carrier Busy Announcement
    busy_res = amd.classify(
        audio_duration_sec=2.8,
        transcript="The number you have dialed is currently busy. Please call back later.",
    )
    print(f"  - Case C (Carrier Busy): Decision={busy_res.decision.value}, Conf={busy_res.confidence:.2f}")
    assert busy_res.decision == AMDDecision.OPERATOR_ANNOUNCEMENT

    # Case D: Voicemail / Answering Machine Greeting
    vm_res = amd.classify(
        audio_duration_sec=4.2,
        transcript="You have reached the voicemail of Deepak. Please leave your message after the tone.",
    )
    print(f"  - Case D (Voicemail): Decision={vm_res.decision.value}, Conf={vm_res.confidence:.2f}, Detected={vm_res.detected_phrases}")
    assert vm_res.decision == AMDDecision.MACHINE_VOICEMAIL

    # Case E: Voicemail Beep Tone Detection (1000Hz pure tone)
    sample_rate = 8000
    pcm = bytearray()
    for i in range(sample_rate // 3):  # 333ms of 1000Hz tone
        val = int(16000 * math.sin(2.0 * math.pi * 1000.0 * i / sample_rate))
        pcm.extend(val.to_bytes(2, byteorder='little', signed=True))

    beep_detected = amd.detect_beep_tone(bytes(pcm), sample_rate=sample_rate)
    print(f"  - Case E (1000Hz Beep Spectral Energy): Detected={beep_detected} (Expected: True)")
    assert beep_detected is True

    beep_res = amd.classify(
        audio_duration_sec=2.5,
        transcript="",
        pcm_bytes=bytes(pcm),
    )
    assert beep_res.decision == AMDDecision.MACHINE_VOICEMAIL
    print(f"  - Case E AMD Result: Decision={beep_res.decision.value}, Reason='{beep_res.reason}'")

    # Case F: Silence / Dead Air
    silence_res = amd.classify(audio_duration_sec=0.1)
    print(f"  - Case F (Dead Air): Decision={silence_res.decision.value}")
    assert silence_res.decision == AMDDecision.SILENCE_TIMEOUT

    print("[PASS] Answering Machine Detection (AMD) engine verified.")


def test_lead_ingestion_and_sanitization():
    print_header("4. Testing Lead Ingestion & DPDP Boundary Sanitization")

    dialer = CampaignDialer()

    sample_csv = """name,phone_number,amount_due,loan_id,due_date
Vikram Malhotra,+91 98765 43210,5420.00,MUTH-8921,2026-09-20
Pooja Mehta,8765432109,3200.50,MUTH-8922,2026-09-21
Invalid Phone,+1 212 555 1234,4500.00,MUTH-8923,2026-09-22
Excessive Amount,+919988776655,9000000.00,MUTH-8924,2026-09-23
Script Injection Phone,+919876543211,1500.00,<script>alert(1)</script>,2026-09-25
"""

    accepted, rejected, errors = dialer.ingest_csv(sample_csv)
    print(f"  - Ingestion result: Accepted={accepted}, Rejected={rejected}")
    for err in errors:
        print(f"    [Scrubbed Input]: {err}")

    assert accepted == 2, f"Expected 2 accepted leads, got {accepted}"
    assert rejected == 3, f"Expected 3 rejected leads, got {rejected}"

    # Verify sanitized leads stored properly
    lead_1 = dialer.leads[0]
    assert lead_1.phone_number == "+919876543210"
    assert lead_1.amount_due == 5420.00
    assert lead_1.loan_id == "MUTH-8921"

    print("[PASS] Lead Ingestion and DPDP boundary validation verified.")


def test_concurrent_campaign_execution():
    print_header("5. Testing Concurrent Multi-Channel Worker Queue Execution")

    config = CampaignConfig(
        campaign_id="CAMP_TEST_001",
        max_concurrent_channels=3,
        enforce_trai_calling_hours=False,  # Allow test execution any hour
        enforce_dnd_check=False,
    )
    dialer = CampaignDialer(config=config)

    # Ingest 6 valid leads
    leads_payload = [
        {"name": f"Borrower {i}", "phone": f"+91987654321{i}", "amount": 2500.0 + (i * 100), "loan_id": f"LOAN_00{i}"}
        for i in range(6)
    ]
    accepted, rejected, _ = dialer.ingest_leads_from_list(leads_payload)
    assert accepted == 6

    # Execute concurrent campaign
    summary = asyncio.run(dialer.run_campaign())

    print(f"  - Campaign Completed: Leads={summary.total_leads}, Dialed={summary.dialed_count}, Connected={summary.connected_count}")
    print(f"  - Human Answered: {summary.human_answered_count}, Promises to Pay: {summary.promise_to_pay_count}")
    print(f"  - Total Recovered / Promised: Rs. {summary.total_amount_recovered:,.2f}")
    print(f"  - Average Call Duration: {summary.average_duration_sec:.2f}s")

    assert summary.total_leads == 6
    assert summary.dialed_count == 6
    assert summary.connected_count == 6
    assert summary.human_answered_count == 6
    assert summary.promise_to_pay_count == 6
    assert summary.total_amount_recovered > 0

    print("[PASS] Concurrent Multi-Channel Campaign execution verified.")


def test_automated_retry_logic():
    print_header("6. Testing Automated Retry Logic with Backoff")

    config = CampaignConfig(
        campaign_id="RETRY_TEST",
        max_concurrent_channels=2,
        max_retries_per_lead=2,
        retry_delay_seconds=0.01,  # Accelerated for unit test
        enforce_trai_calling_hours=False,
        enforce_dnd_check=False,
    )
    dialer = CampaignDialer(config=config)

    busy_lead = Lead(
        lead_id="BUSY_01",
        phone_number="+919876543299",
        name="Busy Borrower",
        loan_id="MUTH-BUSY",
        amount_due=4000.0,
        custom_metadata={"scenario": "busy"},
    )
    dialer.leads.append(busy_lead)

    summary = asyncio.run(dialer.run_campaign())

    print(f"  - Retry test lead: Attempts={busy_lead.call_attempts}, Status={busy_lead.status.value}, Disposition={busy_lead.disposition.value}")
    assert busy_lead.call_attempts == 2, f"Expected 2 attempts, got {busy_lead.call_attempts}"
    assert busy_lead.disposition == CallDisposition.LINE_BUSY
    assert summary.busy_or_no_answer_count >= 1

    print("[PASS] Automated Retry Logic with Backoff verified.")


def test_cdr_export_pii_redaction():
    print_header("7. Testing DPDP-Sanitized Call Detail Record (CDR) Export")

    config = CampaignConfig(campaign_id="CDR_TEST", max_concurrent_channels=1, enforce_trai_calling_hours=False)
    dialer = CampaignDialer(config=config)

    lead = Lead(
        lead_id="CDR_LEAD_01",
        phone_number="+919876543210",
        name="Sunita Rao",
        loan_id="MUTH-9911",
        amount_due=6500.0,
    )
    dialer.leads.append(lead)
    asyncio.run(dialer.run_campaign())

    # Export JSON
    cdr_json = dialer.export_cdr_json(mask_pii=True)
    cdr_data = json.loads(cdr_json)
    records = cdr_data["call_detail_records"]
    assert len(records) == 1

    rec = records[0]
    print(f"  - Masked Phone in CDR: '{rec['phone_number']}' (Expected: '+91*****3210')")
    assert rec["phone_number"] == "+91*****3210"

    # Export CSV
    cdr_csv = dialer.export_cdr_csv(mask_pii=True)
    assert "+91*****3210" in cdr_csv
    assert "+919876543210" not in cdr_csv
    print(f"  - Plaintext Phone Absent in CSV: Verified")

    print("[PASS] DPDP-Sanitized Call Detail Record (CDR) Export verified.")


def test_fastapi_campaign_rest_api():
    print_header("8. Testing FastAPI Campaign & AMD REST Endpoints")

    from fastapi.testclient import TestClient
    from verbalyze.telephony.server import create_app

    test_token = "secret_campaign_token_123"
    app = create_app(auth_token=test_token)
    client = TestClient(app)

    auth_headers = {"Authorization": f"Bearer {test_token}"}

    # 1. Health check includes active_campaigns
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert "active_campaigns" in res_health.json()
    print(f"  - Health Endpoint: {res_health.json()}")

    # 2. Reject unauthenticated campaign start
    res_unauth = client.post("/campaign/start", json={"leads": []})
    assert res_unauth.status_code == 401
    print(f"  - Unauthenticated /campaign/start rejected with 401")

    # 3. Start authenticated campaign
    sample_leads = [
        {"name": "Anil Kumar", "phone": "+919876543210", "amount": 4500.0, "loan_id": "MUTH-API-1"},
        {"name": "Meena Devi", "phone": "+918765432109", "amount": 3200.0, "loan_id": "MUTH-API-2"},
    ]
    res_start = client.post(
        "/campaign/start",
        json={
            "campaign_name": "API Test Recovery",
            "channels": 2,
            "enforce_calling_hours": False,
            "leads": sample_leads,
        },
        headers=auth_headers,
    )
    assert res_start.status_code == 200
    data_start = res_start.json()
    campaign_id = data_start["campaign_id"]
    print(f"  - Campaign started: ID={campaign_id}, Accepted={data_start['accepted_leads']}")
    assert data_start["accepted_leads"] == 2

    # 4. Query campaign status
    res_status = client.get(f"/campaign/status/{campaign_id}", headers=auth_headers)
    assert res_status.status_code == 200
    print(f"  - Status check: Total Leads={res_status.json()['total_leads']}")

    # 5. Query CDRs
    res_cdr = client.get(f"/campaign/cdr/{campaign_id}", headers=auth_headers)
    assert res_cdr.status_code == 200
    print(f"  - CDR check: Count={res_cdr.json()['count']}, PII Masked={res_cdr.json()['pii_masked']}")

    # 6. Standalone AMD classification endpoint
    res_amd = client.post(
        "/telephony/amd",
        json={
            "transcript": "Aapka dial kiya gaya number abhi vyast hai.",
            "speech_duration_sec": 2.5,
        },
        headers=auth_headers,
    )
    assert res_amd.status_code == 200
    amd_data = res_amd.json()
    print(f"  - Standalone AMD Endpoint: Decision={amd_data['decision']}, Confidence={amd_data['confidence']}")
    assert amd_data["decision"] == "OPERATOR_ANNOUNCEMENT"

    print("[PASS] FastAPI Campaign & AMD REST Endpoints verified.")


def main():
    print("=" * 65)
    print("VERBALYZE: OUTBOUND CAMPAIGN BATCH DIALER & AMD SUITE VERIFICATION")
    print("=" * 65)

    test_traI_calling_window()
    test_trai_dnd_and_frequency_capping()
    test_amd_engine()
    test_lead_ingestion_and_sanitization()
    test_concurrent_campaign_execution()
    test_automated_retry_logic()
    test_cdr_export_pii_redaction()
    test_fastapi_campaign_rest_api()

    print("\n" + "=" * 65)
    print("ALL 8/8 CAMPAIGN BATCH DIALER & AMD TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
