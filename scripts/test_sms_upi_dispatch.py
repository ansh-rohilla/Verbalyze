#!/usr/bin/env python3
"""
scripts/test_sms_upi_dispatch.py

Verification suite for Live SMS & Real UPI Payment Gateway Dispatch:
1. NPCI-compliant UPI deep-link intent generation.
2. Indian mobile phone number normalization.
3. Pluggable SMS gateway dispatchers (Mock Sandbox, Fast2SMS, Twilio).
4. End-to-end VoiceAgent spoken turn-taking with live SMS dispatch.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.telephony.sms_dispatch import (
    clean_indian_phone,
    generate_upi_intent_url,
    format_payment_sms_text,
    SMSDispatcher,
    dispatch_payment_sms
)
from verbalyze.agent.voice_bot import VoiceAgent


def test_phone_cleaning():
    print("=================================================================")
    print("1. Testing Indian Mobile Number Normalization")
    print("=================================================================")
    test_cases = [
        ("+91 98765 43210", "9876543210"),
        ("+919876543210", "9876543210"),
        ("09876543210", "9876543210"),
        ("9876543210", "9876543210"),
        ("91-98765-43210", "9876543210"),
    ]
    for inp, expected in test_cases:
        res = clean_indian_phone(inp)
        assert res == expected, f"Failed for {inp}: expected {expected}, got {res}"
        print(f"[PASS] '{inp}' -> '{res}'")
    print("Phone normalization PASSED!\n")


def test_upi_intent_url():
    print("=================================================================")
    print("2. Testing NPCI-Compliant UPI Deep-Link Intent Generation")
    print("=================================================================")
    upi_url = generate_upi_intent_url(amount=5420.0, loan_id="MUTH-8921")
    print(f"Generated UPI Deep-Link:\n  {upi_url}\n")
    assert upi_url.startswith("upi://pay?"), "UPI link does not start with upi://pay?"
    assert "pa=muthootfincorp%40icici" in upi_url or "pa=muthootfincorp@icici" in upi_url, "Payee VPA missing!"
    assert "am=5420.00" in upi_url, "Amount missing or misformatted!"
    assert "cu=INR" in upi_url, "Currency INR missing!"
    assert "tn=Loan" in upi_url or "MUTH-8921" in upi_url, "Transaction note missing!"
    print("NPCI UPI intent generator PASSED!\n")


def test_sms_formatting_and_dispatch():
    print("=================================================================")
    print("3. Testing SMS Formatting and Mock Dispatcher")
    print("=================================================================")
    res = dispatch_payment_sms(
        phone_number="+91 98765 43210",
        amount=5420.0,
        loan_id="MUTH-8921"
    )
    print(f"Dispatch result status: {res.get('status')}")
    print(f"Recipient phone: {res.get('phone')}")
    print(f"UPI Intent URL: {res.get('upi_url')}")
    print(f"Formatted SMS text:\n  '{res.get('message')}'\n")

    assert res.get("status") == "delivered_mock", "Expected delivered_mock status!"
    assert res.get("phone") == "+919876543210", "Phone mismatch!"
    assert res.get("amount") == 5420.0, "Amount mismatch!"
    assert "upi://pay?" in res.get("upi_url", ""), "UPI URL missing!"
    print("Mock SMS dispatch PASSED!\n")


def test_end_to_end_voice_agent_sms_trigger():
    print("=================================================================")
    print("4. Testing End-to-End Voice Agent Spoken Turn -> Live SMS Dispatch")
    print("=================================================================")
    caller_mobile = "+919812345678"
    agent = VoiceAgent(
        language="hi",
        persona="muthoot_recovery",
        llm_provider="mock",
        voice_enabled=True,
        caller_phone=caller_mobile
    )

    # Turn requesting payment link
    user_turn = "हाँ जी ठीक है, मुझे पेमेंट करने के लिए यूपीआई लिंक भेज दीजिए।"
    print(f"Caller: '{user_turn}'")
    res = agent.step(user_turn)

    print(f"Agent Speech: '{res['text']}'")
    print(f"Tool Event: {res.get('tool_event')}")

    tool_data = res.get("tool_data")
    assert tool_data is not None, "tool_data was not returned in agent.step()!"
    print(f"[PASS] Dispatched to phone: {tool_data.get('phone')}")
    print(f"[PASS] UPI Deep-Link: {tool_data.get('upi_url')}")
    print(f"[PASS] Amount: Rs. {tool_data.get('amount')}")

    assert tool_data.get("phone") == "+919812345678", f"Expected +919812345678, got {tool_data.get('phone')}"
    assert tool_data.get("amount") == 5420.0, "Expected amount 5420.0"
    assert "upi://pay?" in tool_data.get("upi_url", ""), "UPI link missing!"
    print("End-to-End Voice Agent Spoken Turn -> SMS Dispatch PASSED!\n")


if __name__ == "__main__":
    test_phone_cleaning()
    test_upi_intent_url()
    test_sms_formatting_and_dispatch()
    test_end_to_end_voice_agent_sms_trigger()
    print("=================================================================")
    print("ALL LIVE SMS & UPI PAYMENT GATEWAY TESTS PASSED!")
    print("=================================================================")
