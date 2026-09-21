"""
scripts/test_whatsapp_settlement_gateway.py

Comprehensive Test Suite for:
1. WhatsApp Interactive Template & NPCI UPI Intent Link Generation.
2. Inbound WhatsApp Delivery Receipts & Quick Reply Button Clicks.
3. Settlement Ledger Order Creation, Status Lifecycle & DPDP PII Masking.
4. Cryptographic HMAC-SHA256 Webhook Verification & Tamper Resistance.
5. End-to-End Payment Reconciliation (Dialer Retry Halt & Supervisor Broadcast).
6. In-Memory Digital PDF Payment Receipt Generation via fpdf2.
7. Voice-to-WhatsApp Omnichannel Fallback on Unreached Calls.
8. FastAPI WhatsApp & Payment Webhook REST Endpoints Integration.

Zero-emoji compliant.
DPDP Act 2023 compliant.
"""

import os
import sys
import json
import time
import hmac
import hashlib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from verbalyze.campaign.models import Lead, LeadStatus, CallDisposition
from verbalyze.campaign.dialer import CampaignDialer, CampaignConfig
from verbalyze.telephony.supervisor import SupervisorManager
from verbalyze.telephony.whatsapp_gateway import (
    WhatsAppGateway,
    WhatsAppButton,
    WhatsAppInteractiveTemplate,
)
from verbalyze.telephony.settlement_engine import (
    SettlementLedger,
    SettlementStatus,
    SettlementTransaction,
)
from verbalyze.telephony.payment_webhooks import (
    verify_razorpay_signature,
    verify_cashfree_signature,
    verify_upi_webhook_signature,
    parse_razorpay_webhook,
    parse_cashfree_webhook,
    parse_upi_callback,
)
from verbalyze.agent.tools import execute_telephony_tool
from verbalyze.telephony.server import create_app


def test_1_whatsapp_interactive_template_and_upi_intent():
    print("\n--- Test 1: WhatsApp Interactive Template & UPI Intent Link Generation ---")
    gateway = WhatsAppGateway()

    # Hindi Template
    hi_tpl = gateway.format_payment_template(
        customer_name="Rajesh Sharma",
        loan_id="MUTH-9812",
        amount=6500.0,
        overdue_days=15,
        language="hi",
        phone_number="+91 98765 43210",
    )

    assert hi_tpl.recipient_phone in ("9876543210", "+919876543210")
    assert "मुथूट फिनकॉर्प" in hi_tpl.header_text
    assert "MUTH-9812" in hi_tpl.body_text
    assert "upi://pay?" in hi_tpl.upi_intent_url
    assert "muthootfincorp" in hi_tpl.upi_intent_url
    assert "am=6500.00" in hi_tpl.upi_intent_url
    assert len(hi_tpl.buttons) == 3
    assert hi_tpl.buttons[0].title == "Pay via UPI"
    assert hi_tpl.buttons[1].title == "Request Callback"
    assert hi_tpl.buttons[2].title == "Raise Dispute"

    # Meta Graph API Payload Format
    meta_payload = hi_tpl.to_meta_payload()
    assert meta_payload["messaging_product"] == "whatsapp"
    assert meta_payload["type"] == "interactive"
    assert meta_payload["interactive"]["type"] == "button"
    assert len(meta_payload["interactive"]["action"]["buttons"]) == 3

    # English Template
    en_tpl = gateway.format_payment_template(
        customer_name="John Doe",
        loan_id="MUTH-1029",
        amount=12450.50,
        overdue_days=30,
        language="en",
        phone_number="9876543210",
    )
    assert "Muthoot Fincorp" in en_tpl.header_text
    assert "am=12450.50" in en_tpl.upi_intent_url
    print("[PASSED] WhatsApp interactive payment templates & NPCI UPI Intent URLs formatted correctly.")


def test_2_inbound_whatsapp_delivery_receipts_and_button_clicks():
    print("\n--- Test 2: Inbound WhatsApp Delivery Receipts & Quick Reply Button Clicks ---")
    gateway = WhatsAppGateway()

    # Dispatched message mock
    dispatch_res = gateway.dispatch_payment_message(
        phone_number="+919876543210",
        customer_name="Rajesh Sharma",
        loan_id="MUTH-7721",
        amount=4500.0,
    )
    msg_id = dispatch_res["message_id"]
    assert dispatch_res["status"] == "sent"

    # 1. Delivery Receipt Webhook
    status_webhook_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "WHATSAPP_BUSINESS_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "15550239999", "phone_number_id": "10001"},
                    "statuses": [{
                        "id": msg_id,
                        "status": "delivered",
                        "timestamp": "1726912345",
                        "recipient_id": "919876543210",
                    }]
                },
                "field": "messages"
            }]
        }]
    }

    event1 = gateway.handle_inbound_webhook(status_webhook_payload)
    assert event1["event_type"] == "status_update"
    assert event1["status"] == "delivered"
    assert gateway.dispatched_messages[msg_id]["status"] == "delivered"

    # 2. Interactive Button Click Webhook (Request Callback)
    button_webhook_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "WHATSAPP_BUSINESS_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "15550239999", "phone_number_id": "10001"},
                    "messages": [{
                        "from": "919876543210",
                        "id": "wamid_incoming_001",
                        "timestamp": "1726912400",
                        "type": "interactive",
                        "interactive": {
                            "type": "button_reply",
                            "button_reply": {
                                "id": "CALLBACK_MUTH-7721",
                                "title": "Request Callback"
                            }
                        }
                    }]
                },
                "field": "messages"
            }]
        }]
    }

    event2 = gateway.handle_inbound_webhook(button_webhook_payload)
    assert event2["event_type"] == "button_click"
    assert event2["button_id"] == "CALLBACK_MUTH-7721"
    assert event2["text"] == "Request Callback"
    assert event2["sender_phone"] == "919876543210"

    print("[PASSED] Inbound WhatsApp delivery status receipts and Quick Reply clicks parsed cleanly.")


def test_3_settlement_ledger_order_creation_and_dpdp_masking():
    print("\n--- Test 3: Settlement Ledger Order Creation & DPDP PII Masking ---")
    ledger = SettlementLedger()

    txn = ledger.create_order(
        loan_id="MUTH-4402",
        amount=8750.0,
        customer_phone="+91 98765 12345",
        customer_name="Amitabh Bachchan",
        gateway="UPI_INTENT",
        metadata={"overdue_days": 21},
    )

    assert txn.transaction_id.startswith("TXN_")
    assert txn.status == SettlementStatus.PENDING
    assert txn.amount == 8750.0
    assert txn.customer_phone_raw == "9876512345"
    assert "*" in txn.customer_phone_masked and "2345" in txn.customer_phone_masked

    # Test DPDP Masking in serialization
    masked_dict = txn.to_dict(mask_pii=True)
    assert masked_dict["customer_phone"] != "9876512345"
    assert "[REDACTED_NAME]" in masked_dict["customer_name"] or masked_dict["customer_name"] != "Amitabh Bachchan"

    # Test Raw serialization (internal only)
    raw_dict = txn.to_dict(mask_pii=False)
    assert raw_dict["customer_phone"] == "9876512345"
    assert raw_dict["customer_name"] == "Amitabh Bachchan"

    print("[PASSED] Settlement ledger orders initialized with robust DPDP Act PII redaction.")


def test_4_cryptographic_hmac_sha256_verification_and_tamper_resistance():
    print("\n--- Test 4: Cryptographic HMAC-SHA256 Verification & Tamper Resistance ---")
    secret = "rzp_webhook_secret_secure_key_2026"
    ledger = SettlementLedger(webhook_secret=secret)

    payload = json.dumps({"event": "payment.captured", "amount": 875000, "loan_id": "MUTH-4402"}).encode("utf-8")
    valid_sig = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    # 1. Valid Signature
    assert ledger.verify_hmac_signature(payload, valid_sig, secret) is True

    # 2. Tampered Payload (e.g. attacker changes amount)
    tampered_payload = json.dumps({"event": "payment.captured", "amount": 100, "loan_id": "MUTH-4402"}).encode("utf-8")
    assert ledger.verify_hmac_signature(tampered_payload, valid_sig, secret) is False

    # 3. Wrong Secret
    assert ledger.verify_hmac_signature(payload, valid_sig, "wrong_secret_key") is False

    # 4. Razorpay Signature Verification Helper
    assert verify_razorpay_signature(payload, valid_sig, secret) is True
    assert verify_razorpay_signature(tampered_payload, valid_sig, secret) is False

    # 5. Cashfree Timestamped Signature Verification Helper
    cf_secret = "cf_secret_key_888"
    ts = str(int(time.time()))
    cf_raw = json.dumps({"type": "PAYMENT_SUCCESS_WEBHOOK", "data": {"order": {"order_id": "MUTH-4402"}}}).encode("utf-8")
    to_sign = ts.encode("utf-8") + cf_raw
    cf_sig = hmac.new(cf_secret.encode("utf-8"), to_sign, hashlib.sha256).hexdigest()

    assert verify_cashfree_signature(cf_raw, cf_sig, ts, cf_secret) is True
    # Verify replay / timestamp tampering fails
    assert verify_cashfree_signature(cf_raw, cf_sig, str(int(ts) + 10), cf_secret) is False

    # 6. UPI Signature Verification Helper
    upi_secret = "upi_secret_999"
    upi_raw = json.dumps({"order_id": "MUTH-4402", "status": "SUCCESS"}).encode("utf-8")
    upi_sig = hmac.new(upi_secret.encode("utf-8"), upi_raw, hashlib.sha256).hexdigest()
    assert verify_upi_webhook_signature(upi_raw, upi_sig, upi_secret) is True
    assert verify_upi_webhook_signature(upi_raw, "bad_signature", upi_secret) is False

    print("[PASSED] Constant-time HMAC-SHA256 verification rejects tampered payloads and unauthorized signers.")


def test_5_end_to_end_payment_reconciliation_and_dialer_retry_halt():
    print("\n--- Test 5: End-to-End Payment Webhook Reconciliation & Dialer Retry Halt ---")
    supervisor = SupervisorManager()
    wa_gw = WhatsAppGateway()
    config = CampaignConfig(campaign_id="CAMP_SETTLE_TEST", campaign_name="Settle Test")
    dialer = CampaignDialer(config=config)

    # Ingest lead into dialer
    leads = [
        {"lead_id": "LEAD_7701", "loan_id": "MUTH-7701", "name": "Ramesh Gupta", "phone_number": "+919876543210", "amount_due": 5200.0}
    ]
    dialer.ingest_leads_from_list(leads)
    assert len(dialer.leads) == 1
    lead = dialer.leads[0]
    assert lead.status == LeadStatus.PENDING

    # Setup Settlement Ledger
    ledger = SettlementLedger(
        campaign_dialer=dialer,
        supervisor_manager=supervisor,
        whatsapp_gateway=wa_gw,
    )

    # Create order
    txn = ledger.create_order(
        loan_id="MUTH-7701",
        amount=5200.0,
        customer_phone="+919876543210",
        customer_name="Ramesh Gupta",
    )

    # Subscribe to supervisor events
    queue = supervisor.subscribe()

    # Reconcile Payment
    payment_ref = "pay_rzp_success_8921"
    success, msg, settled_txn = ledger.reconcile_payment(
        transaction_id=txn.transaction_id,
        gateway_payment_id=payment_ref,
        verify_sig=False,
        gateway="RAZORPAY",
    )

    assert success is True
    assert settled_txn.status == SettlementStatus.SETTLED
    assert settled_txn.receipt_number.startswith("REC_")
    assert settled_txn.gateway_payment_id == payment_ref

    # Verify Campaign Dialer marked lead SETTLED and cancelled retries
    assert lead.status == LeadStatus.SETTLED
    assert lead.disposition == CallDisposition.PAYMENT_SETTLED

    # Verify WhatsApp receipt was automatically dispatched
    receipt_msgs = [m for m in wa_gw.dispatched_messages.values() if m.get("template_type") == "payment_receipt"]
    assert len(receipt_msgs) == 1
    assert receipt_msgs[0]["loan_id"] == "MUTH-7701"
    assert receipt_msgs[0]["receipt_number"] == settled_txn.receipt_number

    # Verify Supervisor Manager broadcasted payment_settled event
    event = None
    try:
        # Check queue
        while not queue.empty():
            msg_str = queue.get_nowait()
            data = json.loads(msg_str)
            if data.get("event") == "payment_settled":
                event = data
                break
    finally:
        supervisor.unsubscribe(queue)

    assert event is not None
    assert event["loan_id"] == "MUTH-7701"
    assert event["amount"] == 5200.0
    assert event["receipt_number"] == settled_txn.receipt_number

    # Verify Idempotency on duplicate callback
    dup_success, dup_msg, _ = ledger.reconcile_payment(
        transaction_id=txn.transaction_id,
        gateway_payment_id=payment_ref,
        verify_sig=False,
    )
    assert dup_success is True
    assert "idempotent" in dup_msg.lower()

    print("[PASSED] End-to-end reconciliation halts dialer, marks lead SETTLED, and dispatches receipt.")


def test_6_in_memory_digital_pdf_receipt_generation():
    print("\n--- Test 6: In-Memory Digital PDF Receipt Generation via fpdf2 ---")
    ledger = SettlementLedger()
    txn = ledger.create_order(
        loan_id="MUTH-3312",
        amount=15420.0,
        customer_phone="+919876543210",
        customer_name="Suresh Raina",
        gateway="NPCI_UPI",
    )

    # Before settlement, receipt generation should return None
    assert ledger.generate_pdf_receipt_bytes(txn.transaction_id) is None

    # Settle transaction
    ledger.reconcile_payment(
        transaction_id=txn.transaction_id,
        gateway_payment_id="upi_txn_00998877",
        verify_sig=False,
    )

    # Generate in-memory PDF
    pdf_bytes = ledger.generate_pdf_receipt_bytes(txn.transaction_id)
    assert pdf_bytes is not None
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF"), "Generated file must have valid %PDF magic header"

    print(f"[PASSED] In-memory PDF receipt generated cleanly ({len(pdf_bytes)} bytes, zero disk bloat).")


def test_7_voice_to_whatsapp_fallback_and_agent_tool():
    print("\n--- Test 7: Voice-to-WhatsApp Omnichannel Fallback & Agent Tool ---")
    gateway = WhatsAppGateway()

    # 1. Omnichannel Voice Fallback
    fb_res = gateway.trigger_voice_fallback(
        phone_number="+91 98765 00000",
        loan_id="MUTH-9011",
        amount=3800.0,
        customer_name="Sunil Gavaskar",
        call_disposition="NO_ANSWER",
        language="hi",
    )
    assert fb_res["success"] is True
    assert fb_res["status"] == "sent"

    # 2. Agent Tool Execution (send_whatsapp_payment_link)
    tool_args = {
        "loan_id": "MUTH-9011",
        "amount": 3800.0,
        "language": "hi",
    }
    terminated, msg, data = execute_telephony_tool(
        "send_whatsapp_payment_link",
        tool_args,
        caller_phone="+919876500000",
    )
    assert terminated is False
    assert "Telephony Event" in msg
    assert data["success"] is True
    assert "upi_intent_url" in data
    assert "upi://pay?" in data["upi_intent_url"]
    assert "*" in data["recipient"]

    print("[PASSED] Omnichannel voice fallback and VoiceAgent payment link tool executed successfully.")


def test_8_fastapi_whatsapp_and_payment_webhooks_integration():
    print("\n--- Test 8: FastAPI WhatsApp & Payment Webhooks REST API Integration ---")
    app = create_app()
    client = TestClient(app)

    # 1. GET /webhook/whatsapp (Meta Challenge Handshake)
    res_challenge = client.get("/webhook/whatsapp?hub.mode=subscribe&hub.verify_token=verbalyze_wa_verify_secret&hub.challenge=11559988")
    assert res_challenge.status_code == 200
    assert res_challenge.text == "11559988"

    # Invalid token rejected
    res_bad_token = client.get("/webhook/whatsapp?hub.mode=subscribe&hub.verify_token=wrong_token&hub.challenge=11559988")
    assert res_bad_token.status_code == 403

    # 2. POST /settlement/order (Create Order via API)
    order_data = {
        "loan_id": "MUTH-API-01",
        "phone": "+919876543210",
        "amount": 7200.0,
        "customer_name": "Kishore Kumar",
    }
    res_order = client.post("/settlement/order", json=order_data)
    assert res_order.status_code == 200
    order_res = res_order.json()
    assert order_res["status"] == "created"
    txn_id = order_res["transaction"]["transaction_id"]

    # 3. GET /settlement/transactions
    res_list = client.get("/settlement/transactions")
    assert res_list.status_code == 200
    assert res_list.json()["count"] >= 1

    # 4. POST /webhook/payment/razorpay (Reconcile via Webhook)
    rzp_secret = "razorpay_secret_key_123"
    rzp_payload_dict = {
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test_rzp_999",
                    "amount": 720000,
                    "currency": "INR",
                    "status": "captured",
                    "order_id": txn_id,
                    "notes": {"loan_id": "MUTH-API-01"},
                }
            }
        }
    }
    rzp_bytes = json.dumps(rzp_payload_dict).encode("utf-8")
    rzp_sig = hmac.new(rzp_secret.encode("utf-8"), rzp_bytes, hashlib.sha256).hexdigest()

    res_rzp = client.post(
        "/webhook/payment/razorpay",
        content=rzp_bytes,
        headers={"Content-Type": "application/json", "x-razorpay-signature": rzp_sig},
    )
    assert res_rzp.status_code == 200
    assert res_rzp.json()["status"] == "reconciled"

    # 5. GET /settlement/receipt/{transaction_id} (Download In-Memory PDF Receipt)
    res_receipt = client.get(f"/settlement/receipt/{txn_id}")
    assert res_receipt.status_code == 200
    assert res_receipt.headers["content-type"] == "application/pdf"
    assert res_receipt.content.startswith(b"%PDF")

    # 6. POST /webhook/payment/upi (Direct UPI Callback)
    # Create another order
    res_order2 = client.post("/settlement/order", json={
        "loan_id": "MUTH-API-02",
        "phone": "+919876543210",
        "amount": 3400.0,
    })
    txn2_id = res_order2.json()["transaction"]["transaction_id"]

    upi_secret = "settlement_webhook_secret_key_123"
    upi_payload_dict = {
        "order_id": txn2_id,
        "loan_id": "MUTH-API-02",
        "amount": 3400.0,
        "upi_ref_id": "upi_ref_44556677",
        "status": "SUCCESS",
    }
    upi_bytes = json.dumps(upi_payload_dict).encode("utf-8")
    upi_sig = hmac.new(upi_secret.encode("utf-8"), upi_bytes, hashlib.sha256).hexdigest()

    res_upi = client.post(
        "/webhook/payment/upi",
        content=upi_bytes,
        headers={"Content-Type": "application/json", "x-webhook-signature": upi_sig},
    )
    assert res_upi.status_code == 200
    assert res_upi.json()["status"] == "reconciled"

    # 7. POST /webhook/whatsapp (Button Click Dispatch to Supervisor)
    wa_button_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "WHATSAPP_BUSINESS_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "messages": [{
                        "from": "919876543210",
                        "id": "wamid_in_009",
                        "type": "interactive",
                        "interactive": {
                            "type": "button_reply",
                            "button_reply": {
                                "id": "DISPUTE_MUTH-API-01",
                                "title": "Raise Dispute"
                            }
                        }
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    res_wa_post = client.post("/webhook/whatsapp", json=wa_button_payload)
    assert res_wa_post.status_code == 200
    assert res_wa_post.json()["status"] == "success"

    print("[PASSED] All FastAPI WhatsApp & Settlement REST/Webhook endpoints verified successfully.")


def run_all_tests():
    print("================================================================================")
    print("RUNNING VERBALYZE OMNICHANNEL WHATSAPP & UPI SETTLEMENT GATEWAY TEST SUITE")
    print("Zero-Emoji Compliant | DPDP Act 2023 Compliant | RBI Sovereign In-Memory PDF")
    print("================================================================================")

    test_1_whatsapp_interactive_template_and_upi_intent()
    test_2_inbound_whatsapp_delivery_receipts_and_button_clicks()
    test_3_settlement_ledger_order_creation_and_dpdp_masking()
    test_4_cryptographic_hmac_sha256_verification_and_tamper_resistance()
    test_5_end_to_end_payment_reconciliation_and_dialer_retry_halt()
    test_6_in_memory_digital_pdf_receipt_generation()
    test_7_voice_to_whatsapp_fallback_and_agent_tool()
    test_8_fastapi_whatsapp_and_payment_webhooks_integration()

    print("\n================================================================================")
    print("ALL 8 OMNICHANNEL WHATSAPP & UPI SETTLEMENT TESTS PASSED SUCCESSFULLY.")
    print("================================================================================")


if __name__ == "__main__":
    run_all_tests()
