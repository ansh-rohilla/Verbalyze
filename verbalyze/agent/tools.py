"""
verbalyze/agent/tools.py

Telephony tools executable by the Voice Agent during telephone calls:
- disconnect_tool: ends the phone call cleanly upon mutual resolution or farewell
- send_payment_link: sends SMS/WhatsApp UPI link for overdue EMI recovery
- schedule_callback: logs a promise-to-pay date or customer callback
"""

import json
from typing import Dict, Any, Tuple, Optional
from verbalyze.telephony.sms_dispatch import dispatch_payment_sms
from verbalyze.security import (
    validate_amount,
    validate_loan_id,
    validate_indian_phone,
    PIIRedactor
)


TELEPHONY_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
          "name": "disconnect_tool",
          "description": "Call this function to hang up the phone call ONLY when the conversation is completely finished, both parties have exchanged farewells, or the user requested to end the call.",
          "parameters": {
              "type": "object",
              "properties": {
                  "reason": {
                      "type": "string",
                      "description": "Brief reason for disconnecting (e.g. 'payment_promise_recorded', 'farewell_exchanged', 'wrong_number', 'user_requested_hangup')"
                  }
              },
              "required": ["reason"]
          }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_payment_link",
            "description": "Sends an instant UPI payment link via SMS/WhatsApp to the customer's registered mobile number for clearing overdue loan EMI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "EMI amount to be paid (e.g. 5420.0)"
                    },
                    "loan_id": {
                        "type": "string",
                        "description": "Customer loan account number"
                    }
                },
                "required": ["amount", "loan_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_callback",
            "description": "Schedules a promised follow-up call or logs a committed payment date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "promised_date": {
                        "type": "string",
                        "description": "Date customer committed to pay (e.g. 'tomorrow', '2026-09-15')"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Reason for delay or customer comment"
                    }
                },
                "required": ["promised_date"]
            }
        }
    }
]


def execute_telephony_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    caller_phone: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Executes a telephony function call with end-to-end security and validation.
    Returns:
        (is_call_terminated: bool, status_message: str, event_data: Optional[Dict[str, Any]])
    """
    if tool_name == "disconnect_tool":
        reason = arguments.get("reason", "normal_hangup")
        msg = f"[Telephony Event] CALL DISCONNECTED: {reason}"
        return True, msg, {"reason": reason, "action": "hangup"}

    elif tool_name == "send_payment_link":
        # 1. Validate Amount
        is_amt_valid, valid_amount, amt_err = validate_amount(arguments.get("amount", 5420.0))
        if not is_amt_valid:
            return False, f"[Security Violation] Rejected payment link: {amt_err}", None

        # 2. Validate Loan ID
        is_loan_valid, valid_loan_id, loan_err = validate_loan_id(arguments.get("loan_id", "MUTH-8921"))
        if not is_loan_valid:
            return False, f"[Security Violation] Rejected payment link: {loan_err}", None

        # 3. Validate Target Phone Number
        target_phone = caller_phone or "9876543210"
        is_phone_valid, valid_phone, phone_err = validate_indian_phone(target_phone)
        if not is_phone_valid:
            return False, f"[Security Violation] Rejected payment link: {phone_err}", None

        # Dispatch real SMS with embedded NPCI UPI intent link
        dispatch_res = dispatch_payment_sms(phone_number=valid_phone, amount=valid_amount, loan_id=valid_loan_id)

        masked_phone = PIIRedactor.mask_phone(valid_phone)
        masked_upi = PIIRedactor.mask_upi_url(dispatch_res.get("upi_url", ""))
        msg = (
            f"[Telephony Event] SMS/UPI link sent to {masked_phone} "
            f"for Rs. {valid_amount:,.2f} on account {valid_loan_id}. (UPI: {masked_upi})"
        )
        return False, msg, dispatch_res

    elif tool_name == "schedule_callback":
        date = arguments.get("promised_date", "next week")
        notes = arguments.get("notes", "")
        redacted_notes = PIIRedactor.redact_text(notes)
        msg = f"[Telephony Event] Promise to pay logged for {date}. Note: {redacted_notes}"
        return False, msg, {"promised_date": date, "notes": redacted_notes, "action": "callback"}

    return False, f"[Telephony Event] Unknown tool call: {tool_name}", None
