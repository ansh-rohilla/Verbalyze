"""
verbalyze/agent/tools.py

Telephony tools executable by the Voice Agent during telephone calls:
- disconnect_tool: ends the phone call cleanly upon mutual resolution or farewell
- send_payment_link: sends SMS/WhatsApp UPI link for overdue EMI recovery
- schedule_callback: logs a promise-to-pay date or customer callback
"""

import json
from typing import Dict, Any, Tuple


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


def execute_telephony_tool(tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Executes a telephony function call.
    Returns:
        (is_call_terminated: bool, status_message: str)
    """
    if tool_name == "disconnect_tool":
        reason = arguments.get("reason", "normal_hangup")
        msg = f"[Telephony Event] CALL DISCONNECTED: {reason}"
        return True, msg

    elif tool_name == "send_payment_link":
        amount = arguments.get("amount", 0)
        loan_id = arguments.get("loan_id", "LOAN-XXXX")
        msg = f"[Telephony Event] SMS/WhatsApp UPI link sent for Rs. {amount:,.2f} on account {loan_id}."
        return False, msg

    elif tool_name == "schedule_callback":
        date = arguments.get("promised_date", "next week")
        notes = arguments.get("notes", "")
        msg = f"[Telephony Event] Promise to pay logged for {date}. Note: {notes}"
        return False, msg

    return False, f"[Telephony Event] Unknown tool call: {tool_name}"
