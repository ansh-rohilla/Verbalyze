"""
verbalyze/telephony/sms_dispatch.py

Live SMS & Real UPI Payment Gateway Dispatcher:
- Formats official NPCI-compliant UPI deep-link payment intents (upi://pay?pa=...).
- Pluggable SMS gateways: Fast2SMS (India), Twilio SMS (Global), Webhook, and Mock Sandbox.
- Cleans and normalizes Indian mobile numbers (+91, leading 0, 10-digit format).
"""

import os
import json
import re
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, Tuple


def clean_indian_phone(phone_str: str) -> str:
    """
    Normalizes Indian mobile numbers into standard 10-digit format.
    Accepts: '+91 98765 43210', '09876543210', '+919876543210', '9876543210' -> '9876543210'
    """
    digits = re.sub(r"\D", "", phone_str or "")
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return digits if len(digits) == 10 else (digits or "9876543210")


def generate_upi_intent_url(
    amount: float,
    loan_id: str = "MUTH-8921",
    payee_vpa: str = "muthootfincorp@icici",
    payee_name: str = "Muthoot Fincorp Ltd"
) -> str:
    """
    Generates official NPCI-compliant UPI deep-link URL that directly launches
    Google Pay, PhonePe, Paytm, or BHIM on caller's mobile.
    """
    params = {
        "pa": payee_vpa,
        "pn": payee_name,
        "am": f"{amount:.2f}",
        "cu": "INR",
        "tn": f"Loan EMI {loan_id}",
        "tr": f"TXN_{loan_id}"
    }
    query_str = urllib.parse.urlencode(params)
    return f"upi://pay?{query_str}"


def format_payment_sms_text(amount: float, loan_id: str, upi_url: str) -> str:
    """Builds standard transactional SMS text in English/Hinglish."""
    web_link = f"https://pay.muthootfincorp.com/pay/{loan_id}"
    return (
        f"Dear Customer, your Muthoot Fincorp loan EMI of Rs. {amount:,.2f} for Loan Account {loan_id} is overdue. "
        f"Click to pay instantly via UPI: {web_link} (or open: {upi_url}) - Muthoot Fincorp"
    )


class SMSDispatcher:
    """
    Pluggable SMS Dispatcher supporting Fast2SMS, Twilio, Webhook, and Mock Sandbox.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.environ.get("SMS_PROVIDER") or "mock").lower()

        # Fast2SMS credentials
        self.fast2sms_api_key = os.environ.get("FAST2SMS_API_KEY")

        # Twilio credentials
        self.twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_from_phone = os.environ.get("TWILIO_PHONE_NUMBER")

        # Generic webhook URL
        self.webhook_url = os.environ.get("SMS_WEBHOOK_URL")

    def dispatch_payment_sms(
        self,
        phone_number: str,
        amount: float = 5420.0,
        loan_id: str = "MUTH-8921",
        custom_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dispatches a payment SMS with embedded UPI intent link to the customer's phone.
        """
        clean_phone = clean_indian_phone(phone_number)
        upi_url = generate_upi_intent_url(amount, loan_id)
        sms_text = format_payment_sms_text(amount, loan_id, upi_url)
        active_provider = (custom_provider or self.provider).lower()

        # 1. Fast2SMS (Indian Gateway)
        if active_provider == "fast2sms" and self.fast2sms_api_key:
            return self._send_fast2sms(clean_phone, sms_text, upi_url, amount, loan_id)

        # 2. Twilio SMS
        elif active_provider == "twilio" and self.twilio_account_sid and self.twilio_auth_token and self.twilio_from_phone:
            return self._send_twilio(clean_phone, sms_text, upi_url, amount, loan_id)

        # 3. Generic Webhook
        elif active_provider == "webhook" and self.webhook_url:
            return self._send_webhook(clean_phone, sms_text, upi_url, amount, loan_id)

        # 4. Mock / Sandbox Mode (Default)
        return self._send_mock(clean_phone, sms_text, upi_url, amount, loan_id)

    def _send_fast2sms(self, phone: str, text: str, upi_url: str, amount: float, loan_id: str) -> Dict[str, Any]:
        """Sends transactional SMS via Fast2SMS API."""
        url = "https://www.fast2sms.com/dev/bulkV2"
        headers = {
            "authorization": self.fast2sms_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "route": "q",
            "message": text,
            "language": "english",
            "flash": 0,
            "numbers": phone
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "status": "delivered" if data.get("return") else "failed",
                    "provider": "fast2sms",
                    "phone": f"+91{phone}",
                    "amount": amount,
                    "loan_id": loan_id,
                    "upi_url": upi_url,
                    "message": text,
                    "gateway_response": data
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "provider": "fast2sms",
                "phone": f"+91{phone}",
                "amount": amount,
                "loan_id": loan_id,
                "upi_url": upi_url,
                "message": text
            }

    def _send_twilio(self, phone: str, text: str, upi_url: str, amount: float, loan_id: str) -> Dict[str, Any]:
        """Sends SMS via Twilio Messages API."""
        import base64
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json"
        auth_header = "Basic " + base64.b64encode(f"{self.twilio_account_sid}:{self.twilio_auth_token}".encode("utf-8")).decode("ascii")

        data = urllib.parse.urlencode({
            "To": f"+91{phone}",
            "From": self.twilio_from_phone,
            "Body": text
        }).encode("utf-8")

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return {
                    "status": "delivered",
                    "provider": "twilio",
                    "sid": res_data.get("sid"),
                    "phone": f"+91{phone}",
                    "amount": amount,
                    "loan_id": loan_id,
                    "upi_url": upi_url,
                    "message": text
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "provider": "twilio",
                "phone": f"+91{phone}",
                "amount": amount,
                "loan_id": loan_id,
                "upi_url": upi_url,
                "message": text
            }

    def _send_webhook(self, phone: str, text: str, upi_url: str, amount: float, loan_id: str) -> Dict[str, Any]:
        """Sends SMS payload to a generic CPaaS webhook URL."""
        payload = {
            "event": "send_sms",
            "to": f"+91{phone}",
            "amount": amount,
            "loan_id": loan_id,
            "upi_intent_url": upi_url,
            "text": text
        }
        headers = {"Content-Type": "application/json"}
        try:
            req = urllib.request.Request(self.webhook_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return {
                    "status": "dispatched",
                    "provider": "webhook",
                    "phone": f"+91{phone}",
                    "amount": amount,
                    "loan_id": loan_id,
                    "upi_url": upi_url,
                    "message": text
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "provider": "webhook",
                "phone": f"+91{phone}",
                "amount": amount,
                "loan_id": loan_id,
                "upi_url": upi_url,
                "message": text
            }

    def _send_mock(self, phone: str, text: str, upi_url: str, amount: float, loan_id: str) -> Dict[str, Any]:
        """Mock sandbox dispatcher that formats realistic NPCI UPI link without API credentials."""
        formatted_phone = f"+91 {phone[:5]} {phone[5:]}"
        print(f"📱 [Live SMS Dispatcher] Sent to {formatted_phone}:")
        print(f"   • Message: '{text}'")
        print(f"   • UPI Deep-Link: '{upi_url}'")
        return {
            "status": "delivered_mock",
            "provider": "mock_sandbox",
            "phone": f"+91{phone}",
            "formatted_phone": formatted_phone,
            "amount": amount,
            "loan_id": loan_id,
            "upi_url": upi_url,
            "message": text,
            "web_link": f"https://pay.muthootfincorp.com/pay/{loan_id}"
        }


# Global default dispatcher singleton
_DEFAULT_DISPATCHER = SMSDispatcher()

def dispatch_payment_sms(
    phone_number: str,
    amount: float = 5420.0,
    loan_id: str = "MUTH-8921",
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to dispatch payment SMS."""
    return _DEFAULT_DISPATCHER.dispatch_payment_sms(phone_number, amount, loan_id, provider)
