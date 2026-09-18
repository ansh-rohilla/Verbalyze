"""
verbalyze/telephony/transfer.py

Carrier SIP REFER & Human Warm Transfer Protocol:
- Formats RFC 3515 SIP REFER directives with X-Verbalyze-Context metadata headers.
- Formats Twilio / Exotel XML <Dial> and <Refer> directives for live carrier handoff.
- Generates bi-directional WebSocket media stream transfer control frames.
"""

import json
import base64
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from verbalyze.security import PIIRedactor


@dataclass
class TransferContext:
    """
    Contextual briefing payload passed to the receiving human agent or queue.
    Eliminates need for the customer to repeat their details or dispute cause.
    """
    call_id: str
    caller_phone: str
    loan_id: str
    amount_due: float
    agitation_score: float
    dispute_type: str
    briefing_summary: str
    target_department: str = "supervisor"
    recent_turns: list = field(default_factory=list)

    def to_header_string(self) -> str:
        """Serializes context to a URL-safe Base64 JSON string for SIP headers."""
        raw_dict = {
            "call_id": self.call_id,
            "caller_phone": self.caller_phone,
            "loan_id": self.loan_id,
            "amount_due": self.amount_due,
            "agitation_score": round(self.agitation_score, 3),
            "dispute_type": self.dispute_type,
            "briefing_summary": self.briefing_summary,
            "target_department": self.target_department,
        }
        json_bytes = json.dumps(raw_dict, ensure_ascii=False).encode("utf-8")
        return base64.urlsafe_b64encode(json_bytes).decode("ascii")

    @classmethod
    def from_header_string(cls, header_val: str) -> "TransferContext":
        """Reconstructs TransferContext from Base64 header string."""
        json_bytes = base64.urlsafe_b64decode(header_val.encode("ascii"))
        data = json.loads(json_bytes.decode("utf-8"))
        return cls(
            call_id=data.get("call_id", ""),
            caller_phone=data.get("caller_phone", ""),
            loan_id=data.get("loan_id", ""),
            amount_due=float(data.get("amount_due", 0.0)),
            agitation_score=float(data.get("agitation_score", 0.0)),
            dispute_type=data.get("dispute_type", "NONE"),
            briefing_summary=data.get("briefing_summary", ""),
            target_department=data.get("target_department", "supervisor"),
        )

    def to_dict(self, mask_pii: bool = True) -> Dict[str, Any]:
        """Returns dictionary representation with optional DPDP PII masking."""
        phone = PIIRedactor.mask_phone(self.caller_phone) if mask_pii else self.caller_phone
        summary = PIIRedactor.redact_text(self.briefing_summary) if mask_pii else self.briefing_summary

        return {
            "call_id": self.call_id,
            "caller_phone": phone,
            "loan_id": self.loan_id,
            "amount_due": round(self.amount_due, 2),
            "agitation_score": round(self.agitation_score, 3),
            "dispute_type": self.dispute_type,
            "briefing_summary": summary,
            "target_department": self.target_department,
            "recent_turns": self.recent_turns,
        }


class SIPTransferDispatcher:
    """
    Constructs carrier-grade call transfer payloads across standard SIP trunks,
    carrier XML gateways, and WebSocket media stream control planes.
    """

    DEFAULT_SUPERVISOR_SIP = "sip:supervisor_queue@bank.internal"
    DEFAULT_SUPERVISOR_PHONE = "+918000012345"

    @classmethod
    def build_sip_refer(
        cls,
        target_uri: Optional[str] = None,
        context: Optional[TransferContext] = None,
        user_agent: str = "Verbalyze-Telephony/0.3.0"
    ) -> Dict[str, str]:
        """
        Builds standard RFC 3515 SIP REFER request headers and metadata:
        - Refer-To: Target SIP URI or telephone number
        - Referred-By: Originating Voicebot SIP identity
        - X-Verbalyze-Context: Encoded briefing metadata
        """
        target = target_uri or cls.DEFAULT_SUPERVISOR_SIP
        context_header = context.to_header_string() if context else ""

        return {
            "Method": "REFER",
            "Refer-To": f"<{target}>",
            "Referred-By": "<sip:verbalyze_bot@trunk.domain>",
            "X-Verbalyze-Context": context_header,
            "User-Agent": user_agent,
            "Content-Type": "application/json",
        }

    @classmethod
    def build_twiml_dial_transfer(
        cls,
        target: Optional[str] = None,
        context: Optional[TransferContext] = None,
        caller_id: Optional[str] = None,
        language: str = "hi",
    ) -> str:
        """
        Generates Twilio / Exotel compatible TwiML XML payload:
        Plays an empathetic transfer announcement, then dials the supervisor queue.
        """
        dest = target or cls.DEFAULT_SUPERVISOR_PHONE
        caller_id_attr = f' callerId="{caller_id}"' if caller_id else ""

        announcement = (
            "कृपया एक क्षण प्रतीक्षा करें, मैं आपकी कॉल वरिष्ठ अधिकारी को ट्रांसफर कर रही हूँ।"
            if language == "hi"
            else "Please hold for a moment while I transfer you to a senior officer."
        )

        if dest.startswith("sip:"):
            context_header = context.to_header_string() if context else ""
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="{language}-IN">{announcement}</Say>
    <Dial{caller_id_attr} timeout="25">
        <Sip>
            {dest}?X-Verbalyze-Context={context_header}
        </Sip>
    </Dial>
</Response>"""
        else:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="{language}-IN">{announcement}</Say>
    <Dial{caller_id_attr} timeout="25">
        <Number>{dest}</Number>
    </Dial>
</Response>"""

    @classmethod
    def build_websocket_transfer_event(
        cls,
        target_uri: Optional[str] = None,
        context: Optional[TransferContext] = None,
    ) -> Dict[str, Any]:
        """
        Generates a carrier WebSocket control frame instructing the telephony bridge
        to transfer the audio stream to another endpoint.
        """
        target = target_uri or cls.DEFAULT_SUPERVISOR_SIP
        return {
            "event": "transfer",
            "action": "sip_refer",
            "target_uri": target,
            "context": context.to_dict(mask_pii=False) if context else {},
        }
