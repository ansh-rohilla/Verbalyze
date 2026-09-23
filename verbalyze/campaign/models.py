"""
verbalyze/campaign/models.py

Data structures, enums, and models for outbound telephony campaign management,
lead tracking, answering machine detection, and Call Detail Records (CDRs).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional


class LeadStatus(str, Enum):
    """Lifecycle status of a campaign lead."""
    PENDING = "PENDING"
    DIALING = "DIALING"
    CONNECTED = "CONNECTED"
    COMPLETED = "COMPLETED"
    SETTLED = "SETTLED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    WHATSAPP_ENGAGED = "WHATSAPP_ENGAGED"
    BUSY = "BUSY"
    NO_ANSWER = "NO_ANSWER"
    AMD_VOICEMAIL = "AMD_VOICEMAIL"
    AMD_OPERATOR = "AMD_OPERATOR"
    DND_BLOCKED = "DND_BLOCKED"
    HOURS_RESTRICTED = "HOURS_RESTRICTED"
    FREQUENCY_EXCEEDED = "FREQUENCY_EXCEEDED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    FAILED = "FAILED"


class CallDisposition(str, Enum):
    """Business outcome disposition for completed or terminated calls."""
    PROMISE_TO_PAY = "PROMISE_TO_PAY"
    PAYMENT_LINK_SENT = "PAYMENT_LINK_SENT"
    PAYMENT_SETTLED = "PAYMENT_SETTLED"
    WHATSAPP_FOLLOWUP_DISPATCHED = "WHATSAPP_FOLLOWUP_DISPATCHED"
    DISPUTE_RAISED = "DISPUTE_RAISED"
    CALLBACK_REQUESTED = "CALLBACK_REQUESTED"
    WRONG_PERSON = "WRONG_PERSON"
    REFUSED_TO_PAY = "REFUSED_TO_PAY"
    CUSTOMER_HANGUP = "CUSTOMER_HANGUP"
    VOICEMAIL = "VOICEMAIL"
    OPERATOR_ANNOUNCEMENT = "OPERATOR_ANNOUNCEMENT"
    LINE_BUSY = "LINE_BUSY"
    NO_ANSWER = "NO_ANSWER"
    DND_REJECTED = "DND_REJECTED"
    HOURS_BLOCKED = "HOURS_BLOCKED"
    TRANSFERRED_TO_SUPERVISOR = "TRANSFERRED_TO_SUPERVISOR"
    LEGAL_DISPUTE_ESCALATED = "LEGAL_DISPUTE_ESCALATED"
    FAILED_CALL = "FAILED_CALL"


class AMDDecision(str, Enum):
    """Classification outcome from the Answering Machine Detection engine."""
    HUMAN_ANSWERED = "HUMAN_ANSWERED"
    MACHINE_VOICEMAIL = "MACHINE_VOICEMAIL"
    OPERATOR_ANNOUNCEMENT = "OPERATOR_ANNOUNCEMENT"
    SILENCE_TIMEOUT = "SILENCE_TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class AMDResult:
    """Detailed result produced by the AMD classifier."""
    decision: AMDDecision
    confidence: float
    reason: str
    latency_ms: float = 0.0
    detected_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "detected_phrases": self.detected_phrases,
        }


@dataclass
class Lead:
    """Individual customer lead record for outbound campaigns."""
    lead_id: str
    phone_number: str
    name: str
    loan_id: str
    amount_due: float
    due_date: str = ""
    status: LeadStatus = LeadStatus.PENDING
    disposition: Optional[CallDisposition] = None
    call_attempts: int = 0
    max_retries: int = 2
    last_called_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    call_duration_sec: float = 0.0
    payment_link_sent: bool = False
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    call_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self, mask_pii: bool = False) -> Dict[str, Any]:
        phone = self.phone_number
        if mask_pii:
            from verbalyze.security import PIIRedactor
            phone = PIIRedactor.mask_phone(phone)

        return {
            "lead_id": self.lead_id,
            "phone_number": phone,
            "name": self.name,
            "loan_id": self.loan_id,
            "amount_due": self.amount_due,
            "due_date": self.due_date,
            "status": self.status.value,
            "disposition": self.disposition.value if self.disposition else None,
            "call_attempts": self.call_attempts,
            "last_called_at": self.last_called_at.isoformat() if self.last_called_at else None,
            "call_duration_sec": round(self.call_duration_sec, 2),
            "payment_link_sent": self.payment_link_sent,
        }


@dataclass
class CampaignConfig:
    """Configuration parameters for an outbound dialing campaign."""
    campaign_id: str = "default_campaign"
    campaign_name: str = "Muthoot Recovery Campaign"
    persona: str = "muthoot_recovery"
    language: str = "hi"
    llm_provider: str = "ollama"
    model_name: Optional[str] = None
    max_concurrent_channels: int = 5
    max_retries_per_lead: int = 2
    retry_delay_seconds: float = 60.0
    exponential_backoff_factor: float = 2.0
    enforce_trai_calling_hours: bool = True
    enforce_dnd_check: bool = True
    max_daily_calls_per_borrower: int = 3
    amd_enabled: bool = True
    carrier_type: str = "simulated"
    strict_sovereignty: bool = False
    auth_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "persona": self.persona,
            "language": self.language,
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            "max_concurrent_channels": self.max_concurrent_channels,
            "max_retries_per_lead": self.max_retries_per_lead,
            "retry_delay_seconds": self.retry_delay_seconds,
            "enforce_trai_calling_hours": self.enforce_trai_calling_hours,
            "enforce_dnd_check": self.enforce_dnd_check,
            "max_daily_calls_per_borrower": self.max_daily_calls_per_borrower,
            "amd_enabled": self.amd_enabled,
            "carrier_type": self.carrier_type,
            "strict_sovereignty": self.strict_sovereignty,
        }


@dataclass
class CallDetailRecord:
    """Structured, immutable audit record for an individual outbound call."""
    call_id: str
    campaign_id: str
    lead_id: str
    phone_number: str
    loan_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    amd_result: Optional[AMDResult]
    final_disposition: CallDisposition
    payment_link_sent: bool
    amount_recovered_or_promised: float
    turns_count: int
    transcript_turns: List[Dict[str, str]] = field(default_factory=list)
    tool_events: List[Dict[str, Any]] = field(default_factory=list)
    agitation_score: float = 0.0
    dispute_type: str = "NONE"
    detected_language: str = "hi"
    is_code_switched: bool = False
    error_message: Optional[str] = None
    trunk_id: Optional[str] = None
    carrier_name: Optional[str] = None
    failover_occurred: bool = False
    failover_count: int = 0
    trunk_mos_score: Optional[float] = None
    biometric_status: str = "NOT_ENROLLED"
    biometric_confidence: float = 0.0
    spoof_type: str = "AUTHENTIC_HUMAN"

    def to_dict(self, mask_pii: bool = True) -> Dict[str, Any]:
        phone = self.phone_number
        transcripts = self.transcript_turns
        tool_events = self.tool_events

        if mask_pii:
            from verbalyze.security import PIIRedactor
            phone = PIIRedactor.mask_phone(phone)
            transcripts = [
                {
                    "role": turn.get("role", "unknown"),
                    "content": PIIRedactor.redact_text(str(turn.get("content", ""))),
                }
                for turn in self.transcript_turns
            ]
            tool_events = []
            for event in self.tool_events:
                if isinstance(event, dict):
                    tool_events.append({
                        k: PIIRedactor.redact_text(str(v)) if isinstance(v, str) else v
                        for k, v in event.items()
                    })
                elif isinstance(event, str):
                    tool_events.append(PIIRedactor.redact_text(event))
                else:
                    tool_events.append(event)

        return {
            "call_id": self.call_id,
            "campaign_id": self.campaign_id,
            "lead_id": self.lead_id,
            "phone_number": phone,
            "loan_id": self.loan_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "amd_decision": self.amd_result.decision.value if self.amd_result else None,
            "amd_confidence": self.amd_result.confidence if self.amd_result else None,
            "final_disposition": self.final_disposition.value,
            "agitation_score": round(self.agitation_score, 3),
            "dispute_type": self.dispute_type,
            "detected_language": self.detected_language,
            "is_code_switched": self.is_code_switched,
            "payment_link_sent": self.payment_link_sent,
            "amount_recovered_or_promised": self.amount_recovered_or_promised,
            "turns_count": self.turns_count,
            "transcripts": transcripts,
            "tool_events": tool_events,
            "error_message": self.error_message,
            "trunk_id": self.trunk_id,
            "carrier_name": self.carrier_name,
            "failover_occurred": self.failover_occurred,
            "failover_count": self.failover_count,
            "trunk_mos_score": self.trunk_mos_score,
            "biometric_status": self.biometric_status,
            "biometric_confidence": round(self.biometric_confidence, 4),
            "spoof_type": self.spoof_type,
        }


@dataclass
class CampaignSummary:
    """Aggregated metrics and disposition breakdown for a campaign run."""
    campaign_id: str
    total_leads: int = 0
    dialed_count: int = 0
    connected_count: int = 0
    human_answered_count: int = 0
    voicemail_count: int = 0
    operator_announcement_count: int = 0
    busy_or_no_answer_count: int = 0
    dnd_blocked_count: int = 0
    hours_blocked_count: int = 0
    transferred_to_supervisor_count: int = 0
    promise_to_pay_count: int = 0
    payment_link_sent_count: int = 0
    total_amount_recovered: float = 0.0
    failed_count: int = 0
    disposition_breakdown: Dict[str, int] = field(default_factory=dict)
    language_breakdown: Dict[str, int] = field(default_factory=dict)
    code_switched_count: int = 0
    average_duration_sec: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "total_leads": self.total_leads,
            "dialed_count": self.dialed_count,
            "connected_count": self.connected_count,
            "human_answered_count": self.human_answered_count,
            "voicemail_count": self.voicemail_count,
            "operator_announcement_count": self.operator_announcement_count,
            "busy_or_no_answer_count": self.busy_or_no_answer_count,
            "dnd_blocked_count": self.dnd_blocked_count,
            "hours_blocked_count": self.hours_blocked_count,
            "promise_to_pay_count": self.promise_to_pay_count,
            "payment_link_sent_count": self.payment_link_sent_count,
            "total_amount_recovered": round(self.total_amount_recovered, 2),
            "failed_count": self.failed_count,
            "disposition_breakdown": self.disposition_breakdown,
            "language_breakdown": self.language_breakdown,
            "code_switched_count": self.code_switched_count,
            "average_duration_sec": round(self.average_duration_sec, 2),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }
