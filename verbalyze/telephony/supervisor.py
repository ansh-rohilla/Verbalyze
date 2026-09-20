"""
verbalyze/telephony/supervisor.py

Enterprise Telephony Supervisor Observability & Live Call Monitoring Engine.
Provides real-time fleet call monitoring, whisper coaching bus, 1-click takeover,
acoustic quality telemetry, and live WebSocket pub/sub broadcasting.
DPDP Act 2023 compliant (automatic PII masking).
Zero-emoji compliant.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any

from verbalyze.security import PIIRedactor
from verbalyze.telephony.transfer import TransferContext, SIPTransferDispatcher


@dataclass
class WhisperMessage:
    """A private supervisor coaching directive injected into an active call."""
    whisper_id: str
    supervisor_id: str
    text: str
    timestamp: float = field(default_factory=time.time)
    delivered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "whisper_id": self.whisper_id,
            "supervisor_id": self.supervisor_id,
            "text": self.text,
            "timestamp": round(self.timestamp, 2),
            "delivered": self.delivered,
        }


@dataclass
class CallSupervisorRecord:
    """Live state record for an active telephony session tracked by supervisors."""
    call_id: str
    caller_phone_masked: str
    persona: str = "muthoot_recovery"
    language: str = "hi"
    start_time: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    turn_count: int = 0
    latest_customer_utterance: str = ""
    latest_agent_reply: str = ""
    sentiment_category: str = "CALM"
    agitation_score: float = 0.0
    dispute_type: str = "NONE"
    is_code_switched: bool = False
    detected_language: str = "hi"
    mos_score: float = 4.2
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    status: str = "IN_PROGRESS"  # IN_PROGRESS, SUPERVISOR_COACHING, SUPERVISOR_TAKEOVER, TRANSFERRED, COMPLETED
    whisper_history: List[WhisperMessage] = field(default_factory=list)
    transcript_history: List[Dict[str, str]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "caller_phone_masked": self.caller_phone_masked,
            "persona": self.persona,
            "language": self.language,
            "start_time": round(self.start_time, 2),
            "duration_seconds": round(time.time() - self.start_time, 1),
            "turn_count": self.turn_count,
            "latest_customer_utterance": self.latest_customer_utterance,
            "latest_agent_reply": self.latest_agent_reply,
            "sentiment_category": self.sentiment_category,
            "agitation_score": round(self.agitation_score, 3),
            "dispute_type": self.dispute_type,
            "is_code_switched": self.is_code_switched,
            "detected_language": self.detected_language,
            "mos_score": round(self.mos_score, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "packet_loss_rate": round(self.packet_loss_rate, 4),
            "status": self.status,
            "whisper_count": len(self.whisper_history),
            "whispers": [w.to_dict() for w in self.whisper_history],
            "transcript_turns": len(self.transcript_history),
            "last_updated": round(self.last_updated, 2),
        }


class SupervisorManager:
    """
    Fleet-wide Telephony Supervisor Observability Hub.
    Maintains real-time state for active calls, orchestrates private whisper coaching,
    triggers takeover escalations, and broadcasts live telemetry over WebSocket.
    """

    def __init__(self):
        self.active_calls: Dict[str, CallSupervisorRecord] = {}
        self.call_history: List[CallSupervisorRecord] = []
        self.subscribers: Set[asyncio.Queue] = set()
        self.call_agents: Dict[str, Any] = {}  # call_id -> VoiceAgent instance

    def register_call(
        self,
        call_id: str,
        caller_phone: Optional[str] = None,
        persona: str = "muthoot_recovery",
        language: str = "hi",
        agent_instance: Optional[Any] = None,
    ) -> CallSupervisorRecord:
        """Registers a newly connected call into the supervisor fleet registry."""
        masked_phone = PIIRedactor.mask_phone(caller_phone) if caller_phone else "Unknown"
        record = CallSupervisorRecord(
            call_id=call_id,
            caller_phone_masked=masked_phone,
            persona=persona,
            language=language,
            start_time=time.time(),
            last_updated=time.time(),
        )
        self.active_calls[call_id] = record
        if agent_instance:
            self.call_agents[call_id] = agent_instance

        self._broadcast({
            "event": "call_started",
            "call_id": call_id,
            "caller_phone_masked": masked_phone,
            "persona": persona,
            "language": language,
            "timestamp": time.time(),
        })
        return record

    def update_turn(
        self,
        call_id: str,
        customer_utterance: str,
        agent_reply: str,
        sentiment: Optional[Dict[str, Any]] = None,
        lid_info: Optional[Dict[str, Any]] = None,
        quality_report: Optional[Dict[str, Any]] = None,
        jitter_stats: Optional[Dict[str, Any]] = None,
    ):
        """Updates live conversational state and telemetry after a dialogue turn."""
        record = self.active_calls.get(call_id)
        if not record:
            return

        redacted_user = PIIRedactor.redact_text(customer_utterance)
        redacted_reply = PIIRedactor.redact_text(agent_reply)

        record.turn_count += 1
        record.latest_customer_utterance = redacted_user
        record.latest_agent_reply = redacted_reply
        record.last_updated = time.time()
        record.duration_seconds = time.time() - record.start_time

        record.transcript_history.append({
            "turn": str(record.turn_count),
            "customer": redacted_user,
            "agent": redacted_reply,
            "timestamp": str(round(time.time(), 2)),
        })

        # Process Sentiment & Agitation
        if sentiment:
            record.sentiment_category = sentiment.get("category", record.sentiment_category)
            record.agitation_score = float(sentiment.get("composite_agitation", record.agitation_score))
            record.dispute_type = sentiment.get("dispute_type", record.dispute_type)

        # Process Language Identification
        if lid_info:
            record.detected_language = lid_info.get("primary_language", record.language)
            record.is_code_switched = bool(lid_info.get("is_code_switched", record.is_code_switched))
            if lid_info.get("language_switched"):
                record.language = record.detected_language

        # Process MOS Quality Report
        if quality_report:
            record.mos_score = float(quality_report.get("composite_mos", record.mos_score))

        # Process Jitter Buffer Metrics
        if jitter_stats:
            record.jitter_ms = float(jitter_stats.get("current_jitter_ms", record.jitter_ms))
            record.packet_loss_rate = float(jitter_stats.get("packet_loss_rate", record.packet_loss_rate))

        # Check for High Agitation or High Jitter Alerts
        alert_type = None
        if record.agitation_score >= 0.70 or record.sentiment_category in ("AGITATED", "HOSTILE"):
            alert_type = "HIGH_AGITATION_ALERT"
        elif record.packet_loss_rate >= 0.15 or record.jitter_ms >= 120.0:
            alert_type = "NETWORK_DEGRADATION_ALERT"

        event_payload = {
            "event": "turn_update",
            "call_id": call_id,
            "turn_count": record.turn_count,
            "latest_customer_utterance": redacted_user,
            "latest_agent_reply": redacted_reply,
            "agitation_score": record.agitation_score,
            "sentiment_category": record.sentiment_category,
            "dispute_type": record.dispute_type,
            "detected_language": record.detected_language,
            "is_code_switched": record.is_code_switched,
            "mos_score": record.mos_score,
            "jitter_ms": record.jitter_ms,
            "status": record.status,
            "alert": alert_type,
            "timestamp": time.time(),
        }
        self._broadcast(event_payload)

    def inject_whisper(
        self,
        call_id: str,
        whisper_text: str,
        supervisor_id: str = "supervisor_1",
    ) -> bool:
        """
        Injects private supervisor coaching directly into an active call.
        The instruction is injected into the VoiceAgent conversational context.
        """
        record = self.active_calls.get(call_id)
        if not record:
            return False

        whisper_id = f"wh_{int(time.time() * 1000)}"
        msg = WhisperMessage(
            whisper_id=whisper_id,
            supervisor_id=supervisor_id,
            text=whisper_text,
            timestamp=time.time(),
            delivered=True,
        )
        record.whisper_history.append(msg)
        record.status = "SUPERVISOR_COACHING"
        record.last_updated = time.time()

        # Inject into agent if reference is active
        agent = self.call_agents.get(call_id)
        if agent and hasattr(agent, "inject_supervisor_whisper"):
            agent.inject_supervisor_whisper(whisper_text)

        self._broadcast({
            "event": "whisper_injected",
            "call_id": call_id,
            "whisper_id": whisper_id,
            "supervisor_id": supervisor_id,
            "text": whisper_text,
            "timestamp": time.time(),
        })
        return True

    def takeover_call(
        self,
        call_id: str,
        supervisor_id: str = "supervisor_1",
        target_sip: Optional[str] = None,
        reason: str = "supervisor_intervention",
    ) -> Dict[str, Any]:
        """
        Executes immediate supervisor takeover of an active call.
        Prepares warm transfer metadata and switches call status.
        """
        record = self.active_calls.get(call_id)
        if not record:
            return {"success": False, "error": "Call not found"}

        record.status = "SUPERVISOR_TAKEOVER"
        record.last_updated = time.time()

        transfer_ctx = TransferContext(
            call_id=call_id,
            caller_phone=record.caller_phone_masked,
            loan_id="MUTH-8921",
            amount_due=5420.0,
            agitation_score=record.agitation_score,
            dispute_type=record.dispute_type,
            briefing_summary=f"Supervisor takeover initiated by {supervisor_id}. Reason: {reason}",
            target_department="supervisor",
        )

        sip_refer = SIPTransferDispatcher.build_sip_refer(
            target_uri=target_sip,
            context=transfer_ctx,
        )

        payload = {
            "event": "takeover_initiated",
            "call_id": call_id,
            "supervisor_id": supervisor_id,
            "target_sip": target_sip or "sip:supervisor@telephony.internal",
            "reason": reason,
            "transfer_context": transfer_ctx.to_dict(mask_pii=True),
            "sip_refer_headers": sip_refer,
            "timestamp": time.time(),
        }
        self._broadcast(payload)
        return {"success": True, "call_id": call_id, "transfer_details": payload}

    def barge_in(self, call_id: str, supervisor_id: str = "supervisor_1") -> bool:
        """Interrupts bot audio playback immediately upon supervisor command."""
        record = self.active_calls.get(call_id)
        if not record:
            return False

        agent = self.call_agents.get(call_id)
        if agent and hasattr(agent, "interrupt"):
            agent.interrupt()

        self._broadcast({
            "event": "barge_in_triggered",
            "call_id": call_id,
            "supervisor_id": supervisor_id,
            "timestamp": time.time(),
        })
        return True

    def terminate_call(self, call_id: str, reason: str = "completed"):
        """Marks call as completed and moves it to history."""
        record = self.active_calls.pop(call_id, None)
        self.call_agents.pop(call_id, None)
        if record:
            record.status = "COMPLETED" if reason == "completed" else "TERMINATED"
            record.duration_seconds = time.time() - record.start_time
            record.last_updated = time.time()
            self.call_history.append(record)
            if len(self.call_history) > 100:
                self.call_history.pop(0)

            self._broadcast({
                "event": "call_ended",
                "call_id": call_id,
                "duration_seconds": round(record.duration_seconds, 1),
                "turn_count": record.turn_count,
                "reason": reason,
                "timestamp": time.time(),
            })

    def get_call(self, call_id: str) -> Optional[CallSupervisorRecord]:
        """Retrieves active call record."""
        return self.active_calls.get(call_id)

    def list_calls(self) -> List[Dict[str, Any]]:
        """Returns snapshot of all active calls."""
        return [record.to_dict() for record in self.active_calls.values()]

    def get_fleet_summary(self) -> Dict[str, Any]:
        """Aggregates fleet-wide live telemetry across all active calls."""
        active = list(self.active_calls.values())
        total_calls = len(active)

        if total_calls == 0:
            return {
                "active_calls_count": 0,
                "high_agitation_calls": 0,
                "code_switched_calls": 0,
                "average_agitation": 0.0,
                "average_jitter_ms": 0.0,
                "average_mos_score": 4.2,
                "status": "idle",
            }

        avg_agitation = sum(r.agitation_score for r in active) / total_calls
        avg_jitter = sum(r.jitter_ms for r in active) / total_calls
        avg_mos = sum(r.mos_score for r in active) / total_calls
        high_agitation = sum(1 for r in active if r.agitation_score >= 0.70)
        code_switched = sum(1 for r in active if r.is_code_switched)

        return {
            "active_calls_count": total_calls,
            "high_agitation_calls": high_agitation,
            "code_switched_calls": code_switched,
            "average_agitation": round(avg_agitation, 3),
            "average_jitter_ms": round(avg_jitter, 1),
            "average_mos_score": round(avg_mos, 2),
            "status": "elevated_alert" if high_agitation > 0 else "nominal",
        }

    def subscribe(self) -> asyncio.Queue:
        """Registers a new WebSocket listener for real-time events."""
        q = asyncio.Queue(maxsize=100)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue):
        """Removes a listener queue."""
        self.subscribers.discard(queue)

    def _broadcast(self, event_dict: Dict[str, Any]):
        """Non-blocking dispatch of event to all connected supervisor WebSocket queues."""
        msg_json = json.dumps(event_dict)
        dead_queues = []
        for q in self.subscribers:
            try:
                q.put_nowait(msg_json)
            except (asyncio.QueueFull, Exception):
                dead_queues.append(q)
        for dq in dead_queues:
            self.subscribers.discard(dq)
