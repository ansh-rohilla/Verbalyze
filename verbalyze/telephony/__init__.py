"""Telephony Webhook Integration for Exotel, Twilio, WebRTC/Browser, WhatsApp, and Unmetered SIP."""

from verbalyze.telephony.transfer import SIPTransferDispatcher, TransferContext
from verbalyze.telephony.jitter_buffer import AdaptiveJitterBuffer, JitterBufferPacket, JitterBufferStats
from verbalyze.telephony.supervisor import SupervisorManager, CallSupervisorRecord, WhisperMessage
from verbalyze.telephony.browser_gateway import BrowserAudioSession
from verbalyze.telephony.whatsapp_gateway import (
    WhatsAppGateway,
    WhatsAppInteractiveTemplate,
    WhatsAppButton,
)
from verbalyze.telephony.settlement_engine import (
    SettlementLedger,
    SettlementTransaction,
    SettlementStatus,
)
from verbalyze.telephony.payment_webhooks import (
    verify_razorpay_signature,
    verify_cashfree_signature,
    verify_upi_webhook_signature,
    parse_razorpay_webhook,
    parse_cashfree_webhook,
    parse_upi_callback,
)
from verbalyze.telephony.call_recorder import DualChannelCallRecorder
from verbalyze.telephony.compliance_qa import (
    ComplianceQAEngine,
    ComplianceQARegistry,
    QAScorecard,
    ComplianceStatus,
    CRMNotes,
    ComplianceInfraction,
    PillarScore,
)
from verbalyze.telephony.dtmf_engine import (
    GoertzelDetector,
    DTMFPad,
    RFC4733EventDecoder,
    RFC4733Event,
    encode_rfc4733_packet,
    decode_rfc4733_packet,
    DTMFToneGenerator,
    mask_digits,
    sanitize_sensitive_dict,
)
from verbalyze.telephony.ivr_tree import (
    IVRNode,
    IVRStateMachine,
    IVRTransitionResult,
    create_default_banking_ivr,
)
from verbalyze.telephony.circuit_breaker import (
    CircuitBreakerState,
    TrunkHealth,
    SIPResponseCategory,
    categorize_sip_code,
    calculate_itu_g107_mos,
    SIPCircuitBreaker,
    SIPCircuitBreakerConfig,
    TrunkQoS,
)
from verbalyze.telephony.trunk_router import (
    TelecomCircle,
    CarrierTrunk,
    MultiTrunkRouter,
    DispatchResult,
    NoAvailableTrunkError,
    detect_telecom_circle,
    create_default_indian_trunk_mesh,
)
from verbalyze.telephony.voice_biometrics import (
    BiometricStatus,
    SpoofType,
    AntiSpoofResult,
    BiometricVerificationResult,
    SpeakerProfile,
    AcousticFeatureExtractor,
    AntiSpoofingDetector,
    AntiSpoofDetector,
    BiometricVerificationEngine,
    SpeakerProfileRegistry,
)

__all__ = [
    "DispatchResult",
    "SIPTransferDispatcher",
    "TransferContext",
    "AdaptiveJitterBuffer",
    "JitterBufferPacket",
    "JitterBufferStats",
    "SupervisorManager",
    "CallSupervisorRecord",
    "WhisperMessage",
    "BrowserAudioSession",
    "WhatsAppGateway",
    "WhatsAppInteractiveTemplate",
    "WhatsAppButton",
    "SettlementLedger",
    "SettlementTransaction",
    "SettlementStatus",
    "verify_razorpay_signature",
    "verify_cashfree_signature",
    "verify_upi_webhook_signature",
    "parse_razorpay_webhook",
    "parse_cashfree_webhook",
    "parse_upi_callback",
    "DualChannelCallRecorder",
    "ComplianceQAEngine",
    "ComplianceQARegistry",
    "QAScorecard",
    "ComplianceStatus",
    "CRMNotes",
    "ComplianceInfraction",
    "PillarScore",
    "GoertzelDetector",
    "DTMFPad",
    "RFC4733EventDecoder",
    "RFC4733Event",
    "encode_rfc4733_packet",
    "decode_rfc4733_packet",
    "DTMFToneGenerator",
    "mask_digits",
    "sanitize_sensitive_dict",
    "IVRNode",
    "IVRStateMachine",
    "IVRTransitionResult",
    "create_default_banking_ivr",
    "CircuitBreakerState",
    "TrunkHealth",
    "SIPResponseCategory",
    "categorize_sip_code",
    "calculate_itu_g107_mos",
    "SIPCircuitBreaker",
    "SIPCircuitBreakerConfig",
    "TrunkQoS",
    "TelecomCircle",
    "CarrierTrunk",
    "MultiTrunkRouter",
    "NoAvailableTrunkError",
    "detect_telecom_circle",
    "create_default_indian_trunk_mesh",
    "BiometricStatus",
    "SpoofType",
    "AntiSpoofResult",
    "BiometricVerificationResult",
    "SpeakerProfile",
    "AcousticFeatureExtractor",
    "AntiSpoofingDetector",
    "AntiSpoofDetector",
    "BiometricVerificationEngine",
    "SpeakerProfileRegistry",
]


