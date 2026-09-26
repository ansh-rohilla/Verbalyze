"""
Verbalyze: Indic Voice AI & Synthetic Data Suite
Carrier-grade conversational voicebots, speech benchmarks, and pure-math telephony DSP engines.
"""

__version__ = "1.0.0"

from verbalyze.agent import (
    VoiceAgent,
    AudioEngine,
    HumanLikenessScorer,
    QualityGate,
    SovereignSTTEngine,
    LanguageIdentificationGate,
    UnifiedSentimentEngine,
)
from verbalyze.telephony import (
    CellularLineQualityClassifier,
    DualChannelDiarizer,
    AutomaticLevelController,
    AcousticWatermarker,
    BandwidthExpander,
    ComfortNoiseGenerator,
    IndicFormantEqualizer,
    PacketLossConcealer,
    AcousticEchoAndNoiseProcessor,
    AdaptiveTurnTakingManager,
    DualChannelCallRecorder,
    ComplianceQAEngine,
    MultiTrunkRouter,
    SIPCircuitBreaker,
    WhatsAppGateway,
    SettlementLedger,
    GoertzelDetector,
    IVRStateMachine,
    BiometricVerificationEngine,
    VoiceBoundaryPredictor,
)

__all__ = [
    "__version__",
    "VoiceAgent",
    "AudioEngine",
    "HumanLikenessScorer",
    "QualityGate",
    "SovereignSTTEngine",
    "LanguageIdentificationGate",
    "UnifiedSentimentEngine",
    "CellularLineQualityClassifier",
    "DualChannelDiarizer",
    "AutomaticLevelController",
    "AcousticWatermarker",
    "BandwidthExpander",
    "ComfortNoiseGenerator",
    "IndicFormantEqualizer",
    "PacketLossConcealer",
    "AcousticEchoAndNoiseProcessor",
    "AdaptiveTurnTakingManager",
    "DualChannelCallRecorder",
    "ComplianceQAEngine",
    "MultiTrunkRouter",
    "SIPCircuitBreaker",
    "WhatsAppGateway",
    "SettlementLedger",
    "GoertzelDetector",
    "IVRStateMachine",
    "BiometricVerificationEngine",
    "VoiceBoundaryPredictor",
]
