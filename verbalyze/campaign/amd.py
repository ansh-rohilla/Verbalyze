"""
verbalyze/campaign/amd.py

Answering Machine Detection (AMD) Engine:
- Dual-stage classification: acoustic cadence analysis + fast lexical announcement detection.
- Distinguishes Human vs Voicemail vs Indian Telecom Operator Announcements vs Silence within 800ms - 1500ms.
- Spectral energy beep tone detection for voicemail recording signals.
"""

import re
import time
import math
from typing import Optional, Tuple, List, Dict, Any
from verbalyze.campaign.models import AMDDecision, AMDResult


# Common telecom network announcements across Airtel, Jio, Vodafone Idea, BSNL, MTNL
OPERATOR_ANNOUNCEMENT_PATTERNS = [
    # Hindi / Hinglish Operator Phrases
    r"switched\s*off",
    r"switch\s*off",
    r"switch\s*off\s*hai",
    r"vyast\s*hai",
    r"dusri\s*call\s*par\s*vyast",
    r"doosri\s*call",
    r"pahunch\s*se\s*bahar",
    r"kripya\s*kuch\s*samay\s*baad",
    r"kripya\s*thodi\s*der\s*baad",
    r"simit\s*dayre",
    r"network\s*coverage",
    r"out\s*of\s*reach",
    r"coverage\s*area",
    r"upalabdha\s*nahi\s*hai",
    r"sewa\s*mein\s*nahi\s*hai",
    r"invalid\s*number",
    r"number\s*exist\s*nahi",
    r"paryapt\s*balance",

    # English Indian Telecom Phrases
    r"number\s*(?:you\s*have\s*)?dialed\s*is\s*(?:currently\s*)?switched\s*off",
    r"number\s*(?:you\s*have\s*)?dialed\s*is\s*(?:currently\s*)?busy",
    r"person\s*(?:you\s*are\s*calling\s*)?is\s*speaking\s*to\s*someone\s*else",
    r"currently\s*out\s*of\s*coverage",
    r"not\s*reachable",
    r"subscriber\s*(?:is\s*)?not\s*reachable",
    r"number\s*does\s*not\s*exist",
    r"temporarily\s*out\s*of\s*service",
    r"call\s*cannot\s*be\s*completed",
    r"all\s*lines\s*are\s*(?:currently\s*)?busy",
]

# Voicemail Greetings & Prompts
VOICEMAIL_PATTERNS = [
    r"leave\s*(?:a|your)?\s*message",
    r"after\s*(?:the)?\s*(?:beep|tone)",
    r"record\s*your\s*message",
    r"voicemail",
    r"voice\s*mail",
    r"mailbox\s*is\s*full",
    r"press\s*(?:one|1|star|\*)",
    r"welcome\s*to\s*(?:the\s*)?voicemail",
    r"not\s*available\s*to\s*take\s*your\s*call",
    r"cannot\s*take\s*your\s*call\s*right\s*now",
]

# Human conversational answer patterns (short, turn-yielding)
HUMAN_ANSWER_PATTERNS = [
    r"^(?:hello|namaste|haan|ha|ji|yes|haan\s*ji|hello\s*kaun|kaun\s*bol\s*rahe\s*hain|yes\s*speaking)[\.\?!]?$",
    r"^(?:hello\s*,?\s*)?(?:sharma\s*ji|bol\s*raha\s*hoon|bol\s*rahi\s*hoon)[\.\?!]?$",
]


class AMDClassifier:
    """
    Real-time Answering Machine & Telecom Announcement Classifier.
    """

    def __init__(
        self,
        min_human_duration_sec: float = 0.3,
        max_human_duration_sec: float = 2.2,
        min_machine_duration_sec: float = 2.6,
        min_pause_after_burst_sec: float = 0.7,
    ):
        self.min_human_duration = min_human_duration_sec
        self.max_human_duration = max_human_duration_sec
        self.min_machine_duration = min_machine_duration_sec
        self.min_pause_after_burst = min_pause_after_burst_sec

        # Compile regexes
        self._operator_regex = re.compile(
            "|".join(OPERATOR_ANNOUNCEMENT_PATTERNS), re.IGNORECASE
        )
        self._voicemail_regex = re.compile(
            "|".join(VOICEMAIL_PATTERNS), re.IGNORECASE
        )
        self._human_regex = re.compile(
            "|".join(HUMAN_ANSWER_PATTERNS), re.IGNORECASE
        )

    def analyze_transcript(self, transcript: str) -> Optional[Tuple[AMDDecision, float, str, List[str]]]:
        """
        Fast lexical classification of early speech turn transcript.
        Returns (decision, confidence, reason, detected_phrases) or None if inconclusive.
        """
        clean_text = transcript.strip()
        if not clean_text:
            return None

        # 1. Check for Operator Announcements
        op_matches = self._operator_regex.findall(clean_text)
        if op_matches:
            return (
                AMDDecision.OPERATOR_ANNOUNCEMENT,
                0.96,
                f"Matched telecom operator announcement pattern: '{op_matches[0]}'",
                op_matches,
            )

        # 2. Check for Voicemail Greetings
        vm_matches = self._voicemail_regex.findall(clean_text)
        if vm_matches:
            return (
                AMDDecision.MACHINE_VOICEMAIL,
                0.95,
                f"Matched voicemail greeting pattern: '{vm_matches[0]}'",
                vm_matches,
            )

        # 3. Check for Short Human Greetings
        if self._human_regex.match(clean_text):
            return (
                AMDDecision.HUMAN_ANSWERED,
                0.92,
                f"Matched natural human greeting: '{clean_text}'",
                [],
            )

        return None

    def analyze_cadence(
        self,
        speech_duration_sec: float,
        silence_after_burst_sec: float = 0.0,
        silence_ratio: float = 0.0,
    ) -> Tuple[AMDDecision, float, str]:
        """
        Evaluates acoustic timing:
        - Human: Short initial burst (0.3s - 2.2s) followed by pause (>0.7s) listening for agent.
        - Voicemail: Long uninterrupted speech (>2.6s) with low pause.
        - Silence: Dead air / immediate hangup (<0.2s speech).
        """
        if speech_duration_sec < 0.2:
            return (
                AMDDecision.SILENCE_TIMEOUT,
                0.88,
                f"Insufficient speech burst ({speech_duration_sec:.2f}s) - dead air detected.",
            )

        if speech_duration_sec >= self.min_machine_duration:
            return (
                AMDDecision.MACHINE_VOICEMAIL,
                0.89,
                f"Extended uninterrupted speech burst ({speech_duration_sec:.2f}s >= {self.min_machine_duration:.2f}s) typical of IVR/voicemail.",
            )

        if self.min_human_duration <= speech_duration_sec <= self.max_human_duration:
            if silence_after_burst_sec >= self.min_pause_after_burst or silence_ratio >= 0.25:
                return (
                    AMDDecision.HUMAN_ANSWERED,
                    0.90,
                    f"Typical human speech burst ({speech_duration_sec:.2f}s) followed by conversational pause ({silence_after_burst_sec:.2f}s).",
                )
            return (
                AMDDecision.HUMAN_ANSWERED,
                0.78,
                f"Short speech burst ({speech_duration_sec:.2f}s) within human greeting boundaries.",
            )

        return (
            AMDDecision.UNKNOWN,
            0.50,
            f"Ambiguous cadence duration ({speech_duration_sec:.2f}s).",
        )

    def detect_beep_tone(self, pcm_bytes: bytes, sample_rate: int = 8000) -> bool:
        """
        Detects sustained sinusoidal beep tone typical of voicemail prompts (425 Hz or 1000 Hz).
        Uses Goertzel algorithm / spectral energy concentration ratio.
        """
        if len(pcm_bytes) < sample_rate // 2:  # Need at least 250ms of audio
            return False

        import array
        samples = array.array('h', pcm_bytes)
        num_samples = len(samples)
        if num_samples == 0:
            return False

        # Target frequencies: 425Hz (Indian dial tone/network) and 1000Hz (Voicemail beep)
        target_freqs = [425.0, 1000.0]

        total_energy = sum(float(s) * float(s) for s in samples)
        if total_energy < 1e5:
            return False  # Dead air

        for freq in target_freqs:
            # Goertzel algorithm for single frequency power
            k = int(0.5 + ((num_samples * freq) / sample_rate))
            omega = (2.0 * math.pi * k) / num_samples
            coeff = 2.0 * math.cos(omega)
            q0, q1, q2 = 0.0, 0.0, 0.0

            for sample in samples:
                q0 = coeff * q1 - q2 + float(sample)
                q2 = q1
                q1 = q0

            power = q1 * q1 + q2 * q2 - q1 * q2 * coeff
            ratio = power / total_energy
            if ratio > 0.35:  # Significant spectral concentration at target tone
                return True

        return False

    def classify(
        self,
        audio_duration_sec: float = 1.0,
        transcript: str = "",
        silence_after_burst_sec: float = 0.0,
        silence_ratio: float = 0.0,
        pcm_bytes: Optional[bytes] = None,
        sample_rate: int = 8000,
    ) -> AMDResult:
        """
        Unified multi-modal classification pipeline:
        1. Transcript lexical analysis (highest precision if words transcribed)
        2. Beep tone spectral detection (detects voicemail beep signals)
        3. Acoustic cadence & pause timing analysis
        """
        t0 = time.perf_counter()

        # Stage 1: Lexical Transcript Matching
        if transcript.strip():
            lexical_res = self.analyze_transcript(transcript)
            if lexical_res is not None:
                decision, conf, reason, detected = lexical_res
                latency = (time.perf_counter() - t0) * 1000.0
                return AMDResult(
                    decision=decision,
                    confidence=conf,
                    reason=reason,
                    latency_ms=latency,
                    detected_phrases=detected,
                )

        # Stage 2: Beep Tone Detection
        if pcm_bytes and self.detect_beep_tone(pcm_bytes, sample_rate=sample_rate):
            latency = (time.perf_counter() - t0) * 1000.0
            return AMDResult(
                decision=AMDDecision.MACHINE_VOICEMAIL,
                confidence=0.94,
                reason="Detected sustained sinusoidal beep tone characteristic of voicemail prompt.",
                latency_ms=latency,
                detected_phrases=["[BEEP_TONE]"],
            )

        # Stage 3: Acoustic Cadence Analysis
        cadence_decision, cadence_conf, cadence_reason = self.analyze_cadence(
            speech_duration_sec=audio_duration_sec,
            silence_after_burst_sec=silence_after_burst_sec,
            silence_ratio=silence_ratio,
        )

        latency = (time.perf_counter() - t0) * 1000.0
        return AMDResult(
            decision=cadence_decision,
            confidence=cadence_conf,
            reason=cadence_reason,
            latency_ms=latency,
            detected_phrases=[],
        )
