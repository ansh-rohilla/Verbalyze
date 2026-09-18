"""
verbalyze/agent/sentiment.py

Dual-Channel Sentiment & Dispute Detection Engine:
- Acoustic Agitation Scorer: RMS volume dynamics, energy variance, and pitch jitter.
- Lexical Dispute Classifier: Detects Indian payment disputes, legal threats, harassment claims, and supervisor requests.
- Composite Agitation Index & Thresholding: CALM, ELEVATED, AGITATED, CRITICAL.
"""

import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


class SentimentCategory(str, Enum):
    """Customer emotional agitation level."""
    CALM = "CALM"
    ELEVATED = "ELEVATED"
    AGITATED = "AGITATED"
    CRITICAL = "CRITICAL"


class DisputeType(str, Enum):
    """Categorization of customer dispute or escalation cause."""
    NONE = "NONE"
    PAYMENT_DISPUTE = "PAYMENT_DISPUTE"
    LEGAL_THREAT = "LEGAL_THREAT"
    HARASSMENT_COMPLAINT = "HARASSMENT_COMPLAINT"
    HUMAN_REQUEST = "HUMAN_REQUEST"
    WRONG_PERSON = "WRONG_PERSON"
    ABUSIVE_LANGUAGE = "ABUSIVE_LANGUAGE"


@dataclass
class SentimentResult:
    """Consolidated emotional and dispute classification result."""
    composite_agitation: float          # 0.00 to 1.00
    acoustic_score: float               # 0.00 to 1.00
    lexical_score: float                # 0.00 to 1.00
    category: SentimentCategory
    dispute_type: DisputeType
    detected_cues: List[str] = field(default_factory=list)
    transfer_recommended: bool = False
    deescalation_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_agitation": round(self.composite_agitation, 3),
            "acoustic_score": round(self.acoustic_score, 3),
            "lexical_score": round(self.lexical_score, 3),
            "category": self.category.value,
            "dispute_type": self.dispute_type.value,
            "detected_cues": self.detected_cues,
            "transfer_recommended": self.transfer_recommended,
            "deescalation_recommended": self.deescalation_recommended,
        }


# ---------------------------------------------------------------------------
# Lexical Dispute & Escalation Regex Patterns (Hindi, English, Hinglish)
# ---------------------------------------------------------------------------

PAYMENT_DISPUTE_PATTERNS = [
    r"already\s*(?:paid|done|given)",
    r"payment\s*(?:kar\s*diya|ho\s*gaya|done|bhej\s*diya|clear|de\s*diya)",
    r"paise?\s*(?:de\s*diya|de\s*diye|jama\s*kar\s*diya|jama\s*kiye|bhej\s*diya|bhej\s*diye|de\s*chuka|bhar\s*diya|bhar\s*chuka)",
    r"paise\s*(?:kat\s*gaye|cut\s*ho\s*gaye|nikal\s*gaye)",
    r"pehle\s*hi\s*(?:de\s*diya|pay\s*kar\s*diya|bhej\s*diya)",
    r"cheque\s*(?:diya\s*tha|clear\s*ho\s*gaya)",
    r"receipt",
    r"statement\s*(?:dekh|check)",
    r"utr\s*(?:number|no)",
    r"transaction\s*(?:id|successful)",
    r"wrong\s*(?:amount|calculation|interest|charges)",
    r"galat\s*(?:amount|byaaj|hisab|charges)",
    r"settlement\s*(?:ho\s*chuka|done)",
    r"noc\s*(?:mil\s*chuki|issued)",
    r"double\s*(?:debit|kata)",
]

LEGAL_THREAT_PATTERNS = [
    r"\bpolice\b",
    r"\bcourt\b",
    r"\blawyer\b",
    r"\bwakeel\b",
    r"\bvakeel\b",
    r"consumer\s*(?:court|forum)",
    r"rbi\s*(?:complaint|ombudsman)",
    r"lokpal",
    r"fir\s*(?:karunga|darj|file)",
    r"case\s*(?:karunga|karenge|file)",
    r"legal\s*(?:notice|action)",
    r"jail\s*bhejoonga",
    r"mental\s*(?:harassment|torture|stress)",
]

HARASSMENT_COMPLAINT_PATTERNS = [
    r"\bharass(?:ing)?\b",
    r"harass\s*kar\s*rahe",
    r"harassment",
    r"pareshan\s*kar\s*rahe",
    r"bar\s*bar\s*call",
    r"subah\s*se\s*(?:shaam|call)",
    r"threaten",
    r"dhamki\s*de\s*rahe",
    r"tameez\s*se\s*baat",
    r"gali\s*mat\s*do",
    r"stop\s*calling\s*me",
    r"don\'?t\s*call\s*(?:me\s*)?again",
]

HUMAN_REQUEST_PATTERNS = [
    r"manager\s*(?:se\s*baat|ko\s*phone|se\s*connect)",
    r"supervisor\s*(?:se\s*baat|ko\s*connect|se\s*connect)",
    r"senior\s*(?:se\s*baat|officer)",
    r"human\s*(?:agent|being|supervisor|manager|person)",
    r"asli\s*insaan",
    r"kisi\s*(?:bhi\s*)?insaan\s*se",
    r"(?:robot|bot|ai|machine)\s*se\s*nahi",
    r"talk\s*to\s*(?:a\s*)?(?:human|person|agent|representative|supervisor|manager)",
    r"connect\s*(?:me\s*)?to\s*(?:a\s*)?(?:human|supervisor|officer|manager)",
    r"(?:human|supervisor|manager)\s*(?:se\s*)?(?:connect\s*karo|baat\s*karao|transfer\s*karo)",
    r"transfer\s*(?:karo|to\s*human|to\s*supervisor)",
]

WRONG_PERSON_PATTERNS = [
    r"wrong\s*number",
    r"galat\s*number",
    r"galat\s*phone",
    r"main\s*wo\s*nahi\s*hoon",
    r"main\s*sharma\s*nahi",
    r"not\s*my\s*(?:loan|number|account)",
    r"don\'?t\s*know\s*any",
    r"kisi\s*aur\s*ka\s*number",
]

ABUSIVE_LANGUAGE_PATTERNS = [
    r"\bbakwas\b",
    r"\bbakwaas\b",
    r"\bfraud\b",
    r"\bchutiy",
    r"\bmadarch",
    r"\bbehench",
    r"\bkamina\b",
    r"\bkutte\b",
    r"\bbullshit\b",
    r"\bidiot\b",
    r"\bshut\s*up\b",
    r"\bloot\s*rahe\b",
]

ELEVATED_ANNOYANCE_PATTERNS = [
    r"bar\s*bar\s*(?:kyu\s*)?(?:phone|call|message)",
    r"baar\s*baar",
    r"roz\s*roz",
    r"kyu\s*(?:phone|call)\s*kar",
    r"thoda\s*time\s*(?:do|dijiye|chahiye)",
    r"itni\s*jaldi",
    r"pareshan\s*mat",
    r"shanti\s*rakho",
    r"dimaag\s*kharab",
    r"chillao\s*mat",
    r"wait\s*karo",
    r"annoying",
    r"why\s*are\s*you\s*calling",
    r"give\s*me\s*some\s*time",
]


class AcousticSentimentAnalyzer:
    """
    Evaluates customer vocal agitation from raw linear PCM frames:
    1. RMS energy level relative to standard speaking volume.
    2. Energy variance across frames (erratic shouting bursts).
    3. High-frequency spectral energy and zero-crossing rate volatility (pitch jitter).
    """

    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate

    def score(self, pcm_bytes: Optional[bytes], speech_duration_sec: float = 1.0) -> float:
        """
        Computes acoustic agitation score between 0.00 (calm) and 1.00 (extreme shouting/screeching).
        """
        if not pcm_bytes or len(pcm_bytes) < 160:
            return 0.20  # Neutral calm default when only transcript is available

        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        except Exception:
            return 0.20

        if len(samples) == 0:
            return 0.20

        # 1. RMS Energy Calculation
        rms = np.sqrt(np.mean(samples ** 2))
        # Normal telephone speaking level is around 800 - 2500 RMS. Shouting exceeds 6000 - 15000 RMS.
        normalized_rms = min(1.0, max(0.0, (rms - 1000.0) / 7000.0))

        # 2. Frame-level Energy Variance
        frame_size = int(self.sample_rate * 0.02)  # 20ms frames
        if len(samples) > frame_size * 2:
            num_frames = len(samples) // frame_size
            trimmed = samples[: num_frames * frame_size].reshape(num_frames, frame_size)
            frame_energies = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-6)
            energy_mean = np.mean(frame_energies)
            energy_std = np.std(frame_energies)
            coef_var = (energy_std / energy_mean) if energy_mean > 0 else 0.0
            variance_score = min(1.0, coef_var / 1.2)
        else:
            variance_score = 0.2

        # 3. Zero-Crossing Volatility (Proxy for pitch jitter and high-frequency shouting)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))
        # Typical voiced speech is 0.05 - 0.15; distressed/screeching shouting exceeds 0.25 - 0.40
        zc_score = min(1.0, max(0.0, (zero_crossings - 0.10) / 0.25))

        # Composite Acoustic Agitation: 50% RMS + 25% Variance + 25% High Frequency
        acoustic_score = (0.50 * normalized_rms) + (0.25 * variance_score) + (0.25 * zc_score)
        return float(np.clip(acoustic_score, 0.0, 1.0))


class LexicalDisputeClassifier:
    """
    Evaluates spoken transcript for emotional sentiment, financial disputes,
    regulatory threats, and human transfer requests.
    """

    def __init__(self):
        self._payment_regex = re.compile("|".join(PAYMENT_DISPUTE_PATTERNS), re.IGNORECASE)
        self._legal_regex = re.compile("|".join(LEGAL_THREAT_PATTERNS), re.IGNORECASE)
        self._harassment_regex = re.compile("|".join(HARASSMENT_COMPLAINT_PATTERNS), re.IGNORECASE)
        self._human_regex = re.compile("|".join(HUMAN_REQUEST_PATTERNS), re.IGNORECASE)
        self._wrong_person_regex = re.compile("|".join(WRONG_PERSON_PATTERNS), re.IGNORECASE)
        self._abusive_regex = re.compile("|".join(ABUSIVE_LANGUAGE_PATTERNS), re.IGNORECASE)
        self._annoyance_regex = re.compile("|".join(ELEVATED_ANNOYANCE_PATTERNS), re.IGNORECASE)

    def classify(self, transcript: str) -> Tuple[float, DisputeType, List[str]]:
        """
        Analyzes customer text for dispute categories and agitation score.
        Returns: (lexical_agitation_score, dispute_type, detected_cues)
        """
        text = transcript.strip()
        if not text:
            return 0.10, DisputeType.NONE, []

        cues = []
        score = 0.15
        dispute = DisputeType.NONE

        # 1. Check Legal Threats (Highest Priority)
        legal_matches = self._legal_regex.findall(text)
        if legal_matches:
            cues.extend([f"legal:{m}" for m in legal_matches])
            score = max(score, 0.90)
            dispute = DisputeType.LEGAL_THREAT

        # 2. Check Harassment Complaints
        harass_matches = self._harassment_regex.findall(text)
        if harass_matches:
            cues.extend([f"harassment:{m}" for m in harass_matches])
            score = max(score, 0.85)
            if dispute == DisputeType.NONE:
                dispute = DisputeType.HARASSMENT_COMPLAINT

        # 3. Check Abusive / Hostile Language
        abusive_matches = self._abusive_regex.findall(text)
        if abusive_matches:
            cues.extend([f"abusive:{m}" for m in abusive_matches])
            score = max(score, 0.82)
            if dispute == DisputeType.NONE:
                dispute = DisputeType.ABUSIVE_LANGUAGE

        # 4. Check Explicit Human / Supervisor Demand
        human_matches = self._human_regex.findall(text)
        if human_matches:
            cues.extend([f"human_demand:{m}" for m in human_matches])
            score = max(score, 0.75)
            if dispute == DisputeType.NONE:
                dispute = DisputeType.HUMAN_REQUEST

        # 5. Check Payment Disputes
        pay_matches = self._payment_regex.findall(text)
        if pay_matches:
            cues.extend([f"payment_dispute:{m}" for m in pay_matches])
            score = max(score, 0.65)
            if dispute == DisputeType.NONE:
                dispute = DisputeType.PAYMENT_DISPUTE

        # 6. Check Wrong Person
        wrong_matches = self._wrong_person_regex.findall(text)
        if wrong_matches:
            cues.extend([f"wrong_person:{m}" for m in wrong_matches])
            score = max(score, 0.60)
            if dispute == DisputeType.NONE:
                dispute = DisputeType.WRONG_PERSON

        # 7. Check Mild / Elevated Annoyance (No formal dispute, but customer is elevated)
        annoyance_matches = self._annoyance_regex.findall(text)
        if annoyance_matches:
            cues.extend([f"annoyance:{m}" for m in annoyance_matches])
            score = max(score, 0.50)

        # 8. Check Punctuation & Exclamation Density
        if text.count("!") >= 2 or (text.isupper() and len(text) > 10):
            score = min(1.0, score + 0.15)
            cues.append("exclamation_intensity")

        return float(min(1.0, score)), dispute, cues


class UnifiedSentimentEngine:
    """
    Combined multi-channel sentiment detection engine:
    Evaluates both acoustic signal dynamics and lexical semantics.
    """

    def __init__(self, sample_rate: int = 8000):
        self.acoustic_analyzer = AcousticSentimentAnalyzer(sample_rate=sample_rate)
        self.lexical_classifier = LexicalDisputeClassifier()

    def analyze(
        self,
        transcript: str = "",
        pcm_bytes: Optional[bytes] = None,
        speech_duration_sec: float = 1.0,
        text: Optional[str] = None,
    ) -> SentimentResult:
        """
        Executes unified sentiment classification across acoustic and lexical channels.
        """
        if text is not None and not transcript:
            transcript = text

        # Acoustic scoring
        acoustic_score = self.acoustic_analyzer.score(pcm_bytes, speech_duration_sec=speech_duration_sec)

        # Lexical scoring
        lexical_score, dispute_type, cues = self.lexical_classifier.classify(transcript)

        # Composite Agitation Index:
        # If PCM bytes are provided, weight 45% acoustic + 55% lexical.
        # If PCM is absent or minimal, weight 15% acoustic + 85% lexical.
        if pcm_bytes and len(pcm_bytes) >= 1600:
            composite = (0.45 * acoustic_score) + (0.55 * lexical_score)
        else:
            composite = (0.15 * acoustic_score) + (0.85 * lexical_score)

        # Critical override: explicit legal threat or harassment complaint immediately elevates score
        if dispute_type in (DisputeType.LEGAL_THREAT, DisputeType.HARASSMENT_COMPLAINT):
            composite = max(composite, 0.85)

        composite = float(np.clip(composite, 0.0, 1.0))

        # Categorize
        if composite >= 0.85:
            category = SentimentCategory.CRITICAL
            transfer_rec = True
            deesc_rec = True
        elif composite >= 0.65:
            category = SentimentCategory.AGITATED
            transfer_rec = True
            deesc_rec = True
        elif composite >= 0.40:
            category = SentimentCategory.ELEVATED
            transfer_rec = (dispute_type in (
                DisputeType.HUMAN_REQUEST,
                DisputeType.LEGAL_THREAT,
                DisputeType.PAYMENT_DISPUTE,
                DisputeType.HARASSMENT_COMPLAINT,
            ))
            deesc_rec = True
        else:
            category = SentimentCategory.CALM
            transfer_rec = False
            deesc_rec = False

        # Specific rule: if user explicitly demands a human, always recommend transfer
        if dispute_type == DisputeType.HUMAN_REQUEST:
            transfer_rec = True

        return SentimentResult(
            composite_agitation=composite,
            acoustic_score=acoustic_score,
            lexical_score=lexical_score,
            category=category,
            dispute_type=dispute_type,
            detected_cues=cues,
            transfer_recommended=transfer_rec,
            deescalation_recommended=deesc_rec,
        )
