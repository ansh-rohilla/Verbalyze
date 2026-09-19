"""
verbalyze/agent/lid_engine.py

Multi-Modal Language Identification (LID) & Code-Switching Engine:
- Acoustic LID: Fast mel-spectrogram language detection via faster-whisper.
- Lexical LID: Unicode script block distribution and Indic romanized lexicon classifiers.
- Multi-Lingual Code-Switching Detection: Hinglish, Gujlish, Tanglish, and Indic-English code-switching.
- Language Transition Detection: Emits dynamic adaptation recommendations for mid-call language switches.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


class ScriptType(str, Enum):
    """Unicode script classification for Indian multilingual text."""
    DEVANAGARI = "devanagari"    # Hindi, Marathi
    GUJARATI = "gujarati"        # Gujarati
    TAMIL = "tamil"              # Tamil
    TELUGU = "telugu"            # Telugu
    BENGALI = "bengali"          # Bengali, Assamese
    KANNADA = "kannada"          # Kannada
    MALAYALAM = "malayalam"      # Malayalam
    GURMUKHI = "gurmukhi"        # Punjabi
    ODIA = "odia"                # Odia
    ARABIC = "arabic"            # Urdu
    LATIN = "latin"              # English, Romanized Indic / Hinglish
    MIXED = "mixed"              # Multiple scripts present
    UNKNOWN = "unknown"


@dataclass
class LanguageIDResult:
    """Structured result of multi-modal language identification."""
    primary_language: str                       # e.g., 'hi', 'en', 'gu', 'ta', 'mr'
    confidence: float                          # 0.00 to 1.00
    is_code_switched: bool = False              # True if customer blends Indic + English (e.g. Hinglish)
    languages_detected: List[str] = field(default_factory=list)  # e.g. ['hi', 'en']
    script: ScriptType = ScriptType.LATIN
    language_switched: bool = False             # True if language switched from conversation baseline
    recommended_voice: Optional[str] = None     # Neural TTS voice to switch to (e.g. 'hi-IN-SwaraNeural')
    distribution: Dict[str, float] = field(default_factory=dict) # Language probability vector

    def __iter__(self):
        yield self.primary_language
        yield self.confidence
        yield self.is_code_switched
        yield self.languages_detected
        yield self.script
        yield self.distribution

    def __getitem__(self, index):
        items = [
            self.primary_language,
            self.confidence,
            self.is_code_switched,
            self.languages_detected,
            self.script,
            self.distribution,
        ]
        return items[index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_language": self.primary_language,
            "confidence": round(self.confidence, 3),
            "is_code_switched": self.is_code_switched,
            "languages_detected": self.languages_detected,
            "script": self.script.value,
            "language_switched": self.language_switched,
            "recommended_voice": self.recommended_voice,
            "distribution": {k: round(v, 3) for k, v in self.distribution.items()},
        }


# ---------------------------------------------------------------------------
# Lexical Marker Dictionaries for Romanized Indic Dialects
# ---------------------------------------------------------------------------

HINDI_ROMAN_MARKERS = {
    "haan", "han", "hai", "hain", "hoon", "hun", "tha", "thi", "kya", "kyu", "kyun",
    "kaise", "kab", "kahan", "kitna", "maine", "mujhe", "mera", "meri", "mere", "aap", "aapka",
    "aapki", "aapke", "aapko", "tum", "tumhara", "kar", "karo", "karna", "kariye", "karta",
    "karti", "karte", "diya", "diye", "denge", "dunga", "doonga", "raha", "rahe", "rahi",
    "nahi", "nahin", "theek", "thik", "bhi", "bahut", "aaj", "kal", "se", "ko", "par",
    "mein", "aur", "lekin", "magar", "agar", "paise", "paisa", "rupaye", "baat", "bol",
    "batao", "dekh", "dijiye", "bhejo", "bheja", "bhej", "jama", "khata", "namaskar",
    "namaste", "dhanyawad", "alvida", "accha", "achha"
}

MARATHI_ROMAN_MARKERS = {
    "aahe", "ahet", "nahi", "hota", "hoti", "hote", "mala", "tumhala", "aamhi",
    "kay", "kasa", "kashi", "kadhi", "dya", "kara", "karto", "karte", "kela",
    "keli", "paise", "aata", "udya", "pan", "aani", "tar", "jhal", "jhala", "jhali"
}

GUJARATI_ROMAN_MARKERS = {
    "chhe", "chho", "chhu", "nathi", "hatun", "hata", "hati", "mane", "maru",
    "tamaru", "tame", "aavse", "karjo", "aapjo", "pan", "ane", "kem", "su",
    "shu", "paisa", "pashi", "kaley", "aaje", "bol", "bolo", "haji"
}

TAMIL_ROMAN_MARKERS = {
    "illai", "aam", "ennoda", "unga", "ungalukku", "enakku", "irukku", "panren",
    "panniten", "pannunga", "sollunga", "theriyum", "varum", "kudunga", "eppadi",
    "yen", "ippo", "naalai", "panam", "kaasu"
}

TELUGU_ROMAN_MARKERS = {
    "undi", "ledu", "kadu", "avunu", "naaku", "meeku", "cheyandi", "chesanu",
    "chesta", "eppudu", "ela", "enduku", "dabbulu", "ivvandi", "vastanu",
    "repu", "eroju", "cheppandi"
}

ENGLISH_COMMON_MARKERS = {
    "the", "is", "are", "was", "were", "have", "has", "had", "will", "would",
    "can", "could", "should", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "their", "our", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "about", "payment", "paid", "already",
    "yesterday", "tomorrow", "today", "receipt", "account", "bank", "branch",
    "loan", "emi", "statement", "clear", "cleared", "check", "please", "call",
    "not", "no", "yes", "ok", "okay", "transferred", "transfer", "officer", "manager",
    "make", "send", "give", "afternoon", "morning", "evening", "speak", "talk", "pay",
    "do", "me"
}

# Standard Neural Voice Mapping across Indic Languages
NEURAL_VOICE_MAP = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural",
    "gu": "gu-IN-DhwaniNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "mr": "mr-IN-AarohiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-OjasNeural",
    "as": "bn-IN-TanishaaNeural",
    "or": "hi-IN-SwaraNeural",
    "ur": "ur-IN-GulNeural",
}


class LexicalLIDClassifier:
    """
    Analyzes written or transcribed customer utterances:
    1. Inspects Unicode script blocks to distinguish native scripts (Devanagari, Gujarati, Tamil, etc.).
    2. Identifies Romanized Indic speech and distinguishes English vs Hinglish vs Gujlish.
    3. Detects code-switching patterns where English loanwords blend with Indic syntax.
    """

    def classify(self, text: str) -> LanguageIDResult:
        """
        Classifies transcript into primary language and code-switch indicators.
        Returns LanguageIDResult (which unpacks into 6 elements for backwards-compatibility).
        """
        stripped = text.strip()
        if not stripped:
            return LanguageIDResult(primary_language="hi", confidence=0.50, is_code_switched=False, languages_detected=["hi"], script=ScriptType.UNKNOWN, distribution={"hi": 0.50})

        script_counts: Dict[ScriptType, int] = {}
        for char in stripped:
            if char.isspace() or unicodedata.category(char).startswith("P"):
                continue
            script = self._detect_char_script(char)
            script_counts[script] = script_counts.get(script, 0) + 1

        total_chars = sum(script_counts.values()) or 1

        # 1. Native Indic Scripts (Non-Latin)
        indic_scripts = [s for s in script_counts.keys() if s not in (ScriptType.LATIN, ScriptType.UNKNOWN)]
        if indic_scripts:
            # Check if text is predominantly a native script
            primary_script = max(script_counts.keys(), key=lambda s: script_counts[s])

            if primary_script == ScriptType.DEVANAGARI:
                # Distinguish Hindi vs Marathi in Devanagari script
                lang = self._distinguish_devanagari(stripped)
                is_mixed = script_counts.get(ScriptType.LATIN, 0) / total_chars > 0.15
                dist = {lang: 0.90, "en": 0.10} if is_mixed else {lang: 0.95}
                return LanguageIDResult(primary_language=lang, confidence=0.95, is_code_switched=is_mixed, languages_detected=[lang, "en"] if is_mixed else [lang], script=ScriptType.DEVANAGARI, distribution=dist)

            elif primary_script == ScriptType.GUJARATI:
                is_mixed = script_counts.get(ScriptType.LATIN, 0) / total_chars > 0.15
                return LanguageIDResult(primary_language="gu", confidence=0.95, is_code_switched=is_mixed, languages_detected=["gu", "en"] if is_mixed else ["gu"], script=ScriptType.GUJARATI, distribution={"gu": 0.95})

            elif primary_script == ScriptType.TAMIL:
                is_mixed = script_counts.get(ScriptType.LATIN, 0) / total_chars > 0.15
                return LanguageIDResult(primary_language="ta", confidence=0.95, is_code_switched=is_mixed, languages_detected=["ta", "en"] if is_mixed else ["ta"], script=ScriptType.TAMIL, distribution={"ta": 0.95})

            elif primary_script == ScriptType.TELUGU:
                is_mixed = script_counts.get(ScriptType.LATIN, 0) / total_chars > 0.15
                return LanguageIDResult(primary_language="te", confidence=0.95, is_code_switched=is_mixed, languages_detected=["te", "en"] if is_mixed else ["te"], script=ScriptType.TELUGU, distribution={"te": 0.95})

            elif primary_script == ScriptType.BENGALI:
                return LanguageIDResult(primary_language="bn", confidence=0.95, is_code_switched=False, languages_detected=["bn"], script=ScriptType.BENGALI, distribution={"bn": 0.95})

            elif primary_script == ScriptType.KANNADA:
                return LanguageIDResult(primary_language="kn", confidence=0.95, is_code_switched=False, languages_detected=["kn"], script=ScriptType.KANNADA, distribution={"kn": 0.95})

            elif primary_script == ScriptType.MALAYALAM:
                return LanguageIDResult(primary_language="ml", confidence=0.95, is_code_switched=False, languages_detected=["ml"], script=ScriptType.MALAYALAM, distribution={"ml": 0.95})

            elif primary_script == ScriptType.GURMUKHI:
                return LanguageIDResult(primary_language="pa", confidence=0.95, is_code_switched=False, languages_detected=["pa"], script=ScriptType.GURMUKHI, distribution={"pa": 0.95})

            elif primary_script == ScriptType.ODIA:
                return LanguageIDResult(primary_language="or", confidence=0.95, is_code_switched=False, languages_detected=["or"], script=ScriptType.ODIA, distribution={"or": 0.95})

            elif primary_script == ScriptType.ARABIC:
                return LanguageIDResult(primary_language="ur", confidence=0.95, is_code_switched=False, languages_detected=["ur"], script=ScriptType.ARABIC, distribution={"ur": 0.95})

        # 2. Latin Script: Classify English vs Hinglish vs Romanized Indic
        words = re.findall(r"[a-zA-Z]+", stripped.lower())
        if not words:
            return LanguageIDResult(primary_language="en", confidence=0.50, is_code_switched=False, languages_detected=["en"], script=ScriptType.LATIN, distribution={"en": 0.50})

        hindi_hits = sum(1 for w in words if w in HINDI_ROMAN_MARKERS)
        marathi_hits = sum(1 for w in words if w in MARATHI_ROMAN_MARKERS)
        gujarati_hits = sum(1 for w in words if w in GUJARATI_ROMAN_MARKERS)
        tamil_hits = sum(1 for w in words if w in TAMIL_ROMAN_MARKERS)
        telugu_hits = sum(1 for w in words if w in TELUGU_ROMAN_MARKERS)
        english_hits = sum(1 for w in words if w in ENGLISH_COMMON_MARKERS)

        total_words = len(words)

        total_indic_hits = hindi_hits + marathi_hits + gujarati_hits + tamil_hits + telugu_hits

        # Pure English: zero Indic markers or overwhelmingly English syntax
        if total_indic_hits == 0:
            en_prob = min(0.98, max(0.85, 0.70 + 0.30 * (english_hits / total_words if total_words else 1.0)))
            return LanguageIDResult(
                primary_language="en",
                confidence=en_prob,
                is_code_switched=False,
                languages_detected=["en"],
                script=ScriptType.LATIN,
                distribution={"en": en_prob, "hi": 1.0 - en_prob},
            )

        # Hinglish Detection (Hindi markers present in Latin text)
        if hindi_hits > 0 and (hindi_hits >= marathi_hits and hindi_hits >= gujarati_hits):
            # If English hits overwhelmingly dominate and Hindi is <= 1 ambiguous word
            if english_hits >= 3 and english_hits > hindi_hits * 2:
                en_prob = min(0.95, 0.50 + (0.45 * (english_hits / total_words)))
                return LanguageIDResult(
                    primary_language="en",
                    confidence=en_prob,
                    is_code_switched=True,
                    languages_detected=["en", "hi"],
                    script=ScriptType.LATIN,
                    distribution={"en": en_prob, "hi": 1.0 - en_prob},
                )
            is_code_switched = (english_hits > 0 or hindi_hits < total_words)
            hi_prob = min(0.95, 0.40 + (0.50 * (hindi_hits / total_words)))
            en_prob = max(0.05, 1.0 - hi_prob)
            langs = ["hi", "en"] if is_code_switched else ["hi"]
            return LanguageIDResult(
                primary_language="hi",
                confidence=hi_prob,
                is_code_switched=is_code_switched,
                languages_detected=langs,
                script=ScriptType.LATIN,
                distribution={"hi": hi_prob, "en": en_prob},
            )

        if gujarati_hits > 0 and gujarati_hits >= hindi_hits:
            is_code_switched = (english_hits > 0 or gujarati_hits < total_words)
            gu_prob = min(0.95, 0.40 + (0.50 * (gujarati_hits / total_words)))
            return LanguageIDResult(
                primary_language="gu",
                confidence=gu_prob,
                is_code_switched=is_code_switched,
                languages_detected=["gu", "en"] if is_code_switched else ["gu"],
                script=ScriptType.LATIN,
                distribution={"gu": gu_prob, "en": 1.0 - gu_prob},
            )

        if marathi_hits > 0:
            is_code_switched = (english_hits > 0 or marathi_hits < total_words)
            mr_prob = min(0.95, 0.40 + (0.50 * (marathi_hits / total_words)))
            return LanguageIDResult(
                primary_language="mr",
                confidence=mr_prob,
                is_code_switched=is_code_switched,
                languages_detected=["mr", "en"] if is_code_switched else ["mr"],
                script=ScriptType.LATIN,
                distribution={"mr": mr_prob, "en": 1.0 - mr_prob},
            )

        if tamil_hits > 0:
            return LanguageIDResult(
                primary_language="ta",
                confidence=0.85,
                is_code_switched=True,
                languages_detected=["ta", "en"],
                script=ScriptType.LATIN,
                distribution={"ta": 0.85, "en": 0.15},
            )

        if telugu_hits > 0:
            return LanguageIDResult(
                primary_language="te",
                confidence=0.85,
                is_code_switched=True,
                languages_detected=["te", "en"],
                script=ScriptType.LATIN,
                distribution={"te": 0.85, "en": 0.15},
            )

        # Default Latin: Pure English
        en_prob = min(0.98, max(0.80, english_hits / total_words if total_words else 0.80))
        return LanguageIDResult(
            primary_language="en",
            confidence=en_prob,
            is_code_switched=False,
            languages_detected=["en"],
            script=ScriptType.LATIN,
            distribution={"en": en_prob, "hi": 1.0 - en_prob},
        )

    def _detect_char_script(self, char: str) -> ScriptType:
        """Determines Unicode script category for a given character."""
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            return ScriptType.DEVANAGARI
        elif 0x0A80 <= cp <= 0x0AFF:
            return ScriptType.GUJARATI
        elif 0x0B80 <= cp <= 0x0BFF:
            return ScriptType.TAMIL
        elif 0x0C00 <= cp <= 0x0C7F:
            return ScriptType.TELUGU
        elif 0x0980 <= cp <= 0x09FF:
            return ScriptType.BENGALI
        elif 0x0C80 <= cp <= 0x0CFF:
            return ScriptType.KANNADA
        elif 0x0D00 <= cp <= 0x0D7F:
            return ScriptType.MALAYALAM
        elif 0x0A00 <= cp <= 0x0A7F:
            return ScriptType.GURMUKHI
        elif 0x0B00 <= cp <= 0x0B7F:
            return ScriptType.ODIA
        elif (0x0600 <= cp <= 0x06FF) or (0xFB50 <= cp <= 0xFDFF):
            return ScriptType.ARABIC
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            return ScriptType.LATIN
        return ScriptType.UNKNOWN

    def _distinguish_devanagari(self, text: str) -> str:
        """Distinguishes Marathi from Hindi in Devanagari script."""
        marathi_markers = {
            "आहे", "नाही", "काय", "हो", "करा", "पाहिजे", "झाले", "झाला", "झाली", "केले",
            "आहेत", "आम्ही", "तुम्ही", "त्यांना", "मला", "तुला", "मी", "उद्या", "आणि",
            "नक्की", "देईन", "घेईन", "पाठवेन", "पावती", "होईल", "करणार", "कसा", "कशी", "कधी"
        }
        words = set(re.findall(r"[\u0900-\u097F]+", text))
        if words.intersection(marathi_markers):
            return "mr"
        return "hi"

    def detect_script(self, text: str) -> Tuple[ScriptType, Dict[str, float]]:
        """Detects the primary script and distribution of scripts in the text."""
        script_counts: Dict[str, int] = {}
        for char in text.strip():
            if char.isspace() or unicodedata.category(char).startswith("P"):
                continue
            script = self._detect_char_script(char)
            script_counts[script.value] = script_counts.get(script.value, 0) + 1
        if not script_counts:
            return ScriptType.UNKNOWN, {}
        total = sum(script_counts.values()) or 1
        dist = {k: v / total for k, v in script_counts.items()}
        primary_val = max(script_counts.keys(), key=lambda k: script_counts[k])
        return ScriptType(primary_val), dist


class AcousticLIDClassifier:
    """
    Evaluates acoustic mel-spectrograms from raw PCM audio using faster-whisper's
    built-in detect_language() method with sub-30ms inference.
    """

    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate

    def detect_language(
        self,
        pcm_bytes: Optional[bytes],
        sample_rate: Optional[int] = None,
        whisper_model: Any = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Convenience alias for classify method."""
        if sample_rate and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
        return self.classify(pcm_bytes, whisper_model=whisper_model)

    def classify(self, pcm_bytes: Optional[bytes], whisper_model: Any = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Detects spoken language from linear PCM bytes.
        Returns: (primary_lang, confidence, language_probabilities)
        """
        if not pcm_bytes or len(pcm_bytes) < 1600 or whisper_model is None:
            return "hi", 0.50, {"hi": 0.50, "en": 0.50}

        try:
            # Upsample 8kHz to 16kHz if needed
            if self.sample_rate != 16000:
                import audioop
                pcm_16k = audioop.ratecv(pcm_bytes, 2, 1, self.sample_rate, 16000, None)[0]
            else:
                pcm_16k = pcm_bytes

            audio_np = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0

            # Run whisper detect_language
            detected_lang, prob, all_probs = whisper_model.detect_language(audio_np)
            prob_dict = {lang: float(p) for lang, p in all_probs[:5]}

            return detected_lang, float(prob), prob_dict
        except Exception:
            return "hi", 0.50, {"hi": 0.50, "en": 0.50}


class LanguageIdentificationGate:
    """
    Unified Multi-Modal Language Identification Gate:
    Fuses acoustic and lexical classification to decide:
    1. Primary language of the spoken turn.
    2. Code-switching presence (e.g. Hinglish, Gujlish).
    3. Whether caller has transitioned languages mid-conversation.
    4. Recommended neural voice adaptation for speech synthesis.
    """

    def __init__(
        self,
        switch_confidence_threshold: float = 0.65,
        default_language: str = "hi",
        min_switch_confidence: Optional[float] = None,
        hysteresis_turns: int = 1,
    ):
        self.switch_threshold = min_switch_confidence if min_switch_confidence is not None else switch_confidence_threshold
        self.default_language = default_language
        self.hysteresis_turns = hysteresis_turns
        self.pending_switch_lang: Optional[str] = None
        self.pending_switch_count: int = 0
        self.lexical_classifier = LexicalLIDClassifier()
        self.acoustic_classifier = AcousticLIDClassifier()

    def identify(
        self,
        transcript: str = "",
        pcm_bytes: Optional[bytes] = None,
        current_language: Optional[str] = None,
        whisper_model: Any = None,
    ) -> LanguageIDResult:
        """
        Executes unified multi-modal Language Identification.
        """
        baseline_lang = current_language or self.default_language

        # 1. Lexical LID
        lex_lang, lex_conf, is_code_switched, detected_langs, script, lex_dist = (
            self.lexical_classifier.classify(transcript)
        )

        # 2. Acoustic LID (if audio and model available)
        ac_lang, ac_conf, ac_dist = "hi", 0.50, {}
        has_acoustic = False
        if pcm_bytes and len(pcm_bytes) >= 3200 and whisper_model is not None:
            ac_lang, ac_conf, ac_dist = self.acoustic_classifier.classify(pcm_bytes, whisper_model)
            has_acoustic = True

        # 3. Fuse Signals
        if has_acoustic and transcript:
            # If native script is detected with high confidence, trust native script
            if script not in (ScriptType.LATIN, ScriptType.UNKNOWN) and lex_conf >= 0.85:
                final_lang = lex_lang
                final_conf = lex_conf
            elif ac_lang == lex_lang:
                final_lang = lex_lang
                final_conf = min(0.99, max(ac_conf, lex_conf) + 0.05)
            else:
                # Discrepancy: check if code-switching or trust higher confidence
                if is_code_switched:
                    final_lang = lex_lang
                    final_conf = lex_conf
                elif ac_conf > lex_conf:
                    final_lang = ac_lang
                    final_conf = ac_conf
                else:
                    final_lang = lex_lang
                    final_conf = lex_conf
        elif transcript:
            final_lang = lex_lang
            final_conf = lex_conf
        elif has_acoustic:
            final_lang = ac_lang
            final_conf = ac_conf
        else:
            final_lang = baseline_lang
            final_conf = 0.50

        # Check Language Switch with hysteresis:
        language_switched = False
        if final_lang != baseline_lang:
            if final_conf >= self.switch_threshold:
                if self.hysteresis_turns <= 1 or final_conf >= 0.90:
                    language_switched = True
                    self.pending_switch_lang = None
                    self.pending_switch_count = 0
                else:
                    if self.pending_switch_lang == final_lang:
                        self.pending_switch_count += 1
                        if self.pending_switch_count >= self.hysteresis_turns:
                            language_switched = True
                            self.pending_switch_lang = None
                            self.pending_switch_count = 0
                    else:
                        self.pending_switch_lang = final_lang
                        self.pending_switch_count = 1
        else:
            self.pending_switch_lang = None
            self.pending_switch_count = 0

        # Select recommended neural voice
        recommended_voice = NEURAL_VOICE_MAP.get(final_lang, NEURAL_VOICE_MAP["hi"])

        # Combined distribution
        combined_dist = {final_lang: final_conf}
        for k, v in lex_dist.items():
            if k not in combined_dist:
                combined_dist[k] = v * 0.5

        return LanguageIDResult(
            primary_language=final_lang,
            confidence=final_conf,
            is_code_switched=is_code_switched,
            languages_detected=detected_langs,
            script=script,
            language_switched=language_switched,
            recommended_voice=recommended_voice,
            distribution=combined_dist,
        )
