#!/usr/bin/env python3
"""
scripts/test_lid_and_code_switching.py

Verification suite for Multi-Lingual Code-Switching STT & Language Identification (LID) Gate:
1. Unicode script classification across 11 scripts (Devanagari, Latin, Gujarati, Tamil, Telugu, etc.)
2. Lexical LID classification (Pure Hindi, English, Hinglish, Gujlish, Tanglish, Marathi, Tamil, Telugu)
3. Code-switching detection (Indic romanized text + English loanwords)
4. STT code-switch prompt conditioning (INDIC_CODE_SWITCH_PROMPTS across 12 languages)
5. Acoustic LID evaluation via WhisperModel / SovereignSTTEngine
6. LanguageIdentificationGate multi-modal fusion and transition hysteresis
7. Dynamic VoiceAgent mid-turn voice/language adaptation
8. Campaign CDR and summary language breakdown tracking
9. FastAPI /telephony/lid endpoint verification
"""

import sys
import os
import math
import asyncio
from typing import Dict, Any

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verbalyze.agent.lid_engine import (
    ScriptType,
    LanguageIDResult,
    LexicalLIDClassifier,
    AcousticLIDClassifier,
    LanguageIdentificationGate,
)
from verbalyze.agent.stt_engine import (
    SovereignSTTEngine,
    INDIC_CODE_SWITCH_PROMPTS,
)
from verbalyze.agent.voice_bot import VoiceAgent
from verbalyze.campaign.models import (
    CallDetailRecord,
    CampaignSummary,
    CallDisposition,
    Lead,
)
from verbalyze.campaign.dialer import CampaignDialer, CampaignConfig


def synthesize_test_pcm(duration_sec: float = 1.0, freq_hz: float = 440.0, sample_rate: int = 8000) -> bytes:
    """Generates synthetic PCM audio bytes for offline test execution."""
    total_samples = int(sample_rate * duration_sec)
    pcm = bytearray()
    for i in range(total_samples):
        val = int(12000 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
        pcm.extend(val.to_bytes(2, byteorder="little", signed=True))
    return bytes(pcm)


def test_unicode_script_classification():
    """Verifies Unicode block classification for Indic scripts."""
    print("[TEST 1] Testing Unicode Script Classification...")
    classifier = LexicalLIDClassifier()

    cases = [
        ("नमस्ते, आप कैसे हैं?", ScriptType.DEVANAGARI),
        ("Hello, how are you?", ScriptType.LATIN),
        ("નમસ્તે, તમે કેમ છો?", ScriptType.GUJARATI),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", ScriptType.TAMIL),
        ("నమస్కారం, మీరు ఎలా ఉన్నారు?", ScriptType.TELUGU),
        ("নমস্কার, আপনি কেমন আছেন?", ScriptType.BENGALI),
        ("ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?", ScriptType.KANNADA),
        ("നമസ്കാരം, സുഖമാണോ?", ScriptType.MALAYALAM),
        ("ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?", ScriptType.GURMUKHI),
        ("ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?", ScriptType.ODIA),
        ("السلام علیکم", ScriptType.ARABIC),
    ]

    for sample_text, expected_script in cases:
        detected_script, dist = classifier.detect_script(sample_text)
        assert detected_script == expected_script, (
            f"Script mismatch for '{sample_text}': expected {expected_script}, got {detected_script}"
        )
        assert dist.get(expected_script.value, 0.0) > 0.5, (
            f"Expected dominant distribution for {expected_script.value}"
        )

    print("   Passed: All 11 script classifications verified.")


def test_lexical_lid_and_code_switching():
    """Verifies lexical analysis across pure Indic, English, and Romanized code-switched inputs."""
    print("[TEST 2] Testing Lexical LID & Code-Switching Detection...")
    classifier = LexicalLIDClassifier()

    # Case A: Pure Hindi in Devanagari
    res_hi = classifier.classify("नमस्कार, मैं कल तक ईएमआई का भुगतान कर दूँगा।")
    assert res_hi.primary_language == "hi"
    assert res_hi.script == ScriptType.DEVANAGARI
    assert not res_hi.is_code_switched
    assert res_hi.confidence >= 0.85

    # Case B: Pure English
    res_en = classifier.classify("I will make the payment for my loan account tomorrow afternoon.")
    assert res_en.primary_language == "en"
    assert res_en.script == ScriptType.LATIN
    assert not res_en.is_code_switched
    assert res_en.confidence >= 0.80

    # Case C: Hinglish Code-Switching (Hindi in Latin script with English loan terms)
    res_hinglish = classifier.classify("haan main online upi payment link se kal pay kar doonga")
    assert res_hinglish.is_code_switched, "Expected Hinglish to be detected as code-switched"
    assert res_hinglish.primary_language in ("hi", "en")
    assert "hi" in res_hinglish.languages_detected
    assert "en" in res_hinglish.languages_detected
    assert res_hinglish.script == ScriptType.LATIN

    # Case D: Gujlish Code-Switching (Gujarati in Latin script)
    res_gujlish = classifier.classify("kem cho bhai mane sms par link moklo hu payment kari dais")
    assert res_gujlish.is_code_switched, "Expected Gujlish to be detected as code-switched"
    assert "gu" in res_gujlish.languages_detected

    # Case E: Pure Gujarati in native script
    res_gu = classifier.classify("નમસ્તે, હું સમયસર ચુકવણી કરી દઈશ.")
    assert res_gu.primary_language == "gu"
    assert res_gu.script == ScriptType.GUJARATI

    # Case F: Pure Tamil in native script
    res_ta = classifier.classify("வணக்கம், நான் கடனை செலுத்துகிறேன்.")
    assert res_ta.primary_language == "ta"
    assert res_ta.script == ScriptType.TAMIL

    # Case G: Pure Telugu in native script
    res_te = classifier.classify("నమస్కారం, నేను చెల్లింపు చేస్తాను.")
    assert res_te.primary_language == "te"
    assert res_te.script == ScriptType.TELUGU

    # Case H: Marathi in Devanagari with Marathi-specific words
    res_mr = classifier.classify("नमस्कार, मी उद्या नक्की पैसे देईन आणि पावती पाठवेन.")
    assert res_mr.primary_language == "mr"
    assert res_mr.script == ScriptType.DEVANAGARI

    print("   Passed: Lexical LID and code-switching logic verified across 8 variations.")


def test_code_switch_prompts_coverage():
    """Verifies STT prompt conditioning covers all 12 Indic languages with banking terminology."""
    print("[TEST 3] Testing STT Code-Switch Prompt Conditioning Map...")
    expected_languages = ["hi", "en", "gu", "ta", "te", "mr", "bn", "kn", "ml", "pa", "or", "ur"]
    for lang in expected_languages:
        assert lang in INDIC_CODE_SWITCH_PROMPTS, f"Missing code-switch prompt for language '{lang}'"
        prompt = INDIC_CODE_SWITCH_PROMPTS[lang]
        assert "UPI" in prompt or "EMI" in prompt or "loan" in prompt or "payment" in prompt, (
            f"Language '{lang}' prompt missing key financial loan terms: {prompt}"
        )

    print(f"   Passed: All {len(expected_languages)} Indic languages conditioned with financial loan vocabulary.")


def test_acoustic_lid_mock():
    """Verifies acoustic classifier handling with synthetic audio and fallback."""
    print("[TEST 4] Testing Acoustic LID Classifier...")
    acoustic = AcousticLIDClassifier()
    pcm = synthesize_test_pcm(duration_sec=1.5, sample_rate=8000)

    lang, conf, dist = acoustic.detect_language(pcm, sample_rate=8000)
    assert isinstance(lang, str)
    assert 0.0 <= conf <= 1.0
    assert isinstance(dist, dict)
    print(f"   Acoustic detection: {lang} (confidence: {conf:.2f})")
    print("   Passed: Acoustic LID execution verified.")


def test_language_identification_gate_fusion_and_transition():
    """Verifies multi-modal fusion, hysteresis gating, and voice adaptation recommendation."""
    print("[TEST 5] Testing LanguageIdentificationGate Fusion & Hysteresis...")
    gate = LanguageIdentificationGate(min_switch_confidence=0.75, hysteresis_turns=2)

    # 1. Start in Hindi, caller speaks Hindi
    res1 = gate.identify(transcript="हाँ जी, मैं कल पेमेंट कर दूँगा।", current_language="hi")
    assert res1.primary_language == "hi"
    assert not res1.language_switched
    assert res1.recommended_voice == "hi-IN-SwaraNeural"

    # 2. Caller speaks English clearly: "Please talk to me in English, I want to pay online."
    # Turn 1 of English (hysteresis count = 1)
    res2 = gate.identify(transcript="Please speak in English, I want to pay online right now.", current_language="hi")
    if res2.language_switched:
        assert res2.primary_language == "en"
        assert res2.recommended_voice == "en-IN-NeerjaNeural"
        print("   Direct high-confidence language switch triggered to 'en'.")
    else:
        # Turn 2 confirms transition
        res3 = gate.identify(transcript="Yes, give me the English instructions please.", current_language="hi")
        assert res3.language_switched
        assert res3.primary_language == "en"
        assert res3.recommended_voice == "en-IN-NeerjaNeural"
        print("   Hysteresis confirmed language switch triggered on turn 2 to 'en'.")

    # 3. Caller switches to Gujarati in native script
    res_gu = gate.identify(transcript="હું ગુજરાતીમાં વાત કરવા માંગુ છું, મને લિંક મોકલો.", current_language="en")
    assert res_gu.primary_language == "gu"
    assert res_gu.recommended_voice == "gu-IN-DhwaniNeural"

    # 4. Serialization check
    d = res_gu.to_dict()
    assert d["primary_language"] == "gu"
    assert "recommended_voice" in d
    assert "is_code_switched" in d
    assert "distribution" in d

    print("   Passed: Multi-modal fusion, hysteresis gating, and voice recommendation verified.")


def test_dynamic_voice_agent_adaptation():
    """Verifies VoiceAgent adapts language and neural voice mid-conversation."""
    print("[TEST 6] Testing VoiceAgent Mid-Call Language & Voice Adaptation...")
    agent = VoiceAgent(language="hi", voice_enabled=True, adaptive_language=True)

    assert agent.language == "hi"
    assert agent.audio_engine.voice == "hi-IN-SwaraNeural"

    # Caller replies in pure English
    res = agent.step("Please speak in English. Can you send the payment link on my phone?")
    assert "language_info" in res
    lang_info = res["language_info"]
    assert lang_info["primary_language"] in ("en", "hi")

    # If switch occurred, agent language and audio engine voice updated
    if lang_info.get("language_switched"):
        assert agent.language == "en"
        assert agent.audio_engine.voice == "en-IN-NeerjaNeural"
        print("   Dynamic mid-call voice switch verified: Switched to en-IN-NeerjaNeural.")

    # Caller switches to Gujarati
    res_gu = agent.step("નમસ્તે, હું કાલે મુથૂટ ફિનકોર્પમાં પૈસા ભરી દઈશ.")
    assert agent.language == "gu"
    assert agent.audio_engine.voice == "gu-IN-DhwaniNeural"
    print("   Dynamic mid-call voice switch verified: Switched to gu-IN-DhwaniNeural.")

    print("   Passed: VoiceAgent dynamic adaptation confirmed.")


def test_campaign_cdr_and_summary_language_tracking():
    """Verifies that CallDetailRecord and CampaignSummary record detected language and code-switching."""
    print("[TEST 7] Testing Campaign CDR & Summary Language Tracking...")

    async def _run_dialer_test():
        leads = [
            Lead(
                lead_id="LEAD_LID_01",
                phone_number="9876543210",
                name="Anil Sharma",
                loan_id="MUTH-101",
                amount_due=5420.0,
                custom_metadata={
                    "scenario": "normal_human",
                    "second_turn": "haan main payment online UPI se kar doonga",  # Hinglish
                }
            ),
            Lead(
                lead_id="LEAD_LID_02",
                phone_number="9876543211",
                name="Praveen Patel",
                loan_id="MUTH-102",
                amount_due=8900.0,
                custom_metadata={
                    "scenario": "normal_human",
                    "second_turn": "હું કાલે પૈસા ભરી દઈશ મને એસએમએસ મોકલો",  # Gujarati
                }
            ),
        ]

        cfg = CampaignConfig(
            campaign_id="CAMP_LID_TEST",
            language="hi",
            llm_provider="mock",
            enforce_trai_calling_hours=False,
            enforce_dnd_check=False,
            amd_enabled=True,
            carrier_type="simulated",
        )

        dialer = CampaignDialer(config=cfg)
        dialer.leads = leads
        summary = await dialer.run_campaign()

        assert summary.dialed_count == 2
        assert len(dialer.cdrs) == 2

        # Check CDRs
        cdr1 = dialer.cdrs[0]
        cdr2 = dialer.cdrs[1]

        assert hasattr(cdr1, "detected_language")
        assert hasattr(cdr1, "is_code_switched")
        assert hasattr(cdr2, "detected_language")
        assert hasattr(cdr2, "is_code_switched")

        d1 = cdr1.to_dict(mask_pii=True)
        assert "detected_language" in d1
        assert "is_code_switched" in d1

        # Check Summary
        sum_dict = summary.to_dict()
        assert "language_breakdown" in sum_dict
        assert "code_switched_count" in sum_dict
        assert len(summary.language_breakdown) > 0
        print(f"   Campaign Summary Language Breakdown: {summary.language_breakdown}")
        print(f"   Campaign Summary Code-Switched Count: {summary.code_switched_count}")

        # Check CSV Export contains columns
        csv_text = dialer.export_cdr_csv(mask_pii=True)
        assert "detected_language" in csv_text
        assert "is_code_switched" in csv_text

    asyncio.run(_run_dialer_test())
    print("   Passed: Campaign CDR and Summary language tracking verified.")


def test_fastapi_lid_endpoint():
    """Verifies POST /telephony/lid endpoint in FastAPI server."""
    print("[TEST 8] Testing FastAPI /telephony/lid Endpoint...")
    from verbalyze.telephony.server import create_app
    from fastapi.testclient import TestClient

    app = create_app(auth_token="TEST_SECRET_TOKEN")
    client = TestClient(app)

    # 1. Unauthorized request
    unauth_res = client.post("/telephony/lid", json={"text": "Hello, how are you?"})
    assert unauth_res.status_code == 401, "Expected 401 without auth token"

    # 2. Authorized request: Hinglish text
    headers = {"Authorization": "Bearer TEST_SECRET_TOKEN"}
    res = client.post(
        "/telephony/lid",
        headers=headers,
        json={
            "text": "haan bhai main online UPI se payment kal kar doonga",
            "current_language": "hi",
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert "primary_language" in data
    assert "is_code_switched" in data
    assert "confidence" in data
    assert "script" in data
    assert data["is_code_switched"] is True
    print(f"   FastAPI LID Output: {data['primary_language']} (code_switched={data['is_code_switched']}, script={data['script']})")

    # 3. Pure Gujarati in native script
    res_gu = client.post(
        "/telephony/lid",
        headers=headers,
        json={
            "text": "નમસ્તે, હું સમયસર ચુકવણી કરી દઈશ.",
            "current_language": "hi",
        }
    )
    assert res_gu.status_code == 200
    data_gu = res_gu.json()
    assert data_gu["primary_language"] == "gu"
    assert data_gu["script"] == "gujarati"
    assert data_gu["language_switched"] is True
    assert data_gu["recommended_voice"] == "gu-IN-DhwaniNeural"
    print(f"   FastAPI LID Gujarati Output: {data_gu['primary_language']} -> Voice: {data_gu['recommended_voice']}")

    # 4. SIP call turn returns language_info
    turn_res = client.post(
        "/webhook/sip/turn",
        headers=headers,
        json={
            "call_id": "sip_test_lid_01",
            "transcript": "Hello, I want to talk in English please.",
        }
    )
    assert turn_res.status_code == 200
    turn_data = turn_res.json()
    assert "language_info" in turn_data
    assert turn_data["language_info"] is not None
    print("   SIP call turn returned language_info successfully.")

    print("   Passed: FastAPI /telephony/lid and SIP turn endpoints verified.")


def main():
    print("=" * 64)
    print("VERBALYZE: MULTI-LINGUAL CODE-SWITCHING & LID TEST SUITE")
    print("=" * 64)

    test_unicode_script_classification()
    test_lexical_lid_and_code_switching()
    test_code_switch_prompts_coverage()
    test_acoustic_lid_mock()
    test_language_identification_gate_fusion_and_transition()
    test_dynamic_voice_agent_adaptation()
    test_campaign_cdr_and_summary_language_tracking()
    test_fastapi_lid_endpoint()

    print("=" * 64)
    print("ALL 8 LID & MULTI-LINGUAL CODE-SWITCHING TESTS PASSED!")
    print("=" * 64)


if __name__ == "__main__":
    main()
