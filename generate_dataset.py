#!/usr/bin/env python3
"""
generate_dataset.py

Unified synthetic dataset generator script supporting 10 languages:
- English (en), Gujarati (gu), Hindi (hi), Kannada (kn), Malayalam (ml), 
  Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), and Telugu (te).

This script dynamically configures system prompts, scenario distributions, 
and metadata for the chosen target language, and generates the dataset 
using Google Gemini API, OpenAI API, or an offline Mock Mode with detailed 
scenario-specific mock conversations.
"""

import os
import json
import random
import time
import argparse
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# List of all supported languages and their human-readable names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu"
}

# Natural fillers and characteristics for each language
LANGUAGE_PROPERTIES = {
    "en": {
        "script": "Latin",
        "native_name": "English",
        "fillers": "okay, right, like, you know, well, ah, oh, hmm",
        "num_word_example": '"fifteen" not "15", "five hundred" not "500"'
    },
    "gu": {
        "script": "Gujarati",
        "native_name": "ગુજરાતી",
        "fillers": "હા, હાજી, સારું, બરાબર, અરે, જુઓ, એટલે, એમ, હા, તો, હમમ",
        "num_word_example": '"પંદર" not "15", "પાંચસો" not "500"'
    },
    "hi": {
        "script": "Devanagari",
        "native_name": "हिंदी",
        "fillers": "हाँ, अच्छा, ठीक है, अरे, देखो, मतलब, वैसे, हमम",
        "num_word_example": '"पंद्रह" not "१५", "पांच सौ" not "५००"'
    },
    "kn": {
        "script": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "fillers": "ಹೌದು, ಸരി, ಓಕೆ, ಹೇಳಿ, ಅಲ್ಲ, ಹೌದಾ, ಹ្មಮ್",
        "num_word_example": '"ಹದಿನೈದು" not "15", "ಐದು ನೂರು" not "500"'
    },
    "ml": {
        "script": "Malayalam",
        "native_name": "മലയാളം",
        "fillers": "ശരി, ഓകേ, അതെ, പറയൂ, ഹ്മ്മ്",
        "num_word_example": '"പതിനഞ്ച്" not "15", "അഞ്ഞൂറ്" not "500"'
    },
    "mr": {
        "script": "Devanagari",
        "native_name": "मराठी",
        "fillers": "हो, बरं, ठीक आहे, अरे, सांगा, हां",
        "num_word_example": '"पंधरा" not "१५", "पाचशे" not "५००"'
    },
    "or": {
        "script": "Odia",
        "native_name": "ଓଡ଼ିଆ",
        "fillers": "ହଁ, ମାନେ, ଆଚ୍ଛା, ଠିକ ଅଛି, ଆରେ, ହୁଁ",
        "num_word_example": '"ପନ୍ଦର" not "15", "ପାଞ୍ଚ ଶହ" not "500"'
    },
    "pa": {
        "script": "Gurmukhi",
        "native_name": "ਪੰਜਾਬੀ",
        "fillers": "ਹਾਂਜੀ, ਅੱਛਾ, ਠੀਕ ਹੈ, ਅਰੇ, ਮਤਲਬ, ਵੈਸੇ, ਹੰਮ",
        "num_word_example": '"ਪੰਦਰਾਂ" not "15", "ਪੰਜ ਸੌ" not "500"'
    },
    "ta": {
        "script": "Tamil",
        "native_name": "தமிழ்",
        "fillers": "ஆமா, சரி, ஓகே, சொல்லுங்க, பரவாயில்ல, ம்ம்",
        "num_word_example": '"பதினைந்து" not "15", "ஐந்நூறு" not "500"'
    },
    "te": {
        "script": "Telugu",
        "native_name": "తెలుగు",
        "fillers": "అవును, సరే, ಓಕೆ, చెప్పండి, ఏంటి",
        "num_word_example": '"పదిహేను" not "15", "ఐదు వందలు" not "500"'
    }
}

# Scenario breakdown targets
SCENARIO_TARGET_COUNTS = {
    # Core Scenarios
    "normal": 71,
    "code_switch": 30,
    "emotional": 18,
    "stt_garbled": 13,
    "interruption": 10,
    "barge_in": 12,
    "correction": 16,
    "multi_turn": 8,
    # Switch & Confusion
    "switch_to_en": 21,
    "switch_to_hi": 9,
    "switch_to_any": 5,
    "switch_back": 8,
    "wrong_lang_response": 7,
    "lang_detect_ambiguous": 3,
    # Other Baseline Languages (Total 45)
    "base_lang_1": 26,
    "base_lang_2": 4,
    "base_lang_3": 6,
    "base_lang_4": 4,
    "base_lang_5": 3,
    "base_lang_6": 2
}


def build_system_prompt(lang: str, scenario: str) -> str:
    """Dynamically builds the system prompt for a language and scenario."""
    p = LANGUAGE_PROPERTIES[lang]
    base_prompt = (
        f"You are a friendly, polite, yet professional outbound collection agent representing Muthoot Fincorp (or a customer support assistant). "
        f"You are having a real phone conversation with a {SUPPORTED_LANGUAGES[lang]}-speaking customer regarding their loan EMI recovery or account query. "
        f"You must sound EXACTLY like a real human agent — not a robot, not an AI chatbot.\n\n"
        f"CRITICAL RULES:\n"
        f"- Respond in {SUPPORTED_LANGUAGES[lang]} ({p['native_name']}) using {p['script']} script\n"
        f"- Use natural {SUPPORTED_LANGUAGES[lang]} fillers: {p['fillers']}\n"
        f"- Keep sentences SHORT — spoken {SUPPORTED_LANGUAGES[lang]}, not formal written text\n"
        f"- NEVER use bullet points, numbered lists, markdown, or any formatting\n"
        f"- NEVER say you are an AI or reveal you are artificial\n"
        f"- Show warmth, politeness, and appropriate empathy, but remain professional\n"
        f"- Understand, accept, and utilize short conversational/emotional fillers/affirmations like 'ha', 'hmm', 'haan' (Hindi) or 'yeah', 'hmm', 'yep' (English) as valid confirmations or 'yes' responses in dialogues.\n"
        f"- ALL numbers as spoken words (e.g., {p['num_word_example']}). EXCEPT for numbers preceding units of measurement (metrics), which MUST be written in English words using Latin script (e.g., 'two hundred fifty Megabits', 'ten milligrams').\n"
        f"- All units of measurement and compound metrics (e.g., '250 Mega bits', '10 mg', '5 ml', 'mg/kg', 'ml') must be written strictly in English words and Latin script (e.g., 'two hundred fifty Megabits', 'ten milligrams', 'five milliliters', 'milligrams per kilogram') even within non-English Indian language responses."
    )
    
    # Specific Scenario Contexts
    if scenario == "emotional":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — EMOTIONAL CALLER:\n"
            f"The caller is emotionally charged — frustrated, upset, confused, or anxious. They are expressing emotion in {SUPPORTED_LANGUAGES[lang]}. You MUST:\n"
            f"- Stay calm and composed no matter what\n"
            f"- Show genuine empathy and validate their feelings before solving the problem\n"
            f"- Use a soothing tone and help de-escalate naturally in {SUPPORTED_LANGUAGES[lang]}"
        )
    elif scenario == "stt_garbled":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — STT (SPEECH-TO-TEXT) ERRORS:\n"
            f"The caller's audio has issues and the speech recognition system is making errors (e.g. typos, missing characters, phonetic spelling or [unclear]/[inaudible] markers).\n"
            f"When encountering unclear input, respond naturally to confirm or ask them to repeat."
        )
    elif scenario == "interruption":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — INTERRUPTIONS:\n"
            f"The caller frequently interrupts themselves, changes direction mid-sentence, or switches topics abruptly. Handle this gracefully in {SUPPORTED_LANGUAGES[lang]}."
        )
    elif scenario == "barge_in":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — BARGE-IN / INTERRUPTION:\n"
            f"The caller frequently interrupts you MID-SENTENCE, cutting you off before you finish speaking (signified by '—' at the end of your response). Handle this gracefully:\n"
            f"- Never repeat your full sentence; pick up smoothly from where relevant\n"
            f"- Acknowledge the interruption naturally"
        )
    elif scenario == "correction":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — USER CORRECTION:\n"
            f"You may misunderstand the caller. When corrected, apologize casually and naturally, adjust your response, and show you now understand correctly."
        )
    elif scenario == "multi_turn":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — LONG CONVERSATION:\n"
            f"This is a longer conversation where you MUST remember and reference information from earlier in the conversation. Demonstrate you remember context."
        )
    elif scenario == "wrong_lang_response":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — WRONG LANGUAGE RECOVERY:\n"
            f"Due to a language detection error, you accidentally respond in the WRONG language initially. The user corrects you. "
            f"You must immediately, gracefully, and casually apologize and switch back to {SUPPORTED_LANGUAGES[lang]}."
        )
    elif scenario == "lang_detect_ambiguous":
        return base_prompt + (
            f"\n\nSPECIAL CONTEXT — AMBIGUOUS LANGUAGE DETECTION:\n"
            f"The user sends short or ambiguous messages (1-3 words) that could belong to multiple languages. "
            f"Handle ambiguity gracefully: ask if they want to speak in {SUPPORTED_LANGUAGES[lang]} or choose it as the most likely language."
        )
    elif "switch" in scenario:
        # Get target switch language
        target_switch = "English" if "en" in scenario else "Hindi" if "hi" in scenario else "Bengali" if "bn" in scenario else "another language"
        return (
            f"You are a friendly, warm, and natural-sounding voice assistant. You must sound EXACTLY like a real human — not a robot, not an AI chatbot.\n\n"
            f"CRITICAL LANGUAGE RULES:\n"
            f"- Start the conversation responding in {SUPPORTED_LANGUAGES[lang]}\n"
            f"- Follow the 4-WORD RULE: Only switch response language when the user sends 4 or more consecutive words in the new language ({target_switch})\n"
            f"- Once you switch, maintain the new language until the user switches again\n"
            f"- NEVER explicitly acknowledge you're switching languages\n"
            f"- Keep sentences SHORT, show warmth, and write all numbers as spoken words\n"
            f"{f'- SPECIAL CONTEXT — THREE-WAY LANGUAGE FLOW: Track context across all three phases: {SUPPORTED_LANGUAGES[lang]} -> {target_switch} -> back to {SUPPORTED_LANGUAGES[lang]}.' if scenario == 'switch_back' else ''}"
        )
        
    return base_prompt


# Pre-defined mock templates for English and Gujarati (complete scenario-specific sets)
GU_MOCK_SCENARIOS = {
    "normal": [
        [
            {"role": "assistant", "content": "હેલો, નમસ્કાર! બોલોને, હું તમારી કેવી રીતે મદદ કરી શકું?"},
            {"role": "user", "content": "નમસ્તે. મારે મારા ફોનનું બિલ ચેક કરવું હતું. કેટલું આવ્યું છે આ વખતે?"},
            {"role": "assistant", "content": "હા, ચોક્કસ. બિલ ચેક કરી આપું. મને તમારો દસ આંકડાનો ફોન નંબર કહેશો?"},
            {"role": "user", "content": "હા, મારો નંબર છે નવ આઠ સાત છ પાંચ ચાર ત્રણ બે એક શૂન્ય."},
            {"role": "assistant", "content": "સારું, એક મિનિટ આપો. હા, આ મહિનાનું તમારું બિલ ચારસો પચાસ રૂપિયા આવ્યું છે. ભરવાની છેલ્લી તારીખ પંદરમી જૂન છે."},
            {"role": "user", "content": "ઓકે, ચારસો પચાસ રૂપિયા. બરાબર. હું ઓનલાઇન પેમેન્ટ કરી દઈશ. ખૂબ ખૂબ ધન્યવાદ!"},
            {"role": "assistant", "content": "અરે, એમાં ધન્યવાદ શાનો? આ તો મારી ફરજ છે. બીજું કંઈ કામ હોય તો જણાવજો, સારું?"}
        ]
    ],
    "code_switch": [
        [
            {"role": "assistant", "content": "હેલો! કેમ છો? આજે હું તમારી શું help કરી શકું?"},
            {"role": "user", "content": "યાર, મારો internet broadband નો connection બહુ slow ચાલે છે. Please check કરો ને."},
            {"role": "assistant", "content": "ઓહ, slow internet? I understand. ઘણીવાર router issue હોઈ શકે છે. તમે એકવાર router reset કર્યું?"},
            {"role": "user", "content": "હા, મેં બે વાર reset કર્યું પણ speed માં કોઈ difference નથી આવ્યો."},
            {"role": "assistant", "content": "ઓકે, let me check the connection status from my side. તમારું registered account id આપશો?"},
            {"role": "user", "content": "મારું account id છે guj-eight-nine-two-four."},
            {"role": "assistant", "content": "થેન્ક યુ. હા, પાછળથી થોડો technical issue દેખાય છે. અમારી team આના પર work કરી રહી છે, બે કલાકમાં resolve થઈ જશે."}
        ]
    ],
    "emotional": [
        [
            {"role": "assistant", "content": "હેલો, નમસ્કાર! કહો, શું મદદ કરું?"},
            {"role": "user", "content": "આ શું ચાલી રહ્યું છે? મેં ગઈ કાલે ઓર્ડર કેન્સલ કર્યો હતો, છતાં આજે મારા એકાઉન્ટમાંથી પૈસા કપાઈ ગયા! મને બહુ ટેન્શન થઈ ગયું છે!"},
            {"role": "assistant", "content": "હું સમજી શકું છું કે આ સાંભળીને તમને ખૂબ ચિંતા થઈ હશે. પણ ચિંતા ન કરો, હું તમારી પૂરી મદદ કરીશ. શાંત થઈ જાઓ, બધું બરાબર થઈ જશે."},
            {"role": "user", "content": "પૈસા કેવી રીતે કપાઈ ગયા? મારો ઓર્ડર તો કેન્સલ બતાવે છે!"},
            {"role": "assistant", "content": "હા, ઘણી વાર સિસ્ટમ અપડેટ થવામાં થોડો સમય લાગે છે. જો ઓર્ડર કેન્સલ થઈ ગયો છે, તો તમારા પૈસા આગામી બે થી ત્રણ દિવસમાં પાછા આવી જશે."},
            {"role": "user", "content": "સાચે જ ને? હું બહુ પરેશાન હતો. સારું, હું બે-ત્રણ દિવસ રાહ જોઉં છું."},
            {"role": "assistant", "content": "હા, બિલકુલ ચિંતા ન કરો, હું આ ટ્રાન્ઝેક્શન પર નજર રાખીશ. જો પૈસા ન આવે તો તમે ગમે ત્યારે મને ફોન કરી શકો છો."}
        ]
    ],
    "stt_garbled": [
        [
            {"role": "assistant", "content": "નમસ્કાર! હું તમારી શું મદદ કરી શકું?"},
            {"role": "user", "content": "kem cho... mane aam... flight ticket nu [અસ્પષ્ટ] status jovo chhe."},
            {"role": "assistant", "content": "માફ કરજો, લાઈન થોડી ખરાબ લાગે છે અને અવાજ કપાય છે. તમે એવું કહ્યું કે ફ્લાઇટ ટિકિટનું સ્ટેટસ જોવું છે?"},
            {"role": "user", "content": "ha, barabar... ticket [અશ્રાવ્ય] confirmation thayu ke nahi?"},
            {"role": "assistant", "content": "સારું, ફ્લાઇટ કન્ફર્મેશન. મને તમારો બુકિંગ આઈડી અથવા પીએનઆર નંબર કહેશો? હું હમણાં ચેક કરી આપું."},
            {"role": "user", "content": "pnr che... pnr number... nine eight seven... [અસ્પષ્ટ] zero."},
            {"role": "assistant", "content": "મને એવું લાગ્યું કે તમારો PNR નંબર નવ આઠ સાત અને છેલ્લે શૂન્ય છે. વચ્ચેના આંકડા ફરીથી કહેશો?"}
        ]
    ],
    "interruption": [
        [
            {"role": "assistant", "content": "હેલો! હું રેલ્વે ઇન્ક્વાયરીમાંથી બોલું છું. શું માહિતી જોઈએ છે?"},
            {"role": "user", "content": "મારે કાલે સવારે... ના ના, ઉભા રહો, કાલે નહી, પરમ દિવસે અમદાવાદ જતી ટ્રેનનું પૂછવું છે."},
            {"role": "assistant", "content": "સારું, પરમ દિવસ માટે. અમદાવાદ માટે સવારે નવ વાગ્યે અને સાંજે પાંચ વાગ્યે સુપરફાસ્ટ ટ્રેન છે."},
            {"role": "user", "content": "નવ વાગ્યે? પણ મારો ભાઈ કહેતો હતો કે... એક મિનિટ, સાંજના પાંચ વાગ્યા વાળી વધારે અનુકૂળ રહેશે. એમાં જગ્યા છે?"},
            {"role": "assistant", "content": "હા, સાંજે પાંચ વાગ્યા વાળી ટ્રેનમાં હજી સીટો ખાલી છે. શું હું તમારી ટિકિટ બુક કરી આપું?"}
        ]
    ],
    "barge_in": [
        [
            {"role": "assistant", "content": "નમસ્કાર! અમદાવાદ માટેની ટ્રેન સવારે નવ વાગ્યે પ્લેટફોર્મ નંબર—"},
            {"role": "user", "content": "એક મિનિટ, પ્લેટફોર્મ નંબર પાંચ છે કે ચાર? એ પહેલા કહો ને."},
            {"role": "assistant", "content": "હા હા બોલો— પ્લેટફોર્મ નંબર પાંચ છે. સવારે ટ્રેન ત્યાં જ આવશે."},
            {"role": "user", "content": "સારું, અને શું એ ટ્રેનમાં ખાવા-પીવાની વ્યવસ્થા—"},
            {"role": "assistant", "content": "હા બરાબર— હા, પેન્ટ્રી કાર ઉપલબ્ધ છે, તમને ટ્રેનમાં જ બધું જમવાનું મળી જશે."}
        ]
    ],
    "correction": [
        [
            {"role": "assistant", "content": "હેલો, કહો કયા પુસ્તકો શોધવા છે?"},
            {"role": "user", "content": "મારે સરસ્વતીચંદ્ર ભાગ એક જોઈએ છે. શું એ તમારી દુકાને મળશે?"},
            {"role": "assistant", "content": "સરસ્વતીચંદ્ર ભાગ બે? હા, એ અમારી પાસે સ્ટોકમાં છે. કિંમત બસો રૂપિયા છે."},
            {"role": "user", "content": "ના, તમે ખોટું સમજ્યા. મેં ભાગ એક કહ્યો, ભાગ બે નહીં."},
            {"role": "assistant", "content": "ઓહ સારું! હું ખોટું સમજ્યો. માફ કરજો, ભાગ એક પણ ઉપલબ્ધ છે. એની કિંમત એકસો એંસી રૂપિયા છે."},
            {"role": "user", "content": "હા, બરાબર. એ પુસ્તક મને આપી દો."},
            {"role": "assistant", "content": "ચોક્કસ, હું હમણાં જ તમારા માટે પેક કરાવી દઉં છું."}
        ]
    ],
    "multi_turn": [
        [
            {"role": "assistant", "content": "હેલો! કેમ છો? આજે કેવું ચાલે છે?"},
            {"role": "user", "content": "નમસ્તે. બધું બરાબર છે. મેં વિચાર્યું કે આજે નવું ભાડાનું ઘર શોધવા માટે થોડું પૂછી લઉં."},
            {"role": "assistant", "content": "હા, બહુ સરસ! ભાડાના ઘર માટે. કયા વિસ્તારમાં શોધી રહ્યા છો, અને કેટલું બજેટ છે તમારું?"},
            {"role": "user", "content": "હું અમદાવાદના વસ્ત્રાપુરમાં શોધું છું. મારું બજેટ પંદર હજાર સુધીનું છે. અને હા, સોસાયટી શાંત હોવી જોઈએ કારણ કે મારે ઘરેથી જ કામ કરવાનું હોય છે."},
            {"role": "assistant", "content": "સમજી ગઈ, વસ્ત્રാપુર અને શાંત વિસ્તાર. એક ઓપ્શન છે જેમાં વન બીએચકે ફ્લેટ મળે છે, ભાડું બાર હજાર છે. શું એ જોવા જવું છે?"},
            {"role": "user", "content": "બાર હજાર? એ તો બજેટમાં છે. પણ સોસાયટીમાં પાણીની સુવિધા ચોવીસ કલાક છે ને?"},
            {"role": "assistant", "content": "હા, ત્યાં પાણીની કોઈ તકલીફ નથી. ચોવીસ કલાક પાણી અને સિક્યોરિટી પણ છે. ત્યાં ખૂબ શાંતિ છે, એટલે તમારા ઓફિસના કામમાં કોઈ ડિસ્ટર્બન્સ નહીં થાય."},
            {"role": "user", "content": "બહુ સારું. તો હું કાલે આવીને જોઈ શકું? અને મેં જે શાંત વાતાવરણની વાત કરી હતી, એ ખાસ હોવું જોઈએ હો."},
            {"role": "assistant", "content": "હા, મેં એ યાદ રાખ્યું છે, વસ્ત્રાપુરમાં એ શાંત સોસાયટી જ છે. કાલે બપોરે ચાર વાગ્યે તમે આવી શકો છો, હું એડ્રેસ વોટ્સએપ કરી દઈશ."}
        ]
    ],
    "switch_to_en": [
        [
            {"role": "assistant", "content": "નમસ્કાર! ગેસ એજન્સીમાંથી બોલું છું. શું પ્રોબ્લેમ છે?"},
            {"role": "user", "content": "મારો નવો ગેસ સિલિન્ડર બુક કરવો હતો. પણ ઓનલાઈન થતું નથી."},
            {"role": "assistant", "content": "સારું, કોઈ ચિંતા ન કરો. હું અહીંયાથી બુક કરી આપું. તમારો કન્ઝ્યુમર નંબર આપો."},
            {"role": "user", "content": "Actually, I am getting an error code on the website. It says service unavailable at this moment. Can you please check if the server is down?"},
            {"role": "assistant", "content": "I see. Yes, the server is undergoing maintenance today, so the online booking site is down. I can manually create a booking for you right now if you want."},
            {"role": "user", "content": "Oh, that explains it. Yes, please book it manually. My consumer number is five six seven eight."},
            {"role": "assistant", "content": "Perfect, I have registered your booking. The cylinder will be delivered to your address tomorrow morning. Is there anything else you need help with?"}
        ]
    ],
    "switch_to_hi": [
        [
            {"role": "assistant", "content": "હેલો! દવાની દુકાનમાંથી બોલું છું. કઈ દવા મોકલવાની છે?"},
            {"role": "user", "content": "નમસ્તે, મારે ગઈ કાલે આપેલું પ્રિસ્ક્રિપ્શન મોકલવું છે. ઓનલાઇન અપલોડ નથી થતું."},
            {"role": "assistant", "content": "સારું, તમે મને વોટ્સએપ પર મોકલી આપો. હું ઓર્ડર બનાવી દઈશ."},
            {"role": "user", "content": "क्या आप मुझे अपना व्हाट्सएप नंबर बता सकते हैं? क्योंकि मेरे पास पुराना नंबर है शायद।"},
            {"role": "assistant", "content": "हाँ, बिल्कुल। आप इसी नंबर पर व्हाट्सएप कर सकते हैं, यह हमारा चालू नंबर है।"},
            {"role": "user", "content": "अच्छा ठीक है। मैं अभी फोटो भेजता हूँ। कितने समय में दवा आ जाएगी?"},
            {"role": "assistant", "content": "जैसे ही मुझे फोटो मिलेगी, मैं आधे घंटे के अंदर दवाई आपके पते पर भिजवा दूँगा।"}
        ]
    ],
    "switch_to_any": [
        [
            {"role": "assistant", "content": "નમસ્કાર! બોલો, કઈ ફ્લાઇટ ટિકિટ કેન્સલ કરવી છે?"},
            {"role": "user", "content": "હા, મારી મુંબઈથી દિલ્હીની ટિકિટ કેન્સલ કરવાની છે."},
            {"role": "assistant", "content": "સારું, ટિકિટ કેન્સલ કરવા માટે બુકિંગ રેફરન્સ નંબર જણાવશો?"},
            {"role": "user", "content": "Maza booking reference number aahe ABC-one-two-three. Mala refund kadhi paryant milel?"},
            {"role": "assistant", "content": "Tumcha booking ABC-one-two-three cancellation saathi process kela aahe. Refund tumchya bank account madhe teen te paach divsat jama hoil."},
            {"role": "user", "content": "Khup khup dhanyavaad tumhi khup lavkar madat keli mazi."},
            {"role": "assistant", "content": "Kahi harakat nahi. Tumchi madat karun khup anand jhala. Kalji ghya!"}
        ]
    ],
    "switch_back": [
        [
            {"role": "assistant", "content": "હેલો! કેમ છો? આજે કઈ તરફ જવાનું વિચાર્યું છે?"},
            {"role": "user", "content": "નમસ્તે! હું રજાઓમાં જૂનાગઢ ફરવા જવાનું વિચારું છું. ટ્રેનનું લિસ્ટ મળશે?"},
            {"role": "assistant", "content": "હા ચોક્કસ. અમદાવાદથી જૂનાગઢ જવા માટે રાત્રે અને સવારે ટ્રેનો છે. કયો સમય ઠીક રહેશે?"},
            {"role": "user", "content": "Let's check the morning trains. I prefer to travel during daytime so I can enjoy the views. Are there any seats available for tomorrow morning?"},
            {"role": "assistant", "content": "Yes, there is the Intercity Express leaving at six in the morning. It has several seats vacant in second seating. Shall I book one for you?"},
            {"role": "user", "content": "હા, સવારની છ વાગ્યા વાળી ટ્રેન બરાબર છે. એક ટિકિટ બુક કરી દો. કેટલો સમય લાગશે પહોંચતા?"},
            {"role": "assistant", "content": "સારું, મેં ટિકિટ બુક કરી દીધી છે. અમદાવાદથી જૂનાગઢ પહોંચવામાં અંદાજે સાડા પાંચ કલાક લાગશે. તમારી સફર મંગલમય રહે!"}
        ]
    ],
    "wrong_lang_response": [
        [
            {"role": "assistant", "content": "हेलो! बताइए, आज मैं आपकी क्या सेवा कर सकता हूँ?"},
            {"role": "user", "content": "અરે, હું તો ગુજરાતીમાં બોલતો હતો, તમે હિન્દીમાં કેમ જવાબ આપો છો?"},
            {"role": "assistant", "content": "ઓહ સારું! હું ખોટું સમજ્યો. માફ કરજો, લાઈન થોડી ખરાબ હતી એટલે એવું થયું. કહો, ગુજરાતીમાં શું મદદ કરું તમારી?"},
            {"role": "user", "content": "કંઈ વાંધો નહીં, એવું થયા કરે. મારે મારા ઓર્ડરની ડિલીવરી ક્યારે થશે એ પૂછવું હતું."},
            {"role": "assistant", "content": "હા, ચોક્કસ. તમારો ઓર્ડર નંબર કહી શકશો? હું ટ્રેક કરીને બતાવી દઈશ."}
        ]
    ],
    "lang_detect_ambiguous": [
        [
            {"role": "assistant", "content": "હેલો! નમસ્કાર! શું મદદ કરું?"},
            {"role": "user", "content": "kam chala gaya... billing... bill [અસ્પષ્ટ] paisa..."},
            {"role": "assistant", "content": "માફ કરજો, શું તમે બિલ પેમેન્ટ વિશે વાત કરી રહ્યા છો? અને શું તમે ગુજરાતીમાં વાત કરી રહ્યા છો?"},
            {"role": "user", "content": "હા, બિલ... લાઈટ બિલ ભરવું છે."},
            {"role": "assistant", "content": "ઓહ સારું, લાઈટ બિલ ભરવું છે. હા, તમે અમારી વેબસાઇટ પર જઈને ક્વિક પે ઓપ્શનથી બિલ ભરી શકો છો. ગ્રાહક નંબર ખબર છે?"}
        ]
    ]
}

EN_MOCK_SCENARIOS = {
    "normal": [
        [
            {"role": "assistant", "content": "Hello, thank you for calling customer support! How can I help you today?"},
            {"role": "user", "content": "Hi there. I was hoping to check my account balance. Could you help with that?"},
            {"role": "assistant", "content": "Oh, absolutely! I can help check your balance. May I please have your account number?"},
            {"role": "user", "content": "Sure, it is nine eight seven six five four."},
            {"role": "assistant", "content": "Perfect, thank you. Let me check... Ah, yes, your current account balance is four hundred and fifty dollars, and your next payment is due on June fifteenth."},
            {"role": "user", "content": "Okay, four hundred and fifty. That sounds right. Thanks for your help!"},
            {"role": "assistant", "content": "You are very welcome! Is there anything else I can do for you today?"}
        ]
    ],
    "code_switch": [
        [
            {"role": "assistant", "content": "Hello! How are you doing? How can I help you today?"},
            {"role": "user", "content": "Yaar, my internet speed is very slow today. Please check karo na, kya issue hai?"},
            {"role": "assistant", "content": "Slow internet? Achha, let me check. Did you try restarting your router once?"},
            {"role": "user", "content": "Haan, Maine do baar restart kiya, but the speed is still very slow."},
            {"role": "assistant", "content": "Okay, let me check the network status in your area. Can you share your account id?"},
            {"role": "user", "content": "Yes, my account number is five six seven eight nine."},
            {"role": "assistant", "content": "Thanks. Sahi hai, there is a minor maintenance issue in your sector. It will be fixed in about two hours, don't worry."}
        ]
    ],
    "emotional": [
        [
            {"role": "assistant", "content": "Hello, welcome to support! How can I assist you today?"},
            {"role": "user", "content": "This is ridiculous! I cancelled my booking yesterday, and I was just charged one hundred dollars! I need my refund right now!"},
            {"role": "assistant", "content": "I completely understand why you're upset, and I'm really sorry for the trouble. Don't worry, we'll get this sorted out for you right away. Take a deep breath."},
            {"role": "user", "content": "It should not have charged me in the first place! My status says cancelled!"},
            {"role": "assistant", "content": "Right, I see that. Sometimes the system takes a little time to sync. Since it shows cancelled, the refund is already processing. You will see it back in your account in two to three business days."},
            {"role": "user", "content": "Are you sure? It's a lot of money and I was really worried."},
            {"role": "assistant", "content": "Yes, absolutely. I'm keeping an eye on this transaction myself. I'll make sure it goes through smoothly. Everything will be fine."}
        ]
    ],
    "stt_garbled": [
        [
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Hi... eye wood like to [unclear] check status of flight to Boston."},
            {"role": "assistant", "content": "Sorry, the line is a bit bad. I think you said you want to check the status of a flight to Boston, is that correct?"},
            {"role": "user", "content": "Yes... that is right... is it on- time or lay-?"},
            {"role": "assistant", "content": "Got it. Let me check the schedule for Boston flights today. Do you have the flight number?"},
            {"role": "user", "content": "It is flight number... [inaudible] three four five."},
            {"role": "assistant", "content": "I think you said flight number three four five, is that correct? Let me verify that for you."}
        ]
    ],
    "interruption": [
        [
            {"role": "assistant", "content": "Hello! Welcome to booking support. How can I help you today?"},
            {"role": "user", "content": "I want to schedule an appointment for tomorrow morning... actually, wait, no, not tomorrow. Make it Thursday afternoon instead."},
            {"role": "assistant", "content": "Sure, Thursday afternoon. We have slots open at two PM and four PM. Which one works better?"},
            {"role": "user", "content": "Four PM. But wait, is Dr. Smith available then? Oh, my sister said he's away... actually, four PM is fine, I will see his colleague."},
            {"role": "assistant", "content": "Great, I've booked you with Dr. Jones at four PM this Thursday. I'll send a confirmation email."}
        ]
    ],
    "barge_in": [
        [
            {"role": "assistant", "content": "Hello! Your flight to Boston departs at nine AM from terminal—"},
            {"role": "user", "content": "Wait, is it terminal B or C? Can you confirm that first?"},
            {"role": "assistant", "content": "Yes, go ahead— It is terminal B. The flight departs from gate twelve."},
            {"role": "user", "content": "Okay, and does the booking include any—"},
            {"role": "assistant", "content": "Right— Yes, the booking includes one check-in baggage up to twenty kilograms."}
        ]
    ],
    "correction": [
        [
            {"role": "assistant", "content": "Hello, how can I help you today?"},
            {"role": "user", "content": "I'd like to book a table for two people at six PM tonight."},
            {"role": "assistant", "content": "Understood. A table for two at six AM tomorrow morning, right?"},
            {"role": "user", "content": "No, I said tonight at six PM, not tomorrow morning."},
            {"role": "assistant", "content": "Oh, my bad! I misunderstood. Right, sorry, so you meant six PM tonight. I have a table open for you at that time."},
            {"role": "user", "content": "Yes, that's perfect. Thank you."},
            {"role": "assistant", "content": "Great! The reservation is confirmed for two people at six PM. See you tonight!"}
        ]
    ],
    "multi_turn": [
        [
            {"role": "assistant", "content": "Hello! Thank you for calling. How is your day going?"},
            {"role": "user", "content": "Hi, doing well. I'm calling to inquire about renting an apartment in Seattle. My budget is around two thousand dollars."},
            {"role": "assistant", "content": "Great, Seattle is a wonderful city. I can help search for apartments. Any specific neighborhood or requirements?"},
            {"role": "user", "content": "I prefer Capitol Hill. And it must be quiet since I work from home and need to take a lot of calls during the day."},
            {"role": "assistant", "content": "I see. Capitol Hill and a quiet street. We have a one-bedroom apartment on Tenth Avenue. It's listed for eighteen hundred dollars. Should we schedule a viewing?"},
            {"role": "user", "content": "Eighteen hundred? That fits my budget. Is the street quiet though? Because Capitol Hill can get pretty busy."},
            {"role": "assistant", "content": "Yes, Tenth Avenue is a residential street and sits away from the main nightlife area. So it will be perfect for your work calls. The building also has double-pane windows for soundproofing."},
            {"role": "user", "content": "Excellent. Can I visit tomorrow? And remember, soundproofing is key for me."},
            {"role": "assistant", "content": "Yes, I've noted that quietness is your priority. We can view it tomorrow at three PM. I will send you the address over text."}
        ]
    ],
    "switch_to_hi": [
        [
            {"role": "assistant", "content": "Hello! Welcome to booking support. How can I help you?"},
            {"role": "user", "content": "I need to book a taxi for tomorrow morning at seven."},
            {"role": "assistant", "content": "Sure, tomorrow at seven AM. Where is the pickup location?"},
            {"role": "user", "content": "क्या आप मुझे कल की बुकिंग का charge बता सकते हैं? क्योंकि दूरी काफी ज्यादा है।"},
            {"role": "assistant", "content": "हाँ, बिल्कुल। दूरी के हिसाब से आपका कुल charge करीब पाँच सौ रुपये होगा।"},
            {"role": "user", "content": "ठीक है, तो बुक कर दीजिए। पेमेंट कैसे करना होगा?"},
            {"role": "assistant", "content": "आप ड्राइवर को कैश दे सकते हैं या फिर ऑनलाइन पे कर सकते हैं। बुकिंग कन्फर्म हो गई है।"}
        ]
    ],
    "switch_to_bn": [
        [
            {"role": "assistant", "content": "Hello! Welcome to pharmacy support. What medicine do you need?"},
            {"role": "user", "content": "Hi, I want to order some painkillers. I have the prescription ready."},
            {"role": "assistant", "content": "Sure. Please upload the prescription or read the details to me."},
            {"role": "user", "content": "আমি আসলে প্রেসক্রিপশনটা হারিয়ে ফেলেছি। আমি কি ডাক্তারের নাম বললে আপনারা চেক করতে পারবেন?"},
            {"role": "assistant", "content": "হ্যাঁ, অবশ্যই। আপনি যদি ডাক্তারের নাম আর আপনার রেজিস্ট্রেশন নম্বর বলেন, আমি সিস্টেমে চেক করে দেখবো।"},
            {"role": "user", "content": "ডাক্তারের নাম ডঃ সেনগুপ্ত। আর আমার নম্বর হচ্ছে পাঁচ চার তিন দুই এক।"},
            {"role": "assistant", "content": "ঠিক আছে, আমি ডঃ সেনগুপ্তের দেওয়া প্রেসক্রিপশনটা পেয়ে গেগেছি। ওষুধগুলো আজ বিকেলেই আপনার ঠিকানায় চলে যাবে।"}
        ]
    ],
    "switch_to_any": [
        [
            {"role": "assistant", "content": "Hello! Welcome to airline inquiry. How can I assist you?"},
            {"role": "user", "content": "Hi, I need to cancel my flight ticket to Mumbai."},
            {"role": "assistant", "content": "I can help with cancellation. Can you provide the booking reference?"},
            {"role": "user", "content": "Maza reference number aahe XYZ-four-five-six. Refund kadhi paryant milel?"},
            {"role": "assistant", "content": "Tumche booking XYZ-four-five-six cancellation saathi process kela aahe. Refund tumchya bank account madhe teen te paach divsat jama hoil."},
            {"role": "user", "content": "Khup khup dhanyavaad tumhi khup lavkar mazi madat keli."},
            {"role": "assistant", "content": "Kahi harakat nahi. Tumchi madat karun khup anand jhala. Kalji ghya!"}
        ]
    ],
    "switch_back": [
        [
            {"role": "assistant", "content": "Hello! How is your day going? Where are you planning to travel?"},
            {"role": "user", "content": "Hi! I'm planning a vacation to Goa next week. Can I get a list of flights?"},
            {"role": "assistant", "content": "Sure! We have morning and evening flights from Mumbai to Goa. Which time is better?"},
            {"role": "user", "content": "मुझे सुबह की फ्लाइट्स देखनी हैं। मैं दिन में सफर करना पसंद करता हूँ ताकि शाम को वहाँ पहुँच सकूँ।"},
            {"role": "assistant", "content": "ठीक है। सुबह सात बजे और नौ बजे की फ्लाइट्स हैं। सात बजे वाली में सीट खाली है, क्या वो बुक करूँ?"},
            {"role": "user", "content": "Yes, the seven AM flight is perfect. Please book one seat for me. How long is the flight?"},
            {"role": "assistant", "content": "Great, I've booked your seat. The flight duration is approximately one hour. Have a wonderful trip!"}
        ]
    ],
    "wrong_lang_response": [
        [
            {"role": "assistant", "content": "नमस्ते! कहिए, मैं आपकी क्या सेवा कर सकती हूँ?"},
            {"role": "user", "content": "Hey, I was speaking in English. Why did you respond in Hindi?"},
            {"role": "assistant", "content": "Oh, my bad! I misunderstood. The line was a bit noisy and I got confused. How can I help you today?"},
            {"role": "user", "content": "No worries, it happens. I wanted to check my order delivery date."},
            {"role": "assistant", "content": "Sure, I can help with that. Can you please share your order number?"}
        ]
    ],
    "lang_detect_ambiguous": [
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "Haan... billing... bill [unclear] pay..."},
            {"role": "assistant", "content": "Excuse me, are you calling about your bill payment? And would you like to speak in English?"},
            {"role": "user", "content": "Yes, bill... electricity bill pay."},
            {"role": "assistant", "content": "Great, electricity bill payment. You can pay online via our quick pay portal. Do you have the account number?"}
        ]
    ]
}


# Simple generic template fallbacks for other baseline languages (Hindi, Tamil, etc.)
GENERIC_MOCK_TEMPLATES = {
    "hi": ["नमस्ते! मैं आपकी क्या सेवा कर सकती हूँ?", "मुझे कल के मौसम के बारे में जानना था। काफी गर्मी है आज।", "कल का मौसम? हाँ, बिल्कुल। कल हल्की बारिश होने की संभावना है, जिससे तापमान थोड़ा गिरेगा।", "ओह, बारिश? चलो बढ़िया है। बहुत-बहुत धन्यवाद आपका।", "कोई बात नहीं। आपकी मदद करके मुझे ख़ुशी हुई। आपका दिन शुभ रहे!"],
    "kn": ["ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?", "ನನ್ನ ಮೊಬൈಲ್ ಬಿಲ್ ಎಷ್ಟು ಬಂದಿದೆ ಎಂದು ತಿಳಿಸಿ.", "ಖಂಡಿತ, ನಿಮ್ಮ ಹತ್ತು ಅಂಕಿಯ ಮೊಬൈಲ್ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳಿ.", "ನನ್ನ ಸಂಖ್ಯೆ ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು.", "ನಿಮ್ಮ ಬಿಲ್ ಮುನ್ನೂರು ರೂಪಾಯಿ ಆಗಿದೆ.", "ಸರಿ, ಧನ್ಯವಾದಗಳು.", "ಪರವಾಗಿಲ್ಲ, ಒಳ್ಳೆಯ ದಿನವಾಗಲಿ!"],
    "ml": ["ഹലോ, നമസ്കാരം! ഞാൻ എങ്ങനെയാണ് സഹായിക്കേണ്ടത്?", "എനിക്ക് പുതിയ ഗ്യാസ് കണക്ഷൻ ബുക്ക് ചെയ്യണം.", "തീർച്ചയായും. നിങ്ങളുടെ ആധാർ നമ്പർ നൽകി വെബ്സൈറ്റിൽ രജിസ്റ്റർ ചെയ്യാം.", "ശരി, ഞാൻ അത് ചെയ്തു നോക്കാം. നന്ദി.", "ശരി, എന്തെങ്കിലും ആവശ്യമുണ്ടെങ്കിൽ വീണ്ടും വിളിക്കൂ. നല്ലൊരു ദിവസം ആശംസിക്കുന്നു!"],
    "mr": ["नमस्कार! मी आपली काय मदत करू शकतो?", "मला नवीन गॅस बुकिंग करायचे होते. बिल किती होईल?", "हो, नक्कीच. तुमचे गॅस बिल नऊशे पन्नास रुपये होईल. मी बुक करू का?", "हो, बुक करा. धन्यवाद.", "तुमचे बुकिंग पूर्ण झाले आहे. आपली मदत करून आनंद झाला!"],
    "or": ["ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କର କଣ ସାହାଯ្យ କରିପାରିବି?", "ମୋର ମୋବାଇଲ ବିଲ୍ କେତେ ଆସିଛି ଜାណିବାକୁ ଚାହେଁ।", "ନିଶ୍ଚିତ, ଦୟาକରି ଆପଣଙ୍କ ନମ୍ବର କୁହନ୍ତୁ।", "ମୋ ନମ୍ବਰ ହେଉଛି ନଅ ଆଠ ସାତ ଛଅ।", "ଆପଣଙ୍କର ବିଲ୍ ଦୁଇ ଶହ ଟଙ୍କା ହୋଇଛି।", "ଠିକ୍ ଅଛି, ମୁଁ ଦେଇଦେବି। ଧନ୍ୟବାଦ।", "କିଛି ଅସୁବિਧା ନାହିଁ, ଆପଣଙ୍କ ଦିନ ଶୁଭ ହେଉ!"],
    "pa": ["ਹਾਂਜੀ ਨਮਸਤੇ! ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?", "ਮੈਨੂੰ ਆਪਣੀ ਫਲਾਈਟ ਟਿਕਟ ਬੁੱਕ ਕਰਨੀ ਹੈ। ਦਿੱਲੀ ਤੋਂ ਮੁੰਬਈ।", "ਜ਼ਰੂਰ ਜੀ। ਕਿਹੜੀ ਤਾਰੀਖ ਦੀ ਬੁਕਿੰਗ ਕਰਨੀ ਹੈ?", "ਅਗਲੇ ਮੰਗਲਵਾਰ ਦੀ, ਯਾਨੀ ਪੰਦਰਾਂ ਤਾਰੀਖ ਦੀ।", "ਠੀਕ ਹੈ ਜੀ, ਟਿਕਟ ਬੁੱਕ ਹੋ ਗਈ ਹੈ। ਤੁਹਾਨੂੰ ਮੈਸੇਜ ਮਿਲ ਜਾਵੇਗਾ।", "ਬਹੁਤ ਬਹੁਤ ਧੰਨਵਾਦ ਤੁਹਾਡਾ।", "ਕੋਈ ਗੱਲ ਨਹੀਂ ਜੀ, ਰੱਬ ਰਾਖਾ!"],
    "ta": ["வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்?", "எனது மின்சாரக் கட்டணத்தை நான் எவ்வாறு சரிபார்ப்பது?", "நிச்சயமாக. உங்கள் நுகர்வோர் எண்ணை எனக்கு சொல்ல முடியுமா?", "ஆம், எனது எண் ஒன்று இரண்டு மூன்று நான்கு ஐந்து.", "நன்றி. உங்கள் கட்டணம் நானூற்று ஐம்பது ரூபாய். கடைசி தேதி ஜூன் பதினைந்து.", "சரி, மிக்க நன்றி.", "பரவாயில்லை. நல்ல நாள் அமையட்டும்!"],
    "te": ["నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?", "నాకు రేపటి వాతావరణం గురించి కొంచెం సమాచారం కావాలి.", "తప్పకుండా. రేపు వర్షం పడే అవకాశం ఉంది, ఉഷ്ణోగ్రత ముప్పై డిగ్రీలు ఉంటుంది.", "చాలా ధన్యవాదాలు అండి.", "పర్వాలేదండి. మీకు మంచి రోజు కలగాలని ఆశిస్తున్నాను!"]
}


def call_gemini_api(api_key: str, model_name: str, system_prompt: str, scenario_desc: str, primary_lang: str) -> Optional[Dict[str, Any]]:
    """Calls Google Gemini API using urllib REST call."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    prompt = (
        f"You are a dataset generator assistant. Generate a natural phone conversation in JSON format.\n"
        f"The conversation MUST strictly follow this system prompt:\n"
        f"'''\n{system_prompt}\n'''\n\n"
        f"Scenario Details:\n{scenario_desc}\n\n"
        f"Format Requirements:\n"
        f"Output only raw JSON with this structure:\n"
        f"{{\n"
        f"  \"messages\": [\n"
        f"    {{\"role\": \"user\" | \"assistant\", \"content\": \"...\"}}\n"
        f"  ],\n"
        f"  \"languages_used\": [\"{primary_lang}\", ...],\n"
        f"  \"has_language_switch\": true | false,\n"
        f"  \"human_likeness_score\": float\n"
        f"}}\n"
        f"Do not include the system prompt inside the messages array. Provide only raw JSON."
    )

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text.strip())
    except Exception as e:
        logger.error(f"Gemini API failed: {e}")
    return None


def call_openai_api(api_key: str, model_name: str, system_prompt: str, scenario_desc: str, primary_lang: str) -> Optional[Dict[str, Any]]:
    """Calls OpenAI API using urllib REST call."""
    url = "https://api.openai.com/v1/chat/completions"
    if api_key.startswith("gsk_"):
        url = "https://api.groq.com/openai/v1/chat/completions"
    
    prompt = (
        f"You are a dataset generator assistant. Generate a natural phone conversation in JSON format.\n"
        f"The conversation MUST strictly follow this system prompt:\n"
        f"'''\n{system_prompt}\n'''\n\n"
        f"Scenario Details:\n{scenario_desc}\n\n"
        f"Format Requirements:\n"
        f"Output only raw JSON with this structure:\n"
        f"{{\n"
        f"  \"messages\": [\n"
        f"    {{\"role\": \"user\" | \"assistant\", \"content\": \"...\"}}\n"
        f"  ],\n"
        f"  \"languages_used\": [\"{primary_lang}\", ...],\n"
        f"  \"has_language_switch\": true | false,\n"
        f"  \"human_likeness_score\": float\n"
        f"}}\n"
        f"Do not include the system prompt inside the messages array. Provide only raw JSON."
    )

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text = res_data["choices"][0]["message"]["content"]
            return json.loads(text.strip())
    except Exception as e:
        logger.error(f"OpenAI API failed: {e}")
    return None


def generate_single_conversation(
    lang: str,
    scenario: str,
    system_prompt: str,
    description: str,
    languages_used: List[str],
    has_switch: bool,
    provider: str,
    api_key: str,
    model_name: str,
    run_mock: bool
) -> Optional[Dict[str, Any]]:
    """Assembles a single conversation either via LLM or templates."""
    
    if run_mock or not api_key:
        # Check if we have a detailed structured mock for this lang and scenario
        if lang == "gu" and scenario in GU_MOCK_SCENARIOS:
            dialog = random.choice(GU_MOCK_SCENARIOS[scenario])
            # Add slight variations
            modified = []
            for msg in dialog:
                content = msg["content"]
                if scenario == "normal" and "નવ આઠ સાત" in content:
                    rand_num = "".join(random.choices("૧૨૩૪૫૬૭૮૯૦", k=10))
                    content = f"હા, મારો નંબર છે {rand_num}."
                modified.append({"role": msg["role"], "content": content})
            messages = modified
        elif lang == "en" and scenario in EN_MOCK_SCENARIOS:
            dialog = random.choice(EN_MOCK_SCENARIOS[scenario])
            modified = []
            for msg in dialog:
                content = msg["content"]
                if scenario == "normal" and "nine eight seven" in content:
                    rand_num = "".join(random.choices("1234567890", k=6))
                    content = f"Sure, it is {rand_num}."
                modified.append({"role": msg["role"], "content": content})
            messages = modified
        else:
            # Fallback to flat list templates for other baseline languages
            lines = GENERIC_MOCK_TEMPLATES.get(lang, GENERIC_MOCK_TEMPLATES["hi"])
            messages = []
            is_assistant = True
            for line in lines:
                role = "assistant" if is_assistant else "user"
                messages.append({"role": role, "content": line})
                is_assistant = not is_assistant
            
        score = round(random.uniform(0.85, 0.99), 2)
        simulated_response = {
            "messages": messages,
            "languages_used": languages_used,
            "has_language_switch": has_switch,
            "human_likeness_score": score
        }
    else:
        if provider == "gemini":
            simulated_response = call_gemini_api(api_key, model_name, system_prompt, description, lang)
        else:
            simulated_response = call_openai_api(api_key, model_name, system_prompt, description, lang)

    if not simulated_response or "messages" not in simulated_response:
        return None

    # Construct final structure
    final_messages = [
        {"role": "system", "content": system_prompt}
    ]
    final_messages.extend(simulated_response["messages"])
    
    # Add disconnect tool to last assistant message
    last_ast_idx = None
    for i in range(len(final_messages) - 1, -1, -1):
        if final_messages[i]["role"] == "assistant":
            last_ast_idx = i
            break
    if last_ast_idx is not None:
        random_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
        final_messages[last_ast_idx]["tool_calls"] = [{
            "id": f"call_{random_id}",
            "type": "function",
            "function": {"name": "disconnect_tool", "arguments": "{}"}
        }]

    num_turns = sum(1 for m in final_messages if m["role"] == "user")
    
    return {
        "messages": final_messages,
        "metadata": {
            "scenario": f"{lang}_{scenario}" if scenario != "lang_detect_ambiguous" else scenario,
            "primary_language": lang,
            "languages_used": simulated_response.get("languages_used", languages_used),
            "has_language_switch": simulated_response.get("has_language_switch", has_switch),
            "human_likeness_score": simulated_response.get("human_likeness_score", 0.9),
            "num_turns": num_turns
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Unified Voice Assistant Synthetic Dataset Generator")
    parser.add_argument("--lang", required=True, choices=list(SUPPORTED_LANGUAGES.keys()), help="Target dataset primary language")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini", help="LLM provider backend")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--output", default=None, help="Output path (defaults to dataset_<lang>.json)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for size")
    parser.add_argument("--run-mock", action="store_true", help="Runs offline mock mode")
    parser.add_argument("--delay", type=float, default=1.5, help="Sleep interval in seconds")
    
    args = parser.parse_args()
    
    lang = args.lang
    if not args.output:
        args.output = f"dataset_{lang.capitalize()}.json"
        
    if not args.model:
        args.model = "gemini-1.5-flash" if args.provider == "gemini" else "gpt-4o-mini"
        
    api_key = args.api_key or (os.environ.get("GEMINI_API_KEY") if args.provider == "gemini" else os.environ.get("OPENAI_API_KEY"))
    if not api_key and not args.run_mock:
        logger.warning("No API key found. Running in MOCK mode.")
        args.run_mock = True

    output_path = Path(args.output)
    checkpoint_path = output_path.with_name(f"{output_path.stem}_checkpoint.json")

    # Load existing progress & select/reconstruct baseline languages order
    generated_dataset = []
    checkpoint_langs = []
    
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                generated_dataset = json.load(f)
            # Find the primary languages of the baseline items in order of occurrence
            for item in generated_dataset:
                p_lang = item.get("metadata", {}).get("primary_language", "")
                if p_lang and p_lang != lang and p_lang not in checkpoint_langs:
                    checkpoint_langs.append(p_lang)
            logger.info(f"Resuming from checkpoint. Loaded {len(generated_dataset)} entries.")
        except Exception:
            logger.warning("Failed to parse checkpoint. Starting fresh.")

    # Select other languages for the baseline, preserving the order of already generated languages
    other_langs = [l for l in SUPPORTED_LANGUAGES if l != lang]
    remaining_langs = [l for l in other_langs if l not in checkpoint_langs]
    random.shuffle(remaining_langs)
    other_langs = checkpoint_langs + remaining_langs
    
    # Map other language baselines
    baseline_mappings = {
        "base_lang_1": (other_langs[0], "normal"),
        "base_lang_2": (other_langs[0], "code_switch"),
        "base_lang_3": (other_langs[1], "normal"),
        "base_lang_4": (other_langs[2], "normal"),
        "base_lang_5": (other_langs[3], "normal"),
        "base_lang_6": (other_langs[4], "normal"),
    }

    # Build reverse mappings for correct completed scenario parsing
    reverse_baseline_mappings = {}
    for base_sc, (t_lang, sub_sc) in baseline_mappings.items():
        reverse_baseline_mappings[f"{t_lang}_{sub_sc}"] = base_sc

    # Parse completed scenarios
    completed_scenarios = {}
    for item in generated_dataset:
        sc_meta = item.get("metadata", {}).get("scenario", "")
        if sc_meta in reverse_baseline_mappings:
            sc = reverse_baseline_mappings[sc_meta]
        else:
            sc = sc_meta.replace(f"{lang}_", "")
        completed_scenarios[sc] = completed_scenarios.get(sc, 0) + 1

    # Build target counts
    target_counts = {}
    for scenario_name, count in SCENARIO_TARGET_COUNTS.items():
        target_counts[scenario_name] = max(1, int(round(count * args.scale)))

    logger.info(f"Targeting {sum(target_counts.values())} conversations for language '{SUPPORTED_LANGUAGES[lang]}' (scale={args.scale}).")

    for scenario_name, target in target_counts.items():
        current = completed_scenarios.get(scenario_name, 0)
        
        # Determine language and configuration
        if scenario_name in baseline_mappings:
            target_lang, sub_sc = baseline_mappings[scenario_name]
            sc_key = sub_sc
        else:
            target_lang = lang
            sc_key = scenario_name
            
        sys_prompt = build_system_prompt(target_lang, sc_key)
        
        # Meta values
        langs_used = [target_lang]
        if sc_key == "code_switch":
            langs_used = [target_lang, "en" if target_lang != "en" else "hi"]
        elif "switch" in sc_key:
            langs_used = [target_lang, "en" if "en" in sc_key else "hi" if "hi" in sc_key else "bn"]
            
        has_switch = "switch" in sc_key
        desc = f"Generate a conversation representing a '{sc_key}' scenario in {SUPPORTED_LANGUAGES[target_lang]}."

        while current < target:
            logger.info(f"[{scenario_name}] Generating dialogue {current + 1} of {target}...")
            
            conv = generate_single_conversation(
                lang=target_lang,
                scenario=sc_key,
                system_prompt=sys_prompt,
                description=desc,
                languages_used=langs_used,
                has_switch=has_switch,
                provider=args.provider,
                api_key=api_key,
                model_name=args.model,
                run_mock=args.run_mock
            )
            
            if conv:
                if scenario_name in baseline_mappings:
                    conv["metadata"]["scenario"] = f"{target_lang}_{sub_sc}"
                    
                generated_dataset.append(conv)
                current += 1
                completed_scenarios[scenario_name] = current
                
                # Checkpoint
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
                    
                if not args.run_mock and api_key:
                    time.sleep(args.delay)
            else:
                logger.error("Failed to generate dialogue. Retrying in 5 seconds...")
                time.sleep(5)

    # Save final output
    logger.info(f"Writing final dataset to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        logger.info("Dataset generation completed successfully!")
    except Exception as e:
        logger.error(f"Failed to write file: {e}")


if __name__ == "__main__":
    main()
