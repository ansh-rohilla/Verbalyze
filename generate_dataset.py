#!/usr/bin/env python3
"""
generate_dataset.py

Unified synthetic dataset generator script supporting 10 languages:
- English (en), Gujarati (gu), Hindi (hi), Kannada (kn), Malayalam (ml), 
  Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), and Telugu (te).

This script dynamically configures system prompts, scenario distributions, 
and metadata for the chosen target language, and generates the dataset 
using Google Gemini API, OpenAI API, or an offline Mock Mode.
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
        "fillers": "ಹೌದು, ಸರಿ, ಓಕೆ, ಹೇಳಿ, ಅಲ್ಲ, ಹೌದಾ, ಹ្មಮ್",
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
        "fillers": "అవును, సరే, ఓకే, చెప్పండి, ఏంటి",
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
        f"You are a friendly, warm, and natural-sounding voice assistant having a real phone conversation with a {SUPPORTED_LANGUAGES[lang]}-speaking person. "
        f"You must sound EXACTLY like a real human — not a robot, not an AI chatbot.\n\n"
        f"CRITICAL RULES:\n"
        f"- Respond in {SUPPORTED_LANGUAGES[lang]} ({p['native_name']}) using {p['script']} script\n"
        f"- Use natural {SUPPORTED_LANGUAGES[lang]} fillers: {p['fillers']}\n"
        f"- Keep sentences SHORT — spoken {SUPPORTED_LANGUAGES[lang]}, not formal written text\n"
        f"- NEVER use bullet points, numbered lists, markdown, or any formatting\n"
        f"- NEVER say you are an AI or reveal you are artificial\n"
        f"- Show warmth, empathy, and personality\n"
        f"- ALL numbers as spoken words (e.g., {p['num_word_example']})"
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


# Simplified templates for Mock Mode for all languages
MOCK_DIALOGUES = {
    "en": ["Hello! How can I help you today?", "I want to check my account statement.", "Sure, let me fetch that. Can I have your account number?", "It is nine eight seven six five.", "Thank you. Your account balance is five hundred dollars.", "Perfect, thank you!", "You're welcome. Have a great day!"],
    "gu": ["હેલો, નમસ્કાર! કહોને, શું મદદ કરું?", "મારે મારું લાઈટ બિલ ચેક કરવું છે.", "ચોક્કસ, મને ગ્રાહક નંબર જણાવશો?", "હા, મારો નંબર છે આઠ સાત છ પાંચ ચાર.", "સારું, તમારું બિલ ચારસો રૂપિયા આવ્યું છે.", "ઠીક છે, હું ભરી દઈશ. આભાર!", "કોઈ વાત નહીં, આપનો દિવસ સારો રહે!"],
    "hi": ["नमस्ते! मैं आपकी क्या सेवा कर सकती हूँ?", "मुझे कल के मौसम के बारे में जानना था। काफी गर्मी है आज।", "कल का मौसम? हाँ, बिल्कुल। कल हल्की बारिश होने की संभावना है, जिससे तापमान थोड़ा गिरेगा।", "ओह, बारिश? चलो बढ़िया है। बहुत-बहुत धन्यवाद आपका।", "कोई बात नहीं। आपकी मदद करके मुझे ख़ुशी हुई। आपका दिन शुभ रहे!"],
    "kn": ["ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?", "ನನ್ನ ಮೊಬೈಲ್ ಬಿಲ್ ಎಷ್ಟು ಬಂದಿದೆ ಎಂದು ತಿಳಿಸಿ.", "ಖಂಡಿತ, ನಿಮ್ಮ ಹತ್ತು ಅಂಕಿಯ ಮೊಬೈಲ್ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳಿ.", "ನನ್ನ ಸಂಖ್ಯೆ ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು.", "ನಿಮ್ಮ ಬಿಲ್ ಮುನ್ನೂರು ರೂಪಾಯಿ ಆಗಿದೆ. ಕಟ್ಟಲು ಕೊನೆಯ ದಿನಾಂಕ ಹದಿನೈದು.", "ಸರಿ, ಧನ್ಯವಾದಗಳು.", "ಪರವಾಗಿಲ್ಲ, ಒಳ್ಳೆಯ ದಿನವಾಗಲಿ!"],
    "ml": ["ഹലോ, നമസ്കാരം! ഞാൻ എങ്ങനെയാണ് സഹായിക്കേണ്ടത്?", "എനിക്ക് പുതിയ ഗ്യാസ് കണക്ഷൻ ബുക്ക് ചെയ്യണം. എന്താണ് ചെയ്യേണ്ടത്?", "തീർച്ചയായും. നിങ്ങളുടെ ആധാർ നമ്പർ നൽകി വെബ്സൈറ്റിൽ രജിസ്റ്റർ ചെയ്യാം.", "ശരി, ഞാൻ അത് ചെയ്തു നോക്കാം. നന്ദി.", "ശരി, എന്തെങ്കിലും ആവശ്യമുണ്ടെങ്കിൽ വീണ്ടും വിളിക്കൂ. നല്ലൊരു ദിവസം ആശംസിക്കുന്നു!"],
    "mr": ["नमस्कार! मी आपली काय मदत करू शकतो?", "मला नवीन गॅस बुकिंग करायचे होते. बिल किती होईल?", "हो, नक्कीच. तुमचे गॅस बिल नऊशे पन्नास रुपये होईल. मी बुक करू का?", "हो, बुक करा. धन्यवाद.", "तुमचे बुकिंग पूर्ण झाले आहे. आपली मदत करून आनंद झाला!"],
    "or": ["ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କର କଣ ସାହାଯ୍ୟ କରିପାରିବି?", "ମୋର ମୋବାଇଲ ବିଲ୍ କେତେ ଆସିଛି ଜାଣିବାକୁ ଚାହେଁ।", "ନିଶ୍ଚିତ, ଦୟାକରି ଆପଣଙ୍କ ନମ୍ବର କୁହନ୍ତୁ।", "ମୋ ନମ୍ବର ହେଉଛି ନଅ ଆଠ ସାତ ଛଅ।", "ଆପଣଙ୍କର ବିଲ୍ ଦୁଇ ଶହ ଟଙ୍କା ହୋଇଛି।", "ଠିକ୍ ଅଛି, ମୁଁ ଦେଇଦେବି। ଧନ୍ୟବାଦ।", "କିଛି ଅସୁବିଧା ନାହିଁ, ଆପଣଙ୍କ ଦିନ ଶୁଭ ହେଉ!"],
    "pa": ["ਹਾਂਜੀ ਨਮਸਤੇ! ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?", "ਮੈਨੂੰ ਆਪਣੀ ਫਲਾਈਟ ਟਿਕਟ ਬੁੱਕ ਕਰਨੀ ਹੈ। ਦਿੱਲੀ ਤੋਂ ਮੁੰਬਈ।", "ਜ਼ਰੂਰ ਜੀ। ਕਿਹੜੀ ਤਾਰੀਖ ਦੀ ਬੁਕਿੰਗ ਕਰਨੀ ਹੈ?", "ਅਗਲੇ ਮੰਗਲਵਾਰ ਦੀ, ਯਾਨੀ ਪੰਦਰਾਂ ਤਾਰੀਖ ਦੀ।", "ਠੀਕ ਹੈ ਜੀ, ਟਿਕਟ ਬੁੱਕ ਹੋ ਗਈ ਹੈ। ਤੁਹਾਨੂੰ ਮੈਸੇਜ ਮਿਲ ਜਾਵੇਗਾ।", "ਬਹੁਤ ਬਹੁਤ ਧੰਨਵਾਦ ਤੁਹਾਡਾ।", "ਕੋਈ ਗੱਲ ਨਹੀਂ ਜੀ, ਰੱਬ ਰਾਖਾ!"],
    "ta": ["வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்?", "எனது மின்சாரக் கட்டணத்தை நான் எவ்வாறு சரிபார்ப்பது?", "நிச்சயமாக. உங்கள் நுகர்வோர் எண்ணை எனக்கு சொல்ல முடியுமா?", "ஆம், எனது எண் ஒன்று இரண்டு மூன்று நான்கு ஐந்து.", "நன்றி. உங்கள் கட்டணம் நானூற்று ஐம்பது ரூபாய். கடைசி தேதி ஜூன் பதினைந்து.", "சரி, மிக்க நன்றி.", "பரவாயில்லை. நல்ல நாள் அமையட்டும்!"],
    "te": ["నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?", "నాకు రేపటి వాతావరణం గురించి కొంచెం సమాచారం కావాలి.", "తప్పకుండా. రేపు వర్ഷం పడే అవకాశం ఉంది, ఉష్ణోగ్రత ముప్పై డిగ్రీలు ఉంటుంది.", "చాలా ధన్యవాదాలు అండి.", "పర్వాలేదండి. మీకు మంచి రోజు కలగాలని ఆశిస్తున్నాను!"]
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

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
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
        # Mock mode fallback
        lines = MOCK_DIALOGUES.get(lang, MOCK_DIALOGUES["en"])
        messages = []
        # Build turns alternating user/assistant
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

    # Load existing progress
    generated_dataset = []
    completed_scenarios = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                generated_dataset = json.load(f)
            for item in generated_dataset:
                sc_meta = item.get("metadata", {}).get("scenario", "")
                # Normalize scenario name
                sc = sc_meta.replace(f"{lang}_", "")
                completed_scenarios[sc] = completed_scenarios.get(sc, 0) + 1
            logger.info(f"Resuming from checkpoint. Loaded {len(generated_dataset)} entries.")
        except Exception:
            logger.warning("Failed to parse checkpoint. Starting fresh.")

    # Select other languages for the baseline
    other_langs = [l for l in SUPPORTED_LANGUAGES if l != lang]
    random.shuffle(other_langs)
    
    # Map other language baselines
    baseline_mappings = {
        "base_lang_1": (other_langs[0], "normal"),
        "base_lang_2": (other_langs[0], "code_switch"),
        "base_lang_3": (other_langs[1], "normal"),
        "base_lang_4": (other_langs[2], "normal"),
        "base_lang_5": (other_langs[3], "normal"),
        "base_lang_6": (other_langs[4], "normal"),
    }

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
                # Override scenario metadata name for baseline mappings to match standard expected keys
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
