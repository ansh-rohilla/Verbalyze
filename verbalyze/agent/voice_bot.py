"""
verbalyze/agent/voice_bot.py

Full-Duplex Conversational Telephony Voice Agent:
- Muthoot Fincorp Outbound Debt Recovery / Banking Support State Machine
- Natural spoken filler injection
- Live telephony action tools (disconnect_tool, send_payment_link, schedule_callback)
- Audio synthesis and interactive terminal call simulator
"""

import os
import re
import sys
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from verbalyze.agent.tools import TELEPHONY_TOOLS_SCHEMA, execute_telephony_tool
from verbalyze.agent.audio_engine import AudioEngine
from verbalyze.security import detect_prompt_injection, PIIRedactor
from verbalyze.agent.sentiment import (
    UnifiedSentimentEngine,
    SentimentResult,
    SentimentCategory,
    DisputeType,
)
from verbalyze.agent.lid_engine import (
    LanguageIdentificationGate,
    LanguageIDResult,
    ScriptType,
)
from verbalyze.telephony.voice_biometrics import (
    BiometricVerificationEngine,
    BiometricVerificationResult,
    BiometricStatus,
    SpeakerProfile,
)

INITIAL_GREETINGS = {
    "muthoot_recovery": {
        "hi": "नमस्कार, क्या मेरी बात मिस्टर शर्मा से हो रही है? मैं मुथूट फिनकॉर्प से बोल रही हूँ।",
        "en": "Hello, am I speaking with Mr. Sharma? I am calling from Muthoot Fincorp regarding your loan EMI.",
        "gu": "નમસ્તે, શું હું મિસ્ટર શર્મા સાથે વાત કરી રહ્યો છું? હું મુથૂટ ફિનકોર્પમાંથી વાત કરું છું.",
        "ta": "வணக்கம், நான் திரு. ஷர்மாவுடன் பேசுகிறேனா? நான் முத்தூட் ஃபின்கார்ப் நிறுவனத்திலிருந்து பேசுகிறேன்.",
        "te": "నమస్కారం, నేను మిస్టర్ శర్మతో మాట్లాడుతున్నానా? నేను ముత్తూట్ ఫిన్‌కార్ప్ నుండి మాట్లాడుతున్నాను.",
        "mr": "नमस्कार, मी मिस्टर शर्मा यांच्याशी बोलत आहे का? मी मुथूट फिनकॉर्पमधून बोलत आहे.",
        "bn": "নমস্কার, আমি কি মিস্টার শর্মার সাথে কথা বলছি? আমি মুথুট ফিনকর্প থেকে বলছি।",
        "kn": "ನಮಸ್ಕಾರ, ನಾನು ಮಿಸ್ಟರ್ ಶರ್ಮಾ ಅವರೊಂದಿಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆಯೇ? ನಾನು ಮುತ್ತೂಟ್ ಫಿನ್‌ಕಾರ್ಪ್‌ನಿಂದ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
        "ml": "നമസ്കാരം, ഞാൻ മിസ്റ്റർ ശർമ്മയോടാണോ സംസാരിക്കുന്നത്? ഞാൻ മുത്തൂറ്റ് ഫിൻകോർപ്പിൽ നിന്നാണ് വിളിക്കുന്നത്.",
        "pa": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਕੀ ਮੈਂ ਮਿਸਟਰ ਸ਼ਰਮਾ ਨਾਲ ਗੱਲ ਕਰ ਰਿਹਾ ਹਾਂ? ਮੈਂ ਮੁਥੂਟ ਫਿਨਕੋਰਪ ਤੋਂ ਬੋਲ ਰਿਹਾ ਹਾਂ।",
        "or": "ନମସ୍କାର, ମୁଁ ମିଷ୍ଟର ଶର୍ମାଙ୍କ ସହ କଥା ହେଉଛି କି? ମୁଁ ମୁଥୁଟ୍ ଫିନକର୍ପରୁ କହୁଛି।",
        "ur": "السلام علیکم، کیا میری بات مسٹر شرما سے ہو रही ہے؟ میں متھوٹ فن کارپ سے بات کر رہا ہوں۔",
    },
    "bank_kyc": {
        "hi": "नमस्कार, मैं बैंक से बोल रहा हूँ। आपके सेविंग्स अकाउंट का री-केवाईसी वेरिफिकेशन पेंडिंग है।",
        "en": "Hello, I am calling from the bank. Your account re-KYC verification update is currently pending.",
        "gu": "નમસ્તે, હું બેંકમાંથી વાત કરું છું. તમારા ખાતાનું રી-કેવાયસી વેરિફિકેશન બાકી છે.",
        "ta": "வணக்கம், நான் வங்கியிலிருந்து பேசுகிறேன். உங்கள் வங்கிக் கணக்கின் ரீ-கேஒய்சி புதுப்பிப்பு நிலுவையில் உள்ளது.",
        "te": "నమస్కారం, నేను బ్యాంకు నుండి మాట్లాడుతున్నాను. మీ ఖాతా రీ-కేవైసీ వెరిఫికేషన్ పెండింగ్‌లో ఉంది.",
        "mr": "नमस्कार, मी बँकेतून बोलत आहे. तुमच्या खात्याचे री-केवायसी पडताळणी बाकी आहे.",
        "bn": "নমস্কার, আমি ব্যাংক থেকে বলছি। আপনার সেভিংস অ্যাকাউন্টের রি-কেওয়াইসি ভেরিফিকেশন বাকি রয়েছে।",
        "kn": "ನಮಸ್ಕಾರ, ನಾನು ಬ್ಯಾಂಕ್‌ನಿಂದ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನಿಮ್ಮ ಖಾತೆಯ ರೀ-ಕೆವೈಸಿ ಪರಿಶೀಲನೆ ಬಾಕಿ ಇದೆ.",
        "ml": "നമസ്കാരം, ഞാൻ ബാങ്കിൽ നിന്നാണ് വിളിക്കുന്നത്. നിങ്ങളുടെ അക്കൗണ്ടിന്റെ റീ-കെവൈസി വെരിഫിക്കേഷൻ തീർപ്പാക്കാനുണ്ട്.",
        "pa": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਮੈਂ ਬੈਂਕ ਤੋਂ ਬੋਲ ਰਿਹਾ ਹਾਂ। ਤੁਹਾਡੇ ਖਾਤੇ ਦੀ ਰੀ-ਕੇਵਾਈਸੀ ਪੈਂਡਿੰਗ ਹੈ।",
        "or": "ନମସ୍କାର, ମୁଁ ବ୍ୟାଙ୍କରୁ କହୁଛି। ଆପଣଙ୍କ ଖାତାର ରି-କେୱାଇସି ଯାଞ୍ଚ ବାକି ଅଛି।",
        "ur": "السلام علیکم، میں بینک سے بات کر رہا ہوں۔ آپ کے اکاؤنٹ کی ری-کے وائی سی ویریفیکیشن پینڈنگ ہے۔",
    },
    "swiggy_delivery": {
        "hi": "नमस्ते सर, मैं स्विगी से डिलीवरी पार्टनर बोल रहा हूँ। आपकी लोकेशन पर गेट नंबर क्या है?",
        "en": "Hello sir, I am your Swiggy delivery partner. Could you please confirm the society gate or flat number?",
        "gu": "નમસ્તે સર, હું સ્વિગી ડિલિવરી પાર્ટનર બોલું છું. તમારી સોસાયટીનો ગેટ નંબર શું છે?",
        "ta": "வணக்கம் சார், நான் ஸ்விக்கி டெலிவரி பார்ட்னர் பேசுகிறேன். உங்கள் அப்பார்ட்மென்ட் கேட் எண் என்ன?",
        "te": "నమస్కారం సర్, నేను స్విగ్గీ డెలివరీ భాగస్వామిని మాట్లాడుతున్నాను. మీ అపార్ట్‌మెంట్ గేట్ నంబర్ ఏమిటి?",
        "mr": "नमस्कार सर, मी स्विगी डिलिव्हरी पार्टनर बोलत आहे. तुमच्या सोसायटीचा गेट नंबर काय आहे?",
        "bn": "নমস্কার স্যার, আমি সুইগি ডেলিভারি পার্টনার বলছি। আপনার গেট নম্বরটি দয়া করে বলবেন?",
        "kn": "ನಮಸ್ಕಾರ ಸರ್, ನಾನು ಸ್ವಿಗ್ಗಿ ಡೆಲಿವರಿ ಪಾರ್ಟ್ನರ್ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನಿಮ್ಮ ಗೇಟ್ ನಂಬರ್ ಯಾವುದು?",
        "ml": "നമസ്കാരം സർ, ഞാൻ സ്വിഗ്ഗി ഡെലിവറി പാർട്ണറാണ് സംസാരിക്കുന്നത്. നിങ്ങളുടെ ഗേറ്റ് നമ്പർ ഏതാണ്?",
        "pa": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਸਰ, ਮੈਂ ਸਵਿੱਗੀ ਡਿਲੀਵਰੀ ਪਾਰਟਨਰ ਬੋਲ ਰਿਹਾ ਹਾਂ। ਤੁਹਾਡਾ ਗੇਟ ਨੰਬਰ ਕੀ ਹੈ?",
        "or": "ନମସ୍କାର ସାର୍, ମୁଁ ସ୍ୱିଗୀ ଡେଲିଭରି ପାର୍ଟନର କହୁଛି। ଆପଣଙ୍କ ଗେଟ୍ ନମ୍ବର କ'ଣ?",
        "ur": "السلام علیکم سر، میں سویگی ڈیلیوری پارٹنر بات کر رہا ہوں۔ آپ کا سوسائٹی گیٹ نمبر کیا ہے؟",
    }
}

DEFAULT_SYSTEM_PROMPTS = {
    "hi": (
        "You are a friendly, polite, yet professional outbound collection agent representing Muthoot Fincorp. "
        "You are having a real phone conversation with a Hindi-speaking customer regarding their overdue loan EMI recovery of Rs. 5,420. "
        "You must sound EXACTLY like a real human agent — not a robot, not an AI chatbot.\n\n"
        "CRITICAL TELEPHONY RULES:\n"
        "- Respond in conversational Hindi (Devanagari script or natural Hinglish)\n"
        "- Keep sentences SHORT (1-2 sentences maximum per turn) — spoken phone dialogue, not long text\n"
        "- Use natural fillers: 'haan', 'hmm', 'theek hai', 'ji'\n"
        "- Do NOT call tools unless specifically needed: only invoke send_payment_link if customer asks for payment link or UPI, and only invoke disconnect_tool if customer explicitly says goodbye or ends the call."
    ),
    "en": (
        "You are a friendly, polite, yet professional outbound collection agent representing Muthoot Fincorp. "
        "You are having a real phone call with an English-speaking customer regarding their overdue loan EMI recovery of Rs. 5,420. "
        "You must sound EXACTLY like a real human agent — not a robot, not an AI chatbot.\n\n"
        "CRITICAL TELEPHONY RULES:\n"
        "- Respond in spoken English\n"
        "- Keep sentences SHORT (1-2 sentences maximum per turn) — spoken phone dialogue, not long text\n"
        "- Use natural fillers: 'yeah', 'right', 'hmm', 'I understand'\n"
        "- Do NOT call tools unless specifically needed: only invoke send_payment_link if customer asks for payment link, and only invoke disconnect_tool if customer explicitly says goodbye or ends the call."
    ),
}


class VoiceAgent:
    def __init__(
        self,
        language: str = "hi",
        persona: str = "muthoot_recovery",
        system_prompt: Optional[str] = None,
        llm_provider: str = "groq",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        ollama_host: Optional[str] = None,
        voice_enabled: bool = True,
        min_human_likeness: float = 0.80,
        simulate_telephony: bool = False,
        caller_phone: Optional[str] = None,
        adaptive_language: bool = True,
        biometrics_engine: Optional[BiometricVerificationEngine] = None,
        enrolled_profile: Optional[SpeakerProfile] = None,
    ):
        self.language = language
        self.persona = persona
        self.caller_phone = caller_phone
        self.min_human_likeness = min_human_likeness
        self.simulate_telephony = simulate_telephony
        self.adaptive_language = adaptive_language
        self.biometrics_engine = biometrics_engine
        self.enrolled_profile = enrolled_profile
        self.last_biometric: Optional[BiometricVerificationResult] = None
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPTS.get(language, DEFAULT_SYSTEM_PROMPTS["hi"])
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

        if model_name:
            self.model_name = model_name
        elif self.llm_provider == "ollama":
            self.model_name = "verbalyze-indic"
        elif self.llm_provider == "groq":
            self.model_name = "llama-3.3-70b-versatile"
        else:
            self.model_name = "gpt-4o-mini"

        self.voice_enabled = voice_enabled
        self.audio_engine = AudioEngine(
            language=language,
            min_human_likeness=min_human_likeness,
            simulate_telephony=simulate_telephony
        ) if voice_enabled else None

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.is_call_active = True
        self.sentiment_engine = UnifiedSentimentEngine()
        self.last_sentiment: Optional[SentimentResult] = None
        self.deescalation_active: bool = False
        self.lid_gate = LanguageIdentificationGate(default_language=language)
        self.last_lid: Optional[LanguageIDResult] = None
        self.pending_whispers: List[str] = []
        self.whisper_history: List[str] = []

    def inject_supervisor_whisper(self, whisper_text: str):
        """
        Injects a private supervisor coaching directive into the agent's context.
        The agent incorporates this instruction into the very next turn generation.
        """
        cleaned = str(whisper_text).strip()
        if cleaned:
            self.pending_whispers.append(cleaned)
            self.whisper_history.append(cleaned)

    def get_initial_greeting(self) -> str:
        """Returns localized initial greeting for the selected persona."""
        persona_greetings = INITIAL_GREETINGS.get(self.persona, INITIAL_GREETINGS["muthoot_recovery"])
        return persona_greetings.get(self.language, persona_greetings.get("en", "Hello, how can I help you today?"))


    def _call_groq_or_openai(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Calls LLM (Ollama, Groq, or OpenAI) with telephony function calling tools."""
        import urllib.request

        if self.llm_provider == "mock":
            return self._mock_llm_response()

        if self.llm_provider == "ollama":
            url = f"{self.ollama_host.rstrip('/')}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer ollama"
            }
        elif self.llm_provider == "groq":
            if not self.api_key:
                return self._mock_llm_response()
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            if not self.api_key:
                return self._mock_llm_response()
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "tools": TELEPHONY_TOOLS_SCHEMA,
            "tool_choice": "auto",
            "temperature": 0.6,
            "max_tokens": 150
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]["message"]
                content = choice.get("content") or ""
                tool_calls = choice.get("tool_calls")
                
                tool_call = tool_calls[0] if tool_calls else None
                return content, tool_call
        except Exception as e:
            if self.llm_provider == "ollama" and self.model_name == "verbalyze-indic":
                # Fallback to base llama3.2:3b if verbalyze-indic is not yet created
                try:
                    payload["model"] = "llama3.2:3b"
                    req_fallback = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST"
                    )
                    with urllib.request.urlopen(req_fallback, timeout=15) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        choice = data["choices"][0]["message"]
                        content = choice.get("content") or ""
                        tool_calls = choice.get("tool_calls")
                        tool_call = tool_calls[0] if tool_calls else None
                        return content, tool_call
                except Exception:
                    pass

            if self.llm_provider == "ollama":
                print(f"[Ollama Notice] Failed connecting to {url}: {e}. Falling back to rule-based telephony response.")
            # Fallback to simulated telephony response
            return self._mock_llm_response()

    def _mock_llm_response(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Rule-based telephony response when offline or without API key."""
        last_user_msg = ""
        for m in reversed(self.messages):
            if m["role"] == "user":
                last_user_msg = m["content"].lower()
                break

        farewell_keywords = ["bye", "alvida", "hang up", "rakhta hoon", "baad me", "disconnect", "अलविदा", "बाय", "रखता", "नमस्कार", "धन्यवाद", "रखती"]
        link_keywords = ["link", "upi", "qr", "online", "bhejo", "लिंक", "भेज", "यूपीआई", "पेमेंट", "कर दो"]
        callback_keywords = ["kal", "tomorrow", "next week", "tarikh", "pay", "कल", "हफ्ते", "तारीख", "सोमवार"]

        # Check for active supervisor coaching directive
        if self.whisper_history:
            latest_whisper = self.whisper_history[-1].lower()
            if any(w in latest_whisper for w in ["waive", "discount", "fee", "penalty", "छूट", "माफ़"]):
                if self.language == "hi":
                    return "जी, हमारे सुपरवाइजर के निर्देशानुसार हम आपका लेट फीस शुल्क माफ कर सकते हैं। क्या आप शेष ₹5,420 का भुगतान अभी कर सकते हैं?", None
                return "Yes, as approved by our supervisor, we can waive your late payment fee. Can you settle the remaining amount now?", None
            elif any(w in latest_whisper for w in ["manager", "supervisor", "escalat", "अधिकारी"]):
                if self.language == "hi":
                    return "जी, हमारे वरिष्ठ अधिकारी इस कॉल की निगरानी कर रहे हैं और हम आपकी समस्या का तुरंत समाधान करेंगे।", None
                return "Yes, our senior supervisor is actively monitoring this call and we will resolve your issue immediately.", None

        if any(w in last_user_msg for w in farewell_keywords):
            tool_call = {
                "function": {
                    "name": "disconnect_tool",
                    "arguments": json.dumps({"reason": "farewell_exchanged"})
                }
            }
            if self.language == "hi":
                return "जी ठीक है, आपका बहुत धन्यवाद। आपका दिन शुभ हो, नमस्कार।", tool_call
            return "Thank you for your time. Have a great day ahead, goodbye.", tool_call

        elif any(w in last_user_msg for w in link_keywords):
            tool_call = {
                "function": {
                    "name": "send_payment_link",
                    "arguments": json.dumps({"amount": 5420.0, "loan_id": "MUTH-8921"})
                }
            }
            if self.language == "hi":
                return "हाँ बिल्कुल, मैंने आपके पंजीकृत मोबाइल नंबर पर ₹5,420 का UPI पेमेंट लिंक भेज दिया है।", tool_call
            return "Sure, I have sent the UPI payment link for Rs. 5,420 to your registered number.", tool_call

        elif any(w in last_user_msg for w in callback_keywords):
            tool_call = {
                "function": {
                    "name": "schedule_callback",
                    "arguments": json.dumps({"promised_date": "tomorrow", "notes": "Customer promised payment"})
                }
            }
            if self.language == "hi":
                return "हाँ जी ठीक है, मैंने कल तक के भुगतान का वादा नोट कर लिया है। कृपया कल तक जमा कर दें।", tool_call
            return "Alright, I have recorded your commitment to pay by tomorrow. Please ensure it is cleared.", tool_call

        # Persona-specific responses
        if self.persona == "bank_kyc":
            if any(w in last_user_msg for w in farewell_keywords):
                tool_call = {"function": {"name": "disconnect_tool", "arguments": json.dumps({"reason": "kyc_guidance_provided"})}}
                return ("जी ठीक है, आपका बहुत धन्यवाद। कृपया आज ही री-केवाईसी पूरा कर लें, नमस्कार।" if self.language == "hi" 
                        else "Thank you for your time. Please complete your re-KYC today. Goodbye."), tool_call
            if any(w in last_user_msg for w in link_keywords):
                tool_call = {"function": {"name": "send_payment_link", "arguments": json.dumps({"link_type": "kyc_portal", "account_id": "BARB-5012"})}}
                return ("हाँ बिल्कुल, मैंने आपके मोबाइल पर सुरक्षित री-केवाईसी वेरिफिकेशन लिंक भेज दिया है।" if self.language == "hi"
                        else "Sure, I have dispatched the secure video re-KYC portal link to your registered mobile."), tool_call
            return ("हाँ जी, आपका खाता एक्टिव रखने के लिए री-केवाईसी जरूरी है। क्या मैं आपको ऑनलाइन वीडियो केवाईसी का लिंक भेज दूँ?" if self.language == "hi"
                    else "Yes, re-KYC is mandatory to keep your account operational. May I send you the online video KYC link?"), None

        elif self.persona == "swiggy_delivery":
            if any(w in last_user_msg for w in farewell_keywords):
                tool_call = {"function": {"name": "disconnect_tool", "arguments": json.dumps({"reason": "order_delivered"})}}
                return ("जी धन्यवाद सर, ऑर्डर डिलीवर हो गया है। आपका दिन शुभ हो!" if self.language == "hi"
                        else "Thank you sir, order has been delivered. Enjoy your meal!"), tool_call
            return ("जी ठीक है सर, मैं 2 मिनट में आपकी लोकेशन पर पहुँच रहा हूँ। कृपया गेट पर रहिए।" if self.language == "hi"
                    else "Sure sir, I am arriving at your building in 2 minutes. Please be available at the gate."), None

        # Default Muthoot loan recovery replies
        if self.language == "hi":
            return "हाँ जी, मैं मुथूट फिनकॉर्प से बोल रहा हूँ। क्या आप आज अपनी बकाया ईएमआई जमा कर पाएंगे?", None
        return "Yes, I am calling from Muthoot Fincorp regarding your pending loan EMI. Could you confirm when you can clear it?", None

    def step(self, user_utterance: str, pcm_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """Processes a single conversational turn from the user with emotion & dispute analysis."""
        if not self.is_call_active:
            return {"text": "[Call already disconnected]", "terminated": True}

        # 0. Conversational Prompt Injection Defense
        is_injection, reason = detect_prompt_injection(user_utterance)
        if is_injection:
            print(f"[Security Guard] Blocked prompt injection: {reason}")
            safe_reply = (
                "क्षमा करें, मैं केवल आपके लोन खाते और ईएमआई भुगतान के संबंध में सहायता कर सकती हूँ।"
                if self.language == "hi"
                else "I apologize, but I am only authorized to assist with your loan account and EMI payments."
            )
            return {
                "text": safe_reply,
                "terminated": False,
                "tool_event": None,
                "audio_path": self.audio_engine.synthesize(safe_reply) if self.audio_engine else None,
                "security_block": True
            }

        # 0.4. Multi-Modal Language Identification & Dynamic Adaptation
        lid_result = self.lid_gate.identify(
            transcript=user_utterance,
            pcm_bytes=pcm_bytes,
            current_language=self.language,
        )
        self.last_lid = lid_result

        # Dynamically adapt language and voice if customer transitioned languages
        if self.adaptive_language and lid_result.language_switched:
            new_lang = lid_result.primary_language
            self.language = new_lang
            if self.audio_engine:
                self.audio_engine.set_language(new_lang, lid_result.recommended_voice)
            self.messages.append({
                "role": "system",
                "content": f"[Language Switch] Customer is now speaking in {new_lang.upper()}. Respond naturally in {new_lang.upper()}."
            })

        # 0.5. Dual-Channel Emotion & Dispute Analysis
        sentiment = self.sentiment_engine.analyze(user_utterance, pcm_bytes=pcm_bytes)
        self.last_sentiment = sentiment

        # 0.6. Passive Voice Biometrics & Anti-Spoofing Verification
        if self.biometrics_engine and self.enrolled_profile and pcm_bytes:
            bio_res = self.biometrics_engine.verify_speaker(pcm_bytes, self.enrolled_profile)
            self.last_biometric = bio_res

            if bio_res.status == BiometricStatus.SPOOF_DETECTED:
                safe_reply = (
                    "सुरक्षा कारणों से आपकी आवाज़ सत्यापित नहीं हो सकी है। कृपया निकटतम शाखा से संपर्क करें।"
                    if self.language == "hi"
                    else "For security reasons, synthetic or replayed voice was detected. Loan disclosure is blocked. Please contact your branch."
                )
                return {
                    "text": safe_reply,
                    "audio_path": self.audio_engine.synthesize(safe_reply) if self.audio_engine else None,
                    "quality_report": None,
                    "tool_event": None,
                    "tool_data": None,
                    "sentiment": sentiment.to_dict(),
                    "language_info": lid_result.to_dict(),
                    "whispers": list(self.whisper_history),
                    "biometric_result": bio_res.to_dict(),
                    "security_block": True,
                    "terminated": False,
                }
            elif bio_res.status == BiometricStatus.MISMATCH_IMPOSTOR:
                safe_reply = (
                    "सुरक्षा कारणों से आपकी आवाज़ लोन धारक के प्रोफ़ाइल से मेल नहीं खा रही है। मैं आगे की जानकारी साझा नहीं कर सकती।"
                    if self.language == "hi"
                    else "For security reasons, your voice does not match the registered borrower profile. Sensitive account details cannot be disclosed."
                )
                return {
                    "text": safe_reply,
                    "audio_path": self.audio_engine.synthesize(safe_reply) if self.audio_engine else None,
                    "quality_report": None,
                    "tool_event": None,
                    "tool_data": None,
                    "sentiment": sentiment.to_dict(),
                    "language_info": lid_result.to_dict(),
                    "whispers": list(self.whisper_history),
                    "biometric_result": bio_res.to_dict(),
                    "security_block": True,
                    "terminated": False,
                }
            elif bio_res.status == BiometricStatus.INDETERMINATE:
                self.messages.append({
                    "role": "system",
                    "content": "[Biometric Alert]: Speaker voice confidence is indeterminate. Require customer to verify last 4 digits of PAN before disclosing overdue amount."
                })

        # 1. Record user turn
        self.messages.append({"role": "user", "content": user_utterance})

        # Inject pending supervisor whisper guidance into system context
        if self.pending_whispers:
            whisper_directive = " | ".join(self.pending_whispers)
            self.messages.append({
                "role": "system",
                "content": f"[SUPERVISOR COACHING]: {whisper_directive}. Follow this instruction immediately."
            })
            self.pending_whispers.clear()

        # Check if automatic escalation or human transfer is triggered
        auto_transfer = False
        if sentiment.transfer_recommended and (
            sentiment.category in (SentimentCategory.CRITICAL, SentimentCategory.AGITATED)
            or sentiment.dispute_type in (DisputeType.LEGAL_THREAT, DisputeType.HARASSMENT_COMPLAINT, DisputeType.HUMAN_REQUEST)
        ):
            auto_transfer = True

        if auto_transfer:
            transfer_reason = sentiment.dispute_type.value if sentiment.dispute_type != DisputeType.NONE else "high_agitation"
            target_dept = "disputes" if sentiment.dispute_type in (DisputeType.PAYMENT_DISPUTE, DisputeType.LEGAL_THREAT) else "supervisor"
            tool_call = {
                "function": {
                    "name": "transfer_to_human",
                    "arguments": json.dumps({
                        "reason": transfer_reason,
                        "customer_sentiment": sentiment.category.value,
                        "summary": f"Customer escalation ({transfer_reason}): {user_utterance[:100]}",
                        "target_department": target_dept,
                    })
                }
            }
            content = (
                "मैं आपकी स्थिति समझ सकती हूँ। कृपया एक क्षण प्रतीक्षा करें, मैं आपकी कॉल वरिष्ठ अधिकारी को ट्रांसफर कर रही हूँ।"
                if self.language == "hi"
                else "I understand your situation. Please hold for a moment while I transfer you to a senior officer."
            )
        else:
            if sentiment.deescalation_recommended:
                self.deescalation_active = True

            # 2. Generate assistant response
            content, tool_call = self._call_groq_or_openai()

            # Apply empathetic de-escalation tone if customer is elevated/agitated
            if self.deescalation_active and content and not tool_call:
                empathy_prefix = (
                    "मैं आपकी चिंता समझ सकती हूँ। "
                    if self.language == "hi"
                    else "I understand your concern. "
                )
                if not content.startswith(empathy_prefix.strip()[:10]):
                    content = empathy_prefix + content

        # Clean up JSON leaks or token fragments from small local SLMs
        if content:
            cleaned = content.strip()
            if cleaned.startswith("};") or (cleaned.startswith("{") and cleaned.endswith("}")):
                content = ""
            elif "};" in cleaned:
                content = cleaned.split("};")[-1].strip()

        # Handle inline tool invocation emitted in speech text by local SLMs
        if not tool_call and content:
            if "send_payment_link" in content:
                tool_call = {
                    "function": {
                        "name": "send_payment_link",
                        "arguments": json.dumps({"amount": 5420.0, "loan_id": "MUTH-8921"})
                    }
                }
                lines = [l for l in content.split("\n") if "send_payment_link" not in l and "आरेख" not in l]
                content = " ".join(lines).strip()
            elif "disconnect_tool" in content:
                tool_call = {
                    "function": {
                        "name": "disconnect_tool",
                        "arguments": json.dumps({"reason": "farewell_exchanged"})
                    }
                }
                lines = [l for l in content.split("\n") if "disconnect_tool" not in l and "आरेख" not in l]
                content = " ".join(lines).strip()
            elif "schedule_callback" in content:
                tool_call = {
                    "function": {
                        "name": "schedule_callback",
                        "arguments": json.dumps({"promised_date": "tomorrow", "notes": "Customer promised payment"})
                    }
                }
                lines = [l for l in content.split("\n") if "schedule_callback" not in l and "आरेख" not in l]
                content = " ".join(lines).strip()

        # If model executed a tool, provide natural spoken confirmation if content is empty
        if tool_call and not content:
            fn_name = tool_call["function"]["name"]
            if fn_name == "send_payment_link":
                content = "हाँ बिल्कुल, मैंने आपके पंजीकृत मोबाइल नंबर पर ₹5,420 का सुरक्षित UPI पेमेंट लिंक भेज दिया है।" if self.language == "hi" else "Sure, I have dispatched the secure payment link to your registered mobile number."
            elif fn_name == "disconnect_tool":
                content = "जी ठीक है, आपका बहुत धन्यवाद। आपका दिन शुभ हो, नमस्कार।" if self.language == "hi" else "Thank you for your time. Have a great day ahead, goodbye."
            elif fn_name == "schedule_callback":
                content = "हाँ जी ठीक है, मैंने कल तक के भुगतान का वादा नोट कर लिया है। कृपया कल तक जमा कर दें।" if self.language == "hi" else "Alright, I have recorded your commitment for tomorrow. Thank you."

        # If content is still empty, provide persona-specific conversational fallback
        if not content:
            if self.persona == "bank_kyc":
                content = "हाँ जी, आपका खाता एक्टिव रखने के लिए री-केवाईसी वेरिफिकेशन जरूरी है।" if self.language == "hi" else "Yes, re-KYC update is needed to keep your account operational."
            elif self.persona == "swiggy_delivery":
                content = "जी ठीक है सर, मैं आपकी लोकेशन पर पहुँच रहा हूँ।" if self.language == "hi" else "Sure sir, I am arriving at your location."
            else:
                content = "हाँ जी, मैं मुथूट फिनकॉर्प से बोल रहा हूँ। क्या आप आज अपनी बकाया ईएमआई जमा कर पाएंगे?" if self.language == "hi" else "Yes, I am calling from Muthoot Fincorp regarding your overdue EMI payment."

        # 3. Handle assistant message
        asst_turn = {"role": "assistant", "content": content}
        if tool_call:
            asst_turn["tool_calls"] = [tool_call]
        self.messages.append(asst_turn)

        # 4. Execute tool call if triggered
        tool_status = ""
        tool_data = None
        if tool_call:
            fn_name = tool_call["function"]["name"]
            try:
                fn_args = json.loads(tool_call["function"]["arguments"])
            except Exception:
                fn_args = {}
            res = execute_telephony_tool(fn_name, fn_args, caller_phone=self.caller_phone)
            if len(res) == 3:
                terminated, tool_status, tool_data = res
            else:
                terminated, tool_status = res
            if terminated:
                self.is_call_active = False

        # 5. Synthesize voice if enabled (verified via Quality Gate)
        audio_path = None
        quality_report = None
        if self.voice_enabled and self.audio_engine and content:
            audio_path = self.audio_engine.synthesize(content)
            quality_report = self.audio_engine.last_quality_report

        return {
            "text": content,
            "audio_path": audio_path,
            "quality_report": quality_report,
            "tool_event": tool_status if tool_call else None,
            "tool_data": tool_data if tool_call else None,
            "sentiment": sentiment.to_dict(),
            "language_info": lid_result.to_dict(),
            "whispers": list(self.whisper_history),
            "biometric_result": self.last_biometric.to_dict() if self.last_biometric else None,
            "terminated": not self.is_call_active
        }

    async def step_stream(
        self,
        user_utterance: str,
        pcm_bytes: Optional[bytes] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes a conversational turn with token streaming, clause-level pipelining,
        and real-time emotion/dispute transfer handling.
        Yields:
        - {"type": "clause", "text": clause_str, "index": int}
        - {"type": "tool_call", "tool_name": str, "tool_event": str, "tool_data": Any, "terminated": bool}
        - {"type": "final", "full_text": str, "sentiment": Dict[str, Any], "terminated": bool}
        """
        if not self.is_call_active:
            yield {"type": "final", "full_text": "[Call already disconnected]", "terminated": True}
            return

        # 0. Conversational Prompt Injection Defense
        is_injection, reason = detect_prompt_injection(user_utterance)
        if is_injection:
            print(f"[Security Guard] Blocked prompt injection in stream: {reason}")
            safe_reply = (
                "क्षमा करें, मैं केवल आपके लोन खाते और ईएमआई भुगतान के संबंध में सहायता कर सकती हूँ।"
                if self.language == "hi"
                else "I apologize, but I am only authorized to assist with your loan account and EMI payments."
            )
            yield {
                "type": "clause",
                "text": safe_reply,
                "index": 1
            }
            yield {
                "type": "final",
                "full_text": safe_reply,
                "terminated": False
            }
            return

        # 0.4. Multi-Modal Language Identification & Dynamic Adaptation
        lid_result = self.lid_gate.identify(
            transcript=user_utterance,
            pcm_bytes=pcm_bytes,
            current_language=self.language,
        )
        self.last_lid = lid_result

        if self.adaptive_language and lid_result.language_switched:
            new_lang = lid_result.primary_language
            self.language = new_lang
            if self.audio_engine:
                self.audio_engine.set_language(new_lang, lid_result.recommended_voice)
            self.messages.append({
                "role": "system",
                "content": f"[Language Switch] Customer is now speaking in {new_lang.upper()}. Respond naturally in {new_lang.upper()}."
            })

        # 0.5. Dual-Channel Emotion & Dispute Analysis
        sentiment = self.sentiment_engine.analyze(user_utterance, pcm_bytes=pcm_bytes)
        self.last_sentiment = sentiment

        # 0.6. Passive Voice Biometrics & Anti-Spoofing Verification
        if self.biometrics_engine and self.enrolled_profile and pcm_bytes:
            bio_res = self.biometrics_engine.verify_speaker(pcm_bytes, self.enrolled_profile)
            self.last_biometric = bio_res
            if bio_res.status in (BiometricStatus.SPOOF_DETECTED, BiometricStatus.MISMATCH_IMPOSTOR):
                safe_reply = (
                    "सुरक्षा कारणों से आपकी आवाज़ सत्यापित नहीं हो सकी है। कृपया निकटतम शाखा से संपर्क करें।"
                    if self.language == "hi"
                    else "For security reasons, your voice could not be verified. Sensitive details cannot be disclosed."
                )
                yield {
                    "type": "clause",
                    "text": safe_reply,
                    "index": 1,
                }
                yield {
                    "type": "final",
                    "full_text": safe_reply,
                    "biometric_result": bio_res.to_dict(),
                    "security_block": True,
                    "terminated": False,
                }
                return

        # Auto-transfer under critical agitation, harassment, legal threat, or human request
        if sentiment.transfer_recommended and (
            sentiment.category in (SentimentCategory.CRITICAL, SentimentCategory.AGITATED)
            or sentiment.dispute_type in (DisputeType.LEGAL_THREAT, DisputeType.HARASSMENT_COMPLAINT, DisputeType.HUMAN_REQUEST)
        ):
            transfer_reason = sentiment.dispute_type.value if sentiment.dispute_type != DisputeType.NONE else "high_agitation"
            dept = "disputes" if sentiment.dispute_type in (DisputeType.PAYMENT_DISPUTE, DisputeType.LEGAL_THREAT) else "supervisor"
            transfer_msg = (
                "मैं आपकी स्थिति समझ सकती हूँ। कृपया एक क्षण प्रतीक्षा करें, मैं आपकी कॉल वरिष्ठ अधिकारी को ट्रांसफर कर रही हूँ।"
                if self.language == "hi"
                else "I understand your situation. Please hold for a moment while I transfer you to a senior officer."
            )
            yield {
                "type": "clause",
                "text": transfer_msg,
                "index": 1
            }

            transfer_args = {
                "reason": transfer_reason,
                "customer_sentiment": sentiment.category.value,
                "summary": f"Customer escalation ({transfer_reason}): {user_utterance[:100]}",
                "target_department": dept,
            }
            term, msg, evt_data = execute_telephony_tool("transfer_to_human", transfer_args, caller_phone=self.caller_phone)
            self.is_call_active = False

            yield {
                "type": "tool_call",
                "tool_name": "transfer_to_human",
                "tool_event": msg,
                "tool_data": evt_data,
                "terminated": True
            }
            yield {
                "type": "final",
                "full_text": transfer_msg,
                "sentiment": sentiment.to_dict(),
                "terminated": True
            }
            return

        if sentiment.deescalation_recommended:
            self.deescalation_active = True

        # 1. Record user turn
        self.messages.append({"role": "user", "content": user_utterance})

        # Inject pending supervisor whisper guidance into system context
        if self.pending_whispers:
            whisper_directive = " | ".join(self.pending_whispers)
            self.messages.append({
                "role": "system",
                "content": f"[SUPERVISOR COACHING]: {whisper_directive}. Follow this instruction immediately."
            })
            self.pending_whispers.clear()

        clause_delimiters = ["\n", "।", ".", "?", "!", ";"]
        clause_buffer = ""
        clause_index = 0
        accumulated_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        # If provider is mock, simulate streaming from _mock_llm_response
        if self.llm_provider == "mock":
            mock_content, mock_tool_call = self._mock_llm_response()
            # Split mock_content into natural clauses (preserving currency commas like ₹5,420)
            raw_clauses = [c.strip() for c in re.split(r'(?<=[।?!.\n])\s*|(?<=,)(?!\d)\s*', mock_content) if c.strip()]
            if not raw_clauses:
                raw_clauses = [mock_content]

            for clause in raw_clauses:
                clause_index += 1
                accumulated_content += (" " if accumulated_content else "") + clause
                yield {
                    "type": "clause",
                    "text": clause,
                    "index": clause_index
                }
                await asyncio.sleep(0.02)

            # Handle tool call
            tool_status = ""
            tool_data = None
            terminated = False
            if mock_tool_call:
                fn_name = mock_tool_call["function"]["name"]
                try:
                    fn_args = json.loads(mock_tool_call["function"]["arguments"])
                except Exception:
                    fn_args = {}
                res = execute_telephony_tool(fn_name, fn_args, caller_phone=self.caller_phone)
                if len(res) == 3:
                    terminated, tool_status, tool_data = res
                else:
                    terminated, tool_status = res
                if terminated:
                    self.is_call_active = False

                yield {
                    "type": "tool_call",
                    "tool_name": fn_name,
                    "tool_event": tool_status,
                    "tool_data": tool_data,
                    "terminated": terminated
                }

            asst_turn = {"role": "assistant", "content": accumulated_content}
            if mock_tool_call:
                asst_turn["tool_calls"] = [mock_tool_call]
            self.messages.append(asst_turn)

            yield {
                "type": "final",
                "full_text": accumulated_content,
                "sentiment": sentiment.to_dict(),
                "language_info": lid_result.to_dict(),
                "biometric_result": self.last_biometric.to_dict() if self.last_biometric else None,
                "terminated": not self.is_call_active
            }
            return

        # Real streaming call to Ollama, Groq, or OpenAI
        if self.llm_provider == "ollama":
            url = f"{self.ollama_host.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": "Bearer ollama"}
        elif self.llm_provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        else:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "tools": TELEPHONY_TOOLS_SCHEMA,
            "tool_choice": "auto",
            "temperature": 0.6,
            "max_tokens": 150,
            "stream": True
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            # Content delta
                            token = delta.get("content") or ""
                            if token:
                                clause_buffer += token
                                accumulated_content += token

                                # Check for clause break
                                split_pos = -1
                                for d in clause_delimiters:
                                    pos = clause_buffer.find(d)
                                    if pos != -1 and (split_pos == -1 or pos < split_pos):
                                        split_pos = pos

                                # Comma delimiter only if clause has sufficient substance (>= 14 chars) and not inside a number
                                comma_pos = clause_buffer.find(",")
                                if comma_pos != -1 and len(clause_buffer[:comma_pos].strip()) >= 14:
                                    is_num_comma = (comma_pos + 1 < len(clause_buffer) and clause_buffer[comma_pos + 1].isdigit())
                                    if not is_num_comma:
                                        if split_pos == -1 or comma_pos < split_pos:
                                            split_pos = comma_pos

                                if split_pos != -1:
                                    clause_text = clause_buffer[:split_pos + 1].strip()
                                    clause_buffer = clause_buffer[split_pos + 1:].lstrip()
                                    if clause_text:
                                        clause_index += 1
                                        yield {
                                            "type": "clause",
                                            "text": clause_text,
                                            "index": clause_index
                                        }

                            # Tool call delta
                            t_calls = delta.get("tool_calls")
                            if t_calls:
                                for tc in t_calls:
                                    idx = tc.get("index", 0)
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "id": tc.get("id", f"call_{idx}"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    fn = tc.get("function", {})
                                    if fn.get("name"):
                                        accumulated_tool_calls[idx]["function"]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        accumulated_tool_calls[idx]["function"]["arguments"] += fn["arguments"]

                        except Exception:
                            continue

        except Exception as e:
            # If Ollama / Groq connection fails, fallback to mock response
            print(f"[Streaming LLM Notice] Streaming failed ({e}). Falling back to rule-based telephony.")
            mock_content, mock_tool_call = self._mock_llm_response()
            clause_index += 1
            accumulated_content = mock_content
            yield {
                "type": "clause",
                "text": mock_content,
                "index": clause_index
            }
            if mock_tool_call:
                accumulated_tool_calls[0] = mock_tool_call

        # Flush any remaining text in clause_buffer
        if clause_buffer.strip():
            clause_index += 1
            yield {
                "type": "clause",
                "text": clause_buffer.strip(),
                "index": clause_index
            }

        # Check for inline tool calls in accumulated content (common with local SLMs)
        resolved_tool_call = None
        if accumulated_tool_calls:
            resolved_tool_call = accumulated_tool_calls[0]
        elif accumulated_content:
            if "send_payment_link" in accumulated_content:
                resolved_tool_call = {
                    "function": {"name": "send_payment_link", "arguments": json.dumps({"amount": 5420.0, "loan_id": "MUTH-8921"})}
                }
            elif "disconnect_tool" in accumulated_content:
                resolved_tool_call = {
                    "function": {"name": "disconnect_tool", "arguments": json.dumps({"reason": "farewell_exchanged"})}
                }
            elif "schedule_callback" in accumulated_content:
                resolved_tool_call = {
                    "function": {"name": "schedule_callback", "arguments": json.dumps({"promised_date": "tomorrow", "notes": "Customer promised payment"})}
                }

        # If tool call was invoked without spoken content, emit spoken confirmation clause
        if resolved_tool_call and (not accumulated_content.strip() or "send_payment_link" in accumulated_content or "disconnect_tool" in accumulated_content):
            fn_name = resolved_tool_call["function"]["name"]
            if fn_name == "send_payment_link":
                spoken = "हाँ बिल्कुल, मैंने आपके पंजीकृत मोबाइल नंबर पर ₹5,420 का सुरक्षित UPI पेमेंट लिंक भेज दिया है।" if self.language == "hi" else "Sure, I have dispatched the secure payment link to your registered mobile number."
            elif fn_name == "disconnect_tool":
                spoken = "जी ठीक है, आपका बहुत धन्यवाद। आपका दिन शुभ हो, नमस्कार।" if self.language == "hi" else "Thank you for your time. Have a great day ahead, goodbye."
            elif fn_name == "schedule_callback":
                spoken = "हाँ जी ठीक है, मैंने कल तक के भुगतान का वादा नोट कर लिया है। कृपया कल तक जमा कर दें।" if self.language == "hi" else "Alright, I have recorded your commitment for tomorrow. Thank you."
            else:
                spoken = "हाँ जी, मैंने आपका अनुरोध प्रोसेस कर दिया है।"
            clause_index += 1
            accumulated_content = spoken
            yield {
                "type": "clause",
                "text": spoken,
                "index": clause_index
            }

        # If still empty, provide persona conversational fallback
        if not accumulated_content.strip():
            if self.persona == "bank_kyc":
                fallback = "हाँ जी, आपका खाता एक्टिव रखने के लिए री-केवाईसी वेरिफिकेशन जरूरी है।" if self.language == "hi" else "Yes, re-KYC update is needed to keep your account operational."
            elif self.persona == "swiggy_delivery":
                fallback = "जी ठीक है सर, मैं आपकी लोकेशन पर पहुँच रहा हूँ।" if self.language == "hi" else "Sure sir, I am arriving at your location."
            else:
                fallback = "हाँ जी, मैं मुथूट फिनकॉर्प से बोल रहा हूँ। क्या आप आज अपनी बकाया ईएमआई जमा कर पाएंगे?" if self.language == "hi" else "Yes, I am calling from Muthoot Fincorp regarding your overdue EMI payment."
            clause_index += 1
            accumulated_content = fallback
            yield {
                "type": "clause",
                "text": fallback,
                "index": clause_index
            }

        # Execute tool call if any
        tool_status = ""
        tool_data = None
        terminated = False
        if resolved_tool_call:
            fn_name = resolved_tool_call["function"]["name"]
            try:
                fn_args = json.loads(resolved_tool_call["function"]["arguments"])
            except Exception:
                fn_args = {}
            res = execute_telephony_tool(fn_name, fn_args, caller_phone=self.caller_phone)
            if len(res) == 3:
                terminated, tool_status, tool_data = res
            else:
                terminated, tool_status = res
            if terminated:
                self.is_call_active = False

            yield {
                "type": "tool_call",
                "tool_name": fn_name,
                "tool_event": tool_status,
                "tool_data": tool_data,
                "terminated": terminated
            }

        # Record assistant turn in messages
        asst_turn = {"role": "assistant", "content": accumulated_content}
        if resolved_tool_call:
            asst_turn["tool_calls"] = [resolved_tool_call]
        self.messages.append(asst_turn)

        # Final yield
        yield {
            "type": "final",
            "full_text": accumulated_content,
            "sentiment": sentiment.to_dict(),
            "language_info": lid_result.to_dict(),
            "whispers": list(self.whisper_history),
            "biometric_result": self.last_biometric.to_dict() if self.last_biometric else None,
            "terminated": not self.is_call_active
        }

    def run_interactive_terminal_call(self):
        """Starts an interactive telephony simulation call in the terminal."""
        print("\n========================================================")
        print("  VERBALYZE: OUTBOUND TELEPHONY VOICEBOT SIMULATOR      ")
        print("  Scenario: Muthoot Fincorp Loan EMI Recovery ($5,420)  ")
        print(f"  Language: {self.language.upper()} | Min Human-Likeness: {self.min_human_likeness*100:.0f}%")
        print("  Type 'quit' or 'bye' to exit simulation               ")
        print("========================================================\n")

        # Initial outbound greeting
        if self.language == "hi":
            initial_greeting = "नमस्कार, क्या मेरी बात मिस्टर शर्मा से हो रही है? मैं मुथूट फिनकॉर्प से बोल रही हूँ।"
        else:
            initial_greeting = "Hello, am I speaking with Mr. Sharma? I am calling from Muthoot Fincorp."

        print(f"[Agent]: {initial_greeting}")
        if self.voice_enabled and self.audio_engine:
            audio_file = self.audio_engine.synthesize(initial_greeting)
            if self.audio_engine.last_quality_report:
                q = self.audio_engine.last_quality_report
                print(f"   [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) Accepted")
            if audio_file:
                self.audio_engine.play(audio_file)
                time.sleep(0.5)

        self.messages.append({"role": "assistant", "content": initial_greeting})

        while self.is_call_active:
            try:
                user_input = input("\n[Customer]: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    print("\n[Simulator ended by user]")
                    break

                res = self.step(user_input)
                print(f"[Agent]: {res['text']}")
                
                if res.get("quality_report"):
                    q = res["quality_report"]
                    print(f"   [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) Accepted")
                elif self.voice_enabled:
                    print("   [Quality Gate] Audio rejected: fell below human-likeness threshold.")

                if res.get("tool_event"):
                    print(f"   [Tool]: {res['tool_event']}")

                if res.get("audio_path") and self.audio_engine:
                    self.audio_engine.play(res["audio_path"])
                    time.sleep(0.5)

                if res.get("terminated"):
                    print("\n[CALL TERMINATED - Phone Hung Up]")
            except (KeyboardInterrupt, EOFError):
                print("\n[Call interrupted]")
                break
        print("\n========================================================\n")

    def run_live_microphone_call(self, mode: str = "push_to_talk", enable_barge_in: bool = True):
        """
        Runs a hands-free conversational voice session on Mac with Barge-In Interruption.
        Listens to microphone, transcribes speech, reasons, and speaks back through speakers.
        Supports real-time interruption (<150ms cutoff) when the customer speaks during agent playback.
        """
        from verbalyze.agent.mic_listener import MicrophoneListener

        listener = MicrophoneListener(language=self.language)

        print("\n" + "=" * 64)
        print("  VERBALYZE: LIVE HANDS-FREE VOICEBOT ON MAC")
        print(f"  Persona:  {self.persona.upper()} (Muthoot Loan Recovery INR 5,420)")
        print(f"  Language: {self.language.upper()} | Voice: {self.audio_engine.voice if self.audio_engine else 'Default'}")
        print(f"  Input:    MacBook Microphone ({listener.locale})")
        print(f"  Output:   MacBook Speakers (Neural Edge-TTS via afplay)")
        print(f"  Mode:     {'Push-to-Talk [ENTER to record]' if mode == 'push_to_talk' else 'Auto Voice Detection (VAD)'}")
        print(f"  Barge-In: {'Active (<150ms cutoff)' if enable_barge_in else 'Disabled'}")
        print("  (Type 'q' or 'quit' at any prompt to hang up)")
        print("=" * 64 + "\n")

        # Initial outbound greeting
        initial_greeting = self.get_initial_greeting()
        print(f"[Agent]: {initial_greeting}")
        self.messages.append({"role": "assistant", "content": initial_greeting})

        pending_user_utterance = None

        if self.voice_enabled and self.audio_engine:
            audio_file = self.audio_engine.synthesize(initial_greeting)
            if self.audio_engine.last_quality_report:
                q = self.audio_engine.last_quality_report
                print(f"   [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) Accepted")
            if audio_file:
                if enable_barge_in:
                    interrupted, int_wav, cutoff_ms = listener.monitor_barge_in_and_record(
                        self.audio_engine, audio_file, mode=mode
                    )
                    if interrupted and int_wav:
                        int_text = listener.transcribe(int_wav)
                        try:
                            os.remove(int_wav)
                        except Exception:
                            pass
                        if int_text and int_text.strip():
                            pending_user_utterance = int_text
                else:
                    self.audio_engine.play(audio_file)
                    time.sleep(0.5)

        while self.is_call_active:
            try:
                user_utterance = None
                wav_path = None

                if pending_user_utterance:
                    user_utterance = pending_user_utterance
                    pending_user_utterance = None
                else:
                    print("\n[Customer]:")
                    if mode == "auto":
                        wav_path = listener.record_auto_vad(silence_seconds=1.2)
                    else:
                        prompt = input("   Press [ENTER] to speak into mic (or type text directly): ").strip()
                        if prompt.lower() in ["q", "quit", "exit"]:
                            print("\n[Call ended by user]")
                            break
                        elif prompt:
                            user_utterance = prompt
                        else:
                            wav_path = listener.record_push_to_talk()

                    if wav_path:
                        print("[STT] Transcribing your speech...", end="\r", flush=True)
                        user_utterance = listener.transcribe(wav_path)
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass

                if not user_utterance or not user_utterance.strip():
                    print("[Notice] Could not detect speech clearly. Please try again")
                    continue

                print(f"[Customer Transcribed]: \"{user_utterance}\"")

                # Step agent
                res = self.step(user_utterance)
                print(f"[Agent]: {res['text']}")

                if res.get("quality_report"):
                    q = res["quality_report"]
                    print(f"   [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) Accepted")
                elif self.voice_enabled:
                    print("   [Quality Gate] Audio rejected: fell below human-likeness threshold.")

                if res.get("tool_event"):
                    print(f"   [Tool]: {res['tool_event']}")

                if res.get("audio_path") and self.audio_engine:
                    if enable_barge_in and not res.get("terminated"):
                        interrupted, int_wav, cutoff_ms = listener.monitor_barge_in_and_record(
                            self.audio_engine, res["audio_path"], mode=mode
                        )
                        if interrupted and int_wav:
                            int_text = listener.transcribe(int_wav)
                            try:
                                os.remove(int_wav)
                            except Exception:
                                pass
                            if int_text and int_text.strip():
                                pending_user_utterance = int_text
                    else:
                        self.audio_engine.play(res["audio_path"])
                        time.sleep(0.5)

                if res.get("terminated"):
                    print("\n[CALL TERMINATED - Phone Hung Up]")
                    break

            except (KeyboardInterrupt, EOFError):
                print("\n[Call disconnected by user]")
                break

        print("\n" + "=" * 64 + "\n")



if __name__ == "__main__":
    bot = VoiceAgent(language="hi")
    bot.run_interactive_terminal_call()
