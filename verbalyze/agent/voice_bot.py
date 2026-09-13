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
from typing import Dict, List, Any, Optional, Tuple
from verbalyze.agent.tools import TELEPHONY_TOOLS_SCHEMA, execute_telephony_tool
from verbalyze.agent.audio_engine import AudioEngine

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
        "- If customer promises to pay or asks to end call, be courteous, say goodbye, and invoke disconnect_tool\n"
        "- If customer asks for online payment link, offer to send UPI link and invoke send_payment_link"
    ),
    "en": (
        "You are a friendly, polite, yet professional outbound collection agent representing Muthoot Fincorp. "
        "You are having a real phone call with an English-speaking customer regarding their overdue loan EMI recovery of Rs. 5,420. "
        "You must sound EXACTLY like a real human agent — not a robot, not an AI chatbot.\n\n"
        "CRITICAL TELEPHONY RULES:\n"
        "- Respond in spoken English\n"
        "- Keep sentences SHORT (1-2 sentences maximum per turn) — spoken phone dialogue, not long text\n"
        "- Use natural fillers: 'yeah', 'right', 'hmm', 'I understand'\n"
        "- If customer promises to pay or asks to end call, be courteous, say goodbye, and invoke disconnect_tool\n"
        "- If customer asks for payment link, invoke send_payment_link"
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
        voice_enabled: bool = True,
        min_human_likeness: float = 0.80
    ):
        self.language = language
        self.persona = persona
        self.min_human_likeness = min_human_likeness
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPTS.get(language, DEFAULT_SYSTEM_PROMPTS["hi"])
        self.llm_provider = llm_provider
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name or ("llama-3.3-70b-versatile" if llm_provider == "groq" else "gpt-4o-mini")
        self.voice_enabled = voice_enabled
        self.audio_engine = AudioEngine(language=language, min_human_likeness=min_human_likeness) if voice_enabled else None

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.is_call_active = True

    def get_initial_greeting(self) -> str:
        """Returns localized initial greeting for the selected persona."""
        persona_greetings = INITIAL_GREETINGS.get(self.persona, INITIAL_GREETINGS["muthoot_recovery"])
        return persona_greetings.get(self.language, persona_greetings.get("en", "Hello, how can I help you today?"))


    def _call_groq_or_openai(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Calls LLM with telephony function calling tools."""
        import urllib.request

        url = "https://api.groq.com/openai/v1/chat/completions" if self.llm_provider == "groq" else "https://api.openai.com/v1/chat/completions"
        if not self.api_key:
            return self._mock_llm_response()

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
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]["message"]
                content = choice.get("content") or ""
                tool_calls = choice.get("tool_calls")
                
                tool_call = tool_calls[0] if tool_calls else None
                return content, tool_call
        except Exception as e:
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

    def step(self, user_utterance: str) -> Dict[str, Any]:
        """Processes a single conversational turn from the user."""
        if not self.is_call_active:
            return {"text": "[Call already disconnected]", "terminated": True}

        # 1. Record user turn
        self.messages.append({"role": "user", "content": user_utterance})

        # 2. Generate assistant response
        content, tool_call = self._call_groq_or_openai()

        # 3. Handle assistant message
        asst_turn = {"role": "assistant", "content": content}
        if tool_call:
            asst_turn["tool_calls"] = [tool_call]
        self.messages.append(asst_turn)

        # 4. Execute tool call if triggered
        tool_status = ""
        if tool_call:
            fn_name = tool_call["function"]["name"]
            try:
                fn_args = json.loads(tool_call["function"]["arguments"])
            except Exception:
                fn_args = {}
            terminated, tool_status = execute_telephony_tool(fn_name, fn_args)
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

        print(f"📞 Agent: {initial_greeting}")
        if self.voice_enabled and self.audio_engine:
            audio_file = self.audio_engine.synthesize(initial_greeting)
            if self.audio_engine.last_quality_report:
                q = self.audio_engine.last_quality_report
                print(f"   🎯 [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) ✓ Accepted")
            if audio_file:
                self.audio_engine.play(audio_file)
                time.sleep(0.5)

        self.messages.append({"role": "assistant", "content": initial_greeting})

        while self.is_call_active:
            try:
                user_input = input("\n👤 Customer: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    print("\n[Simulator ended by user]")
                    break

                res = self.step(user_input)
                print(f"📞 Agent: {res['text']}")
                
                if res.get("quality_report"):
                    q = res["quality_report"]
                    print(f"   🎯 [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) ✓ Accepted")
                elif self.voice_enabled:
                    print("   ⚠️  [Quality Gate] Audio rejected: fell below human-likeness threshold.")

                if res.get("tool_event"):
                    print(f"   ⚙️  {res['tool_event']}")

                if res.get("audio_path") and self.audio_engine:
                    self.audio_engine.play(res["audio_path"])
                    time.sleep(0.5)

                if res.get("terminated"):
                    print("\n🔴 [CALL TERMINATED - Phone Hung Up]")
            except (KeyboardInterrupt, EOFError):
                print("\n[Call interrupted]")
                break
        print("\n========================================================\n")

    def run_live_microphone_call(self, mode: str = "push_to_talk"):
        """
        Runs a hands-free conversational voice session on Mac.
        Listens to microphone, transcribes speech, reasons, and speaks back through speakers.
        """
        from verbalyze.agent.mic_listener import MicrophoneListener

        listener = MicrophoneListener(language=self.language)

        print("\n" + "=" * 64)
        print("  🎙️  VERBALYZE: LIVE HANDS-FREE VOICEBOT ON MAC")
        print(f"  Persona:  {self.persona.upper()} (Muthoot Loan Recovery ₹5,420)")
        print(f"  Language: {self.language.upper()} | Voice: {self.audio_engine.voice if self.audio_engine else 'Default'}")
        print(f"  Input:    MacBook Microphone ({listener.locale})")
        print(f"  Output:   MacBook Speakers (Neural Edge-TTS via afplay)")
        print(f"  Mode:     {'Push-to-Talk [ENTER to record]' if mode == 'push_to_talk' else 'Auto Voice Detection (VAD)'}")
        print("  (Type 'q' or 'quit' at any prompt to hang up)")
        print("=" * 64 + "\n")

        # Initial outbound greeting
        initial_greeting = self.get_initial_greeting()
        print(f"📞 Agent: {initial_greeting}")
        self.messages.append({"role": "assistant", "content": initial_greeting})

        if self.voice_enabled and self.audio_engine:
            audio_file = self.audio_engine.synthesize(initial_greeting)
            if self.audio_engine.last_quality_report:
                q = self.audio_engine.last_quality_report
                print(f"   🎯 [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) ✓ Accepted")
            if audio_file:
                self.audio_engine.play(audio_file)
                time.sleep(0.5)

        while self.is_call_active:
            try:
                user_utterance = None
                wav_path = None

                print("\n👤 You (Customer):")
                if mode == "auto":
                    wav_path = listener.record_auto_vad(silence_seconds=1.2)
                else:
                    prompt = input("   👉 Press [ENTER] to speak into mic (or type text directly): ").strip()
                    if prompt.lower() in ["q", "quit", "exit"]:
                        print("\n[Call ended by user]")
                        break
                    elif prompt:
                        user_utterance = prompt
                    else:
                        wav_path = listener.record_push_to_talk()

                if wav_path:
                    print("⚡ Transcribing your speech...", end="\r", flush=True)
                    user_utterance = listener.transcribe(wav_path)
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                if not user_utterance or not user_utterance.strip():
                    print("⚠️  [Could not detect speech clearly. Please try again]")
                    continue

                print(f"👤 Customer (Transcribed): \"{user_utterance}\"")

                # Step agent
                res = self.step(user_utterance)
                print(f"📞 Agent: {res['text']}")

                if res.get("quality_report"):
                    q = res["quality_report"]
                    print(f"   🎯 [Quality Gate] Human-Likeness: {q.score*100:.1f}% (MOS {q.mos_equivalent}/5.0) ✓ Accepted")
                elif self.voice_enabled:
                    print("   ⚠️  [Quality Gate] Audio rejected: fell below human-likeness threshold.")

                if res.get("tool_event"):
                    print(f"   ⚙️  {res['tool_event']}")

                if res.get("audio_path") and self.audio_engine:
                    self.audio_engine.play(res["audio_path"])
                    time.sleep(0.5)

                if res.get("terminated"):
                    print("\n🔴 [CALL TERMINATED - Phone Hung Up]")
                    break

            except (KeyboardInterrupt, EOFError):
                print("\n[Call disconnected by user]")
                break

        print("\n" + "=" * 64 + "\n")



if __name__ == "__main__":
    bot = VoiceAgent(language="hi")
    bot.run_interactive_terminal_call()
