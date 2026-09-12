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
        system_prompt: Optional[str] = None,
        llm_provider: str = "groq",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        voice_enabled: bool = True
    ):
        self.language = language
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPTS.get(language, DEFAULT_SYSTEM_PROMPTS["hi"])
        self.llm_provider = llm_provider
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name or ("llama-3.3-70b-versatile" if llm_provider == "groq" else "gpt-4o-mini")
        self.voice_enabled = voice_enabled
        self.audio_engine = AudioEngine(language=language) if voice_enabled else None

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.is_call_active = True

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

        # Default conversational reply
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

        # 5. Synthesize voice if enabled
        audio_path = None
        if self.voice_enabled and self.audio_engine and content:
            audio_path = self.audio_engine.synthesize(content)

        return {
            "text": content,
            "audio_path": audio_path,
            "tool_event": tool_status if tool_call else None,
            "terminated": not self.is_call_active
        }

    def run_interactive_terminal_call(self):
        """Starts an interactive telephony simulation call in the terminal."""
        print("\n========================================================")
        print("  VERBALYZE: OUTBOUND TELEPHONY VOICEBOT SIMULATOR      ")
        print("  Scenario: Muthoot Fincorp Loan EMI Recovery ($5,420)  ")
        print("  Language: " + self.language.upper())
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
            if audio_file:
                self.audio_engine.play(audio_file)

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
                
                if res.get("tool_event"):
                    print(f"   ⚙️  {res['tool_event']}")

                if res.get("audio_path") and self.audio_engine:
                    self.audio_engine.play(res["audio_path"])

                if res.get("terminated"):
                    print("\n🔴 [CALL TERMINATED - Phone Hung Up]")
                    break

            except (KeyboardInterrupt, EOFError):
                print("\n[Call interrupted]")
                break
        print("\n========================================================\n")


if __name__ == "__main__":
    bot = VoiceAgent(language="hi")
    bot.run_interactive_terminal_call()
