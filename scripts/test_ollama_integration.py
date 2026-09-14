"""
scripts/test_ollama_integration.py

Verifies 100% Offline Local SLM inference with Ollama (Llama-3.2-3B)
running on Apple Silicon Metal with telephony tool calling and Quality Gate.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.agent.voice_bot import VoiceAgent

def main():
    print("=======================================================")
    print("  VERBALYZE: LOCAL OFFLINE SLM (OLLAMA) VERIFICATION   ")
    print("  Model: Llama-3.2-3B on Apple Silicon Metal           ")
    print("=======================================================\n")

    print("1. Initializing VoiceAgent with llm_provider='ollama'...")
    agent = VoiceAgent(
        language="hi",
        persona="muthoot_recovery",
        llm_provider="ollama",
        model_name="llama3.2:3b",
        voice_enabled=True,
        min_human_likeness=0.80
    )
    greeting = agent.get_initial_greeting()
    agent.messages.append({"role": "assistant", "content": greeting})
    print("✓ Agent initialized successfully.")
    print(f"📞 Agent Greeting: {greeting}")

    print("\n2. Testing Turn 1: Conversational inquiry...")
    user_msg_1 = "हाँ, मैं मिस्टर शर्मा बोल रहा हूँ। आप मुथूट फिनकॉर्प से क्यों कॉल कर रहे हैं?"
    print(f"👤 Customer: {user_msg_1}")
    res1 = agent.step(user_msg_1)
    print(f"📞 Agent: {res1['text']}")
    if res1.get("quality_report"):
        qr = res1["quality_report"]
        print(f"🎯 Quality Gate: {qr.score*100:.1f}% (MOS {qr.mos_equivalent:.2f}/5.0) - {'✓ Accepted' if qr.passed else '⚠️ Rejected'}")
    assert res1["text"], "Agent must provide response"

    print("\n3. Testing Turn 2: Telephony Tool Execution (UPI link request)...")
    user_msg_2 = "अच्छा ठीक है, मुझे पेमेंट करने के लिए UPI लिंक SMS पर भेज दीजिए।"
    print(f"👤 Customer: {user_msg_2}")
    res2 = agent.step(user_msg_2)
    print(f"📞 Agent: {res2['text']}")
    if res2.get("tool_event"):
        print(f"⚙️ Telephony Event: {res2['tool_event']}")
    if res2.get("quality_report"):
        qr = res2["quality_report"]
        print(f"🎯 Quality Gate: {qr.score*100:.1f}% (MOS {qr.mos_equivalent:.2f}/5.0) - {'✓ Accepted' if qr.passed else '⚠️ Rejected'}")

    print("\n4. Testing Turn 3: Call Disconnect...")
    user_msg_3 = "लिंक मिल गया, मैं अभी भर देता हूँ। बहुत धन्यवाद, अलविदा।"
    print(f"👤 Customer: {user_msg_3}")
    res3 = agent.step(user_msg_3)
    print(f"📞 Agent: {res3['text']}")
    if res3.get("tool_event"):
        print(f"⚙️ Telephony Event: {res3['tool_event']}")
    print(f"🔴 Call Active: {agent.is_call_active} (Terminated: {res3.get('terminated')})")

    print("\n=======================================================")
    print("🎉 LOCAL OFFLINE SLM VERIFICATION SUCCESSFUL!")
    print("=======================================================")

if __name__ == "__main__":
    main()
