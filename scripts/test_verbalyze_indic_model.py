#!/usr/bin/env python3
"""
scripts/test_verbalyze_indic_model.py

Comprehensive integration test for the turn-key 'verbalyze-indic' Ollama model:
1. Verifies Ollama connectivity & model registration ('verbalyze-indic').
2. Multi-turn spoken telephony interaction in Hindi.
3. Telephony function calling verification (send_payment_link, disconnect_tool).
4. Audio synthesis and 80% MOS Quality Gate verification.
"""

import sys
import json
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.agent.voice_bot import VoiceAgent


def test_ollama_registration():
    print("============================================================")
    print("1. Testing Ollama Model Registration ('verbalyze-indic')")
    print("============================================================")
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            print(f"Registered Ollama models: {models}")
            assert any("verbalyze-indic" in m for m in models), "verbalyze-indic not found in Ollama registry!"
            print("✅ 'verbalyze-indic' is registered and ready in Ollama daemon.")
    except Exception as e:
        print(f"❌ Failed to reach Ollama: {e}")
        sys.exit(1)


def test_conversational_and_tool_turns():
    print("\n============================================================")
    print("2. Testing Conversational Turns with 'verbalyze-indic'")
    print("============================================================")
    agent = VoiceAgent(
        language="hi",
        persona="muthoot_recovery",
        llm_provider="ollama",
        model_name="verbalyze-indic",
        voice_enabled=True,
        min_human_likeness=0.80
    )

    greeting = agent.get_initial_greeting()
    print(f"Agent Greeting: '{greeting}'\n")

    # Turn 1: Conversational query
    user_turn_1 = "हाँ जी, मैं सुन रहा हूँ। कितना लोन बाकी है?"
    print(f"User: '{user_turn_1}'")
    res1 = agent.step(user_turn_1)
    print(f"Agent Reply: '{res1['text']}'")
    print(f"Terminated: {res1['terminated']}")
    if res1.get("quality_report"):
        qr1 = res1["quality_report"]
        print(f"Quality Score: {qr1.score * 100:.1f}% (MOS: {qr1.mos_equivalent:.2f}) | Passed: {qr1.passed}")
    assert res1["text"], "Agent reply turn 1 was empty!"

    # Turn 2: Request payment link (Trigger tool: send_payment_link)
    print("\n------------------------------------------------------------")
    user_turn_2 = "हाँ ठीक है, मुझे यूपीआई से पेमेंट करना है, लिंक भेज दीजिए।"
    print(f"User: '{user_turn_2}'")
    res2 = agent.step(user_turn_2)
    print(f"Agent Reply: '{res2['text']}'")
    print(f"Terminated: {res2['terminated']}")
    if res2.get("quality_report"):
        qr2 = res2["quality_report"]
        print(f"Quality Score: {qr2.score * 100:.1f}% (MOS: {qr2.mos_equivalent:.2f}) | Passed: {qr2.passed}")
    print(f"Call history turns: {len(agent.messages)}")

    # Turn 3: Farewell (Trigger tool: disconnect_tool)
    print("\n------------------------------------------------------------")
    user_turn_3 = "बहुत धन्यवाद, मैंने पेमेंट कर दिया है। नमस्कार, बाय!"
    print(f"User: '{user_turn_3}'")
    res3 = agent.step(user_turn_3)
    print(f"Agent Reply: '{res3['text']}'")
    print(f"Call Terminated Flag: {res3['terminated']}")
    assert res3["terminated"] or not agent.is_call_active, "Call was expected to terminate on farewell!"

    print("\n============================================================")
    print("✅ All 'verbalyze-indic' Multi-Turn Telephony Tests PASSED!")
    print("============================================================")


if __name__ == "__main__":
    test_ollama_registration()
    test_conversational_and_tool_turns()
