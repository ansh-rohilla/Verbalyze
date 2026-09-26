"""
scripts/demo_barge_in_live.py

Live Audible Demonstration of Real-Time Barge-In Interruption:
1. Agent starts speaking a long Muthoot EMI explanation through MacBook speakers.
2. After 2.0 seconds (midway through speech), a customer interruption is triggered.
3. Playback is cut off immediately (< 10ms).
4. Local SLM (Ollama) processes the interruption and speaks the clarification.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.agent.audio_engine import AudioEngine
from verbalyze.agent.voice_bot import VoiceAgent

def run_live_barge_in_demo():
    print("\n" + "=" * 65)
    print("  VERBALYZE: LIVE BARGE-IN INTERRUPTION AUDIBLE DEMO")
    print("  Listen to your MacBook speakers...")
    print("=" * 65 + "\n")

    engine = AudioEngine(language="hi", min_human_likeness=0.80)
    agent = VoiceAgent(language="hi", llm_provider="ollama", voice_enabled=True)

    # 1. Agent speaks a long explanation (~6 seconds long)
    long_speech = (
        "नमस्कार मिस्टर शर्मा, मैं मुथूट फिनकॉर्प के हेड ऑफिस से बात कर रही हूँ। "
        "आपके पर्सनल लोन अकाउंट नंबर 8921 पर ₹5,420 की ईएमआई ओवरड्यू हो चुकी है, "
        "और यदि आज शाम 5 बजे तक इसका भुगतान नहीं हुआ तो लेट पेनल्टी लग जाएगी।"
    )

    print("1. Synthesizing Agent's long explanation...")
    audio_path = engine.synthesize(long_speech)
    assert audio_path and os.path.exists(audio_path), "Synthesis failed"

    print("2. Starting Agent speech through your Mac speakers...")
    print(f"Agent speaking: \"{long_speech}\"\n")
    proc = engine.play(audio_path, block=False)

    # Let the agent speak for 2.0 seconds
    for i in range(20):
        time.sleep(0.1)
        if i % 3 == 0:
            print(f"   [Agent speaking... {(i+1)*0.1:.1f}s]", flush=True)

    # 3. Simulate customer barging in
    customer_interruption = "अरे रुकिए एक मिनट! मुझे यह बताइए कि कितना अमाउंट पे करना है?"
    print("\n" + "!" * 65)
    print(f"Customer Barges In (Speaking over Agent):")
    print(f"   \"{customer_interruption}\"")
    print("!" * 65)

    # Trigger atomic cutoff
    t0 = time.perf_counter()
    stopped = engine.stop_playback()
    cutoff_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[BARGE-IN TRIGGERED] Audio playback killed in {cutoff_ms:.2f} ms!")
    print(f"   Voice cut off instantly: {'SUCCESS' if stopped else 'FAILED'}")
    assert stopped, "Failed to halt playback"

    time.sleep(0.4)

    # 4. Agent processes the interruption with Ollama
    print("\n3. Local SLM (Llama-3.2-3B) processing customer interruption...")
    agent_reply = agent.step(customer_interruption)
    reply_text = agent_reply["text"]
    print(f"Agent (Immediate Clarification): \"{reply_text}\"")

    # 5. Play agent's response out loud
    if agent_reply.get("audio_path"):
        print("\nPlaying Agent's response on speakers...")
        engine.play(agent_reply["audio_path"], block=True)

    print("\n" + "=" * 65)
    print("LIVE AUDIBLE DEMO COMPLETE!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_live_barge_in_demo()
