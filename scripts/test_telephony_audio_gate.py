"""
scripts/test_telephony_audio_gate.py

Verifies:
1. 8kHz G.711 A-law Telephony Channel Degradation on Indic Speech (Hindi/Hinglish).
2. Quality Gate Acoustic Scoring under telecom bandpass & quantization distortion.
3. Unmetered SIP Trunk Webhooks (/webhook/sip/inbound & /webhook/sip/turn).
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verbalyze.agent.audio_quality import HumanLikenessScorer, TelephonyChannelSimulator
from verbalyze.agent.audio_engine import AudioEngine
from verbalyze.agent.voice_bot import VoiceAgent

def run_telephony_gate_verification():
    print("=" * 65)
    print("  VERBALYZE: 8kHz G.711 TELEPHONY AUDIO & SIP TRUNK TEST")
    print("=" * 65)

    test_text = "नमस्कार मिस्टर शर्मा, मैं मुथूट फिनकॉर्प से आपकी बकाया ईएमआई के संदर्भ में बात कर रही हूँ।"
    print(f"\n1. Synthesizing test utterance in Hindi...")
    print(f"📝 Text: {test_text}")

    # Initialize standard AudioEngine (clean wideband)
    clean_engine = AudioEngine(language="hi", min_human_likeness=0.80, simulate_telephony=False)
    clean_audio = clean_engine.synthesize(test_text)

    if not clean_audio or not os.path.exists(clean_audio):
        print("❌ Error: Synthesis failed to generate audio.")
        sys.exit(1)

    clean_report = clean_engine.last_quality_report
    print(f"✓ Clean Audio Generated: {clean_audio}")
    print(f"   • Baseline Score: {clean_report.score * 100:.1f}% (MOS {clean_report.mos_equivalent:.2f}/5.0)")
    print(f"   • Cadence: {clean_report.cadence_score * 100:.1f}% ({clean_report.wpm:.1f} WPM)")
    print(f"   • Pause Ratio: {clean_report.pause_ratio * 100:.1f}%")

    print(f"\n2. Passing audio through 8kHz G.711 A-law Telephony Channel...")
    simulator = TelephonyChannelSimulator(sample_rate=8000, A=87.6, packet_loss_rate=0.015)
    
    # Degrade and evaluate
    scorer = HumanLikenessScorer(min_threshold=0.80, simulate_telephony=True)
    telephony_report = scorer.evaluate(clean_audio, test_text, simulate_telephony=True)

    print(f"✓ Telephony Channel Applied:")
    print(f"   • Resampling: 8,000 Hz Telecom Carrier Standard")
    print(f"   • Bandpass Filter: 300 Hz - 3,400 Hz (ITU-T G.712)")
    print(f"   • Quantization: 8-bit ITU-T G.711 A-law logarithmic companding (A=87.6)")
    print(f"   • Packet Jitter/Loss: 1.5% simulated RTP frame drop")
    print(f"   • Telephony Quality Score: {telephony_report.score * 100:.1f}% (MOS {telephony_report.mos_equivalent:.2f}/5.0)")
    print(f"   • Quality Gate Decision: {'✓ ACCEPTED' if telephony_report.passed else '⚠️ REJECTED'}")
    print(f"   • Telephony Feedback: {telephony_report.feedback}")

    assert telephony_report.passed, f"Telephony audio failed quality gate with score {telephony_report.score}"
    assert telephony_report.telephony_simulated, "telephony_simulated flag must be True"

    print(f"\n3. Testing AudioEngine in native --telephony-sim mode...")
    telephony_engine = AudioEngine(language="hi", min_human_likeness=0.80, simulate_telephony=True)
    sim_dest = tempfile.mktemp(suffix=".mp3")
    sim_out = telephony_engine.synthesize(test_text, output_path=sim_dest)
    assert sim_out is not None and os.path.exists(sim_out), "Native telephony engine synthesis failed"
    print(f"✓ Native Telephony Engine generated audio at: {sim_out}")
    print(f"   • Score: {telephony_engine.last_quality_report.score * 100:.1f}% (MOS {telephony_engine.last_quality_report.mos_equivalent:.2f}/5.0)")

    print(f"\n4. Testing Unmetered SIP Trunk Gateway Endpoints (RingTrunk / Asterisk)...")
    from verbalyze.telephony.server import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    # Inbound SIP call
    inbound_resp = client.post(
        "/webhook/sip/inbound",
        json={
            "call_id": "test_sip_trunk_099",
            "lang": "hi",
            "persona": "muthoot_recovery",
            "provider": "mock"
        }
    )
    assert inbound_resp.status_code == 200
    inbound_data = inbound_resp.json()
    print(f"✓ SIP Inbound Handshake Successful:")
    print(f"   • Trunk Type: {inbound_data.get('trunk_type')}")
    print(f"   • Provider: {inbound_data.get('provider')}")
    print(f"   • Greeting: {inbound_data.get('greeting_text')}")

    # Spoken turn via SIP
    turn_resp = client.post(
        "/webhook/sip/turn",
        json={
            "call_id": "test_sip_trunk_099",
            "transcript": "अच्छा ठीक है, मुझे पेमेंट करने के लिए UPI लिंक SMS पर भेज दीजिए।"
        }
    )
    assert turn_resp.status_code == 200
    turn_data = turn_resp.json()
    print(f"✓ SIP Spoken Turn Handshake Successful:")
    print(f"   • Agent Reply: {turn_data.get('agent_response')}")
    print(f"   • Tool Event: {turn_data.get('tool_event')}")
    print(f"   • Hangup Status: {turn_data.get('hangup')}")

    print("\n" + "=" * 65)
    print("🎉 8kHz TELEPHONY CODEC & UNMETERED SIP TRUNK TEST PASSED!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_telephony_gate_verification()
