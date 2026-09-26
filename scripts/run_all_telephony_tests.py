#!/usr/bin/env python3
"""
scripts/run_all_telephony_tests.py

Master Telephony & DSP Test Suite Runner for Verbalyze.
Sequentially executes all 27 automated test suites across:
- Core DSP algorithms (PESQ quality, ALC, Diarization, Watermarking, BWE, CNG, PLC, AEC)
- Telephony protocols (SIP WebSocket, Media Streams, DTMF IVR, Outbound AMD)
- Safety & Compliance (Regulatory QA, Sentiment Transfer, Biometrics, Tamper Seal)
- Enterprise Integrations (SMS UPI, WhatsApp Gateway, Multi-trunk Failover)
"""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

TEST_SUITES = [
    ("Acoustic Watermarker & DSSS Tamper Seal", "scripts/test_acoustic_watermarker.py"),
    ("Dual-Channel Active Speaker Diarization", "scripts/test_active_speaker_diarizer.py"),
    ("Automatic Level Control & Multi-Speaker Normalizer", "scripts/test_automatic_level_controller.py"),
    ("Pure-Math Artificial Bandwidth Expansion (BWE)", "scripts/test_bandwidth_expander.py"),
    ("Sub-10ms Barge-In Interruption Engine", "scripts/test_barge_in_engine.py"),
    ("Campaign Auto-Dialer & Predictive AMD", "scripts/test_campaign_dialer_amd.py"),
    ("Circle-Based LCR Trunks & Circuit Breaker", "scripts/test_carrier_trunks_circuit_breaker.py"),
    ("Adaptive Comfort Noise Generator (CNG)", "scripts/test_comfort_noise_generator.py"),
    ("Dual-Tone Multi-Frequency (DTMF) IVR Engine", "scripts/test_dtmf_ivr_engine.py"),
    ("Normalized LMS Acoustic Echo Canceller & Noise Gate", "scripts/test_echo_canceller_and_noise_suppression.py"),
    ("End-to-End Telecom Security & Audio Cloaking", "scripts/test_end_to_end_security.py"),
    ("Indic Formant & Vowel Resonance Equalizer", "scripts/test_indic_formant_equalizer.py"),
    ("Language ID & Acoustic Code-Switching", "scripts/test_lid_and_code_switching.py"),
    ("Cellular Line Quality Classifier & P.862 PESQ", "scripts/test_line_quality_classifier.py"),
    ("Twilio / Plivo Media Stream WebSocket Pipeline", "scripts/test_media_stream_websocket.py"),
    ("Packet Loss Concealment (PLC) & Adaptive Jitter Buffer", "scripts/test_packet_loss_concealment_and_jitter.py"),
    ("RBI / TRAI Regulatory QA & Compliance Guard", "scripts/test_regulatory_qa_engine.py"),
    ("Real-Time Sentiment Analysis & Human Transfer", "scripts/test_sentiment_and_transfer.py"),
    ("Multi-Channel SMS & NPCI UPI Link Dispatch", "scripts/test_sms_upi_dispatch.py"),
    ("Sovereign Local STT & Micro-Whisper Fallback", "scripts/test_sovereign_stt.py"),
    ("Full-Duplex Streaming Pipeline Orchestrator", "scripts/test_streaming_pipeline.py"),
    ("WebRTC Supervisor Audio & Browser Gateway", "scripts/test_supervisor_and_browser_gateway.py"),
    ("8kHz G.711 Telephony Audio Gate & MOS Filter", "scripts/test_telephony_audio_gate.py"),
    ("Spoken Turn-Taking & Sub-200ms Latency Budget", "scripts/test_turn_taking_and_latency.py"),
    ("Verbalyze Indic SLM Integration & Function Calling", "scripts/test_verbalyze_indic_model.py"),
    ("Voice Biometrics, Anti-Spoofing & Replay Guard", "scripts/test_voice_biometrics_and_anti_spoof.py"),
    ("WhatsApp Settlement & Dispute Resolution Gateway", "scripts/test_whatsapp_settlement_gateway.py"),
    ("Acoustic End-of-Turn & Voice Boundary Predictor", "scripts/test_voice_boundary_predictor.py"),
]


def run_all_tests():
    print("=" * 80)
    print("VERBALYZE: MASTER TELEPHONY & DSP TEST SUITE RUNNER")
    print(f"Total Suites to Execute: {len(TEST_SUITES)}")
    print("=" * 80)

    results = []
    overall_start = time.perf_counter()

    for idx, (suite_name, script_rel_path) in enumerate(TEST_SUITES, 1):
        script_full_path = PROJECT_ROOT / script_rel_path
        if not script_full_path.exists():
            print(f"[{idx:02d}/{len(TEST_SUITES):02d}] MISSING: {script_rel_path}")
            results.append((suite_name, script_rel_path, "MISSING", 0.0, "File not found"))
            continue

        print(f"\n[{idx:02d}/{len(TEST_SUITES):02d}] Running: {suite_name} ({script_rel_path})...")
        t0 = time.perf_counter()
        
        proc = subprocess.run(
            [sys.executable, str(script_full_path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        elapsed = time.perf_counter() - t0

        if proc.returncode == 0:
            if "[SKIP]" in proc.stdout or "[SKIP]" in proc.stderr:
                status = "SKIPPED"
            else:
                status = "PASSED"
            print(f"[{status}] in {elapsed:.2f}s")
            results.append((suite_name, script_rel_path, status, elapsed, ""))
        else:
            status = "FAILED"
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            # Grab last non-empty line of error
            last_err = [line for line in err_msg.splitlines() if line.strip()][-1] if err_msg else "Unknown error"
            print(f"[FAILED] in {elapsed:.2f}s: {last_err}")
            results.append((suite_name, script_rel_path, status, elapsed, last_err))

    overall_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 96)
    print("VERBALYZE TELEPHONY & DSP TEST EXECUTION SUMMARY")
    print("=" * 96)
    print(f"{'#':<3} | {'Suite Name':<52} | {'Status':<8} | {'Time (s)':<8}")
    print("-" * 96)

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, (suite_name, _, status, elapsed, _) in enumerate(results, 1):
        if status == "PASSED":
            passed_count += 1
        elif status == "SKIPPED":
            skipped_count += 1
        else:
            failed_count += 1
        print(f"{idx:<3} | {suite_name:<52} | {status:<8} | {elapsed:>7.2f}s")

    print("-" * 96)
    print(f"Total: {len(results)} suites | Passed: {passed_count} | Skipped: {skipped_count} | Failed: {failed_count} | Total Time: {overall_elapsed:.2f}s")
    print("=" * 96)

    if failed_count > 0:
        print("\n[ERROR] One or more test suites failed. Please inspect logs above.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All telephony test suites passed cleanly!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
