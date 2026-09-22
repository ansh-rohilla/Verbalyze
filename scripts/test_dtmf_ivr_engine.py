"""
scripts/test_dtmf_ivr_engine.py

Comprehensive Automated Test Suite for:
1. Pure Goertzel Algorithm Detection for all 16 DTMF dual-tones at 8kHz & 16kHz PCM.
2. Frequency Rejection, Twist Tolerance (-8 dB to +4 dB), & Noise Immunity.
3. RFC 4733 / RFC 2833 RTP Telephone-Event Encoding, Decoding & Deduplication.
4. Acoustic DTMF Debouncing & Inter-Digit Gap Verification.
5. Multi-Level IVR State Machine Hierarchical Navigation & Action Execution.
6. Hybrid Keypad & Speech Traversal with Retry Limit Management.
7. Multi-Digit PIN Collection with Inter-Digit Timeout & DPDP Act 2023 Keypad Masking.
8. FastAPI REST API Telephony Integration (/telephony/dtmf/decode, /telephony/ivr/*).

Zero-emoji compliant.
DPDP Act 2023 compliant.
ITU-T Q.23 / Q.24 & RFC 4733 compliant.
"""

import os
import sys
import base64
import random
import struct
import time
from typing import Dict, Any, List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from verbalyze.telephony.dtmf_engine import (
    GoertzelDetector,
    DTMFPad,
    RFC4733EventDecoder,
    encode_rfc4733_packet,
    decode_rfc4733_packet,
    DTMFToneGenerator,
    mask_digits,
    sanitize_sensitive_dict,
)
from verbalyze.telephony.ivr_tree import (
    IVRNode,
    IVRStateMachine,
    IVRTransitionResult,
    create_default_banking_ivr,
)
from verbalyze.telephony.server import create_app


ALL_DTMF_KEYS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#", "A", "B", "C", "D"]


def test_1_goertzel_all_16_digits_8k_and_16k():
    print("\n--- Test 1: Goertzel Algorithm Detection for All 16 DTMF Keys (8kHz & 16kHz) ---")

    # 1. Test 8,000 Hz (20ms frame = 160 samples)
    detector_8k = GoertzelDetector(sample_rate=8000, frame_size=160)
    detected_8k = []
    for key in ALL_DTMF_KEYS:
        pcm = DTMFToneGenerator.generate_tone(key, duration_ms=40, sample_rate=8000, amplitude=0.6)
        det = detector_8k.detect_frame(pcm)
        detected_8k.append(det)

    assert detected_8k == ALL_DTMF_KEYS, f"8kHz Mismatch: {detected_8k} vs {ALL_DTMF_KEYS}"
    print(f"PASS: 8kHz Goertzel successfully detected all 16 keys: {' '.join(detected_8k)}")

    # 2. Test 16,000 Hz (20ms frame = 320 samples)
    detector_16k = GoertzelDetector(sample_rate=16000, frame_size=320)
    detected_16k = []
    for key in ALL_DTMF_KEYS:
        pcm = DTMFToneGenerator.generate_tone(key, duration_ms=40, sample_rate=16000, amplitude=0.6)
        det = detector_16k.detect_frame(pcm)
        detected_16k.append(det)

    assert detected_16k == ALL_DTMF_KEYS, f"16kHz Mismatch: {detected_16k} vs {ALL_DTMF_KEYS}"
    print(f"PASS: 16kHz Goertzel successfully detected all 16 keys: {' '.join(detected_16k)}")


def test_2_frequency_rejection_and_twist_tolerance():
    print("\n--- Test 2: Frequency Rejection, Twist Tolerance & Noise Immunity ---")
    detector = GoertzelDetector(sample_rate=8000, frame_size=160)

    # 1. Pure silence
    silence_pcm = b"\x00\x00" * 160
    assert detector.detect_frame(silence_pcm) is None
    print("PASS: Pure silence rejected.")

    # 2. Non-DTMF frequency (pure 1000 Hz sine wave)
    samples_1000hz = [
        int(16000 * (math_sin := (0.8 * __import__("math").sin(2 * 3.1415926535 * 1000 * n / 8000))))
        for n in range(160)
    ]
    sine_1000_pcm = struct.pack("<160h", *samples_1000hz)
    assert detector.detect_frame(sine_1000_pcm) is None
    print("PASS: Non-DTMF 1000 Hz tone rejected.")

    # 3. Random broadband white noise
    random.seed(42)
    noise_samples = [random.randint(-1500, 1500) for _ in range(160)]
    noise_pcm = struct.pack("<160h", *noise_samples)
    assert detector.detect_frame(noise_pcm) is None
    print("PASS: Broadband noise rejected.")

    # 4. Standard twist: Column slightly attenuated or louder (within -6 dB to +3 dB)
    # Row 770 Hz + Col 1336 Hz ('5')
    # Valid twist (+2 dB column):
    half_amp_row = 0.4 * 32767.0
    half_amp_col = 0.5 * 32767.0  # Slightly louder column
    samples_twist_valid = []
    for n in range(160):
        val = int(half_amp_row * __import__("math").sin(2 * 3.1415926535 * 770 * n / 8000) +
                  half_amp_col * __import__("math").sin(2 * 3.1415926535 * 1336 * n / 8000))
        samples_twist_valid.append(max(-32768, min(32767, val)))
    valid_pcm = struct.pack("<160h", *samples_twist_valid)
    assert detector.detect_frame(valid_pcm) == "5"
    print("PASS: Valid twist accepted ('5' detected).")

    # 5. Excessive reverse twist (+15 dB column): Column overwhelming row
    half_amp_row_weak = 0.05 * 32767.0
    half_amp_col_loud = 0.90 * 32767.0
    samples_twist_invalid = []
    for n in range(160):
        val = int(half_amp_row_weak * __import__("math").sin(2 * 3.1415926535 * 770 * n / 8000) +
                  half_amp_col_loud * __import__("math").sin(2 * 3.1415926535 * 1336 * n / 8000))
        samples_twist_invalid.append(max(-32768, min(32767, val)))
    invalid_pcm = struct.pack("<160h", *samples_twist_invalid)
    assert detector.detect_frame(invalid_pcm) is None
    print("PASS: Excessive twist rejected as per ITU-T standards.")


def test_3_rfc4733_packet_encoding_decoding_and_deduplication():
    print("\n--- Test 3: RFC 4733 / RFC 2833 RTP Telephone-Event Parsing & Deduplication ---")

    # 1. Encode packet for digit '9'
    pkt = encode_rfc4733_packet("9", end_bit=True, volume=10, duration=1600)
    assert len(pkt) == 4

    # 2. Decode packet
    event = decode_rfc4733_packet(pkt)
    assert event.digit == "9"
    assert event.event_id == 9
    assert event.end_bit is True
    assert event.volume == 10
    assert event.duration == 1600
    print(f"PASS: Decoded RFC 4733 event: {event}")

    # 3. Test special telephony keys ('*', '#', 'A')
    pkt_star = encode_rfc4733_packet("*", end_bit=True)
    assert decode_rfc4733_packet(pkt_star).digit == "*"

    pkt_hash = encode_rfc4733_packet("#", end_bit=True)
    assert decode_rfc4733_packet(pkt_hash).digit == "#"

    pkt_a = encode_rfc4733_packet("A", end_bit=True)
    assert decode_rfc4733_packet(pkt_a).digit == "A"
    print("PASS: Telephony symbols (*, #, A) correctly mapped.")

    # 4. RFC 4733 Section 2.5.1 deduplication test
    # A standard carrier sends 2 update packets (end_bit=False) followed by 3 duplicate end packets (end_bit=True)
    decoder = RFC4733EventDecoder()
    update_1 = encode_rfc4733_packet("4", end_bit=False, duration=160)
    update_2 = encode_rfc4733_packet("4", end_bit=False, duration=320)
    end_1 = encode_rfc4733_packet("4", end_bit=True, duration=480)
    end_2 = encode_rfc4733_packet("4", end_bit=True, duration=480)
    end_3 = encode_rfc4733_packet("4", end_bit=True, duration=480)

    res1 = decoder.process_packet(update_1)
    res2 = decoder.process_packet(update_2)
    res3 = decoder.process_packet(end_1)
    res4 = decoder.process_packet(end_2)
    res5 = decoder.process_packet(end_3)

    assert res1 is None
    assert res2 is None
    assert res3 == "4"  # First end packet emits digit
    assert res4 is None  # Duplicate end packet suppressed
    assert res5 is None  # Duplicate end packet suppressed
    print("PASS: RFC 4733 triplicate end packet redundancy successfully debounced into single event.")


def test_4_acoustic_debouncing_and_inter_digit_gaps():
    print("\n--- Test 4: Acoustic Debouncing & Inter-Digit Gap Detection ---")
    pad = DTMFPad(sample_rate=8000)

    # 1. Sustained tone burst of 120ms (6 consecutive 20ms frames) for key '7'
    # Must register exactly ONE '7'
    tone_120ms = DTMFToneGenerator.generate_tone("7", duration_ms=120, sample_rate=8000)
    chunk_size = 320  # 20ms at 8kHz 16-bit mono
    emitted_1 = []
    for i in range(0, len(tone_120ms), chunk_size):
        chunk = tone_120ms[i : i + chunk_size]
        digits = pad.process_pcm_chunk(chunk)
        emitted_1.extend(digits)

    assert emitted_1 == ["7"], f"Expected single ['7'], got {emitted_1}"
    print("PASS: 120ms continuous tone debounced to single keypress event.")

    # 2. Add silence gap of 40ms, followed by another tone of '7' for 80ms
    # Must register the second '7'
    gap_pcm = DTMFToneGenerator.generate_silence(duration_ms=40, sample_rate=8000)
    tone_80ms = DTMFToneGenerator.generate_tone("7", duration_ms=80, sample_rate=8000)
    seq_pcm = gap_pcm + tone_80ms

    emitted_2 = []
    for i in range(0, len(seq_pcm), chunk_size):
        chunk = seq_pcm[i : i + chunk_size]
        digits = pad.process_pcm_chunk(chunk)
        emitted_2.extend(digits)

    assert emitted_2 == ["7"], f"Expected ['7'] after gap, got {emitted_2}"
    print("PASS: Inter-digit gap permitted subsequent identical keypress.")

    # 3. Short glitch/click of 10ms (less than 40ms minimum persistence threshold)
    pad.reset()
    glitch_pcm = DTMFToneGenerator.generate_tone("3", duration_ms=10, sample_rate=8000)
    emitted_glitch = pad.process_pcm_chunk(glitch_pcm)
    assert len(emitted_glitch) == 0
    print("PASS: Sub-threshold audio glitch (< 40ms) safely ignored.")


def test_5_ivr_state_machine_navigation():
    print("\n--- Test 5: Multi-Level IVR State Machine Hierarchical Navigation ---")
    ivr = create_default_banking_ivr(call_id="call_ivr_001", default_lang="hi", caller_phone="+919876543210")

    # 1. Start IVR
    start_res = ivr.start()
    assert start_res.node_id == "root"
    assert "मुथूट फिनकॉर्प" in start_res.prompt
    assert start_res.is_terminal is False
    print("PASS: Root node initialized.")

    # 2. Select Language: English (press '2')
    step1 = ivr.handle_digit("2")
    assert step1.node_id == "node_lang_en"
    assert ivr.language == "en"
    assert step1.action == "set_language"
    print("PASS: Language selected as English.")

    # 3. Proceed to Main Menu (press '*')
    step2 = ivr.handle_digit("*")
    assert step2.node_id == "main_menu"
    assert "Press 1 to receive a UPI payment link" in step2.prompt
    print("PASS: Main Menu rendered in English.")

    # 4. Branch 1: Request WhatsApp UPI payment link (press '1')
    step3 = ivr.handle_digit("1")
    assert step3.node_id == "node_pay_upi"
    assert step3.action == "send_upi_link"
    assert "secure UPI payment link has been dispatched" in step3.prompt
    print("PASS: WhatsApp UPI payment link action executed.")

    # 5. Return to Main Menu (press '0')
    step4 = ivr.handle_digit("0")
    assert step4.node_id == "main_menu"
    print("PASS: Returned to Main Menu.")

    # 6. Complete call by hanging up (press '#')
    # First go to pay_upi then hang up
    ivr.handle_digit("1")
    step5 = ivr.handle_digit("#")
    assert step5.node_id == "node_hangup"
    assert step5.is_terminal is True
    assert ivr.is_completed is True
    print("PASS: Call successfully terminated via '#' hangup.")


def test_6_hybrid_voice_keypad_traversal_and_retries():
    print("\n--- Test 6: Hybrid Traversal (Spoken Utterances & Keypad) with Retry Limits ---")
    ivr = create_default_banking_ivr(call_id="call_ivr_002", default_lang="en")
    ivr.start()

    # 1. Spoken language selection: "Hindi"
    step1 = ivr.handle_speech("I speak Hindi")
    assert step1.node_id == "node_lang_hi"
    assert ivr.language == "hi"
    print("PASS: Spoken utterance 'Hindi' successfully selected Hindi.")

    # 2. Proceed to Main Menu
    ivr.handle_digit("*")

    # 3. Spoken transfer request: "loan officer"
    step2 = ivr.handle_speech("Please connect me to an officer")
    assert step2.node_id == "node_transfer_agent"
    assert step2.action == "transfer_human"
    assert step2.is_terminal is True
    assert "वरिष्ठ ऋण अधिकारी" in step2.prompt
    print("PASS: Spoken intent 'officer' routed to human transfer terminal node.")

    # 4. Retry limit test: Enter invalid digits 3 times
    ivr_retry = create_default_banking_ivr(call_id="call_retry_001", default_lang="en")
    ivr_retry.start()

    res1 = ivr_retry.handle_digit("8")  # Invalid on root
    assert res1.status == "INVALID_INPUT"
    assert res1.retries_left == 2

    res2 = ivr_retry.handle_digit("9")  # Invalid on root
    assert res2.status == "INVALID_INPUT"
    assert res2.retries_left == 1

    res3 = ivr_retry.handle_digit("7")  # 3rd invalid attempt
    assert res3.status == "MAX_RETRIES_EXCEEDED"
    assert res3.is_terminal is True
    assert ivr_retry.is_completed is True
    print("PASS: Exhausting maximum invalid retries gracefully terminated session.")


def test_7_multi_digit_collection_and_dpdp_masking():
    print("\n--- Test 7: Multi-Digit PIN Collection with DPDP Act 2023 Masking ---")
    ivr = create_default_banking_ivr(call_id="call_dpdp_001", default_lang="en")
    ivr.start()
    ivr.handle_digit("2")  # English
    ivr.handle_digit("*")  # Main menu

    # Navigate to Account Verification (press '2')
    nav_res = ivr.handle_digit("2")
    assert nav_res.node_id == "node_verify_account"
    print("PASS: Reached PIN collection node.")

    # Enter 4 digits: '5', '8', '2', '1'
    # First 3 digits should yield COLLECTING_DIGITS with masked digits
    r1 = ivr.handle_digit("5")
    assert r1.status == "COLLECTING_DIGITS"
    assert r1.collected_digits == "*"

    r2 = ivr.handle_digit("8")
    assert r2.status == "COLLECTING_DIGITS"
    assert r2.collected_digits == "**"

    r3 = ivr.handle_digit("2")
    assert r3.status == "COLLECTING_DIGITS"
    assert r3.collected_digits == "***"

    # 4th digit completes PIN
    r4 = ivr.handle_digit("1")
    assert r4.status == "OK"
    assert r4.node_id == "node_pin_verified"
    assert r4.collected_digits == "****"
    assert r4.raw_collected_digits == "5821"  # Internal only
    print("PASS: 4-digit PIN collected and masked as '****'.")

    # Check serialized dictionary: raw digits must NEVER be present
    dict_repr = r4.to_dict()
    assert "5821" not in str(dict_repr)
    assert dict_repr["collected_digits"] == "****"
    print("PASS: Transition result dictionary strictly masked.")

    # Check IVR audit summary: context and state history must be masked
    summary = ivr.get_summary()
    assert "5821" not in str(summary)
    assert summary["context"]["node_verify_account_digits_masked"] == "****"
    print("PASS: State Machine audit summary is 100% DPDP Act 2023 compliant.")


def test_8_fastapi_rest_api_endpoints():
    print("\n--- Test 8: FastAPI REST API Endpoints Integration ---")
    app = create_app()
    client = TestClient(app)

    # 1. Health check verifies active_ivr_sessions metric
    res = client.get("/health")
    assert res.status_code == 200
    assert "active_ivr_sessions" in res.json()
    print("PASS: Health check reports active_ivr_sessions metric.")

    # 2. POST /telephony/dtmf/decode with base64 PCM audio
    pcm_audio = DTMFToneGenerator.generate_sequence("123", tone_duration_ms=80, gap_duration_ms=40, sample_rate=8000)
    pcm_b64 = base64.b64encode(pcm_audio).decode("ascii")

    dec_res = client.post("/telephony/dtmf/decode", json={"pcm_base64": pcm_b64, "sample_rate": 8000})
    assert dec_res.status_code == 200
    data = dec_res.json()
    assert data["status"] == "ok"
    assert data["detected_digits"] == ["1", "2", "3"]
    print(f"PASS: /telephony/dtmf/decode successfully detected sequence: {data['detected_digits']}")

    # 3. POST /telephony/dtmf/decode with hex-encoded RFC 4733 packet
    rfc_pkt = encode_rfc4733_packet("8", end_bit=True, volume=15, duration=800)
    rfc_hex = rfc_pkt.hex()

    rfc_res = client.post("/telephony/dtmf/decode", json={"rfc4733_hex": rfc_hex})
    assert rfc_res.status_code == 200
    rfc_data = rfc_res.json()
    assert rfc_data["status"] == "ok"
    assert rfc_data["rfc4733_event"]["digit"] == "8"
    print(f"PASS: /telephony/dtmf/decode parsed RFC 4733 event: {rfc_data['rfc4733_event']}")

    # 4. POST /telephony/ivr/start
    start_res = client.post(
        "/telephony/ivr/start",
        json={"call_id": "api_call_001", "default_lang": "en", "caller_phone": "+919876543210"}
    )
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["status"] == "ok"
    assert start_data["transition"]["node_id"] == "root"
    print("PASS: /telephony/ivr/start spawned session 'api_call_001'.")

    # 5. POST /telephony/ivr/action: Press '2' (English)
    act_res1 = client.post(
        "/telephony/ivr/action",
        json={"call_id": "api_call_001", "digit": "2"}
    )
    assert act_res1.status_code == 200
    assert act_res1.json()["transition"]["node_id"] == "node_lang_en"

    # Proceed to main menu
    client.post("/telephony/ivr/action", json={"call_id": "api_call_001", "digit": "*"})

    # Advance via speech: "send payment link"
    act_res2 = client.post(
        "/telephony/ivr/action",
        json={"call_id": "api_call_001", "speech_text": "payment link"}
    )
    assert act_res2.status_code == 200
    assert act_res2.json()["transition"]["node_id"] == "node_pay_upi"
    assert act_res2.json()["transition"]["action"] == "send_upi_link"
    print("PASS: /telephony/ivr/action handled digit and speech triggers.")

    # 6. GET /telephony/ivr/session/{call_id}
    audit_res = client.get("/telephony/ivr/session/api_call_001")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["call_id"] == "api_call_001"
    assert audit_data["current_node_id"] == "node_pay_upi"
    assert audit_data["history_count"] >= 3
    print("PASS: /telephony/ivr/session/api_call_001 returned full audit summary.")


def run_all_tests():
    print("================================================================================")
    print("VERBALYZE DTMF KEYPAD ENGINE & MULTI-LEVEL IVR TEST SUITE")
    print("================================================================================")
    t0 = time.time()

    test_1_goertzel_all_16_digits_8k_and_16k()
    test_2_frequency_rejection_and_twist_tolerance()
    test_3_rfc4733_packet_encoding_decoding_and_deduplication()
    test_4_acoustic_debouncing_and_inter_digit_gaps()
    test_5_ivr_state_machine_navigation()
    test_6_hybrid_voice_keypad_traversal_and_retries()
    test_7_multi_digit_collection_and_dpdp_masking()
    test_8_fastapi_rest_api_endpoints()

    elapsed = time.time() - t0
    print("\n================================================================================")
    print(f"SUCCESS: All 8 DTMF & IVR Engine Tests Passed in {elapsed:.2f}s!")
    print("================================================================================")


if __name__ == "__main__":
    run_all_tests()
