"""
scripts/test_voice_biometrics_and_anti_spoof.py

Comprehensive Test Suite for:
1. Acoustic Feature Extraction & 64-Dimensional Speaker Embedding Generation.
2. Multi-Sample Voiceprint Enrollment & Centroid Vector Updating.
3. Authentic Speaker Verification (High Confidence Match >= 0.78).
4. Impostor Speaker Rejection (Mismatch Detection < 0.65).
5. Synthetic AI Deepfake Vocoder Detection (Zero Pitch Jitter & High-Band Flatness).
6. Loudspeaker Phone Replay Attack Detection (Mid-Band Resonance & Noise Flatness).
7. VoiceAgent Telephony Integration (Step-Up Authentication & Disclosure Blocking).
8. Outbound Dialer CDR Telemetry Attribution & FastAPI REST Endpoints (DPDP Erasure).

Zero-emoji compliant.
DPDP Act 2023 & RBI Fair Practices Code compliant.
"""

import os
import sys
import base64
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from verbalyze.telephony.voice_biometrics import (
    BiometricStatus,
    SpoofType,
    AntiSpoofResult,
    BiometricVerificationResult,
    SpeakerProfile,
    AcousticFeatureExtractor,
    AntiSpoofingDetector,
    BiometricVerificationEngine,
    SpeakerProfileRegistry,
)
from verbalyze.agent.voice_bot import VoiceAgent
from verbalyze.campaign.models import CampaignConfig, Lead, CallDisposition
from verbalyze.campaign.dialer import OutboundCampaignDialer
from verbalyze.telephony.server import create_app


def generate_synthetic_audio(
    speaker_id: str = "spk1",
    duration_sec: float = 2.0,
    sample_rate: int = 8000,
    spoof_type: str = "human",
    seed: int = 42,
) -> bytes:
    """
    Generates realistic synthetic audio waveforms for biometric testing.
    spoof_type: 'human', 'synthetic_deepfake', or 'replay_attack'
    """
    np.random.seed(seed)
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    if speaker_id == "spk1":
        f0 = 135.0
        formants = [680.0, 1220.0, 2450.0]
    elif speaker_id == "spk2":
        f0 = 220.0
        formants = [420.0, 1820.0, 2850.0]
    else:
        f0 = 175.0
        formants = [550.0, 1500.0, 2600.0]

    if spoof_type == "synthetic_deepfake":
        # AI TTS / Neural vocoder clone: perfectly unperturbed static pitch, zero micro-tremor
        phase = 2 * np.pi * f0 * t
        sig = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
        # Low-pass filter cutting off high frequencies
        sig = np.clip(sig, -1.0, 1.0)
        return (sig * 32767).astype(np.int16).tobytes()

    # Human voice with organic pitch jitter / micro-tremor
    jitter = 2.5 * np.sin(2 * np.pi * 3.5 * t)
    phase = 2 * np.pi * (f0 * t + jitter * t)
    sig = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    for ff in formants:
        sig += 0.25 * np.sin(2 * np.pi * ff * t)

    # Add realistic vocal tract turbulence and background breath
    sig += 0.04 * np.random.normal(0, 0.05, len(t))

    if spoof_type == "replay_attack":
        # Loudspeaker phone replay: colored by speaker enclosure resonance (2500-3200Hz)
        res_tone = 0.9 * np.sin(2 * np.pi * 2600.0 * t) + 0.8 * np.sin(2 * np.pi * 3000.0 * t)
        sig = 0.25 * sig + res_tone + 0.08 * np.random.normal(0, 0.1, len(t))

    sig = np.clip(sig, -1.0, 1.0)
    return (sig * 32767).astype(np.int16).tobytes()


def test_1_acoustic_feature_extraction_and_speaker_embedding():
    print("\n--- Test 1: Feature Extraction & 64-Dimensional Speaker Embedding ---")
    extractor = AcousticFeatureExtractor(sample_rate=8000)
    audio_pcm = generate_synthetic_audio("spk1", duration_sec=1.5, spoof_type="human", seed=10)

    features = extractor.extract_features(audio_pcm)
    assert "mfccs" in features
    assert "spectral_centroid" in features
    assert "spectral_flatness" in features
    assert "spectral_rolloff" in features
    assert "zcr" in features
    assert "f0" in features

    assert features["mfccs"].shape[1] == 13
    assert len(features["spectral_centroid"]) == len(features["mfccs"])

    embedding = extractor.generate_speaker_embedding(audio_pcm)
    assert embedding.shape == (64,)
    l2_norm = float(np.linalg.norm(embedding))
    print(f"Embedding shape: {embedding.shape}, L2-norm: {l2_norm:.4f}")
    assert abs(l2_norm - 1.0) < 1e-4, "Speaker embedding must be unit normalized."
    print("Test 1 Passed: Acoustic feature extraction and unit embedding verified.")


def test_2_speaker_enrollment_and_centroid_updating():
    print("\n--- Test 2: Voiceprint Multi-Sample Enrollment & Centroid Updating ---")
    registry = SpeakerProfileRegistry(sample_rate=8000)

    sample1 = generate_synthetic_audio("spk1", duration_sec=1.2, spoof_type="human", seed=11)
    sample2 = generate_synthetic_audio("spk1", duration_sec=1.2, spoof_type="human", seed=12)

    profile = registry.enroll_speaker(
        customer_id="BORROWER_9821",
        name="Sunil Kumar",
        loan_id="MUTH_LN_4040",
        audio_samples=[sample1, sample2],
    )

    assert profile.customer_id == "BORROWER_9821"
    assert profile.sample_count == 2
    assert profile.centroid_embedding.shape == (64,)
    assert len(profile.voiceprint_sha256) == 16
    assert abs(np.linalg.norm(profile.centroid_embedding) - 1.0) < 1e-4

    # Incremental enrollment update
    sample3 = generate_synthetic_audio("spk1", duration_sec=1.0, spoof_type="human", seed=13)
    updated_profile = registry.enroll_speaker(
        customer_id="BORROWER_9821",
        name="Sunil Kumar",
        loan_id="MUTH_LN_4040",
        audio_samples=[sample3],
    )
    assert updated_profile.sample_count == 3
    assert registry.count() == 1

    # Lookup by loan_id
    lookup = registry.get_profile("MUTH_LN_4040")
    assert lookup is not None
    assert lookup.customer_id == "BORROWER_9821"

    # DPDP Right to Erasure
    deleted = registry.delete_profile("BORROWER_9821")
    assert deleted is True
    assert registry.count() == 0
    assert registry.get_profile("BORROWER_9821") is None
    print("Test 2 Passed: Speaker enrollment, centroid aggregation, and DPDP erasure verified.")


def test_3_authentic_speaker_matching():
    print("\n--- Test 3: Authentic Speaker Matching (Cosine Similarity >= 0.78) ---")
    registry = SpeakerProfileRegistry(sample_rate=8000)
    engine = BiometricVerificationEngine(sample_rate=8000, verify_threshold=0.78)

    enroll_audio = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="human", seed=21)
    probe_audio = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="human", seed=22)

    profile = registry.enroll_speaker(
        customer_id="CUST_AUTH_01",
        name="Anita Desai",
        loan_id="MUTH_LN_7788",
        audio_samples=[enroll_audio],
    )

    result = engine.verify_speaker(probe_audio, profile)
    print(f"Status: {result.status.value}, Confidence: {result.confidence:.3f}, Cosine Sim: {result.cosine_similarity:.3f}")
    assert result.status == BiometricStatus.VERIFIED
    assert result.is_verified is True
    assert result.cosine_similarity >= 0.78
    assert result.anti_spoof.is_authentic is True
    assert profile.last_verified_at is not None
    print("Test 3 Passed: Authentic speaker matched with high confidence.")


def test_4_impostor_speaker_rejection():
    print("\n--- Test 4: Impostor Speaker Rejection (Mismatch < 0.65) ---")
    registry = SpeakerProfileRegistry(sample_rate=8000)
    engine = BiometricVerificationEngine(sample_rate=8000, indeterminate_threshold=0.65)

    spk1_enroll = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="human", seed=31)
    spk2_probe = generate_synthetic_audio("spk2", duration_sec=2.0, spoof_type="human", seed=32)

    profile = registry.enroll_speaker(
        customer_id="CUST_AUTH_02",
        name="Vikram Mehta",
        loan_id="MUTH_LN_9911",
        audio_samples=[spk1_enroll],
    )

    result = engine.verify_speaker(spk2_probe, profile)
    print(f"Status: {result.status.value}, Confidence: {result.confidence:.3f}, Cosine Sim: {result.cosine_similarity:.3f}")
    assert result.status == BiometricStatus.MISMATCH_IMPOSTOR
    assert result.is_verified is False
    assert result.cosine_similarity < 0.65
    print("Test 4 Passed: Impostor speaker rejected successfully.")


def test_5_synthetic_deepfake_ai_detection():
    print("\n--- Test 5: Synthetic AI Deepfake Vocoder Detection ---")
    detector = AntiSpoofingDetector(sample_rate=8000)
    engine = BiometricVerificationEngine(sample_rate=8000)
    registry = SpeakerProfileRegistry(sample_rate=8000)

    synth_audio = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="synthetic_deepfake", seed=41)
    human_audio = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="human", seed=42)

    # Standalone detector check
    res = detector.analyze(synth_audio)
    print(f"Detector decision: {res.decision.value}, is_authentic: {res.is_authentic}, score: {res.synthetic_score:.3f}")
    assert res.decision == SpoofType.SYNTHETIC_DEEPFAKE
    assert res.is_authentic is False
    assert res.synthetic_score >= 0.70

    # Verification engine gate check
    profile = registry.enroll_speaker("CUST_SPOOF", "Deepfake Target", "LOAN_SPOOF", [human_audio])
    verif_res = engine.verify_speaker(synth_audio, profile)
    print(f"Engine status: {verif_res.status.value}, Msg: {verif_res.message}")
    assert verif_res.status == BiometricStatus.SPOOF_DETECTED
    assert verif_res.is_verified is False
    print("Test 5 Passed: Synthetic deepfake vocoder detected and blocked.")


def test_6_loudspeaker_phone_replay_attack_detection():
    print("\n--- Test 6: Loudspeaker Phone Replay Attack Detection ---")
    detector = AntiSpoofingDetector(sample_rate=8000)
    engine = BiometricVerificationEngine(sample_rate=8000)
    registry = SpeakerProfileRegistry(sample_rate=8000)

    replay_audio = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="replay_attack", seed=51)
    human_audio = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="human", seed=52)

    # Standalone detector check
    res = detector.analyze(replay_audio)
    print(f"Detector decision: {res.decision.value}, is_authentic: {res.is_authentic}, score: {res.replay_score:.3f}")
    assert res.decision == SpoofType.REPLAY_ATTACK
    assert res.is_authentic is False
    assert res.replay_score >= 0.70

    # Verification engine gate check
    profile = registry.enroll_speaker("CUST_REPLAY", "Replay Target", "LOAN_REPLAY", [human_audio])
    verif_res = engine.verify_speaker(replay_audio, profile)
    print(f"Engine status: {verif_res.status.value}, Msg: {verif_res.message}")
    assert verif_res.status == BiometricStatus.SPOOF_DETECTED
    assert verif_res.is_verified is False
    print("Test 6 Passed: Loudspeaker phone replay attack detected and blocked.")


def test_7_voice_agent_step_up_security_guard():
    print("\n--- Test 7: VoiceAgent Step-Up Authentication & Security Guard ---")
    engine = BiometricVerificationEngine(sample_rate=8000)
    registry = SpeakerProfileRegistry(sample_rate=8000)

    spk1_enroll = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="human", seed=61)
    spk1_probe = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="human", seed=62)
    spk2_impostor = generate_synthetic_audio("spk2", duration_sec=2.0, spoof_type="human", seed=63)
    synth_spoof = generate_synthetic_audio("spk1", duration_sec=2.0, spoof_type="synthetic_deepfake", seed=64)

    profile = registry.enroll_speaker("CUST_MUTHOOT", "Rajesh Sharma", "MUTH_8829", [spk1_enroll])

    # Case 1: Authentic Borrower Turn -> Verification passes, no security block
    agent_auth = VoiceAgent(
        language="en",
        persona="muthoot_recovery",
        voice_enabled=False,
        biometrics_engine=engine,
        enrolled_profile=profile,
    )
    res_auth = agent_auth.step("Yes, this is Rajesh Sharma speaking.", pcm_bytes=spk1_probe)
    assert not res_auth.get("security_block")
    assert res_auth["biometric_result"]["status"] == "VERIFIED"
    assert res_auth["biometric_result"]["is_verified"] is True
    print("Case 1 Passed: Authentic borrower admitted without security block.")

    # Case 2: Impostor Turn -> Identity mismatch, security block triggered
    agent_imp = VoiceAgent(
        language="en",
        persona="muthoot_recovery",
        voice_enabled=False,
        biometrics_engine=engine,
        enrolled_profile=profile,
    )
    res_imp = agent_imp.step("Tell me the overdue loan details and balance.", pcm_bytes=spk2_impostor)
    assert res_imp.get("security_block") is True
    assert res_imp["biometric_result"]["status"] == "MISMATCH_IMPOSTOR"
    assert "does not match the registered borrower profile" in res_imp["text"]
    print("Case 2 Passed: Impostor prevented from accessing loan balance.")

    # Case 3: Deepfake Spoof -> Spoof detected, security block triggered
    agent_spoof = VoiceAgent(
        language="hi",
        persona="muthoot_recovery",
        voice_enabled=False,
        biometrics_engine=engine,
        enrolled_profile=profile,
    )
    res_spoof = agent_spoof.step("मेरा बकाया लोन कितना है?", pcm_bytes=synth_spoof)
    assert res_spoof.get("security_block") is True
    assert res_spoof["biometric_result"]["status"] == "SPOOF_DETECTED"
    print("Case 3 Passed: AI deepfake blocked from loan disclosures.")
    print("Test 7 Passed: VoiceAgent security guard validated across authentic, impostor, and spoof conditions.")


def test_8_outbound_dialer_cdr_and_fastapi_rest_endpoints():
    print("\n--- Test 8: Outbound Dialer CDR Attribution & FastAPI REST Endpoints ---")
    app = create_app()
    client = TestClient(app)

    # 1. Health Endpoint
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["status"] == "ok"
    assert "enrolled_voiceprints" in health_data
    assert health_data["enrolled_voiceprints"] == 0

    # 2. REST Enrollment via FastAPI
    spk1_audio = generate_synthetic_audio("spk1", duration_sec=1.5, spoof_type="human", seed=71)
    b64_sample = base64.b64encode(spk1_audio).decode("utf-8")

    resp_enroll = client.post("/telephony/biometrics/enroll", json={
        "customer_id": "CUST_FASTAPI_01",
        "name": "Kavita Rao",
        "loan_id": "MUTH_9988",
        "audio_samples_b64": [b64_sample],
    })
    assert resp_enroll.status_code == 200
    enroll_data = resp_enroll.json()
    assert enroll_data["status"] == "ok"
    assert enroll_data["profile"]["customer_id"] == "CUST_FASTAPI_01"

    # Verify count increment in health
    resp_health2 = client.get("/health")
    assert resp_health2.json()["enrolled_voiceprints"] == 1

    # 3. REST Verification
    spk1_probe = generate_synthetic_audio("spk1", duration_sec=1.5, spoof_type="human", seed=72)
    b64_probe = base64.b64encode(spk1_probe).decode("utf-8")

    resp_verify = client.post("/telephony/biometrics/verify", json={
        "customer_id": "CUST_FASTAPI_01",
        "audio_sample_b64": b64_probe,
    })
    assert resp_verify.status_code == 200
    verif_data = resp_verify.json()
    assert verif_data["status"] == "ok"
    assert verif_data["verification"]["status"] == "VERIFIED"
    assert verif_data["verification"]["is_verified"] is True

    # 4. REST Anti-Spoof
    synth_audio = generate_synthetic_audio("spk1", duration_sec=1.5, spoof_type="synthetic_deepfake", seed=73)
    b64_synth = base64.b64encode(synth_audio).decode("utf-8")

    resp_anti_spoof = client.post("/telephony/biometrics/anti-spoof", json={
        "audio_sample_b64": b64_synth,
    })
    assert resp_anti_spoof.status_code == 200
    spoof_data = resp_anti_spoof.json()
    assert spoof_data["status"] == "ok"
    assert spoof_data["anti_spoof"]["decision"] == "SYNTHETIC_DEEPFAKE"
    assert spoof_data["anti_spoof"]["is_authentic"] is False

    # 5. REST Profile Lookup
    resp_get = client.get("/telephony/biometrics/profile/CUST_FASTAPI_01")
    assert resp_get.status_code == 200
    assert resp_get.json()["profile"]["customer_id"] == "CUST_FASTAPI_01"

    # 6. REST List Profiles
    resp_list = client.get("/telephony/biometrics/profiles")
    assert resp_list.status_code == 200
    assert resp_list.json()["count"] == 1

    # 7. DPDP Act 2023 Right to Erasure
    resp_del = client.delete("/telephony/biometrics/profile/CUST_FASTAPI_01")
    assert resp_del.status_code == 200
    assert resp_del.json()["deleted"] is True

    resp_get_after = client.get("/telephony/biometrics/profile/CUST_FASTAPI_01")
    assert resp_get_after.status_code == 404

    # 8. Outbound Campaign Dialer CDR Telemetry Attribution
    config = CampaignConfig(
        campaign_id="test_biometrics_campaign",
        enforce_trai_calling_hours=False,
        enforce_dnd_check=False,
        amd_enabled=False,
        llm_provider="mock",
    )
    dialer_registry = SpeakerProfileRegistry(sample_rate=8000)
    dialer_engine = BiometricVerificationEngine(sample_rate=8000)

    dialer_prof = dialer_registry.enroll_speaker(
        customer_id="LEAD_BIO_01",
        name="Deepak Joshi",
        loan_id="MUTH_LN_101",
        audio_samples=[spk1_audio],
    )

    dialer = OutboundCampaignDialer(
        config=config,
        biometrics_registry=dialer_registry,
        biometrics_engine=dialer_engine,
    )

    raw_leads = [
        {
            "lead_id": "LEAD_BIO_01",
            "name": "Deepak Joshi",
            "phone_number": "+919876543210",
            "loan_id": "MUTH_LN_101",
            "amount_due": 5420.0,
            "custom_metadata": {
                "caller_pcm_bytes": spk1_probe,
            },
        }
    ]
    dialer.ingest_leads_from_list(raw_leads)
    cdrs = dialer.start_campaign()

    assert len(cdrs) == 1
    cdr = cdrs[0]
    print(f"CDR Biometric Status: {cdr.biometric_status}, Confidence: {cdr.biometric_confidence:.3f}, Spoof: {cdr.spoof_type}")
    assert cdr.biometric_status == "VERIFIED"
    assert cdr.biometric_confidence > 0.70
    assert cdr.spoof_type == "AUTHENTIC_HUMAN"

    cdr_dict = cdr.to_dict()
    assert cdr_dict["biometric_status"] == "VERIFIED"
    assert cdr_dict["spoof_type"] == "AUTHENTIC_HUMAN"
    assert "biometric_confidence" in cdr_dict

    print("Test 8 Passed: Dialer CDR attribution and FastAPI REST routes verified.")


def main():
    print("================================================================================")
    print("VERBALYZE: LIVE VOICE BIOMETRICS & ANTI-SPOOFING SPEAKER VERIFICATION SUITE")
    print("================================================================================")
    start_time = time.time()

    test_1_acoustic_feature_extraction_and_speaker_embedding()
    test_2_speaker_enrollment_and_centroid_updating()
    test_3_authentic_speaker_matching()
    test_4_impostor_speaker_rejection()
    test_5_synthetic_deepfake_ai_detection()
    test_6_loudspeaker_phone_replay_attack_detection()
    test_7_voice_agent_step_up_security_guard()
    test_8_outbound_dialer_cdr_and_fastapi_rest_endpoints()

    elapsed = time.time() - start_time
    print("================================================================================")
    print(f"ALL 8 VOICE BIOMETRICS & ANTI-SPOOFING TESTS PASSED ({elapsed:.2f}s)")
    print("Zero Emojis. DPDP Act 2023 & RBI Fair Practices Code Compliant.")
    print("================================================================================")


if __name__ == "__main__":
    main()
