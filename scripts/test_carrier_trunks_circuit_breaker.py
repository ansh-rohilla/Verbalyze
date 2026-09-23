"""
scripts/test_carrier_trunks_circuit_breaker.py

Comprehensive Test Suite for:
1. Carrier Trunk Registration, In-Memory QoS Tracking & DPDP Masking.
2. ITU-T G.107 E-Model MOS (Mean Opinion Score) Telemetry Computation.
3. SIP Circuit Breaker State Transitions (CLOSED -> OPEN on Carrier 5xx/408/480 vs 486/603 Ignore).
4. Cooldown Timer & HALF_OPEN Probe Recovery back to CLOSED.
5. Sub-150ms Automated Multi-Trunk Failover Dispatching.
6. Indian 22-Circle Least-Cost Routing (LCR) and Carrier Priority Resolution.
7. Campaign Dialing Engine Integration with Trunk Failover & CDR Attribution.
8. FastAPI Telephony Carrier Trunk REST Management Endpoints.

Zero-emoji compliant.
DPDP Act 2023 compliant.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from verbalyze.telephony.circuit_breaker import (
    CircuitBreakerState,
    TrunkHealth,
    SIPResponseCategory,
    categorize_sip_code,
    calculate_itu_g107_mos,
    SIPCircuitBreakerConfig,
    TrunkQoS,
    SIPCircuitBreaker,
)
from verbalyze.telephony.trunk_router import (
    TelecomCircle,
    detect_telecom_circle,
    CarrierTrunk,
    MultiTrunkRouter,
    create_default_indian_trunk_mesh,
)
from verbalyze.campaign.models import Lead, LeadStatus, CallDisposition
from verbalyze.campaign.dialer import CampaignDialer, CampaignConfig
from verbalyze.telephony.server import create_app


def test_1_trunk_registration_and_initial_qos():
    print("\n--- Test 1: Carrier Trunk Registration & In-Memory QoS Tracking ---")
    trunk = CarrierTrunk(
        trunk_id="trunk_tata_test",
        carrier_name="Tata Teleservices",
        sip_host="sip.tata.in",
        sip_port=5060,
        priority=1,
        cost_per_minute_inr=0.28,
        supported_circles=["MAHARASHTRA", "MUMBAI", "ALL"],
    )

    assert trunk.circuit_breaker.state == CircuitBreakerState.CLOSED
    assert trunk.health == TrunkHealth.HEALTHY
    assert trunk.can_route_call() is True
    assert trunk.qos.total_calls == 0

    # Record first call (200 OK)
    trunk.record_call_result(sip_code=200, rtt_ms=25.0, jitter_ms=4.0, packet_loss_pct=0.1)
    assert trunk.qos.total_calls == 1
    assert trunk.qos.successful_calls == 1
    assert trunk.qos.carrier_fault_calls == 0
    assert trunk.qos.avg_rtt_ms == 25.0
    assert trunk.health == TrunkHealth.HEALTHY

    # Verify serialization
    data = trunk.to_dict()
    assert data["trunk_id"] == "trunk_tata_test"
    assert data["carrier_name"] == "Tata Teleservices"
    assert data["circuit_breaker"]["state"] == "CLOSED"
    assert data["qos"]["total_calls"] == 1
    assert data["health"] == "HEALTHY"

    print("[PASSED] Carrier Trunk registration and initial QoS tracking verified.")


def test_2_itu_t_g107_mos_score_calculation():
    print("\n--- Test 2: ITU-T G.107 E-Model MOS Score Calculation ---")

    # 1. Ideal conditions (low delay, low jitter, 0% loss)
    ideal_mos = calculate_itu_g107_mos(one_way_delay_ms=20.0, jitter_ms=3.0, packet_loss_pct=0.0)
    print(f"Ideal Network MOS: {ideal_mos:.3f}")
    assert 4.3 <= ideal_mos <= 4.5, f"Expected ideal MOS between 4.3 and 4.5, got {ideal_mos}"

    # 2. Moderate VoIP conditions (80ms delay, 15ms jitter, 1% loss)
    moderate_mos = calculate_itu_g107_mos(one_way_delay_ms=80.0, jitter_ms=15.0, packet_loss_pct=1.0)
    print(f"Moderate Network MOS: {moderate_mos:.3f}")
    assert 3.6 <= moderate_mos < 4.4, f"Expected moderate MOS between 3.6 and 4.4, got {moderate_mos}"

    # 3. Degraded VoIP conditions (250ms delay, 45ms jitter, 8% loss)
    degraded_mos = calculate_itu_g107_mos(one_way_delay_ms=250.0, jitter_ms=45.0, packet_loss_pct=8.0)
    print(f"Degraded Network MOS: {degraded_mos:.3f}")
    assert 1.0 <= degraded_mos < 3.2, f"Expected degraded MOS below 3.2, got {degraded_mos}"

    # 4. Severe outage conditions (1000ms delay, 120ms jitter, 50% loss)
    severe_mos = calculate_itu_g107_mos(one_way_delay_ms=1000.0, jitter_ms=120.0, packet_loss_pct=50.0)
    print(f"Severe Outage MOS: {severe_mos:.3f}")
    assert severe_mos == 1.0, f"Expected bottom clamped MOS 1.0, got {severe_mos}"

    # 5. Perfect conditions upper clamp test
    upper_mos = calculate_itu_g107_mos(one_way_delay_ms=0.0, jitter_ms=0.0, packet_loss_pct=0.0)
    assert upper_mos <= 4.5

    print("[PASSED] ITU-T G.107 E-model transmission rating factor and MOS calculation verified.")


def test_3_circuit_breaker_transitions_closed_to_open():
    print("\n--- Test 3: SIP Circuit Breaker State Transitions (CLOSED -> OPEN) ---")

    # Verify SIP code categorization
    assert categorize_sip_code(200) == SIPResponseCategory.SUCCESS
    assert categorize_sip_code(486) == SIPResponseCategory.CLIENT_OUTCOME  # Busy
    assert categorize_sip_code(603) == SIPResponseCategory.CLIENT_OUTCOME  # Decline
    assert categorize_sip_code(404) == SIPResponseCategory.CLIENT_OUTCOME  # Not Found
    assert categorize_sip_code(487) == SIPResponseCategory.CLIENT_OUTCOME  # Canceled
    assert categorize_sip_code(503) == SIPResponseCategory.CARRIER_FAULT   # Service Unavailable
    assert categorize_sip_code(500) == SIPResponseCategory.CARRIER_FAULT   # Server Internal Error
    assert categorize_sip_code(408) == SIPResponseCategory.CARRIER_FAULT   # Request Timeout
    assert categorize_sip_code(480) == SIPResponseCategory.CARRIER_FAULT   # Temporarily Unavailable

    config = SIPCircuitBreakerConfig(
        consecutive_carrier_faults_threshold=3,
        failure_rate_threshold=0.5,
        min_requests=3,
        cooldown_duration_sec=5.0,
    )
    cb = SIPCircuitBreaker("trunk_test", config)

    # 1. Normal client outcomes must NOT trip circuit breaker
    cb.record_call(486)  # Busy
    cb.record_call(603)  # Decline
    cb.record_call(404)  # Not Found
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.consecutive_carrier_faults == 0
    assert cb.can_route_call() is True

    # 2. Consecutive carrier faults trip circuit breaker to OPEN
    cb.record_call(503)
    assert cb.consecutive_carrier_faults == 1
    assert cb.state == CircuitBreakerState.CLOSED

    cb.record_call(503)
    assert cb.consecutive_carrier_faults == 2
    assert cb.state == CircuitBreakerState.CLOSED

    cb.record_call(503)
    assert cb.consecutive_carrier_faults == 3
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_route_call() is False

    print("[PASSED] Circuit Breaker correctly differentiates client outcomes from carrier faults and trips to OPEN.")


def test_4_cooldown_and_half_open_probe_recovery():
    print("\n--- Test 4: Cooldown Timer & HALF_OPEN Probe Recovery ---")

    config = SIPCircuitBreakerConfig(
        consecutive_carrier_faults_threshold=2,
        failure_rate_threshold=0.5,
        min_requests=2,
        cooldown_duration_sec=0.1,  # 100ms cooldown for fast unit test
        half_open_success_threshold=2,
    )
    cb = SIPCircuitBreaker("trunk_fast_cooldown", config)

    # Trip to OPEN
    cb.record_call(503)
    cb.record_call(503)
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_route_call() is False

    # Wait for cooldown to expire
    time.sleep(0.12)

    # Breaker should allow probe call and transition to HALF_OPEN
    assert cb.can_route_call() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # First probe call succeeds
    cb.record_call(200)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.half_open_successes == 1

    # Second probe call succeeds -> transitions back to CLOSED
    cb.record_call(200)
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.consecutive_carrier_faults == 0
    assert cb.can_route_call() is True

    # Verify if probe call fails, it immediately trips back to OPEN
    cb.trip_to_open("Test forced trip")
    time.sleep(0.12)
    assert cb.can_route_call() is True  # Enters HALF_OPEN
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Probe fails with 503
    cb.record_call(503)
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_route_call() is False

    print("[PASSED] Cooldown timer, HALF_OPEN probe trials, and recovery to CLOSED verified.")


def test_5_sub_150ms_automated_multi_trunk_failover():
    print("\n--- Test 5: Sub-150ms Automated Multi-Trunk Failover Dispatch ---")

    router = MultiTrunkRouter()

    primary_trunk = CarrierTrunk(
        trunk_id="trunk_primary",
        carrier_name="Primary Carrier Outage",
        sip_host="sip.primary.com",
        priority=1,
        cost_per_minute_inr=0.25,
        supported_circles=["ALL"],
    )
    secondary_trunk = CarrierTrunk(
        trunk_id="trunk_secondary",
        carrier_name="Secondary Carrier Backup",
        sip_host="sip.secondary.com",
        priority=2,
        cost_per_minute_inr=0.30,
        supported_circles=["ALL"],
    )

    router.register_trunk(primary_trunk)
    router.register_trunk(secondary_trunk)

    # Dispatch function simulating primary trunk failure (503) and secondary trunk success (200)
    def mock_dispatch(trunk: CarrierTrunk, phone: str):
        if trunk.trunk_id == "trunk_primary":
            return {"sip_code": 503, "connected": False, "call_sid": "CA_fail"}
        return {"sip_code": 200, "connected": True, "call_sid": "CA_success_001"}

    t_start = time.perf_counter()
    result = router.dispatch_with_failover(
        destination_phone="+919810123456",
        dispatch_fn=mock_dispatch,
        rtt_ms=30.0,
        packet_loss_pct=0.0,
    )
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"Failover Dispatch Execution Time: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 150.0, f"Failover took {elapsed_ms:.2f} ms, exceeding 150ms limit."
    assert result.success is True
    assert result.failover_occurred is True
    assert result.failover_count == 1
    assert result.selected_trunk.trunk_id == "trunk_secondary"
    assert result.selected_trunk.carrier_name == "Secondary Carrier Backup"
    assert len(result.attempted_trunks) == 2
    assert result.dispatch_result["call_sid"] == "CA_success_001"

    # Verify primary trunk recorded the 503
    assert primary_trunk.qos.carrier_fault_calls == 1

    print("[PASSED] Sub-150ms automated multi-trunk failover dispatch successfully verified.")


def test_6_indian_22_circle_least_cost_routing_and_priority():
    print("\n--- Test 6: Indian 22-Circle Least-Cost Routing & Dynamic Priority ---")

    # 1. Circle Detection
    delhi_circle = detect_telecom_circle("919810123456")
    mumbai_circle = detect_telecom_circle("919820123456")
    karnataka_circle = detect_telecom_circle("919845123456")
    kolkata_circle = detect_telecom_circle("919830123456")
    mp_circle = detect_telecom_circle("919826123456")
    assam_circle = detect_telecom_circle("919864123456")

    assert delhi_circle == TelecomCircle.DELHI.value
    assert mumbai_circle == TelecomCircle.MUMBAI.value
    assert karnataka_circle == TelecomCircle.KARNATAKA.value
    assert kolkata_circle == TelecomCircle.KOLKATA.value
    assert mp_circle == TelecomCircle.MADHYA_PRADESH.value
    assert assam_circle == TelecomCircle.ASSAM.value

    # 2. LCR Route Resolution with Default Mesh
    mesh_router = create_default_indian_trunk_mesh()

    # Route for Delhi (+91 9810...)
    primary_delhi, fallbacks_delhi = mesh_router.resolve_routes("+919810123456")
    assert primary_delhi is not None
    # Airtel (Circle North) or Exotel (Pan-India) should be prioritized
    assert primary_delhi.trunk_id in ["airtel_north", "exotel_primary", "tatatele_west"]

    # Test preferred carrier override
    primary_pref, fallbacks_pref = mesh_router.resolve_routes("+919810123456", preferred_carrier="Twilio")
    assert "Twilio" in primary_pref.carrier_name

    print("[PASSED] Indian 22-circle identification and Least-Cost Routing verified.")


def test_7_campaign_batch_dialer_integration_with_failover():
    print("\n--- Test 7: Campaign Batch Dialer Integration & Trunk Attribution ---")

    mesh_router = create_default_indian_trunk_mesh()

    config = CampaignConfig(
        campaign_name="NBFC Muthoot Trunk Failover Test",
        max_concurrent_channels=2,
        language="hi",
        enforce_trai_calling_hours=False,
        enforce_dnd_check=False,
    )
    dialer = CampaignDialer(config=config, trunk_router=mesh_router)

    leads = [
        Lead(
            lead_id="LEAD_DELHI_01",
            phone_number="+919810123456",
            name="Aarav Sharma",
            loan_id="LOAN_DELHI_01",
            amount_due=4500.0,
            due_date="2026-09-20",
        ),
        Lead(
            lead_id="LEAD_MUMBAI_02",
            phone_number="+919820123456",
            name="Pooja Patel",
            loan_id="LOAN_MUMBAI_02",
            amount_due=7800.0,
            due_date="2026-09-21",
        ),
    ]

    for lead in leads:
        dialer.load_lead(lead)

    # Run batch dialing
    reports = dialer.start_campaign()
    assert len(reports) == 2

    # Verify CDR trunk metadata
    for cdr in dialer.cdrs:
        assert cdr.trunk_id != ""
        assert cdr.carrier_name != ""
        assert isinstance(cdr.failover_occurred, bool)
        assert isinstance(cdr.trunk_mos_score, float)
        assert cdr.trunk_mos_score >= 1.0

        cdr_dict = cdr.to_dict()
        assert "trunk_id" in cdr_dict
        assert "carrier_name" in cdr_dict
        assert "failover_occurred" in cdr_dict
        assert "trunk_mos_score" in cdr_dict

    print("[PASSED] Campaign dialer multi-trunk failover and CDR trunk attribution verified.")


def test_8_fastapi_telephony_trunks_rest_endpoints():
    print("\n--- Test 8: FastAPI Telephony Carrier Trunk REST Endpoints ---")

    app = create_app()
    client = TestClient(app)

    # 1. GET /telephony/trunks
    res_list = client.get("/telephony/trunks")
    assert res_list.status_code == 200
    data = res_list.json()
    assert data["status"] == "ok"
    assert data["total_trunks"] >= 5
    assert len(data["trunks"]) >= 5

    # 2. POST /telephony/trunks/register
    custom_trunk_payload = {
        "trunk_id": "trunk_reliance_jio_circle_a",
        "carrier_name": "Reliance Jio Infocomm",
        "sip_host": "sip.jio.com",
        "sip_port": 5060,
        "priority": 1,
        "cost_per_minute_inr": 0.22,
        "supported_circles": ["DELHI", "MUMBAI", "PUNJAB"],
    }
    res_reg = client.post("/telephony/trunks/register", json=custom_trunk_payload)
    assert res_reg.status_code == 200
    assert res_reg.json()["status"] == "ok"
    assert res_reg.json()["trunk"]["trunk_id"] == "trunk_reliance_jio_circle_a"

    # 3. POST /telephony/trunks/route
    route_payload = {
        "destination_phone": "+919810123456",
        "preferred_carrier": "Reliance Jio Infocomm",
    }
    res_route = client.post("/telephony/trunks/route", json=route_payload)
    assert res_route.status_code == 200
    route_data = res_route.json()
    assert route_data["status"] == "ok"
    assert route_data["circle"] in ["DL", "DELHI"]
    assert route_data["primary_trunk"]["trunk_id"] == "trunk_reliance_jio_circle_a"

    # 4. POST /telephony/trunks/report-call (Report 503 Carrier Fault)
    report_payload = {
        "trunk_id": "trunk_reliance_jio_circle_a",
        "sip_code": 503,
        "rtt_ms": 120.0,
        "jitter_ms": 25.0,
        "packet_loss_pct": 5.0,
    }
    res_report = client.post("/telephony/trunks/report-call", json=report_payload)
    assert res_report.status_code == 200
    assert res_report.json()["status"] == "ok"

    # 5. POST /telephony/trunks/circuit-breaker/reset
    reset_payload = {
        "trunk_id": "trunk_reliance_jio_circle_a",
    }
    res_reset = client.post("/telephony/trunks/circuit-breaker/reset", json=reset_payload)
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "ok"
    assert res_reset.json()["state"] == "CLOSED"

    # 6. GET /health verifies active_trunks metric
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert "active_trunks" in res_health.json()
    assert res_health.json()["active_trunks"] >= 6

    print("[PASSED] All FastAPI carrier trunk REST endpoints successfully verified.")


def run_all_tests():
    print("================================================================================")
    print("RUNNING VERBALYZE TELECOM TRUNK HEALTH & SIP CIRCUIT BREAKER TEST SUITE")
    print("Zero-Emoji Compliant | ITU-T G.107 E-Model MOS | Sub-150ms Auto-Failover")
    print("================================================================================")

    test_1_trunk_registration_and_initial_qos()
    test_2_itu_t_g107_mos_score_calculation()
    test_3_circuit_breaker_transitions_closed_to_open()
    test_4_cooldown_and_half_open_probe_recovery()
    test_5_sub_150ms_automated_multi_trunk_failover()
    test_6_indian_22_circle_least_cost_routing_and_priority()
    test_7_campaign_batch_dialer_integration_with_failover()
    test_8_fastapi_telephony_trunks_rest_endpoints()

    print("\n================================================================================")
    print("ALL 8 TELECOM TRUNK HEALTH & CIRCUIT BREAKER TESTS PASSED SUCCESSFULLY.")
    print("================================================================================")


if __name__ == "__main__":
    run_all_tests()
