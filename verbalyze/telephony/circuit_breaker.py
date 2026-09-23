"""
verbalyze/telephony/circuit_breaker.py

Dynamic SIP Circuit Breaker & Real-Time QoS Telemetry Engine:
1. SIP Circuit Breaker Pattern (CLOSED, OPEN, HALF_OPEN):
   - Categorizes SIP response codes: SUCCESS (2xx, 18x), CLIENT_TERMINAL (486, 603, 404), CARRIER_FAULT (503, 500, 502, 504, 408, 480).
   - Prevents cascading call failures during carrier outages by tripping OPEN on threshold breach.
   - Automatically probes carrier recovery via HALF_OPEN state after cooldown period.
2. Real-Time Telecom QoS Tracking & ITU-T E-Model G.107:
   - Tracks Round-Trip Time (RTT / latency), jitter, and packet loss percentage.
   - Computes ITU-T G.107 Transmission Rating Factor (R) and Mean Opinion Score (MOS, 1.0 to 4.5).
3. Zero-emoji compliant.
"""

import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Deque
from collections import deque


class CircuitBreakerState(str, Enum):
    """Lifecycle states of the SIP Circuit Breaker."""
    CLOSED = "CLOSED"        # Normal operation: all calls routed to this trunk
    OPEN = "OPEN"            # Tripped: carrier trunk down, calls immediately bypassed/failed over
    HALF_OPEN = "HALF_OPEN"  # Cooldown expired: trial probe calls permitted to test recovery


class TrunkHealth(str, Enum):
    """Composite health classification based on QoS and breaker state."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class SIPResponseCategory(str, Enum):
    """Categorization of SIP telephony response status codes."""
    SUCCESS = "SUCCESS"                  # 200 OK, 180 Ringing, 183 Session Progress
    CLIENT_TERMINAL = "CLIENT_TERMINAL"  # 486 Busy, 603 Decline, 404 Not Found (normal call outcomes)
    CLIENT_OUTCOME = "CLIENT_TERMINAL"   # Alias for CLIENT_TERMINAL
    CARRIER_FAULT = "CARRIER_FAULT"      # 503, 500, 502, 504, 408, 480 (carrier trunk outage)


def categorize_sip_code(code: int) -> SIPResponseCategory:
    """
    Categorizes a SIP status code according to carrier reliability standards.
    Non-carrier client events (e.g. borrower busy, line declined) do not count as carrier failures.
    """
    if code in (200, 202, 180, 183):
        return SIPResponseCategory.SUCCESS
    elif code in (486, 603, 404, 487, 484, 410, 488):
        # 486: Busy Here, 603: Decline, 404: Not Found, 487: Request Terminated
        return SIPResponseCategory.CLIENT_TERMINAL
    elif code in (500, 502, 503, 504, 408, 480):
        # 503: Service Unavailable, 408: Request Timeout, 480: Temporarily Unavailable
        return SIPResponseCategory.CARRIER_FAULT
    elif 500 <= code <= 599:
        return SIPResponseCategory.CARRIER_FAULT
    elif 200 <= code <= 299:
        return SIPResponseCategory.SUCCESS
    else:
        return SIPResponseCategory.CLIENT_TERMINAL


def calculate_itu_g107_mos(
    latency_ms: float = 25.0,
    jitter_ms: float = 5.0,
    packet_loss_pct: float = 0.0,
    one_way_delay_ms: Optional[float] = None,
) -> float:
    """
    Calculates Mean Opinion Score (MOS) based on ITU-T G.107 E-Model:
    1. Base transmission rating: R0 = 93.2
    2. Delay impairment Id calculated from one-way delay + jitter buffer delay.
    3. Equipment & packet loss impairment Ie-eff calculated from loss percentage.
    4. R-factor mapped to MOS on standard 1.0 to 4.5 scale.
    """
    if one_way_delay_ms is not None:
        one_way_delay = max(0.5, float(one_way_delay_ms)) + (2.0 * max(0.0, float(jitter_ms)))
    else:
        rtt = max(1.0, float(latency_ms))
        jitter = max(0.0, float(jitter_ms))
        one_way_delay = (rtt / 2.0) + (2.0 * jitter)

    loss = max(0.0, min(100.0, float(packet_loss_pct)))

    # Delay impairment (Id)
    if one_way_delay > 100.0:
        delay_penalty = 0.024 * one_way_delay
        if one_way_delay > 177.3:
            delay_penalty += 0.11 * (one_way_delay - 177.3)
    else:
        delay_penalty = 0.0

    # Equipment & packet loss impairment (Ie-eff) for standard G.711 / PCM
    # At 0% loss, Ie = 0; loss increases impairment non-linearly
    if loss > 0.0:
        loss_impairment = 95.0 * (loss / (loss + 4.3))
    else:
        loss_impairment = 0.0

    # Overall Transmission Rating Factor (R)
    r_factor = 93.2 - delay_penalty - loss_impairment
    r_factor = max(0.0, min(100.0, r_factor))

    # Convert R to MOS (ITU-T G.107 standard conversion formula)
    if r_factor <= 0.0:
        mos = 1.0
    elif r_factor >= 100.0:
        mos = 4.5
    else:
        mos = 1.0 + (0.035 * r_factor) + (r_factor * (r_factor - 60.0) * (100.0 - r_factor) * 7.0e-6)

    return round(max(1.0, min(4.5, mos)), 2)


@dataclass
class SIPCircuitBreakerConfig:
    """Configuration parameters for SIP Circuit Breaker."""
    failure_threshold_rate: float = 0.5        # 50% failure rate in window trips breaker
    min_requests_to_evaluate: int = 4          # Minimum requests in window before rate check
    consecutive_failure_threshold: int = 3     # 3 consecutive carrier faults immediately trip breaker
    cooldown_duration_sec: float = 15.0        # Duration in OPEN state before transitioning to HALF_OPEN
    half_open_success_threshold: int = 2       # Consecutive successful probes in HALF_OPEN to reset CLOSED
    window_size: int = 20                      # Number of recent calls in sliding window

    def __init__(
        self,
        failure_threshold_rate: float = 0.5,
        min_requests_to_evaluate: int = 4,
        consecutive_failure_threshold: int = 3,
        cooldown_duration_sec: float = 15.0,
        half_open_success_threshold: int = 2,
        window_size: int = 20,
        **kwargs,
    ):
        self.failure_threshold_rate = kwargs.get("failure_rate_threshold", failure_threshold_rate)
        self.min_requests_to_evaluate = kwargs.get("min_requests", min_requests_to_evaluate)
        self.consecutive_failure_threshold = kwargs.get("consecutive_carrier_faults_threshold", consecutive_failure_threshold)
        self.cooldown_duration_sec = cooldown_duration_sec
        self.half_open_success_threshold = half_open_success_threshold
        self.window_size = window_size


@dataclass
class TrunkQoS:
    """Container for real-time telecom QoS telemetry."""
    latency_ms: float = 25.0
    jitter_ms: float = 5.0
    packet_loss_pct: float = 0.0
    mos_score: float = 4.41
    total_calls: int = 0
    successful_calls: int = 0
    carrier_fault_calls: int = 0

    @property
    def avg_rtt_ms(self) -> float:
        return self.latency_ms

    def update(self, latency_ms: float, jitter_ms: float, packet_loss_pct: float, is_carrier_fault: bool = False):
        """Updates QoS metrics and recalculates ITU-T G.107 MOS score."""
        self.total_calls += 1
        if is_carrier_fault:
            self.carrier_fault_calls += 1
        else:
            self.successful_calls += 1

        # Exponential moving average (alpha = 0.3) for smooth telemetry
        alpha = 0.3
        self.latency_ms = round((alpha * latency_ms) + ((1.0 - alpha) * self.latency_ms), 1)
        self.jitter_ms = round((alpha * jitter_ms) + ((1.0 - alpha) * self.jitter_ms), 1)
        self.packet_loss_pct = round((alpha * packet_loss_pct) + ((1.0 - alpha) * self.packet_loss_pct), 2)
        self.mos_score = calculate_itu_g107_mos(self.latency_ms, self.jitter_ms, self.packet_loss_pct)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "jitter_ms": self.jitter_ms,
            "packet_loss_pct": self.packet_loss_pct,
            "mos_score": self.mos_score,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "carrier_fault_calls": self.carrier_fault_calls,
        }


class SIPCircuitBreaker:
    """
    High-Performance State Machine governing carrier trunk availability.
    Protects outbound campaigns from stalling on degraded carrier trunks.
    """

    def __init__(self, trunk_id: str, config: Optional[SIPCircuitBreakerConfig] = None):
        self.trunk_id = trunk_id
        self.config = config or SIPCircuitBreakerConfig()

        self._state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._history: Deque[bool] = deque(maxlen=self.config.window_size)  # True = success/client, False = carrier fault
        self._consecutive_failures: int = 0
        self._half_open_successes: int = 0
        self._state_change_time: float = time.time()
        self._last_failure_reason: str = ""
        self._total_requests: int = 0
        self._total_carrier_faults: int = 0

    @property
    def consecutive_carrier_faults(self) -> int:
        return self._consecutive_failures

    @property
    def half_open_successes(self) -> int:
        return self._half_open_successes

    @property
    def state(self) -> CircuitBreakerState:
        """Returns the current state, dynamically transitioning from OPEN to HALF_OPEN if cooldown has elapsed."""
        if self._state == CircuitBreakerState.OPEN:
            elapsed = time.time() - self._state_change_time
            if elapsed >= self.config.cooldown_duration_sec:
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_successes = 0
                self._state_change_time = time.time()
        return self._state

    def can_execute(self) -> bool:
        """Returns True if trunk is permitted to accept calls."""
        curr_state = self.state
        if curr_state == CircuitBreakerState.CLOSED:
            return True
        elif curr_state == CircuitBreakerState.HALF_OPEN:
            # Allow trial probe calls
            return True
        return False

    def can_route_call(self) -> bool:
        """Alias for can_execute()."""
        return self.can_execute()

    def record_call(
        self,
        sip_code: int,
        rtt_ms: float = 0.0,
        jitter_ms: float = 0.0,
        packet_loss_pct: float = 0.0,
    ) -> CircuitBreakerState:
        """Alias for record_result()."""
        return self.record_result(sip_code, rtt_ms, jitter_ms, packet_loss_pct)

    def trip_to_open(self, reason: str = "Manual trip"):
        """Alias for trip_open()."""
        self.trip_open(reason)

    def record_result(
        self,
        sip_code: int,
        rtt_ms: float = 0.0,
        jitter_ms: float = 0.0,
        packet_loss_pct: float = 0.0,
    ) -> CircuitBreakerState:
        """
        Records the outcome of a SIP call on this trunk and evaluates circuit breaker state transitions.
        """
        self._total_requests += 1
        category = categorize_sip_code(sip_code)
        curr_state = self.state

        if category == SIPResponseCategory.CARRIER_FAULT:
            self._total_carrier_faults += 1
            self._consecutive_failures += 1
            self._history.append(False)
            self._last_failure_reason = f"SIP {sip_code} carrier fault"

            if curr_state == CircuitBreakerState.HALF_OPEN:
                # Probe failed: immediate return to OPEN with fresh cooldown
                self.trip_open(f"Trial probe failed with SIP {sip_code}")
            elif curr_state == CircuitBreakerState.CLOSED:
                # Check consecutive failures threshold
                if self._consecutive_failures >= self.config.consecutive_failure_threshold:
                    self.trip_open(f"{self._consecutive_failures} consecutive carrier faults (SIP {sip_code})")
                # Check sliding window failure rate
                elif len(self._history) >= self.config.min_requests_to_evaluate:
                    fault_count = sum(1 for is_ok in self._history if not is_ok)
                    fault_rate = fault_count / len(self._history)
                    if fault_rate >= self.config.failure_threshold_rate:
                        self.trip_open(f"Failure rate {fault_rate:.1%} breached threshold {self.config.failure_threshold_rate:.1%}")

        else:
            # SUCCESS or CLIENT_TERMINAL (both indicate healthy trunk signaling)
            self._consecutive_failures = 0
            self._history.append(True)

            if curr_state == CircuitBreakerState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.half_open_success_threshold:
                    # Successful probe trials completed: restore CLOSED state
                    self.reset()

        return self.state

    def trip_open(self, reason: str = "Manual trip"):
        """Forces the circuit breaker into the OPEN state."""
        self._state = CircuitBreakerState.OPEN
        self._state_change_time = time.time()
        self._last_failure_reason = reason

    def reset(self):
        """Resets the circuit breaker back to the normal CLOSED state."""
        self._state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self._half_open_successes = 0
        self._state_change_time = time.time()
        self._last_failure_reason = ""

    def get_health(self, qos: Optional[TrunkQoS] = None) -> TrunkHealth:
        """Determines health classification based on breaker state and QoS metrics."""
        curr_state = self.state
        if curr_state == CircuitBreakerState.OPEN:
            return TrunkHealth.UNAVAILABLE
        elif curr_state == CircuitBreakerState.HALF_OPEN:
            return TrunkHealth.DEGRADED

        if qos:
            if qos.mos_score < 3.6 or qos.packet_loss_pct > 5.0 or qos.latency_ms > 250.0:
                return TrunkHealth.DEGRADED

        return TrunkHealth.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Returns serialized status dictionary."""
        curr_state = self.state
        recent_fault_count = sum(1 for ok in self._history if not ok) if self._history else 0
        recent_rate = (recent_fault_count / len(self._history)) if self._history else 0.0

        return {
            "trunk_id": self.trunk_id,
            "state": curr_state.value,
            "can_execute": self.can_execute(),
            "consecutive_failures": self._consecutive_failures,
            "half_open_successes": self._half_open_successes,
            "recent_failure_rate": round(recent_rate, 2),
            "total_requests": self._total_requests,
            "total_carrier_faults": self._total_carrier_faults,
            "last_failure_reason": self._last_failure_reason,
            "cooldown_remaining_sec": max(
                0.0,
                round(self.config.cooldown_duration_sec - (time.time() - self._state_change_time), 1)
            ) if curr_state == CircuitBreakerState.OPEN else 0.0,
        }
