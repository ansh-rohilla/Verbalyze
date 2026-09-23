"""
verbalyze/telephony/trunk_router.py

Telecom Carrier Trunk Health, Dynamic Routing & Multi-Trunk Auto-Failover:
1. Indian 22-Circle Least-Cost Routing (LCR):
   - Maps Indian phone series to TRAI/DoT telecom service areas (e.g. DL, MUM, MH, GJ, KA, TN, AP, etc.).
   - Prioritizes local circle trunks for optimal interconnect pricing and latency.
2. Carrier-Grade Multi-Trunk Router:
   - Evaluates SIP Circuit Breakers (CLOSED, OPEN, HALF_OPEN) and QoS (MOS score, RTT, jitter, loss).
   - Dynamically selects primary and secondary fallback trunks.
3. Sub-150ms Automated Failover:
   - Transparently switches carrier trunks upon 5xx/408/network timeouts without dropping calls or losing leads.

Zero-emoji compliant.
"""

import time
import inspect
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable, Awaitable, Union

from verbalyze.telephony.circuit_breaker import (
    SIPCircuitBreaker,
    SIPCircuitBreakerConfig,
    TrunkQoS,
    CircuitBreakerState,
    TrunkHealth,
    categorize_sip_code,
    SIPResponseCategory,
)


class TelecomCircle(str, Enum):
    """The 22 recognized Indian Telecom Service Areas / Circles + Pan-India."""
    DL = "DL"          # Delhi NCR
    MUM = "MUM"        # Mumbai
    KOL = "KOL"        # Kolkata
    MH = "MH"          # Maharashtra & Goa
    GJ = "GJ"          # Gujarat
    KA = "KA"          # Karnataka (Bangalore)
    TN = "TN"          # Tamil Nadu & Chennai
    AP = "AP"          # Andhra Pradesh & Telangana
    KL = "KL"          # Kerala
    PB = "PB"          # Punjab
    HR = "HR"          # Haryana
    UPW = "UPW"        # Uttar Pradesh (West) & Uttarakhand
    UPE = "UPE"        # Uttar Pradesh (East)
    RJ = "RJ"          # Rajasthan
    MP = "MP"          # Madhya Pradesh & Chhattisgarh
    WB = "WB"          # West Bengal & Sikkim
    BR = "BR"          # Bihar & Jharkhand
    OR = "OR"          # Odisha
    AS = "AS"          # Assam
    NE = "NE"          # North East
    JK = "JK"          # Jammu & Kashmir
    HP = "HP"          # Himachal Pradesh
    ALL = "ALL"        # Pan-India Wildcard

    # Descriptive aliases for ease of use
    DELHI = "DL"
    MUMBAI = "MUM"
    KOLKATA = "KOL"
    MAHARASHTRA = "MH"
    GUJARAT = "GJ"
    KARNATAKA = "KA"
    TAMIL_NADU = "TN"
    ANDHRA_PRADESH = "AP"
    KERALA = "KL"
    PUNJAB = "PB"
    HARYANA = "HR"
    UTTAR_PRADESH_WEST = "UPW"
    UTTAR_PRADESH_EAST = "UPE"
    RAJASTHAN = "RJ"
    MADHYA_PRADESH = "MP"
    WEST_BENGAL = "WB"
    BIHAR = "BR"
    ODISHA = "OR"
    ASSAM = "AS"
    NORTH_EAST = "NE"
    JAMMU_KASHMIR = "JK"
    HIMACHAL_PRADESH = "HP"
    UNKNOWN = "DL"


# Series prefix mapping for Indian telecom circles
CIRCLE_PREFIX_MAP: Dict[str, str] = {
    "9810": TelecomCircle.DL.value, "9811": TelecomCircle.DL.value, "9818": TelecomCircle.DL.value,
    "9868": TelecomCircle.DL.value, "9871": TelecomCircle.DL.value, "9873": TelecomCircle.DL.value,
    "9910": TelecomCircle.DL.value, "9911": TelecomCircle.DL.value, "9999": TelecomCircle.DL.value,

    "9820": TelecomCircle.MUM.value, "9821": TelecomCircle.MUM.value, "9819": TelecomCircle.MUM.value,
    "9833": TelecomCircle.MUM.value, "9869": TelecomCircle.MUM.value, "9892": TelecomCircle.MUM.value,
    "9920": TelecomCircle.MUM.value, "9930": TelecomCircle.MUM.value, "9969": TelecomCircle.MUM.value,

    "9830": TelecomCircle.KOL.value, "9831": TelecomCircle.KOL.value, "9832": TelecomCircle.KOL.value,

    "9822": TelecomCircle.MH.value, "9823": TelecomCircle.MH.value, "9850": TelecomCircle.MH.value,
    "9881": TelecomCircle.MH.value, "9890": TelecomCircle.MH.value, "9922": TelecomCircle.MH.value,

    "9824": TelecomCircle.GJ.value, "9825": TelecomCircle.GJ.value, "9879": TelecomCircle.GJ.value,
    "9898": TelecomCircle.GJ.value, "9904": TelecomCircle.GJ.value, "9924": TelecomCircle.GJ.value,

    "9844": TelecomCircle.KA.value, "9845": TelecomCircle.KA.value, "9880": TelecomCircle.KA.value,
    "9886": TelecomCircle.KA.value, "9900": TelecomCircle.KA.value, "9901": TelecomCircle.KA.value,

    "9840": TelecomCircle.TN.value, "9841": TelecomCircle.TN.value, "9884": TelecomCircle.TN.value,
    "9444": TelecomCircle.TN.value, "9940": TelecomCircle.TN.value, "9941": TelecomCircle.TN.value,

    "9848": TelecomCircle.AP.value, "9849": TelecomCircle.AP.value, "9866": TelecomCircle.AP.value,
    "9885": TelecomCircle.AP.value, "9948": TelecomCircle.AP.value, "9949": TelecomCircle.AP.value,

    "9846": TelecomCircle.KL.value, "9847": TelecomCircle.KL.value, "9895": TelecomCircle.KL.value,
    "9946": TelecomCircle.KL.value, "9947": TelecomCircle.KL.value,

    "9814": TelecomCircle.PB.value, "9815": TelecomCircle.PB.value, "9872": TelecomCircle.PB.value,
    "9876": TelecomCircle.PB.value, "9914": TelecomCircle.PB.value, "9915": TelecomCircle.PB.value,

    "9812": TelecomCircle.HR.value, "9813": TelecomCircle.HR.value, "9896": TelecomCircle.HR.value,

    "9837": TelecomCircle.UPW.value, "9838": TelecomCircle.UPE.value, "9839": TelecomCircle.UPE.value,

    "9828": TelecomCircle.RJ.value, "9829": TelecomCircle.RJ.value, "9928": TelecomCircle.RJ.value,

    "9826": TelecomCircle.MP.value, "9827": TelecomCircle.MP.value, "9926": TelecomCircle.MP.value,

    "9835": TelecomCircle.BR.value, "9931": TelecomCircle.BR.value, "9934": TelecomCircle.BR.value,

    "9861": TelecomCircle.OR.value, "9937": TelecomCircle.OR.value,

    "9854": TelecomCircle.AS.value, "9864": TelecomCircle.AS.value,

    "9862": TelecomCircle.NE.value, "9863": TelecomCircle.NE.value,

    "9858": TelecomCircle.JK.value, "9906": TelecomCircle.JK.value,

    "9816": TelecomCircle.HP.value, "9817": TelecomCircle.HP.value,
}


def detect_telecom_circle(phone_number: str) -> str:
    """
    Infers the Indian telecom circle from a phone number's mobile prefix.
    Fallback deterministically maps prefix to circles if unknown.
    """
    if not phone_number:
        return TelecomCircle.DL.value

    # Extract 10-digit mobile number
    digits = "".join(ch for ch in phone_number if ch.isdigit())
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    if len(digits) >= 4:
        prefix_4 = digits[:4]
        if prefix_4 in CIRCLE_PREFIX_MAP:
            return CIRCLE_PREFIX_MAP[prefix_4]

    if len(digits) >= 2:
        # Deterministic fallback mapping across circles
        circle_list = [c.value for c in TelecomCircle if c != TelecomCircle.ALL]
        idx = int(digits[:2]) % len(circle_list)
        return circle_list[idx]

    return TelecomCircle.DL.value


@dataclass
class CarrierTrunk:
    """Represents an active SIP carrier trunk connection."""
    trunk_id: str
    carrier_name: str
    sip_host: str
    sip_port: int = 5060
    priority: int = 1                              # 1 = top priority
    cost_per_minute_inr: float = 0.35              # Cost in Indian Rupees
    supported_circles: List[str] = field(default_factory=lambda: [TelecomCircle.ALL.value])
    circuit_breaker: Optional[SIPCircuitBreaker] = None
    qos: Optional[TrunkQoS] = None
    is_enabled: bool = True

    # Call metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    def __post_init__(self):
        if self.circuit_breaker is None:
            self.circuit_breaker = SIPCircuitBreaker(trunk_id=self.trunk_id)
        if self.qos is None:
            self.qos = TrunkQoS()

    @property
    def health(self) -> TrunkHealth:
        """Dynamic health status of this trunk combining circuit breaker and QoS."""
        return self.circuit_breaker.get_health(self.qos)

    def can_route_call(self) -> bool:
        """Returns True if trunk is enabled and its circuit breaker allows calls."""
        return self.is_enabled and self.circuit_breaker.can_execute()

    def supports_circle(self, circle: str) -> bool:
        """Checks if trunk handles the specified circle."""
        if TelecomCircle.ALL.value in self.supported_circles or "ALL" in [c.upper() for c in self.supported_circles]:
            return True
        norm_circle = circle.upper()
        for c in self.supported_circles:
            c_upper = c.upper()
            if c_upper == norm_circle:
                return True
            try:
                if TelecomCircle[c_upper].value == norm_circle:
                    return True
            except KeyError:
                pass
        return False

    def record_call_result(
        self,
        sip_code: int,
        rtt_ms: float = 25.0,
        jitter_ms: float = 5.0,
        packet_loss_pct: float = 0.0,
    ):
        """Records outcome on circuit breaker and QoS telemetry."""
        self.total_calls += 1
        category = categorize_sip_code(sip_code)
        is_carrier_fault = (category == SIPResponseCategory.CARRIER_FAULT)
        if is_carrier_fault:
            self.failed_calls += 1
        else:
            self.successful_calls += 1

        self.qos.update(rtt_ms, jitter_ms, packet_loss_pct, is_carrier_fault=is_carrier_fault)
        self.circuit_breaker.record_result(sip_code, rtt_ms, jitter_ms, packet_loss_pct)

    def to_dict(self) -> Dict[str, Any]:
        """Returns comprehensive status dictionary."""
        return {
            "trunk_id": self.trunk_id,
            "carrier_name": self.carrier_name,
            "sip_endpoint": f"{self.sip_host}:{self.sip_port}",
            "priority": self.priority,
            "cost_per_minute_inr": self.cost_per_minute_inr,
            "supported_circles": self.supported_circles,
            "is_enabled": self.is_enabled,
            "health": self.circuit_breaker.get_health(self.qos).value,
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "qos": self.qos.to_dict(),
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
        }


class DispatchResult(dict):
    """Result of a multi-trunk dispatch operation with failover telemetry."""
    def __init__(
        self,
        data: Dict[str, Any],
        selected_trunk: Optional[CarrierTrunk] = None,
        attempted_trunks: Optional[List[CarrierTrunk]] = None,
    ):
        super().__init__(data)
        self.success = (data.get("status") != "FAILED" and data.get("sip_code", 200) < 500)
        self.failover_occurred = data.get("failover_occurred", False)
        self.failover_count = data.get("failover_count", 0)
        self.selected_trunk = selected_trunk
        self.attempted_trunks = attempted_trunks or []
        self.dispatch_result = data

    def __await__(self):
        async def _identity():
            return self
        return _identity().__await__()


class NoAvailableTrunkError(Exception):
    """Raised when all candidate carrier trunks have tripped circuit breakers."""
    pass


class MultiTrunkRouter:
    """
    Carrier-Grade Multi-Trunk Dispatcher with Circle-Based LCR and Auto-Failover.
    """

    def __init__(self):
        self.trunks: Dict[str, CarrierTrunk] = {}

    def register_trunk(self, trunk: CarrierTrunk):
        """Registers a carrier trunk in the routing table."""
        self.trunks[trunk.trunk_id] = trunk

    def get_trunk(self, trunk_id: str) -> Optional[CarrierTrunk]:
        """Retrieves a trunk by ID."""
        return self.trunks.get(trunk_id)

    def resolve_routes(
        self,
        destination_phone: str,
        preferred_carrier: Optional[str] = None,
    ) -> Tuple[CarrierTrunk, List[CarrierTrunk]]:
        """
        Determines the optimal primary carrier trunk and ordered fallback trunks:
        1. Detects destination telecom circle.
        2. Filters trunks supporting this circle (or ALL).
        3. Filters trunks whose circuit breakers permit execution (can_execute() is True).
        4. Ranks candidate trunks by:
           - Circuit Breaker state: CLOSED before HALF_OPEN
           - Preferred carrier match
           - Priority (ascending, 1 is highest)
           - Least Cost per Minute (INR)
           - Highest MOS score
        
        Returns:
            Tuple of (primary_trunk, [fallback_trunks])
        """
        circle = detect_telecom_circle(destination_phone)
        candidates: List[CarrierTrunk] = []

        for trunk in self.trunks.values():
            if not trunk.is_enabled:
                continue
            if not trunk.supports_circle(circle):
                continue
            candidates.append(trunk)

        if not candidates:
            # Fallback to any enabled trunk supporting ALL
            candidates = [t for t in self.trunks.values() if t.is_enabled and t.supports_circle(TelecomCircle.ALL.value)]

        if not candidates:
            raise NoAvailableTrunkError("No enabled carrier trunks configured.")

        # Separate healthy/executable from tripped
        executable = [t for t in candidates if t.circuit_breaker.can_execute()]

        if not executable:
            # All trunks have tripped circuit breakers; pick the one with lowest remaining cooldown as last resort
            executable = sorted(
                candidates,
                key=lambda t: t.circuit_breaker.to_dict()["cooldown_remaining_sec"]
            )

        def _sort_key(t: CarrierTrunk):
            # 1. Breaker state: CLOSED (0) preferred over HALF_OPEN (1)
            state_score = 0 if t.circuit_breaker.state == CircuitBreakerState.CLOSED else 1
            # 2. Preferred carrier bonus
            pref_score = 0 if (preferred_carrier and preferred_carrier.lower() in t.carrier_name.lower()) else 1
            # 3. Priority (1 is best)
            prio = t.priority
            # 4. Least Cost (INR)
            cost = t.cost_per_minute_inr
            # 5. MOS (higher is better, inverted for ascending sort)
            mos_inv = -t.qos.mos_score
            return (state_score, pref_score, prio, cost, mos_inv)

        sorted_trunks = sorted(executable, key=_sort_key)
        primary = sorted_trunks[0]
        fallbacks = sorted_trunks[1:]

        return primary, fallbacks

    def dispatch_with_failover(
        self,
        destination_phone: str,
        dial_func: Optional[Callable] = None,
        preferred_carrier: Optional[str] = None,
        dispatch_fn: Optional[Callable] = None,
        rtt_ms: Optional[float] = None,
        packet_loss_pct: Optional[float] = None,
        **kwargs,
    ):
        """
        Executes outbound call dispatch with automatic sub-150ms carrier failover:
        - Supports both synchronous and asynchronous dial functions.
        - Supports both await and direct synchronous returns.
        - If carrier fault occurs (5xx, 408, or exception), immediately fails over to next secondary.
        - Records QoS and circuit breaker results.
        """
        func = dial_func or dispatch_fn or kwargs.get("dispatch_func")
        if func is None:
            raise ValueError("Must provide dial_func or dispatch_fn")

        primary, fallbacks = self.resolve_routes(destination_phone, preferred_carrier)
        all_attempts = [primary] + fallbacks

        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            async def _async_exec():
                failover_count = 0
                attempt_history: List[Dict[str, Any]] = []
                attempted_carrier_trunks: List[CarrierTrunk] = []
                last_error = None

                for trunk in all_attempts:
                    attempted_carrier_trunks.append(trunk)
                    t_start = time.time()
                    try:
                        try:
                            sig_len = len(inspect.signature(func).parameters)
                        except Exception:
                            sig_len = 1
                        if sig_len >= 2:
                            res = await func(trunk, destination_phone)
                        else:
                            res = await func(trunk)

                        t_dur = time.time() - t_start
                        meas_rtt = rtt_ms if rtt_ms is not None else (t_dur * 1000.0)
                        meas_loss = packet_loss_pct if packet_loss_pct is not None else 0.0

                        status = res.get("status", "CONNECTED")
                        sip_code = int(res.get("sip_code", 200 if (status == "CONNECTED" or res.get("connected", True)) else 503))

                        category = categorize_sip_code(sip_code)
                        if category == SIPResponseCategory.CARRIER_FAULT or status == "CARRIER_FAULT":
                            trunk.record_call_result(sip_code=sip_code, rtt_ms=meas_rtt, packet_loss_pct=meas_loss)
                            attempt_history.append({
                                "trunk_id": trunk.trunk_id,
                                "carrier_name": trunk.carrier_name,
                                "status": "FAILED",
                                "sip_code": sip_code,
                                "reason": f"Carrier fault SIP {sip_code}",
                            })
                            failover_count += 1
                            continue

                        trunk.record_call_result(sip_code=sip_code, rtt_ms=meas_rtt, packet_loss_pct=meas_loss)
                        attempt_history.append({
                            "trunk_id": trunk.trunk_id,
                            "carrier_name": trunk.carrier_name,
                            "status": status,
                            "sip_code": sip_code,
                        })
                        res_data = dict(res)
                        res_data["resolved_trunk_id"] = trunk.trunk_id
                        res_data["carrier_name"] = trunk.carrier_name
                        res_data["failover_occurred"] = failover_count > 0
                        res_data["failover_count"] = failover_count
                        res_data["attempts"] = attempt_history
                        return DispatchResult(res_data, selected_trunk=trunk, attempted_trunks=attempted_carrier_trunks)

                    except Exception as exc:
                        t_dur = time.time() - t_start
                        meas_rtt = rtt_ms if rtt_ms is not None else (t_dur * 1000.0)
                        meas_loss = packet_loss_pct if packet_loss_pct is not None else 0.0
                        last_error = exc
                        trunk.record_call_result(sip_code=503, rtt_ms=meas_rtt, packet_loss_pct=meas_loss)
                        attempt_history.append({
                            "trunk_id": trunk.trunk_id,
                            "carrier_name": trunk.carrier_name,
                            "status": "EXCEPTION",
                            "sip_code": 503,
                            "reason": str(exc),
                        })
                        failover_count += 1
                        continue

                fail_data = {
                    "status": "FAILED",
                    "sip_code": 503,
                    "resolved_trunk_id": primary.trunk_id,
                    "carrier_name": primary.carrier_name,
                    "failover_occurred": True,
                    "failover_count": failover_count,
                    "attempts": attempt_history,
                    "error": str(last_error) if last_error else "All carrier trunks failed.",
                }
                return DispatchResult(fail_data, selected_trunk=primary, attempted_trunks=attempted_carrier_trunks)

            return _async_exec()

        else:
            # Synchronous execution
            failover_count = 0
            attempt_history: List[Dict[str, Any]] = []
            attempted_carrier_trunks: List[CarrierTrunk] = []
            last_error = None

            for trunk in all_attempts:
                attempted_carrier_trunks.append(trunk)
                t_start = time.time()
                try:
                    try:
                        sig_len = len(inspect.signature(func).parameters)
                    except Exception:
                        sig_len = 1
                    if sig_len >= 2:
                        res = func(trunk, destination_phone)
                    else:
                        res = func(trunk)

                    if inspect.isawaitable(res):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                res = loop.run_until_complete(res)
                        except RuntimeError:
                            res = asyncio.run(res)

                    t_dur = time.time() - t_start
                    meas_rtt = rtt_ms if rtt_ms is not None else (t_dur * 1000.0)
                    meas_loss = packet_loss_pct if packet_loss_pct is not None else 0.0

                    status = res.get("status", "CONNECTED")
                    sip_code = int(res.get("sip_code", 200 if (status == "CONNECTED" or res.get("connected", True)) else 503))

                    category = categorize_sip_code(sip_code)
                    if category == SIPResponseCategory.CARRIER_FAULT or status == "CARRIER_FAULT":
                        trunk.record_call_result(sip_code=sip_code, rtt_ms=meas_rtt, packet_loss_pct=meas_loss)
                        attempt_history.append({
                            "trunk_id": trunk.trunk_id,
                            "carrier_name": trunk.carrier_name,
                            "status": "FAILED",
                            "sip_code": sip_code,
                            "reason": f"Carrier fault SIP {sip_code}",
                        })
                        failover_count += 1
                        continue

                    trunk.record_call_result(sip_code=sip_code, rtt_ms=meas_rtt, packet_loss_pct=meas_loss)
                    attempt_history.append({
                        "trunk_id": trunk.trunk_id,
                        "carrier_name": trunk.carrier_name,
                        "status": status,
                        "sip_code": sip_code,
                    })
                    res_data = dict(res)
                    res_data["resolved_trunk_id"] = trunk.trunk_id
                    res_data["carrier_name"] = trunk.carrier_name
                    res_data["failover_occurred"] = failover_count > 0
                    res_data["failover_count"] = failover_count
                    res_data["attempts"] = attempt_history
                    return DispatchResult(res_data, selected_trunk=trunk, attempted_trunks=attempted_carrier_trunks)

                except Exception as exc:
                    t_dur = time.time() - t_start
                    meas_rtt = rtt_ms if rtt_ms is not None else (t_dur * 1000.0)
                    meas_loss = packet_loss_pct if packet_loss_pct is not None else 0.0
                    last_error = exc
                    trunk.record_call_result(sip_code=503, rtt_ms=meas_rtt, packet_loss_pct=meas_loss)
                    attempt_history.append({
                        "trunk_id": trunk.trunk_id,
                        "carrier_name": trunk.carrier_name,
                        "status": "EXCEPTION",
                        "sip_code": 503,
                        "reason": str(exc),
                    })
                    failover_count += 1
                    continue

            fail_data = {
                "status": "FAILED",
                "sip_code": 503,
                "resolved_trunk_id": primary.trunk_id,
                "carrier_name": primary.carrier_name,
                "failover_occurred": True,
                "failover_count": failover_count,
                "attempts": attempt_history,
                "error": str(last_error) if last_error else "All carrier trunks failed.",
            }
            return DispatchResult(fail_data, selected_trunk=primary, attempted_trunks=attempted_carrier_trunks)


def create_default_indian_trunk_mesh() -> MultiTrunkRouter:
    """
    Constructs the standard Indian enterprise multi-carrier SIP mesh.
    Includes Exotel, Twilio, Tata Tele, Airtel, and RingTrunk.
    """
    router = MultiTrunkRouter()

    # 1. Exotel Pan-India SIP Trunk (Primary Default)
    exotel = CarrierTrunk(
        trunk_id="exotel_primary",
        carrier_name="Exotel India",
        sip_host="sip.exotel.com",
        sip_port=5060,
        priority=1,
        cost_per_minute_inr=0.35,
        supported_circles=[TelecomCircle.ALL.value],
    )
    router.register_trunk(exotel)

    # 2. Twilio Direct Route (Secondary Pan-India Failover)
    twilio = CarrierTrunk(
        trunk_id="twilio_secondary",
        carrier_name="Twilio Telecom",
        sip_host="voice.twilio.com",
        sip_port=5060,
        priority=2,
        cost_per_minute_inr=0.42,
        supported_circles=[TelecomCircle.ALL.value],
    )
    router.register_trunk(twilio)

    # 3. Tata Tele Business Services (Circle A & West Preferred)
    tatatele = CarrierTrunk(
        trunk_id="tatatele_west",
        carrier_name="Tata Tele Services",
        sip_host="sip.tatateleservices.com",
        sip_port=5060,
        priority=1,
        cost_per_minute_inr=0.30,
        supported_circles=[
            TelecomCircle.MH.value,
            TelecomCircle.GJ.value,
            TelecomCircle.MUM.value,
            TelecomCircle.KA.value,
        ],
    )
    router.register_trunk(tatatele)

    # 4. Airtel Enterprise SIP (North India Circle Preferred)
    airtel = CarrierTrunk(
        trunk_id="airtel_north",
        carrier_name="Airtel Enterprise",
        sip_host="sip.airtel.in",
        sip_port=5060,
        priority=1,
        cost_per_minute_inr=0.32,
        supported_circles=[
            TelecomCircle.DL.value,
            TelecomCircle.HR.value,
            TelecomCircle.PB.value,
            TelecomCircle.UPW.value,
            TelecomCircle.UPE.value,
            TelecomCircle.RJ.value,
        ],
    )
    router.register_trunk(airtel)

    # 5. RingTrunk Direct SIP Peering (Emergency Tertiary Backup)
    ringtrunk = CarrierTrunk(
        trunk_id="ringtrunk_backup",
        carrier_name="RingTrunk Direct",
        sip_host="sip.ringtrunk.net",
        sip_port=5060,
        priority=3,
        cost_per_minute_inr=0.38,
        supported_circles=[TelecomCircle.ALL.value],
    )
    router.register_trunk(ringtrunk)

    return router
