"""
verbalyze/campaign/trai_compliance.py

Telecom Regulatory Authority of India (TRAI) & RBI Compliance Engine:
- Enforces lawful calling windows: 09:00:00 to 19:00:00 IST (UTC+05:30).
- National Customer Preference Register (NCPR / DND registry) filtering.
- Frequency capping: Maximum 3 call attempts per borrower per calendar day.
"""

from datetime import datetime, timezone, timedelta, time as dtime
from typing import Set, Optional, Tuple
from verbalyze.campaign.models import Lead, LeadStatus

# Indian Standard Time (IST) is UTC+05:30
IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
TRAI_CALLING_START = dtime(9, 0, 0)   # 09:00 AM IST
TRAI_CALLING_END = dtime(19, 0, 0)    # 07:00 PM IST


def get_current_ist_time(dt: Optional[datetime] = None) -> datetime:
    """Returns current datetime localized to Indian Standard Time (IST)."""
    if dt is None:
        return datetime.now(timezone.utc).astimezone(IST_TIMEZONE)
    if dt.tzinfo is None:
        # Assume UTC if naive, then convert to IST
        return dt.replace(tzinfo=timezone.utc).astimezone(IST_TIMEZONE)
    return dt.astimezone(IST_TIMEZONE)


class TRAIComplianceEngine:
    """
    Evaluates regulatory eligibility of leads against TRAI tele-calling guidelines
    and the RBI Fair Practices Code for automated collections.
    """

    def __init__(self, dnd_numbers: Optional[Set[str]] = None):
        """
        Initializes the compliance engine with optional in-memory DND registry.
        Numbers are normalized to the last 10 digits for lookup.
        """
        self._dnd_registry: Set[str] = set()
        if dnd_numbers:
            for num in dnd_numbers:
                self.add_dnd_number(num)

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Extracts the last 10 digits of an Indian telephone number."""
        digits = "".join(c for c in phone if c.isdigit())
        return digits[-10:] if len(digits) >= 10 else digits

    def add_dnd_number(self, phone: str) -> None:
        """Adds a telephone number to the National Customer Preference Register."""
        normalized = self._normalize_phone(phone)
        if normalized:
            self._dnd_registry.add(normalized)

    def remove_dnd_number(self, phone: str) -> None:
        """Removes a telephone number from the DND registry."""
        normalized = self._normalize_phone(phone)
        self._dnd_registry.discard(normalized)

    def is_dnd_registered(self, phone: str) -> bool:
        """Checks whether a number is listed in the DND registry."""
        normalized = self._normalize_phone(phone)
        return normalized in self._dnd_registry

    def is_within_calling_window(self, dt: Optional[datetime] = None) -> bool:
        """
        Checks whether the specified time (or current time if omitted)
        falls between 09:00:00 and 19:00:00 Indian Standard Time.
        """
        ist_now = get_current_ist_time(dt)
        current_time = ist_now.time()
        return TRAI_CALLING_START <= current_time < TRAI_CALLING_END

    def check_daily_frequency(
        self,
        lead: Lead,
        max_per_day: int = 3,
        now: Optional[datetime] = None
    ) -> bool:
        """
        Ensures the borrower has not been called more than max_per_day times
        in the current calendar day (IST).
        """
        ist_now = get_current_ist_time(now)
        today_date = ist_now.date()

        attempts_today = 0
        for entry in lead.call_history:
            attempt_time_raw = entry.get("timestamp")
            if isinstance(attempt_time_raw, datetime):
                attempt_ist = get_current_ist_time(attempt_time_raw)
            elif isinstance(attempt_time_raw, str):
                try:
                    dt = datetime.fromisoformat(attempt_time_raw)
                    attempt_ist = get_current_ist_time(dt)
                except Exception:
                    continue
            else:
                continue

            if attempt_ist.date() == today_date:
                attempts_today += 1

        # If call_history is empty but call_attempts is tracked and last_called_at is today
        if not lead.call_history and lead.last_called_at:
            last_called_ist = get_current_ist_time(lead.last_called_at)
            if last_called_ist.date() == today_date:
                attempts_today = lead.call_attempts

        return attempts_today < max_per_day

    def validate_lead_for_dialing(
        self,
        lead: Lead,
        enforce_hours: bool = True,
        enforce_dnd: bool = True,
        max_daily_calls: int = 3,
        now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Comprehensive TRAI validation pipeline:
        1. Calling hours window (09:00 - 19:00 IST)
        2. DND / NCPR registry verification
        3. Daily attempt frequency capping (max 3/day)

        Returns (is_approved, reason_message)
        """
        # 1. Calling Hours Check
        if enforce_hours and not self.is_within_calling_window(now):
            ist_time_str = get_current_ist_time(now).strftime("%H:%M:%S IST")
            return False, f"HOURS_RESTRICTED: Outside TRAI lawful window (09:00-19:00 IST). Current: {ist_time_str}"

        # 2. DND Registry Check
        if enforce_dnd and self.is_dnd_registered(lead.phone_number):
            return False, "DND_BLOCKED: Registered in National Customer Preference Register (NCPR/DND)"

        # 3. Daily Frequency Capping
        if not self.check_daily_frequency(lead, max_per_day=max_daily_calls, now=now):
            return False, f"FREQUENCY_EXCEEDED: Maximum daily call attempts reached ({max_daily_calls} calls/day limit)"

        return True, "APPROVED: Compliance checks passed"
