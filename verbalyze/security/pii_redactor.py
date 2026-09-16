"""
verbalyze/security/pii_redactor.py

End-to-End Security & Privacy Guard for Verbalyze:
1. PII Redaction Engine:
   - Indian Mobile Numbers (masks to +91*****3210)
   - Aadhaar Numbers (12 digits masked to ****-****-1234)
   - PAN Cards (10-char alphanumeric masked to ABCDE****F)
   - Bank Account Numbers (masked to ********1234)
   - UPI Deep Links (masks payee VPA and transaction parameters)
2. Constant-Time HMAC Token Verification (timing-attack protection)
3. Tool Execution Input Validation (amount range, loan ID syntax, E.164 phone format)
4. Conversational Prompt Injection Guard (blocks system prompt override attempts)
"""

import re
import hmac
import urllib.parse
from typing import Optional, Tuple, Any, Dict

# Regex Patterns for Indian PII Detection
RE_PHONE = re.compile(
    r"(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b|(?:\+91[\s\-]?)?[6-9]\d{9}\b"
)
# UIDAI Aadhaar: 12 digits (never starts with 0 or 1), formatted with space/dash or continuous
RE_AADHAAR = re.compile(
    r"\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b|\b[2-9]\d{11}\b"
)
RE_PAN = re.compile(
    r"\b([A-Z]{5})(\d{4})([A-Z])\b"
)
RE_BANK_ACCOUNT = re.compile(
    r"\b(?:account|ac|a/c|खाता|acc\s*no)[\s:#\-]*(\d{9,18})\b",
    re.IGNORECASE
)
RE_UPI_URL = re.compile(
    r"upi://pay\?[^\s\'\"<>]+"
)

# Prompt Injection Keywords
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(system|previous)\s+prompts?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"system\s*:\s*override", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?safety\s+filters", re.IGNORECASE),
    re.compile(r"simulate\s+jailbreak", re.IGNORECASE),
]


class PIIRedactor:
    """
    High-speed regex PII Redactor for logging, transcripts, and storage.
    Complies with Indian Digital Personal Data Protection (DPDP) Act 2023
    and Reserve Bank of India (RBI) Data Localization Guidelines.
    """

    @staticmethod
    def mask_phone(phone_str: str) -> str:
        """
        Masks Indian phone number, revealing only the last 4 digits.
        Example: '+91 98765 43210' -> '+91*****3210'
        """
        if not phone_str:
            return ""
        digits = re.sub(r"\D", "", phone_str)
        if len(digits) >= 10:
            last4 = digits[-4:]
            return f"+91*****{last4}"
        return "****"

    @staticmethod
    def mask_aadhaar(aadhaar_str: str) -> str:
        """
        Masks 12-digit Indian Aadhaar card number.
        Example: '1234 5678 9012' -> '****-****-9012'
        """
        digits = re.sub(r"\D", "", aadhaar_str)
        if len(digits) == 12:
            return f"****-****-{digits[-4:]}"
        return "****-****-****"

    @staticmethod
    def mask_pan(pan_str: str) -> str:
        """
        Masks 10-character Indian PAN card.
        Example: 'ABCDE1234F' -> 'ABCDE****F'
        """
        cleaned = pan_str.strip().upper()
        if len(cleaned) == 10 and RE_PAN.match(cleaned):
            return f"{cleaned[:5]}****{cleaned[-1]}"
        return "**********"

    @staticmethod
    def mask_upi_url(upi_url: str) -> str:
        """
        Masks payee VPA and customer identifiers in UPI deep-links.
        Example: 'upi://pay?pa=muthootfincorp@icici&am=5420.00'
              -> 'upi://pay?pa=m***p@icici&am=5420.00'
        """
        try:
            parsed = urllib.parse.urlparse(upi_url)
            params = urllib.parse.parse_qs(parsed.query)
            if "pa" in params and params["pa"]:
                vpa = params["pa"][0]
                if "@" in vpa:
                    handle, bank = vpa.split("@", 1)
                    if len(handle) > 2:
                        masked_vpa = f"{handle[0]}***{handle[-1]}@{bank}"
                    else:
                        masked_vpa = f"***@{bank}"
                    params["pa"] = [masked_vpa]
            new_query = urllib.parse.urlencode(params, doseq=True, safe="@*")
            return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
        except Exception:
            return "upi://pay?pa=masked@upi"

    @classmethod
    def redact_text(cls, text: str) -> str:
        """
        Scans a text stream and redacts all instances of Indian PII.
        Safe for writing to audit logs, terminal output, and external telemetry.
        """
        if not text:
            return text

        # 1. Mask UPI links
        def _sub_upi(m):
            return cls.mask_upi_url(m.group(0))
        text = RE_UPI_URL.sub(_sub_upi, text)

        # 2. Mask contextual Bank Account numbers
        def _sub_account(m):
            acc_num = m.group(1)
            return m.group(0).replace(acc_num, f"********{acc_num[-4:]}")
        text = RE_BANK_ACCOUNT.sub(_sub_account, text)

        # 3. Mask Phone Numbers
        def _sub_phone(m):
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) >= 10:
                return f"+91*****{digits[-4:]}"
            return raw
        text = RE_PHONE.sub(_sub_phone, text)

        # 4. Mask PAN card IDs
        def _sub_pan(m):
            return f"{m.group(1)}****{m.group(3)}"
        text = RE_PAN.sub(_sub_pan, text)

        # 5. Mask Aadhaar numbers
        def _sub_aadhaar(m):
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) == 12:
                return f"****-****-{digits[-4:]}"
            return raw
        text = RE_AADHAAR.sub(_sub_aadhaar, text)

        return text


# ---------------------------------------------------------------------------
# Constant-Time Authentication & Token Verification
# ---------------------------------------------------------------------------

def verify_auth_token(provided_token: Optional[str], expected_token: Optional[str]) -> bool:
    """
    Validates authentication token using timing-attack resistant hmac.compare_digest.
    If expected_token is None or empty, returns True (open development mode).
    """
    if not expected_token:
        return True
    if not provided_token:
        return False
    return hmac.compare_digest(provided_token.strip(), expected_token.strip())


# ---------------------------------------------------------------------------
# Tool Execution Input Validation & Sanitization
# ---------------------------------------------------------------------------

def validate_amount(
    amount_val: Any,
    min_amount: float = 1.0,
    max_amount: float = 500000.0
) -> Tuple[bool, float, str]:
    """
    Validates EMI / payment amount within authorized bounds.
    Returns: (is_valid, sanitized_amount, error_message)
    """
    try:
        cleaned = str(amount_val).replace(",", "").replace("₹", "").replace("Rs.", "").strip()
        val = float(cleaned)
    except (ValueError, TypeError):
        return False, 0.0, f"Invalid amount format: {amount_val}"

    if val < min_amount:
        return False, val, f"Amount Rs. {val:.2f} is below minimum allowed Rs. {min_amount:.2f}"
    if val > max_amount:
        return False, val, f"Amount Rs. {val:.2f} exceeds maximum allowed Rs. {max_amount:.2f}"

    return True, round(val, 2), "OK"


def validate_loan_id(loan_id_val: Any) -> Tuple[bool, str, str]:
    """
    Validates and sanitizes loan account identifier.
    Restricts to alphanumeric characters, dashes, and underscores (3 to 30 chars).
    Prevents path traversal, SQL injection, script tags, and header injection.
    """
    cleaned = str(loan_id_val or "").strip()
    if not cleaned:
        return False, "", "Loan ID cannot be empty"

    if not re.match(r"^[A-Za-z0-9\-_]{3,30}$", cleaned):
        return False, "", f"Loan ID '{cleaned}' contains invalid characters or length. Must be 3-30 alphanumeric characters."

    return True, cleaned, "OK"


def validate_indian_phone(phone_val: Any) -> Tuple[bool, str, str]:
    """
    Validates Indian mobile number according to National Numbering Plan:
    10 digits, starting with 6, 7, 8, or 9.
    Returns: (is_valid, 10_digit_phone, error_message)
    """
    digits = re.sub(r"\D", "", str(phone_val or ""))
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    if len(digits) != 10:
        return False, "", f"Invalid phone length: expected 10 digits, got {len(digits)}"

    if digits[0] not in ("6", "7", "8", "9"):
        return False, "", f"Invalid Indian mobile prefix: must start with 6-9, got '{digits[0]}'"

    return True, digits, "OK"


# ---------------------------------------------------------------------------
# Conversational Prompt Injection Guard
# ---------------------------------------------------------------------------

def detect_prompt_injection(user_utterance: str) -> Tuple[bool, str]:
    """
    Scans user speech utterance for conversational prompt injection or jailbreak attempts.
    Returns: (is_injection, reason)
    """
    if not user_utterance:
        return False, ""

    for pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(user_utterance)
        if match:
            return True, f"Detected prompt injection pattern: '{match.group(0)}'"

    return False, ""
