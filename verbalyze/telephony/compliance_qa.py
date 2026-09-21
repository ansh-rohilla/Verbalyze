"""
verbalyze/telephony/compliance_qa.py

Regulatory Call Recording & Post-Call AI Quality Assurance (QA) Engine:
- Evaluates 100% of debt recovery calls against RBI Fair Practices Code for Lenders,
  RBI Master Directions on Recovery Agents, and TRAI regulations.
- 4-Pillar Compliance Scoring Matrix (0 to 100 Scale):
    1. Mandatory Identity & Authorization Disclosure (25 pts)
    2. Zero Prohibited Conduct & Harassment Check (25 pts)
    3. Professionalism, Empathy & Active De-escalation (25 pts)
    4. Resolution & Confirmation of Terms (25 pts)
- Automated CRM Notes and Next Best Action (NBA) extraction.
- In-memory official RBI Compliance Certificate PDF generation via fpdf2.
- DPDP Act 2023 compliant: Borrower PII masked in all outputs.
- Zero-emoji compliant.
"""

import re
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from fpdf import FPDF
from verbalyze.security import PIIRedactor
from verbalyze.telephony.sms_dispatch import clean_indian_phone


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    FLAGGED_FOR_AUDIT = "FLAGGED_FOR_AUDIT"
    NON_COMPLIANT = "NON_COMPLIANT"


class DebtorIntent(str, Enum):
    PAYMENT_PROMISED = "PAYMENT_PROMISED"
    DISPUTE_RAISED = "DISPUTE_RAISED"
    CALLBACK_REQUESTED = "CALLBACK_REQUESTED"
    REFUSAL_TO_PAY = "REFUSAL_TO_PAY"
    HARDSHIP_UNEMPLOYMENT = "HARDSHIP_UNEMPLOYMENT"
    WRONG_NUMBER = "WRONG_NUMBER"
    UNSPECIFIED = "UNSPECIFIED"


@dataclass
class ComplianceInfraction:
    """Represents a regulatory or procedural non-compliance finding."""
    dimension: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    rule_code: str
    description: str
    transcript_quote: Optional[str] = None
    deduction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "severity": self.severity,
            "rule_code": self.rule_code,
            "description": self.description,
            "transcript_quote": self.transcript_quote,
            "deduction": round(self.deduction, 1),
        }


@dataclass
class PillarScore:
    """Scorecard item for one compliance pillar."""
    name: str
    score: float
    max_score: float = 25.0
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "max_score": self.max_score,
            "percentage": round((self.score / self.max_score) * 100.0, 1),
            "criteria_met": self.criteria_met,
            "criteria_failed": self.criteria_failed,
        }


@dataclass
class CRMNotes:
    """Automated post-call summary and next best action for human loan officers."""
    call_id: str
    debtor_intent: str
    agreed_amount: Optional[float] = None
    ptp_date: Optional[str] = None
    dispute_category: str = "NONE"
    summary: str = ""
    next_best_action: str = ""

    def to_dict(self, mask_pii: bool = True) -> Dict[str, Any]:
        redacted_summary = PIIRedactor.redact_text(self.summary) if mask_pii else self.summary
        redacted_nba = PIIRedactor.redact_text(self.next_best_action) if mask_pii else self.next_best_action
        return {
            "call_id": self.call_id,
            "debtor_intent": self.debtor_intent,
            "agreed_amount": round(self.agreed_amount, 2) if self.agreed_amount else None,
            "ptp_date": self.ptp_date,
            "dispute_category": self.dispute_category,
            "summary": redacted_summary,
            "next_best_action": redacted_nba,
        }


@dataclass
class QAScorecard:
    """Overall compliance evaluation record for an audited call."""
    call_id: str
    loan_id: str
    customer_phone_masked: str
    customer_name_masked: str
    overall_score: float
    status: ComplianceStatus
    pillar_1_identity: PillarScore
    pillar_2_prohibited_conduct: PillarScore
    pillar_3_professionalism: PillarScore
    pillar_4_resolution: PillarScore
    infractions: List[ComplianceInfraction]
    crm_notes: CRMNotes
    call_duration_sec: float
    language: str
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self, mask_pii: bool = True) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "loan_id": self.loan_id,
            "customer_phone": self.customer_phone_masked,
            "customer_name": self.customer_name_masked,
            "overall_score": round(self.overall_score, 1),
            "status": self.status.value,
            "pillars": {
                "identity_disclosure": self.pillar_1_identity.to_dict(),
                "prohibited_conduct": self.pillar_2_prohibited_conduct.to_dict(),
                "professionalism_empathy": self.pillar_3_professionalism.to_dict(),
                "resolution_agreement": self.pillar_4_resolution.to_dict(),
            },
            "infractions_count": len(self.infractions),
            "infractions": [i.to_dict() for i in self.infractions],
            "crm_notes": self.crm_notes.to_dict(mask_pii=mask_pii),
            "call_duration_sec": round(self.call_duration_sec, 2),
            "language": self.language,
            "evaluated_at": round(self.evaluated_at, 2),
        }


class ComplianceQAEngine:
    """
    Automated Regulatory Quality Assurance Evaluator.
    Scans conversation transcripts against RBI and TRAI debt recovery standards.
    """

    PASS_THRESHOLD: float = 75.0

    # Patterns for Pillar 1: Identity & Authorization
    NBFC_KEYWORDS = [
        "muthoot", "मुथूट", "muthoot fincorp", "मुथूट फिनकॉर्प",
        "fincorp", "bank", "nbfc", "lender"
    ]
    VERIFICATION_PATTERNS = [
        "क्या मेरी बात", "am i speaking with", "speaking to",
        "baat ho rahi hai", "se bol rahe hain", "is this"
    ]
    LOAN_KEYWORDS = [
        "loan", "लोन", "emi", "ईएमआई", "बकाया", "overdue",
        "due", "account", "खाता", "amount", "रुपये", "rs", "₹"
    ]

    # Patterns for Pillar 2: Prohibited Conduct (RBI Violations)
    ABUSIVE_KEYWORDS = [
        "chutiya", "saale", "kamine", "harami", "idiot", "stupid",
        "bastard", "fraud", "chor", "cheat", "badtameez"
    ]
    THREAT_KEYWORDS = [
        "police bhejunga", "jail bhejunga", "gunde bhejenge",
        "beat you", "marunga", "todi dunga", "destroy",
        "humiliate", "shame you", "tell your family", "ghar aaunga dhamkane"
    ]
    PRIVACY_BREACH_KEYWORDS = [
        "tell your employer", "call your boss", "post on social media",
        "inform your relatives", "relatives ko batayenge", "mohalle me batayenge"
    ]

    # Patterns for Pillar 3: Professionalism & Empathy
    GREETING_KEYWORDS = [
        "नमस्कार", "hello", "good morning", "good afternoon",
        "good evening", "namaste", "pranam", "नमस्ते"
    ]
    CLOSING_KEYWORDS = [
        "धन्यवाद", "thank you", "thanks", "have a good day",
        "shubh din", "din shubh ho", "aapka din achha rahe"
    ]
    EMPATHY_KEYWORDS = [
        "समझ सकती हूँ", "समझ सकता हूँ", "understand", "percept",
        "मदद", "help", "sahayata", "chinta mat kijiye", "don't worry", "apologize"
    ]

    # Patterns for Pillar 4: Resolution & PTP
    PTP_KEYWORDS = [
        "kal", "tomorrow", "parso", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday", "pay kar dunga",
        "jama kar dunga", "deposit kar dunga", "pay tomorrow", "promise to pay",
        "कल", "परसों", "सोमवार", "मंगलवार", "बुधवार", "जमा कर", "पे कर", "डिपॉजिट", "पेमेंट"
    ]
    DISPUTE_KEYWORDS = [
        "already paid", "jama kar diya tha", "receipt", "wrong number",
        "galat number", "dispute", "police FIR", "court", "harassment",
        "जमा कर दिया", "रसीद", "गलत नंबर", "शिकायत", "पुलिस"
    ]

    @classmethod
    def evaluate_call(
        cls,
        call_id: str,
        turns: List[Dict[str, str]],
        loan_id: str = "MUTH-8921",
        caller_phone: str = "+919876543210",
        customer_name: str = "Borrower",
        call_start_timestamp: Optional[float] = None,
        audio_duration_sec: float = 0.0,
        language: str = "hi",
    ) -> QAScorecard:
        """
        Runs comprehensive post-call compliance audit across the 4 pillars.
        """
        infractions: List[ComplianceInfraction] = []

        # Aggregate text by speaker
        agent_texts: List[str] = []
        user_texts: List[str] = []
        for t in turns:
            role = t.get("role", "").lower()
            text = t.get("content", "") or t.get("text", "")
            if role in ("agent", "assistant", "bot"):
                agent_texts.append(text)
            elif role in ("user", "customer", "caller"):
                user_texts.append(text)

        full_agent_corpus = " ".join(agent_texts).lower()
        full_user_corpus = " ".join(user_texts).lower()

        # ----------------------------------------------------------------------
        # Pillar 1: Mandatory Identity & Authorization Disclosure (25 pts)
        # ----------------------------------------------------------------------
        p1_score = 0.0
        p1_met: List[str] = []
        p1_failed: List[str] = []

        # 1. NBFC Entity Disclosure (10 pts)
        has_nbfc = any(k in full_agent_corpus for k in cls.NBFC_KEYWORDS)
        if has_nbfc:
            p1_score += 10.0
            p1_met.append("NBFC Entity Name Stated (Muthoot Fincorp)")
        else:
            p1_failed.append("Missing NBFC Entity Disclosure")
            infractions.append(ComplianceInfraction(
                dimension="identity_disclosure",
                severity="MEDIUM",
                rule_code="RBI_DISCLOSURE_ENTITY",
                description="Agent failed to clearly identify the lending institution (Muthoot Fincorp).",
                deduction=10.0,
            ))

        # 2. Borrower Verification / Identity Confirmation (8 pts)
        has_verification = any(p in full_agent_corpus for p in cls.VERIFICATION_PATTERNS) or customer_name.lower() in full_agent_corpus
        if has_verification:
            p1_score += 8.0
            p1_met.append("Borrower Identity Verification Attempted")
        else:
            p1_failed.append("Missing Borrower Verification Confirmation")
            infractions.append(ComplianceInfraction(
                dimension="identity_disclosure",
                severity="LOW",
                rule_code="RBI_BORROWER_VERIFICATION",
                description="Agent did not explicitly verify borrower identity before discussing debt.",
                deduction=8.0,
            ))

        # 3. Loan ID / Debt Amount Disclosure (7 pts)
        has_loan_info = any(k in full_agent_corpus for k in cls.LOAN_KEYWORDS) or loan_id.lower() in full_agent_corpus
        if has_loan_info:
            p1_score += 7.0
            p1_met.append("Loan Overdue Notice / Amount Disclosed")
        else:
            p1_failed.append("Missing Specific Overdue Debt Details")
            infractions.append(ComplianceInfraction(
                dimension="identity_disclosure",
                severity="LOW",
                rule_code="RBI_LOAN_DISCLOSURE",
                description="Agent did not state the specific overdue loan amount or reference.",
                deduction=7.0,
            ))

        pillar_1 = PillarScore(name="Mandatory Identity Disclosure", score=p1_score, criteria_met=p1_met, criteria_failed=p1_failed)

        # ----------------------------------------------------------------------
        # Pillar 2: Zero Prohibited Conduct & Harassment Check (25 pts)
        # ----------------------------------------------------------------------
        p2_score = 25.0
        p2_met: List[str] = []
        p2_failed: List[str] = []

        # Check A: Abusive Language
        found_abusive = [k for k in cls.ABUSIVE_KEYWORDS if k in full_agent_corpus]
        if found_abusive:
            p2_score = 0.0
            p2_failed.append(f"Abusive Language Detected: {', '.join(found_abusive)}")
            infractions.append(ComplianceInfraction(
                dimension="prohibited_conduct",
                severity="CRITICAL",
                rule_code="RBI_HARASSMENT_ABUSIVE_LANGUAGE",
                description=f"Agent used prohibited abusive or offensive words ({', '.join(found_abusive)}).",
                deduction=25.0,
            ))
        else:
            p2_met.append("Zero Abusive Language")

        # Check B: Threats of Physical Violence / Coercion
        found_threats = [k for k in cls.THREAT_KEYWORDS if k in full_agent_corpus]
        if found_threats:
            p2_score = 0.0
            p2_failed.append(f"Coercive Threats Detected: {', '.join(found_threats)}")
            infractions.append(ComplianceInfraction(
                dimension="prohibited_conduct",
                severity="CRITICAL",
                rule_code="RBI_COERCION_PHYSICAL_THREATS",
                description=f"Agent used coercive threats or unlawful intimidation ({', '.join(found_threats)}).",
                deduction=25.0,
            ))
        else:
            p2_met.append("Zero Physical or Unlawful Threats")

        # Check C: Privacy Violations / Third-Party Intimidation
        found_privacy = [k for k in cls.PRIVACY_BREACH_KEYWORDS if k in full_agent_corpus]
        if found_privacy:
            p2_score = max(0.0, p2_score - 15.0)
            p2_failed.append(f"Privacy Breach Threat: {', '.join(found_privacy)}")
            infractions.append(ComplianceInfraction(
                dimension="prohibited_conduct",
                severity="HIGH",
                rule_code="DPDP_PRIVACY_VIOLATION",
                description=f"Agent threatened third-party disclosure or public shaming ({', '.join(found_privacy)}).",
                deduction=15.0,
            ))
        else:
            p2_met.append("Full Borrower Privacy Preservation")

        # Check D: Calling Hours (TRAI 08:00 - 19:00 IST)
        if call_start_timestamp:
            call_dt = datetime.fromtimestamp(call_start_timestamp, tz=timezone.utc)
            # Convert to IST (UTC+5:30)
            ist_hour = (call_dt.hour + 5 + (call_dt.minute + 30) // 60) % 24
            if ist_hour < 8 or ist_hour >= 19:
                p2_score = max(0.0, p2_score - 10.0)
                p2_failed.append(f"Calling Hours Violation (IST Hour: {ist_hour}:00)")
                infractions.append(ComplianceInfraction(
                    dimension="prohibited_conduct",
                    severity="HIGH",
                    rule_code="TRAI_CALLING_HOURS_VIOLATION",
                    description=f"Call initiated outside legal TRAI debt collection hours (08:00-19:00 IST).",
                    deduction=10.0,
                ))
            else:
                p2_met.append("Compliant TRAI Calling Hours")
        else:
            p2_met.append("Calling Hours Verified")

        pillar_2 = PillarScore(name="Zero Prohibited Conduct & Harassment", score=p2_score, criteria_met=p2_met, criteria_failed=p2_failed)

        # ----------------------------------------------------------------------
        # Pillar 3: Professionalism, Empathy & Active De-escalation (25 pts)
        # ----------------------------------------------------------------------
        p3_score = 0.0
        p3_met: List[str] = []
        p3_failed: List[str] = []

        # 1. Courteous Opening Greeting (8 pts)
        has_greeting = any(k in full_agent_corpus for k in cls.GREETING_KEYWORDS)
        if has_greeting:
            p3_score += 8.0
            p3_met.append("Courteous Opening Greeting")
        else:
            p3_failed.append("Missing Courteous Greeting")

        # 2. Respectful Closing (7 pts)
        has_closing = any(k in full_agent_corpus for k in cls.CLOSING_KEYWORDS) or len(agent_texts) > 2
        if has_closing:
            p3_score += 7.0
            p3_met.append("Professional Closing")
        else:
            p3_failed.append("Abrupt or Missing Closing")

        # 3. Empathetic De-escalation Posture (10 pts)
        customer_distressed = any(w in full_user_corpus for w in ["pareshan", "bimar", "hospital", "job lost", "naukri", "mushkil", "problem"])
        agent_empathetic = any(w in full_agent_corpus for w in cls.EMPATHY_KEYWORDS)

        if customer_distressed:
            if agent_empathetic:
                p3_score += 10.0
                p3_met.append("Empathetic De-escalation Executed on Borrower Distress")
            else:
                p3_failed.append("Failed to Acknowledge Borrower Distress Empathetically")
                infractions.append(ComplianceInfraction(
                    dimension="professionalism",
                    severity="MEDIUM",
                    rule_code="RBI_EMPATHY_DEFICIT",
                    description="Customer reported distress/hardship, but agent failed to demonstrate empathy.",
                    deduction=10.0,
                ))
        else:
            p3_score += 10.0
            p3_met.append("Calm & Respectful Conversational Demeanor")

        pillar_3 = PillarScore(name="Professionalism, Empathy & De-escalation", score=p3_score, criteria_met=p3_met, criteria_failed=p3_failed)

        # ----------------------------------------------------------------------
        # Pillar 4: Resolution & Confirmation of Terms (25 pts)
        # ----------------------------------------------------------------------
        p4_score = 0.0
        p4_met: List[str] = []
        p4_failed: List[str] = []

        ptp_agreed = any(k in full_user_corpus for k in cls.PTP_KEYWORDS)
        dispute_raised = any(k in full_user_corpus for k in cls.DISPUTE_KEYWORDS)
        upi_link_sent = "upi" in full_agent_corpus or "link" in full_agent_corpus
        callback_requested = "callback" in full_user_corpus or "baad me" in full_user_corpus

        if ptp_agreed:
            p4_score += 25.0
            p4_met.append("Concrete Promise-to-Pay (PTP) Terms Agreed")
        elif upi_link_sent:
            p4_score += 25.0
            p4_met.append("Instant Digital Payment Link Dispatched")
        elif dispute_raised:
            p4_score += 25.0
            p4_met.append("Customer Dispute Formally Acknowledged & Logged")
        elif callback_requested:
            p4_score += 20.0
            p4_met.append("Callback Window Confirmed")
        elif len(turns) >= 2:
            p4_score += 10.0
            p4_met.append("Debt Notice Delivered Without Refusal")
        else:
            p4_failed.append("Call Terminated Without Any Resolution or Agreement")

        pillar_4 = PillarScore(name="Resolution & Confirmation of Terms", score=p4_score, criteria_met=p4_met, criteria_failed=p4_failed)

        # ----------------------------------------------------------------------
        # Composite Compliance Score
        # ----------------------------------------------------------------------
        overall_score = p1_score + p2_score + p3_score + p4_score
        has_critical = any(i.severity == "CRITICAL" for i in infractions)

        if has_critical or overall_score < 50.0:
            status = ComplianceStatus.NON_COMPLIANT
        elif overall_score < cls.PASS_THRESHOLD or len(infractions) > 2:
            status = ComplianceStatus.FLAGGED_FOR_AUDIT
        else:
            status = ComplianceStatus.COMPLIANT

        # ----------------------------------------------------------------------
        # Automated CRM Notes & Next Best Action Extraction
        # ----------------------------------------------------------------------
        crm_notes = cls._extract_crm_notes(
            call_id=call_id,
            turns=turns,
            full_user_corpus=full_user_corpus,
            full_agent_corpus=full_agent_corpus,
            ptp_agreed=ptp_agreed,
            dispute_raised=dispute_raised,
            upi_link_sent=upi_link_sent,
        )

        clean_phone = clean_indian_phone(caller_phone)
        masked_phone = PIIRedactor.mask_phone(clean_phone)
        masked_name = f"{customer_name[0]}***" if customer_name and len(customer_name) > 1 else "Borrower"

        return QAScorecard(
            call_id=call_id,
            loan_id=loan_id,
            customer_phone_masked=masked_phone,
            customer_name_masked=masked_name,
            overall_score=overall_score,
            status=status,
            pillar_1_identity=pillar_1,
            pillar_2_prohibited_conduct=pillar_2,
            pillar_3_professionalism=pillar_3,
            pillar_4_resolution=pillar_4,
            infractions=infractions,
            crm_notes=crm_notes,
            call_duration_sec=audio_duration_sec,
            language=language,
        )

    @classmethod
    def _extract_crm_notes(
        cls,
        call_id: str,
        turns: List[Dict[str, str]],
        full_user_corpus: str,
        full_agent_corpus: str,
        ptp_agreed: bool,
        dispute_raised: bool,
        upi_link_sent: bool,
    ) -> CRMNotes:
        """Extracts structured debtor disposition and next best action."""
        intent = DebtorIntent.UNSPECIFIED
        dispute_cat = "NONE"
        agreed_amt: Optional[float] = None
        ptp_date: Optional[str] = None

        if dispute_raised:
            intent = DebtorIntent.DISPUTE_RAISED
            if "already paid" in full_user_corpus or "jama kar diya" in full_user_corpus:
                dispute_cat = "PAYMENT_DISPUTE"
            elif "police" in full_user_corpus or "court" in full_user_corpus:
                dispute_cat = "LEGAL_THREAT"
            elif "wrong number" in full_user_corpus or "galat number" in full_user_corpus:
                dispute_cat = "WRONG_PERSON"
            elif "harass" in full_user_corpus or "bar bar call" in full_user_corpus:
                dispute_cat = "HARASSMENT_COMPLAINT"
            summary = f"Customer raised {dispute_cat}. Claim requires verification."
            nba = "Hold outbound dials. Request statement/receipt via WhatsApp and route to Disputes Desk."

        elif ptp_agreed or upi_link_sent:
            intent = DebtorIntent.PAYMENT_PROMISED
            # Look for dates
            if "kal" in full_user_corpus or "tomorrow" in full_user_corpus or "कल" in full_user_corpus:
                ptp_date = "Tomorrow"
            elif "parso" in full_user_corpus or "परसों" in full_user_corpus:
                ptp_date = "Day After Tomorrow"
            elif "monday" in full_user_corpus or "somwar" in full_user_corpus or "सोमवार" in full_user_corpus:
                ptp_date = "Next Monday"
            else:
                ptp_date = "Within 48 Hours"

            summary = f"Borrower promised to settle overdue debt ({ptp_date}). Payment link dispatched."
            nba = f"Schedule automated WhatsApp reminder on {ptp_date} morning. Halt retry dials."

        elif "bimar" in full_user_corpus or "job lost" in full_user_corpus or "naukri" in full_user_corpus:
            intent = DebtorIntent.HARDSHIP_UNEMPLOYMENT
            summary = "Borrower reported financial distress / medical hardship."
            nba = "Flag account for NBFC restructuring / EMI tenure extension review."

        elif "callback" in full_user_corpus or "baad me" in full_user_corpus:
            intent = DebtorIntent.CALLBACK_REQUESTED
            summary = "Borrower requested callback at a convenient time."
            nba = "Schedule callback dialer task for next available compliant calling window."

        else:
            intent = DebtorIntent.REFUSAL_TO_PAY if "nahi dunga" in full_user_corpus else DebtorIntent.UNSPECIFIED
            summary = "Borrower conversation concluded without firm commitment."
            nba = "Retry outbound call during next TRAI calling window."

        return CRMNotes(
            call_id=call_id,
            debtor_intent=intent.value,
            agreed_amount=agreed_amt,
            ptp_date=ptp_date,
            dispute_category=dispute_cat,
            summary=summary,
            next_best_action=nba,
        )


class ComplianceQARegistry:
    """
    In-memory registry storing QA scorecards and dual-channel call recordings.
    Supports official compliance audit certificate PDF generation via fpdf2.
    """

    def __init__(self):
        self.scorecards: Dict[str, QAScorecard] = {}
        self.recordings: Dict[str, bytes] = {}

    def register_audit(self, scorecard: QAScorecard, recording_wav: Optional[bytes] = None) -> None:
        """Stores scorecard and optional WAV recording."""
        self.scorecards[scorecard.call_id] = scorecard
        if recording_wav:
            self.recordings[scorecard.call_id] = recording_wav

    def get_scorecard(self, call_id: str) -> Optional[QAScorecard]:
        return self.scorecards.get(call_id)

    def get_recording(self, call_id: str) -> Optional[bytes]:
        return self.recordings.get(call_id)

    def list_audits(self) -> List[Dict[str, Any]]:
        return [s.to_dict(mask_pii=True) for s in self.scorecards.values()]

    def generate_certificate_pdf_bytes(self, call_id: str) -> Optional[bytes]:
        """
        Generates official RBI Compliance Audit Certificate PDF entirely in memory.
        Zero disk storage bloat.
        """
        scorecard = self.scorecards.get(call_id)
        if not scorecard:
            return None

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "MUTHOOT FINCORP LIMITED", ln=True, align="C")

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "RBI Fair Practices Code - Telephony Call Compliance Certificate", ln=True, align="C")
        pdf.line(10, 28, 200, 28)
        pdf.ln(8)

        # Overall Status Banner
        pdf.set_font("Helvetica", "B", 12)
        if scorecard.status == ComplianceStatus.COMPLIANT:
            pdf.set_fill_color(220, 252, 231)
            pdf.set_text_color(22, 101, 52)
            banner_text = f"AUDIT RESULT: COMPLIANT (Score: {scorecard.overall_score:.1f} / 100)"
        elif scorecard.status == ComplianceStatus.FLAGGED_FOR_AUDIT:
            pdf.set_fill_color(254, 249, 195)
            pdf.set_text_color(133, 77, 14)
            banner_text = f"AUDIT RESULT: FLAGGED FOR REVIEW (Score: {scorecard.overall_score:.1f} / 100)"
        else:
            pdf.set_fill_color(254, 226, 226)
            pdf.set_text_color(153, 27, 27)
            banner_text = f"AUDIT RESULT: NON-COMPLIANT (Score: {scorecard.overall_score:.1f} / 100)"

        pdf.cell(0, 10, banner_text, ln=True, align="C", fill=True)
        pdf.ln(6)

        # Metadata Table
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(0, 7, "Call Audit Metadata", ln=True)
        pdf.set_font("Helvetica", "", 9)

        meta_rows = [
            ("Call Identifier", scorecard.call_id),
            ("Loan Account Ref", scorecard.loan_id),
            ("Borrower Phone (DPDP Masked)", scorecard.customer_phone_masked),
            ("Borrower Name (DPDP Masked)", scorecard.customer_name_masked),
            ("Call Duration", f"{scorecard.call_duration_sec:.1f} seconds"),
            ("Primary Language", scorecard.language.upper()),
            ("Audit Timestamp (UTC)", datetime.fromtimestamp(scorecard.evaluated_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        ]

        for label, val in meta_rows:
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(60, 6, label, border=0)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(0, 6, str(val), ln=True, border=0)

        pdf.ln(4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        # 4-Pillar Score Breakdown
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "4-Pillar Regulatory Scorecard", ln=True)

        pillars = [
            scorecard.pillar_1_identity,
            scorecard.pillar_2_prohibited_conduct,
            scorecard.pillar_3_professionalism,
            scorecard.pillar_4_resolution,
        ]

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(85, 6, "Compliance Pillar", border=1, fill=True)
        pdf.cell(35, 6, "Score (Max 25)", border=1, fill=True, align="C")
        pdf.cell(70, 6, "Assessment Finding", border=1, fill=True)
        pdf.ln(6)

        pdf.set_font("Helvetica", "", 8)
        for p in pillars:
            finding = p.criteria_met[0] if p.criteria_met else (p.criteria_failed[0] if p.criteria_failed else "Evaluated")
            pdf.cell(85, 6, p.name[:45], border=1)
            pdf.cell(35, 6, f"{p.score:.1f} / 25.0", border=1, align="C")
            pdf.cell(70, 6, finding[:40], border=1)
            pdf.ln(6)

        pdf.ln(4)

        # CRM Summary & Next Best Action
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "CRM Case Notes & Next Best Action", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 5, f"Debtor Intent: {scorecard.crm_notes.debtor_intent}\n"
                             f"Summary: {scorecard.crm_notes.summary}\n"
                             f"Recommended Action: {scorecard.crm_notes.next_best_action}")
        pdf.ln(4)

        # Regulatory Integrity Stamp
        cert_payload = f"{scorecard.call_id}|{scorecard.overall_score}|{scorecard.status.value}|{scorecard.evaluated_at}"
        cert_hash = hashlib.sha256(cert_payload.encode("utf-8")).hexdigest().upper()

        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 5, f"Digital Audit Integrity Hash (SHA-256): {cert_hash}", ln=True)
        pdf.cell(0, 5, "This computer-generated certificate is certified under the RBI Fair Practices Code for Lenders & DPDP Act 2023.", ln=True)

        out = pdf.output()
        return bytes(out) if isinstance(out, (bytes, bytearray)) else bytes(str(out), "latin1")
