"""
verbalyze/benchmark/metrics.py

Evaluation metrics for Speech-to-Text and Inverse Text Normalization (ITN):
- Word Error Rate (WER)
- Character Error Rate (CER)
- Digit Sequence Exact Match Accuracy (OTPs, PINs, phone numbers)
- Acronym Retention Rate (KYC, UPI, NEFT, IFSC, PAN)
"""

import re
from typing import List, Tuple, Dict


def _levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """Computes Levenshtein edit distance between two sequences."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = matrix[x - 1][y - 1]
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,      # deletion
                    matrix[x][y - 1] + 1,      # insertion
                    matrix[x - 1][y - 1] + 1   # substitution
                )
    return matrix[size_x - 1][size_y - 1]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calculates Word Error Rate (WER) between reference and hypothesis text."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein(ref_words, hyp_words)
    return min(1.0, dist / len(ref_words))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Calculates Character Error Rate (CER) between reference and hypothesis text."""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = _levenshtein(ref_chars, hyp_chars)
    return min(1.0, dist / len(ref_chars))


def extract_digits(text: str) -> str:
    """Extracts all consecutive digit sequences from text."""
    return "".join(re.findall(r"\d+", text))


def compute_digit_accuracy(reference: str, hypothesis: str) -> Tuple[bool, bool]:
    """
    Checks whether digit sequences in reference are preserved in hypothesis.
    Returns:
        (has_digits, is_exact_match)
    """
    ref_digits = extract_digits(reference)
    hyp_digits = extract_digits(hypothesis)
    if not ref_digits:
        return False, False
    return True, (ref_digits == hyp_digits)


KNOWN_ACRONYMS = {
    "KYC", "PAN", "UPI", "IFSC", "NEFT", "RTGS", "IMPS", "ATM", "PIN", "OTP",
    "CVV", "EMI", "NACH", "FD", "RD", "CIBIL", "GST", "TDS", "SIP", "NAV",
    "IPO", "RBI", "SEBI", "BP", "HR", "ECG", "EKG", "MRI", "CT", "CBC",
    "LFT", "KFT", "ICU", "OPD", "SOS", "BD", "OD", "COPD", "COVID"
}


def extract_acronyms(text: str) -> List[str]:
    """Extracts known business/medical acronyms from text."""
    tokens = re.findall(r"\b[A-Za-z0-9]+\b", text)
    found = []
    for t in tokens:
        upper = t.upper()
        if upper in KNOWN_ACRONYMS:
            found.append(upper)
    return found


def compute_acronym_retention(reference: str, hypothesis: str) -> Tuple[int, int]:
    """
    Computes (total_acronyms_in_ref, retained_acronyms_in_hyp).
    """
    ref_acronyms = extract_acronyms(reference)
    if not ref_acronyms:
        return 0, 0
    hyp_upper = hypothesis.upper()
    retained = sum(1 for ac in ref_acronyms if ac in hyp_upper)
    return len(ref_acronyms), retained


def evaluate_batch(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Computes aggregate benchmark scores across a batch."""
    total_wer = 0.0
    total_cer = 0.0
    digit_trials = 0
    digit_success = 0
    acronym_total = 0
    acronym_retained = 0

    n = len(references)
    for ref, hyp in zip(references, hypotheses):
        total_wer += compute_wer(ref, hyp)
        total_cer += compute_cer(ref, hyp)

        has_digits, match = compute_digit_accuracy(ref, hyp)
        if has_digits:
            digit_trials += 1
            if match:
                digit_success += 1

        tot_ac, ret_ac = compute_acronym_retention(ref, hyp)
        acronym_total += tot_ac
        acronym_retained += ret_ac

    return {
        "samples": n,
        "wer": round((total_wer / n) * 100, 2) if n > 0 else 0.0,
        "cer": round((total_cer / n) * 100, 2) if n > 0 else 0.0,
        "digit_accuracy": round((digit_success / digit_trials) * 100, 2) if digit_trials > 0 else 100.0,
        "acronym_retention": round((acronym_retained / acronym_total) * 100, 2) if acronym_total > 0 else 100.0,
    }
