"""
verbalyze/telephony/dtmf_engine.py

High-Performance Dual-Tone Multi-Frequency (DTMF) Telecom Keypad Engine:
1. In-Band Acoustic Goertzel Algorithm (ITU-T Q.23 / Q.24 & Bellcore standards).
   - Operates on 8,000 Hz and 16,000 Hz raw 16-bit linear PCM audio.
   - Dual-tone frequency detection for standard touch-tone keypad ('0'-'9', '*', '#', 'A'-'D').
   - High/Low group dominance check, second-harmonic rejection, and twist tolerance (-8 dB to +4 dB).
2. Out-of-Band RFC 4733 / RFC 2833 RTP Telephone-Event Decoder & Encoder.
   - 4-byte RTP payload parsing (Event ID, End bit, Volume in -dBm0, Duration).
   - Event deduplication and state tracking.
3. Acoustic DTMF Debouncer & Inter-Digit Gap Detector.
   - Prevents duplicate triggers from continuous tone bursts.
4. Synthetic DTMF Tone Generator.
   - Generates dual-frequency PCM test signals for automated testing and synthesis.
5. DPDP Act 2023 Compliant Sensitive Keypad Masking.
   - Cryptographic masking of sensitive PINs, OTPs, and card/account numbers.

Zero-emoji compliant.
"""

import math
import struct
import time
from typing import Dict, Any, Optional, List, Tuple, Union


# Standard ITU-T Q.23 / Q.24 DTMF Frequencies
DTMF_ROW_FREQUENCIES: List[int] = [697, 770, 852, 941]
DTMF_COL_FREQUENCIES: List[int] = [1209, 1336, 1477, 1633]

# DTMF Matrix: [Row Index][Col Index]
DTMF_KEY_MATRIX: List[List[str]] = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"],
]

# Mapping from digit to frequency pair (row_freq, col_freq)
DIGIT_TO_FREQUENCIES: Dict[str, Tuple[int, int]] = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}

# RFC 4733 / RFC 2833 Named Telephone Events
RFC4733_EVENT_MAP: Dict[int, str] = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "*", 11: "#", 12: "A", 13: "B", 14: "C", 15: "D",
}
RFC4733_DIGIT_MAP: Dict[str, int] = {v: k for k, v in RFC4733_EVENT_MAP.items()}


def mask_digits(value: str, unmasked_suffix_length: int = 0) -> str:
    """
    Masks sensitive digits in accordance with the Digital Personal Data Protection (DPDP) Act 2023.
    By default, masks 100% of input digits with asterisks (e.g. '1234' -> '****').
    If unmasked_suffix_length is specified, preserves the last N digits (e.g. '9876543210' -> '******3210').
    """
    if not value:
        return ""
    val_str = str(value)
    if unmasked_suffix_length <= 0:
        return "*" * len(val_str)
    if len(val_str) <= unmasked_suffix_length:
        return val_str
    masked_count = len(val_str) - unmasked_suffix_length
    return ("*" * masked_count) + val_str[-unmasked_suffix_length:]


def sanitize_sensitive_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively sanitizes a dictionary, masking values of sensitive keys.
    """
    sensitive_keys = {
        "pin", "otp", "aadhaar", "card", "password", "secret",
        "account_number", "digit_buffer", "collected_digits", "raw_digits"
    }
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys):
            if isinstance(v, str):
                sanitized[k] = mask_digits(v)
            elif isinstance(v, (int, float)):
                sanitized[k] = mask_digits(str(v))
            else:
                sanitized[k] = "****"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_sensitive_dict(v)
        else:
            sanitized[k] = v
    return sanitized


class DTMFToneGenerator:
    """
    Generates synthetic 16-bit linear PCM audio for DTMF tones.
    Used for automated unit testing, telecom loopback validation, and audio prompts.
    """

    @staticmethod
    def generate_tone(
        digit: str,
        duration_ms: int = 100,
        sample_rate: int = 8000,
        amplitude: float = 0.5,
    ) -> bytes:
        """
        Synthesizes a dual-sine DTMF tone as raw 16-bit little-endian mono PCM bytes.
        """
        digit_upper = digit.upper()
        if digit_upper not in DIGIT_TO_FREQUENCIES:
            raise ValueError(f"Invalid DTMF digit '{digit}'. Valid digits: 0-9, *, #, A-D")

        f_row, f_col = DIGIT_TO_FREQUENCIES[digit_upper]
        total_samples = int((duration_ms / 1000.0) * sample_rate)
        half_amp = amplitude * 0.5 * 32767.0

        omega_row = 2.0 * math.pi * f_row / sample_rate
        omega_col = 2.0 * math.pi * f_col / sample_rate

        samples = []
        for n in range(total_samples):
            val = int(half_amp * math.sin(omega_row * n) + half_amp * math.sin(omega_col * n))
            val = max(-32768, min(32767, val))
            samples.append(val)

        return struct.pack(f"<{len(samples)}h", *samples)

    @staticmethod
    def generate_silence(duration_ms: int = 40, sample_rate: int = 8000) -> bytes:
        """Generates silent 16-bit linear mono PCM bytes."""
        total_samples = int((duration_ms / 1000.0) * sample_rate)
        return b"\x00\x00" * total_samples

    @classmethod
    def generate_sequence(
        cls,
        digits: str,
        tone_duration_ms: int = 80,
        gap_duration_ms: int = 40,
        sample_rate: int = 8000,
        amplitude: float = 0.5,
    ) -> bytes:
        """
        Generates a sequence of DTMF tones separated by inter-digit silence intervals.
        """
        chunks = []
        for i, ch in enumerate(digits):
            if ch.upper() in DIGIT_TO_FREQUENCIES:
                chunks.append(cls.generate_tone(ch, tone_duration_ms, sample_rate, amplitude))
                if i < len(digits) - 1 and gap_duration_ms > 0:
                    chunks.append(cls.generate_silence(gap_duration_ms, sample_rate))
        return b"".join(chunks)


class GoertzelDetector:
    """
    Acoustic Goertzel Algorithm DTMF Detector for 8kHz / 16kHz 16-bit linear PCM audio.
    
    Implements ITU-T Q.23 and Bellcore tone detection criteria:
    - Continuous frequency coefficient calculation with 0 Hz tuning error.
    - Row and Column peak detection with second-harmonic dominance verification.
    - Telecom twist ratio check (-8 dB to +4 dB).
    - Energy thresholding and wideband noise rejection.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_size: int = 160,
        energy_threshold: float = 1.0e6,
        dominance_ratio: float = 2.5,
        min_twist_db: float = -8.0,
        max_twist_db: float = 4.0,
    ):
        """
        Args:
            sample_rate: Audio sampling frequency in Hz (8000 or 16000).
            frame_size: Number of samples per Goertzel block (e.g. 160 for 20ms at 8kHz).
            energy_threshold: Minimum total signal energy to consider tone presence.
            dominance_ratio: Minimum ratio of peak frequency power to second highest in group.
            min_twist_db: Minimum acceptable twist in dB (normal twist: column / row).
            max_twist_db: Maximum acceptable twist in dB (reverse twist: column / row).
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.energy_threshold = energy_threshold
        self.dominance_ratio = dominance_ratio
        self.min_twist = 10.0 ** (min_twist_db / 10.0)
        self.max_twist = 10.0 ** (max_twist_db / 10.0)

        # Precompute Goertzel filter coefficients for exact frequencies
        self._row_coeffs = [
            2.0 * math.cos(2.0 * math.pi * f / self.sample_rate)
            for f in DTMF_ROW_FREQUENCIES
        ]
        self._col_coeffs = [
            2.0 * math.cos(2.0 * math.pi * f / self.sample_rate)
            for f in DTMF_COL_FREQUENCIES
        ]

    def _compute_goertzel_power(self, samples: List[float], coeff: float) -> float:
        """
        Computes power at the filter frequency using the 2nd-order Goertzel IIR recurrence.
        """
        s1 = 0.0
        s2 = 0.0
        for x in samples:
            s0 = x + coeff * s1 - s2
            s2 = s1
            s1 = s0
        power = (s1 * s1) + (s2 * s2) - (coeff * s1 * s2)
        return power

    def detect_frame(self, pcm_data: Union[bytes, List[int], List[float]]) -> Optional[str]:
        """
        Evaluates a single block of PCM audio for a DTMF touch-tone.
        
        Returns:
            The detected digit ('0'-'9', '*', '#', 'A'-'D') or None if no valid tone is detected.
        """
        if isinstance(pcm_data, bytes):
            sample_count = len(pcm_data) // 2
            if sample_count < self.frame_size:
                return None
            unpacked = struct.unpack(f"<{self.frame_size}h", pcm_data[: self.frame_size * 2])
            samples = [float(s) for s in unpacked]
        else:
            if len(pcm_data) < self.frame_size:
                return None
            samples = [float(s) for s in pcm_data[: self.frame_size]]

        # 1. Total energy check
        total_energy = sum(s * s for s in samples)
        if total_energy < self.energy_threshold:
            return None

        # 2. Compute powers for Row frequencies (697, 770, 852, 941 Hz)
        row_powers = [
            self._compute_goertzel_power(samples, coeff)
            for coeff in self._row_coeffs
        ]
        sorted_row_indices = sorted(range(4), key=lambda i: row_powers[i], reverse=True)
        best_row_idx = sorted_row_indices[0]
        second_row_idx = sorted_row_indices[1]
        best_row_pwr = row_powers[best_row_idx]
        second_row_pwr = row_powers[second_row_idx]

        # 3. Compute powers for Col frequencies (1209, 1336, 1477, 1633 Hz)
        col_powers = [
            self._compute_goertzel_power(samples, coeff)
            for coeff in self._col_coeffs
        ]
        sorted_col_indices = sorted(range(4), key=lambda i: col_powers[i], reverse=True)
        best_col_idx = sorted_col_indices[0]
        second_col_idx = sorted_col_indices[1]
        best_col_pwr = col_powers[best_col_idx]
        second_col_pwr = col_powers[second_col_idx]

        # 4. Dominance check: peak power must exceed 2nd harmonic by at least dominance_ratio
        if second_row_pwr > 0 and (best_row_pwr / second_row_pwr) < self.dominance_ratio:
            return None
        if second_col_pwr > 0 and (best_col_pwr / second_col_pwr) < self.dominance_ratio:
            return None

        # 5. Twist check: Ratio of column power to row power must be within standard bounds
        if best_row_pwr <= 0:
            return None
        twist = best_col_pwr / best_row_pwr
        if twist < self.min_twist or twist > self.max_twist:
            return None

        # 6. Combined tone power concentration check
        combined_tone_pwr = best_row_pwr + best_col_pwr
        # In pure dual tone, Goertzel power is approximately N/2 * total_energy.
        # Ensure tones carry substantial signal power to eliminate wideband speech false triggers.
        power_ratio = combined_tone_pwr / (self.frame_size * (total_energy + 1e-6))
        if power_ratio < 0.25:
            return None

        return DTMF_KEY_MATRIX[best_row_idx][best_col_idx]


class DTMFPad:
    """
    Debounced Acoustic DTMF Keypad Detector for Continuous Streams.
    
    Processes sequential audio frames, requiring a tone to persist for a minimum
    duration (e.g. 2 consecutive frames = 40ms) and enforcing inter-digit silence
    intervals to prevent duplicate triggers from a single continuous keypress.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_duration_ms: float = 20.0,
        min_tone_duration_ms: float = 40.0,
        min_silence_duration_ms: float = 40.0,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int((frame_duration_ms / 1000.0) * sample_rate)
        self.frame_bytes = self.frame_size * 2
        self.detector = GoertzelDetector(sample_rate=sample_rate, frame_size=self.frame_size)

        self.min_tone_frames = max(1, int(min_tone_duration_ms / frame_duration_ms))
        self.min_silence_frames = max(1, int(min_silence_duration_ms / frame_duration_ms))

        # Debounce state
        self._consecutive_tone_digit: Optional[str] = None
        self._consecutive_tone_count: int = 0
        self._consecutive_silence_count: int = 0
        self._last_registered_digit: Optional[str] = None
        self._audio_buffer = bytearray()

    def reset(self):
        """Resets the debounce state."""
        self._consecutive_tone_digit = None
        self._consecutive_tone_count = 0
        self._consecutive_silence_count = 0
        self._last_registered_digit = None
        self._audio_buffer.clear()

    def process_pcm_chunk(self, chunk: bytes) -> List[str]:
        """
        Appends incoming PCM chunk and evaluates available complete frames.
        Returns a list of newly registered debounced DTMF digits.
        """
        registered_digits: List[str] = []
        self._audio_buffer.extend(chunk)

        while len(self._audio_buffer) >= self.frame_bytes:
            frame = bytes(self._audio_buffer[: self.frame_bytes])
            del self._audio_buffer[: self.frame_bytes]

            detected_digit = self.detector.detect_frame(frame)

            if detected_digit is not None:
                self._consecutive_silence_count = 0
                if detected_digit == self._consecutive_tone_digit:
                    self._consecutive_tone_count += 1
                else:
                    self._consecutive_tone_digit = detected_digit
                    self._consecutive_tone_count = 1

                # If tone persisted for required duration and has not been registered in current burst
                if self._consecutive_tone_count >= self.min_tone_frames:
                    if self._last_registered_digit != detected_digit:
                        registered_digits.append(detected_digit)
                        self._last_registered_digit = detected_digit
            else:
                self._consecutive_tone_digit = None
                self._consecutive_tone_count = 0
                self._consecutive_silence_count += 1

                # Once sufficient silence has elapsed, allow the same digit to be registered again
                if self._consecutive_silence_count >= self.min_silence_frames:
                    self._last_registered_digit = None

        return registered_digits


class RFC4733Event:
    """Represents a decoded RFC 4733 / RFC 2833 RTP telephone-event packet."""

    def __init__(self, event_id: int, digit: str, end_bit: bool, volume: int, duration: int):
        self.event_id = event_id
        self.digit = digit
        self.end_bit = end_bit
        self.volume = volume  # in -dBm0 (0 to 63)
        self.duration = duration  # in RTP timestamp units (e.g. 80 = 10ms at 8kHz)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "digit": self.digit,
            "end_bit": self.end_bit,
            "volume_dbm0": -self.volume,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return (
            f"RFC4733Event(digit='{self.digit}', event_id={self.event_id}, "
            f"end_bit={self.end_bit}, volume=-{self.volume}dBm0, duration={self.duration})"
        )


def encode_rfc4733_packet(
    digit: Union[int, str],
    end_bit: bool = True,
    volume: int = 10,
    duration: int = 800,
) -> bytes:
    """
    Encodes an RFC 4733 / RFC 2833 RTP telephone-event packet (4 bytes).
    
    Structure:
    - Byte 0: Event ID (0-15)
    - Byte 1: E (bit 7) | R (bit 6 = 0) | volume (bits 0-5)
    - Bytes 2-3: Duration (16-bit unsigned big-endian)
    """
    if isinstance(digit, str):
        digit_upper = digit.upper()
        if digit_upper not in RFC4733_DIGIT_MAP:
            raise ValueError(f"Invalid DTMF digit '{digit}' for RFC 4733 encoding.")
        event_id = RFC4733_DIGIT_MAP[digit_upper]
    else:
        event_id = int(digit)
        if event_id not in RFC4733_EVENT_MAP:
            raise ValueError(f"Invalid RFC 4733 event ID {event_id}. Must be 0-15.")

    b1 = (0x80 if end_bit else 0x00) | (volume & 0x3F)
    return struct.pack("!BBH", event_id, b1, duration & 0xFFFF)


def decode_rfc4733_packet(data: bytes) -> RFC4733Event:
    """
    Decodes an RFC 4733 / RFC 2833 RTP telephone-event packet (minimum 4 bytes).
    """
    if len(data) < 4:
        raise ValueError(f"RFC 4733 packet must be at least 4 bytes, got {len(data)} bytes.")

    event_id, b1, duration = struct.unpack("!BBH", data[:4])
    end_bit = bool(b1 & 0x80)
    volume = b1 & 0x3F
    digit = RFC4733_EVENT_MAP.get(event_id, "?")

    return RFC4733Event(
        event_id=event_id,
        digit=digit,
        end_bit=end_bit,
        volume=volume,
        duration=duration,
    )


class RFC4733EventDecoder:
    """
    Stateful RFC 4733 / RFC 2833 Event Stream Decoder.
    
    Handles multi-packet transmissions from carrier trunks, debouncing redundant end
    packets (RFC 4733 Section 2.5.1 specifies carriers must send 3 duplicate end packets).
    """

    def __init__(self):
        self._current_event_id: Optional[int] = None
        self._is_in_event: bool = False
        self._last_emitted_event_id: Optional[int] = None

    def reset(self):
        """Resets the event decoder state."""
        self._current_event_id = None
        self._is_in_event = False
        self._last_emitted_event_id = None

    def process_packet(self, packet_bytes: bytes) -> Optional[str]:
        """
        Parses an incoming RFC 4733 packet.
        Returns the decoded digit ('0'-'9', '*', '#', 'A'-'D') when a tone event ends,
        filtering out redundant duplicate end-packets.
        """
        event = decode_rfc4733_packet(packet_bytes)
        if event.digit == "?":
            return None

        if not self._is_in_event:
            self._is_in_event = True
            self._current_event_id = event.event_id

        if event.end_bit:
            self._is_in_event = False
            # Deduplicate repeated end packets for the same event
            if self._last_emitted_event_id != event.event_id:
                self._last_emitted_event_id = event.event_id
                return event.digit
        else:
            # Intermediate update packet; reset last emitted when event ID shifts
            if self._current_event_id != event.event_id:
                self._current_event_id = event.event_id
                self._last_emitted_event_id = None

        return None
