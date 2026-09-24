"""
verbalyze/telephony/jitter_buffer.py

Adaptive Telecom Jitter Buffer & Packet Loss Concealment (PLC) Engine.
Optimized for variable-latency Indian cellular networks (2G/3G/4G GSM corridors).
Implements RFC 3550 Inter-Arrival Jitter calculations and dynamic playout adaptation.
Zero-emoji compliant.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from verbalyze.telephony.packet_loss_concealer import (
    PacketLossConcealer,
    PLCTelemetry,
    G711AppendixIPLC,
)


@dataclass
class JitterBufferPacket:
    """Represents an individual audio packet received over RTP or WebSocket."""
    sequence_number: int
    timestamp_ms: float
    pcm_data: bytes
    arrival_time_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    is_concealed: bool = False


@dataclass
class JitterBufferStats:
    """Network quality and jitter telemetry for an active audio stream."""
    current_jitter_ms: float = 0.0
    average_jitter_ms: float = 0.0
    max_jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    total_packets_received: int = 0
    total_packets_expected: int = 0
    packets_lost: int = 0
    packets_reordered: int = 0
    concealed_frames_count: int = 0
    underrun_count: int = 0
    overrun_count: int = 0
    current_target_delay_ms: float = 60.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "current_jitter_ms": round(self.current_jitter_ms, 2),
            "average_jitter_ms": round(self.average_jitter_ms, 2),
            "max_jitter_ms": round(self.max_jitter_ms, 2),
            "packet_loss_rate": round(self.packet_loss_rate, 4),
            "total_packets_received": self.total_packets_received,
            "packets_lost": self.packets_lost,
            "packets_reordered": self.packets_reordered,
            "concealed_frames_count": self.concealed_frames_count,
            "underrun_count": self.underrun_count,
            "overrun_count": self.overrun_count,
            "current_target_delay_ms": round(self.current_target_delay_ms, 1),
        }


class AdaptiveJitterBuffer:
    """
    Adaptive telecom jitter buffer adhering to RFC 3550 principles.

    Absorbs network timing jitter on mobile connections, reorders packets arriving
    out of sequence, dynamically expands/contracts playout delay between min_delay_ms
    and max_delay_ms, and performs Packet Loss Concealment (PLC) when packets are dropped.
    """

    def __init__(
        self,
        frame_duration_ms: float = 20.0,
        sample_rate: int = 8000,
        bytes_per_sample: int = 2,
        min_delay_ms: float = 40.0,
        max_delay_ms: float = 200.0,
        nominal_delay_ms: float = 60.0,
        adaptation_rate: float = 0.05,
    ):
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.frame_bytes = int((sample_rate * (frame_duration_ms / 1000.0)) * bytes_per_sample)

        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.target_delay_ms = nominal_delay_ms
        self.adaptation_rate = adaptation_rate

        # Packet storage: sorted by sequence_number
        self.buffer: List[JitterBufferPacket] = []
        self.max_buffer_packets = int(max_delay_ms / frame_duration_ms) * 2

        # Sequence tracking
        self.last_played_sequence: Optional[int] = None
        self.highest_sequence_received: Optional[int] = None
        self.base_sequence: Optional[int] = None

        # RFC 3550 Jitter Estimation Variables
        self.estimated_jitter_ms: float = 0.0
        self.prev_transit_ms: Optional[float] = None
        self.jitter_samples: List[float] = []

        # Concealment memory for PLC interpolation
        self.last_valid_pcm: Optional[bytes] = None

        # Packet Loss Concealer (ITU-T G.711 Appendix I)
        self.plc = PacketLossConcealer(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
        )

        # Telemetry counters
        self.stats = JitterBufferStats(current_target_delay_ms=self.target_delay_ms)

    def calculate_rfc3550_jitter(self, packet_timestamp_ms: float, arrival_time_ms: float) -> float:
        """
        Calculates inter-arrival jitter per RFC 3550 section 6.4.1.
        D(i, j) = (R_j - R_i) - (S_j - S_i) = (R_j - S_j) - (R_i - S_i)
        J(i) = J(i-1) + (|D(i, j)| - J(i-1)) / 16
        """
        current_transit_ms = arrival_time_ms - packet_timestamp_ms
        if self.prev_transit_ms is not None:
            difference = abs(current_transit_ms - self.prev_transit_ms)
            # RFC 3550 smoothing factor 1/16 = 0.0625
            self.estimated_jitter_ms += (difference - self.estimated_jitter_ms) / 16.0
        self.prev_transit_ms = current_transit_ms
        return self.estimated_jitter_ms

    def adapt_playout_delay(self):
        """
        Dynamically adjusts playout delay based on current network jitter.
        target_delay = clamp(min_delay, estimated_jitter * 2.5 + safety_margin, max_delay)
        """
        safety_margin = self.frame_duration_ms * 1.5
        ideal_delay = self.estimated_jitter_ms * 2.5 + safety_margin
        ideal_delay = max(self.min_delay_ms, min(self.max_delay_ms, ideal_delay))

        # Smooth transition towards ideal delay to avoid abrupt pitch warps
        self.target_delay_ms += (ideal_delay - self.target_delay_ms) * self.adaptation_rate
        self.stats.current_target_delay_ms = self.target_delay_ms

    def push(
        self,
        pcm_data: bytes,
        sequence_number: int,
        timestamp_ms: Optional[float] = None,
        arrival_time_ms: Optional[float] = None,
    ):
        """
        Inserts an incoming audio packet into the jitter buffer.
        Handles packet reordering and updates RFC 3550 jitter telemetry.
        """
        now_ms = arrival_time_ms if arrival_time_ms is not None else (time.time() * 1000.0)
        ts_ms = timestamp_ms if timestamp_ms is not None else (sequence_number * self.frame_duration_ms)

        packet = JitterBufferPacket(
            sequence_number=sequence_number,
            timestamp_ms=ts_ms,
            pcm_data=pcm_data,
            arrival_time_ms=now_ms,
            is_concealed=False,
        )

        self.stats.total_packets_received += 1

        # Track first sequence
        if self.base_sequence is None:
            self.base_sequence = sequence_number

        # Check for out-of-order reordering
        if self.highest_sequence_received is not None:
            if sequence_number < self.highest_sequence_received:
                self.stats.packets_reordered += 1
            else:
                self.highest_sequence_received = sequence_number
        else:
            self.highest_sequence_received = sequence_number

        # Discard duplicate or already-played packet
        if self.last_played_sequence is not None and sequence_number <= self.last_played_sequence:
            return

        # Calculate RFC 3550 Jitter
        current_jitter = self.calculate_rfc3550_jitter(ts_ms, now_ms)
        self.jitter_samples.append(current_jitter)
        if len(self.jitter_samples) > 200:
            self.jitter_samples.pop(0)

        self.stats.current_jitter_ms = current_jitter
        self.stats.average_jitter_ms = sum(self.jitter_samples) / max(1, len(self.jitter_samples))
        self.stats.max_jitter_ms = max(self.stats.max_jitter_ms, current_jitter)

        # Update expected packets and loss rate
        if self.base_sequence is not None and self.highest_sequence_received is not None:
            self.stats.total_packets_expected = (self.highest_sequence_received - self.base_sequence) + 1
            self.stats.packets_lost = max(0, self.stats.total_packets_expected - self.stats.total_packets_received)
            if self.stats.total_packets_expected > 0:
                self.stats.packet_loss_rate = self.stats.packets_lost / float(self.stats.total_packets_expected)

        # Adapt target playout delay
        self.adapt_playout_delay()

        # Insert packet maintaining sorted sequence order
        inserted = False
        for i, existing in enumerate(self.buffer):
            if existing.sequence_number == sequence_number:
                # Duplicate packet, ignore
                return
            if existing.sequence_number > sequence_number:
                self.buffer.insert(i, packet)
                inserted = True
                break
        if not inserted:
            self.buffer.append(packet)

        # Guard against buffer overflow (overrun)
        if len(self.buffer) > self.max_buffer_packets:
            self.stats.overrun_count += 1
            dropped = self.buffer.pop(0)
            if self.last_played_sequence is None or dropped.sequence_number > self.last_played_sequence:
                self.last_played_sequence = dropped.sequence_number

    def synthesize_concealment_frame(self) -> bytes:
        """
        Packet Loss Concealment (PLC) engine.
        Synthesizes a missing 20ms audio frame using ITU-T G.711 Appendix I
        pitch-synchronous waveform replication and progressive attenuation.
        """
        self.stats.concealed_frames_count += 1
        concealed_pcm, telemetry = self.plc.conceal_frame()
        self.last_valid_pcm = concealed_pcm
        return concealed_pcm

    def pop(self, current_time_ms: Optional[float] = None) -> Tuple[bytes, bool]:
        """
        Pulls the next scheduled audio frame for playout.
        Returns: (pcm_data, is_concealed)
        """
        now_ms = current_time_ms if current_time_ms is not None else (time.time() * 1000.0)

        if not self.buffer:
            self.stats.underrun_count += 1
            concealed = self.synthesize_concealment_frame()
            return concealed, True

        next_packet = self.buffer[0]
        expected_seq = (self.last_played_sequence + 1) if self.last_played_sequence is not None else next_packet.sequence_number

        if next_packet.sequence_number == expected_seq:
            packet = self.buffer.pop(0)
            self.last_played_sequence = packet.sequence_number
            resynced_pcm, _ = self.plc.ingest_good_frame(packet.pcm_data)
            self.last_valid_pcm = resynced_pcm
            return resynced_pcm, False

        if next_packet.sequence_number > expected_seq:
            if self.last_played_sequence is not None:
                self.last_played_sequence += 1
            else:
                self.last_played_sequence = expected_seq
            concealed = self.synthesize_concealment_frame()
            return concealed, True

        self.buffer.pop(0)
        return self.pop(current_time_ms=now_ms)

    def get_all_available_pcm(self) -> bytes:
        """Flushes all queued audio sequentially into a contiguous PCM byte string."""
        chunks = []
        while self.buffer:
            frame, _ = self.pop()
            chunks.append(frame)
        return b"".join(chunks)

    def get_stats(self) -> JitterBufferStats:
        """Returns current buffer and network telemetry."""
        return self.stats

    def clear(self):
        """Resets the jitter buffer queue and PLC state."""
        self.buffer.clear()
        self.last_valid_pcm = None
        self.plc.reset()
