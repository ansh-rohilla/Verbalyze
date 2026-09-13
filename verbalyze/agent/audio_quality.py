"""
verbalyze/agent/audio_quality.py

Automated Human-Likeness Quality Gate & Audio Evaluation for Verbalyze:
Evaluates cadence, pause naturalness, prosodic dynamics, harmonic smoothness,
and signal integrity. Only accepts speech audio that achieves a verified quality threshold.
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pydub


@dataclass
class AudioQualityReport:
    score: float                # 0.00 to 1.00
    mos_equivalent: float       # 1.00 to 5.00
    passed: bool                # True if score >= min_human_likeness
    cadence_score: float        # 0.00 to 1.00
    pause_score: float          # 0.00 to 1.00
    prosody_score: float        # 0.00 to 1.00
    smoothness_score: float     # 0.00 to 1.00
    signal_score: float         # 0.00 to 1.00
    wpm: float                  # Words per minute
    duration_seconds: float     # Audio length in seconds
    pause_ratio: float          # Silence frames / total frames
    feedback: str               # Actionable description

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HumanLikenessScorer:
    """Quantitative acoustic & conversational evaluator for telephony speech."""

    def __init__(self, min_threshold: float = 0.80):
        self.min_threshold = min_threshold

    def evaluate(self, audio_path: str, transcript: str) -> AudioQualityReport:
        """Evaluates an audio file against a text transcript and produces a quality report."""
        if not os.path.exists(audio_path):
            return AudioQualityReport(
                score=0.0,
                mos_equivalent=1.0,
                passed=False,
                cadence_score=0.0,
                pause_score=0.0,
                prosody_score=0.0,
                smoothness_score=0.0,
                signal_score=0.0,
                wpm=0.0,
                duration_seconds=0.0,
                pause_ratio=0.0,
                feedback="Audio file does not exist."
            )

        try:
            seg = pydub.AudioSegment.from_file(audio_path)
        except Exception as e:
            return AudioQualityReport(
                score=0.0,
                mos_equivalent=1.0,
                passed=False,
                cadence_score=0.0,
                pause_score=0.0,
                prosody_score=0.0,
                smoothness_score=0.0,
                signal_score=0.0,
                wpm=0.0,
                duration_seconds=0.0,
                pause_ratio=0.0,
                feedback=f"Failed to decode audio: {e}"
            )

        dur = len(seg) / 1000.0
        if dur <= 0.1:
            return AudioQualityReport(
                score=0.0,
                mos_equivalent=1.0,
                passed=False,
                cadence_score=0.0,
                pause_score=0.0,
                prosody_score=0.0,
                smoothness_score=0.0,
                signal_score=0.0,
                wpm=0.0,
                duration_seconds=dur,
                pause_ratio=0.0,
                feedback="Audio duration is too short (empty or click)."
            )

        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        max_val = float(2 ** (seg.sample_width * 8 - 1))

        # ----------------------------------------------------------------------
        # 1. Cadence Score (30% weight) - Target: 80 - 150 WPM for Indic Speech
        # ----------------------------------------------------------------------
        words = max(1, len(transcript.split()))
        wpm = words / (dur / 60.0)

        if 80 <= wpm <= 150:
            s_cadence = 1.0
        elif wpm < 80:
            s_cadence = max(0.3, 1.0 - (80 - wpm) / 50.0)
        else:
            s_cadence = max(0.3, 1.0 - (wpm - 150) / 60.0)

        # ----------------------------------------------------------------------
        # 2. Pause & Breath Phrasing (25% weight)
        # ----------------------------------------------------------------------
        frame_len = int(seg.frame_rate * 0.025) # 25ms frames
        num_frames = len(samples) // frame_len
        energies = np.array([np.sqrt(np.mean(samples[i*frame_len:(i+1)*frame_len]**2)) for i in range(num_frames)])
        max_e = np.max(energies) if len(energies) > 0 else 1.0

        is_silent = energies < (max_e * 0.05)
        pause_ratio = float(np.sum(is_silent) / max(1, num_frames))

        # Measure variance in pause lengths (natural speech has diverse pause lengths)
        silent_runs = []
        curr_run = 0
        for s in is_silent:
            if s:
                curr_run += 1
            elif curr_run > 0:
                silent_runs.append(curr_run * 0.025)
                curr_run = 0
        if curr_run > 0:
            silent_runs.append(curr_run * 0.025)

        pause_var = float(np.std(silent_runs)) if len(silent_runs) > 1 else 0.0

        # Ratio score: optimal between 18% and 38%
        if 0.18 <= pause_ratio <= 0.38:
            r_score = 1.0
        elif pause_ratio < 0.18:
            r_score = max(0.3, pause_ratio / 0.18)
        else:
            r_score = max(0.3, 1.0 - (pause_ratio - 0.38) / 0.3)

        # Pause variance score (avoids rigid, equal robotic silence intervals)
        v_score = min(1.0, max(0.25, pause_var / 0.25))
        s_pause = 0.65 * r_score + 0.35 * v_score

        # ----------------------------------------------------------------------
        # 3. Prosodic Energy & Dynamic Inflection (25% weight)
        # ----------------------------------------------------------------------
        speech_e = energies[~is_silent]
        if len(speech_e) > 0:
            dyn_std = float(np.std(speech_e) / (np.mean(speech_e) + 1e-6))
            if 0.35 <= dyn_std <= 0.72:
                s_prosody = 1.0
            elif dyn_std < 0.35:
                # Monotone flat speech
                s_prosody = max(0.3, dyn_std / 0.35)
            else:
                # Wildly erratic volume fluctuations
                s_prosody = max(0.4, 1.0 - (dyn_std - 0.72) / 0.5)
        else:
            s_prosody = 0.2

        # ----------------------------------------------------------------------
        # 4. Vocal Continuity & Harmonic Smoothness (10% weight)
        # ----------------------------------------------------------------------
        diffs = np.abs(np.diff(energies)) / (max_e + 1e-6)
        jitter = float(np.mean(diffs))
        if jitter < 0.085:
            s_smooth = 1.0
        else:
            # Penalizes mechanical concatenative step jumps
            s_smooth = max(0.2, 1.0 - (jitter - 0.085) * 8.0)

        # ----------------------------------------------------------------------
        # 5. Signal Integrity (10% weight)
        # ----------------------------------------------------------------------
        clip_count = np.sum(np.abs(samples) >= (max_val - 2))
        clip_ratio = clip_count / max(1, len(samples))
        s_signal = 1.0 if clip_ratio < 0.001 else max(0.2, 1.0 - clip_ratio * 100)

        # ----------------------------------------------------------------------
        # Composite Human-Likeness Calculation
        # ----------------------------------------------------------------------
        composite = round(
            0.30 * s_cadence +
            0.25 * s_pause +
            0.25 * s_prosody +
            0.10 * s_smooth +
            0.10 * s_signal,
            3
        )
        mos = round(1.0 + 4.0 * composite, 2)
        passed = bool(composite >= self.min_threshold)

        # Actionable feedback
        issues = []
        if s_cadence < 0.8:
            issues.append(f"Pacing is {'too fast' if wpm > 150 else 'too slow'} ({round(wpm, 1)} WPM)")
        if s_pause < 0.8:
            issues.append(f"Pause distribution unnatural ({round(pause_ratio*100, 1)}% silence)")
        if s_prosody < 0.8:
            issues.append("Prosody lacks inflection (monotone dynamic range)")
        if s_smooth < 0.8:
            issues.append("Frame transition roughness detected")
        if s_signal < 0.8:
            issues.append("Digital clipping detected")

        feedback = " ✓ Excellent human-likeness." if not issues else f"Issues: {', '.join(issues)}"

        return AudioQualityReport(
            score=composite,
            mos_equivalent=mos,
            passed=passed,
            cadence_score=round(s_cadence, 3),
            pause_score=round(s_pause, 3),
            prosody_score=round(s_prosody, 3),
            smoothness_score=round(s_smooth, 3),
            signal_score=round(s_signal, 3),
            wpm=round(wpm, 1),
            duration_seconds=round(dur, 2),
            pause_ratio=round(pause_ratio, 3),
            feedback=feedback
        )
