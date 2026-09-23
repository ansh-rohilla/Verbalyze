"""
verbalyze/benchmark package

Indic Speech & ITN Benchmark Evaluation Suite:
- Word Error Rate (WER) and Character Error Rate (CER) calculation
- Entity / digit sequence accuracy metrics
- Acronym retention metrics
- Automated batch evaluation and leaderboards
"""

from verbalyze.benchmark.metrics import (
    compute_wer,
    compute_cer,
    compute_digit_accuracy,
    compute_acronym_retention,
    evaluate_batch,
)
from verbalyze.benchmark.evaluator import (
    BaseASRClient,
    MockASRClient,
    BenchmarkRunner,
)

__all__ = [
    "compute_wer",
    "compute_cer",
    "compute_digit_accuracy",
    "compute_acronym_retention",
    "evaluate_batch",
    "BaseASRClient",
    "MockASRClient",
    "BenchmarkRunner",
]
