"""
verbalyze/benchmark/evaluator.py

Benchmark evaluation runner for Speech-to-Text (ASR) engines.
Evaluates model predictions across 11 Indic speech scenarios and produces
scenario-level scorecards and Markdown leaderboards.
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from verbalyze.benchmark.metrics import evaluate_batch, compute_wer, compute_cer


class BaseASRClient:
    """Base class for ASR client adapters."""
    def transcribe(self, text_or_audio: str, language: str) -> str:
        raise NotImplementedError


class MockASRClient(BaseASRClient):
    """
    Simulates real-world ASR failure modes:
    - Code-switching script confusion
    - Minor phonetic noise in digits/acronyms
    - Random dropping of filler words
    """
    def __init__(self, accuracy_level: float = 0.85, seed: int = 42):
        self.accuracy = accuracy_level
        self.rng = random.Random(seed)

    def transcribe(self, text_or_audio: str, language: str) -> str:
        words = text_or_audio.split()
        out_words = []
        for w in words:
            if self.rng.random() < self.accuracy:
                out_words.append(w)
            else:
                # Introduce realistic perturbation
                action = self.rng.choice(["drop", "swap", "case"])
                if action == "drop":
                    continue
                elif action == "case":
                    out_words.append(w.lower() if w.isupper() else w.title())
                else:
                    out_words.append(w)
        return " ".join(out_words) if out_words else text_or_audio


class BenchmarkRunner:
    """Orchestrates scenario evaluations across languages."""
    def __init__(self, dataset_base_dir: str = "stt_dataset"):
        self.base_dir = Path(dataset_base_dir)

    def load_scenario_samples(
        self,
        lang: str,
        limit_per_scenario: int = 50,
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, str]]]:
        """Loads balanced test samples grouped by scenario."""
        csv_path = self.base_dir / lang / f"metadata_{lang}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset for language '{lang}': {csv_path}")

        grouped: Dict[str, List[Dict[str, str]]] = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sc = row.get("scenario", "general")
                if scenarios and sc not in scenarios:
                    continue
                grouped.setdefault(sc, [])
                if len(grouped[sc]) < limit_per_scenario:
                    grouped[sc].append(row)
        return grouped

    def run_benchmark(
        self,
        model_name: str,
        client: BaseASRClient,
        lang: str = "hi",
        limit_per_scenario: int = 50,
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluates an ASR client across all scenarios for a given language."""
        samples_by_sc = self.load_scenario_samples(lang, limit_per_scenario, scenarios)
        
        scenario_results = {}
        all_refs = []
        all_hyps = []

        print(f"\n========================================================")
        print(f"Running Verbalyze STT Benchmark: {model_name.upper()}")
        print(f"Language: {lang} | Samples per Scenario: {limit_per_scenario}")
        print(f"========================================================")

        for sc, samples in samples_by_sc.items():
            refs = [s["transcript"] for s in samples]
            hyps = [client.transcribe(s["transcript"], language=lang) for s in samples]
            
            sc_metrics = evaluate_batch(refs, hyps)
            scenario_results[sc] = sc_metrics

            all_refs.extend(refs)
            all_hyps.extend(hyps)

            print(f"  {sc:<30} WER: {sc_metrics['wer']:>5.1f}% | CER: {sc_metrics['cer']:>5.1f}% | Digits: {sc_metrics['digit_accuracy']:>5.1f}%")

        aggregate = evaluate_batch(all_refs, all_hyps)
        print("--------------------------------------------------------")
        print(f"  OVERALL AGGREGATE           WER: {aggregate['wer']:>5.1f}% | CER: {aggregate['cer']:>5.1f}% | Digits: {aggregate['digit_accuracy']:>5.1f}%")
        print("========================================================\n")

        return {
            "model_name": model_name,
            "language": lang,
            "aggregate": aggregate,
            "scenarios": scenario_results,
        }

    def generate_leaderboard_markdown(self, benchmark_results: List[Dict[str, Any]], output_file: Optional[str] = None) -> str:
        """Generates a GitHub-flavored Markdown leaderboard table."""
        lines = [
            "# Indic ASR & Speech Benchmark Leaderboard",
            "",
            "Benchmark evaluation across real-world Indic telephony scenarios: code-mixing, numeric normalization, abbreviations, and domain vocabulary.",
            "",
            "| Model | Language | Samples | Overall WER (%) | CER (%) | Digit Accuracy (%) | Acronym Retention (%) |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for res in benchmark_results:
            agg = res["aggregate"]
            lines.append(
                f"| **{res['model_name']}** | `{res['language']}` | {agg['samples']} | {agg['wer']}% | {agg['cer']}% | {agg['digit_accuracy']}% | {agg['acronym_retention']}% |"
            )

        lines.extend([
            "",
            "### Scenario Breakdown",
            "",
            "| Scenario | Samples | Avg WER (%) | Digit Accuracy (%) |",
            "|:---|:---:|:---:|:---:|",
        ])

        # Aggregate across models for each scenario
        if benchmark_results:
            first_res = benchmark_results[0]
            for sc, data in first_res["scenarios"].items():
                lines.append(f"| `{sc}` | {data['samples']} | {data['wer']}% | {data['digit_accuracy']}% |")

        content = "\n".join(lines) + "\n"
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[Benchmark] Saved Leaderboard report to: {output_file}")
        return content
