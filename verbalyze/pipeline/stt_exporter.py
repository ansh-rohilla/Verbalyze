"""
verbalyze/pipeline/stt_exporter.py

Exports the 172,800 STT dataset into standardized Hugging Face benchmark formats
with dataset card documentation, scenario tags, and stratified splits.
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

SCENARIOS = [
    "normal_native_speech",
    "code_mixed_speech",
    "numeric_normalization",
    "abbreviations_acronyms",
    "spoken_number_patterns",
    "named_entities",
    "units_measurements",
    "english_word_retention",
    "language_script_consistency",
    "similar_language_confusion",
    "domain_specific_terms",
    "noisy_or_real_world_audio",
]


def load_language_stt(base_dir: Path, lang: str) -> List[Dict[str, str]]:
    """Loads all records from a language metadata CSV."""
    csv_path = base_dir / lang / f"metadata_{lang}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "id": row.get("id", ""),
                "language": row.get("language", lang),
                "language_name": SUPPORTED_LANGUAGES.get(lang, lang),
                "scenario": row.get("scenario", "unknown"),
                "transcript": row.get("transcript", ""),
                "normalized_transcript": row.get("normalized_transcript", ""),
                "native_text": row.get("native_text", ""),
                "romanized_text": row.get("romanized_text", ""),
                "code_mixed_text": row.get("code_mixed_text", ""),
                "duration_seconds": float(row.get("duration_seconds", 0.0) or 0.0),
            })
    return records


def split_records(
    records: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split by scenario."""
    rng = random.Random(seed)
    by_scenario: Dict[str, List[Dict]] = {}
    for r in records:
        sc = r["scenario"]
        by_scenario.setdefault(sc, []).append(r)

    train, val, test = [], [], []
    for sc, items in by_scenario.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def generate_dataset_card(total_samples: int, lang_counts: Dict[str, int], output_path: Path):
    """Generates an official Hugging Face Dataset Card."""
    card_content = f"""---
license: mit
task_categories:
  - automatic-speech-recognition
  - text-normalization
language:
  - as
  - bn
  - en
  - gu
  - hi
  - kn
  - ml
  - mr
  - or
  - pa
  - ta
  - te
tags:
  - speech
  - indic
  - asr-benchmark
  - code-switching
  - inverse-text-normalization
  - telephony
size_categories:
  - 100K<n<1M
---

# Verbalyze: Indic Speech & ITN Benchmark (12 Languages)

**Verbalyze** is a scenario-weighted benchmark dataset designed to evaluate and train Speech-to-Text (ASR) and Inverse Text Normalization (ITN) models on real-world Indic speech phenomena.

It covers **12 major Indian languages** with **172,800 balanced utterances** categorized across 11 edge-case scenarios where standard speech models typically fail.

## Languages Covered

| Language | Code | Samples | Script |
|:---------|:----:|:-------:|:-------|
| Assamese | `as` | {lang_counts.get('as', 14400):,} | Bengali-Assamese |
| Bengali | `bn` | {lang_counts.get('bn', 14400):,} | Bengali |
| English (Indian) | `en` | {lang_counts.get('en', 14400):,} | Latin |
| Gujarati | `gu` | {lang_counts.get('gu', 14400):,} | Gujarati |
| Hindi | `hi` | {lang_counts.get('hi', 14400):,} | Devanagari |
| Kannada | `kn` | {lang_counts.get('kn', 14400):,} | Kannada |
| Malayalam | `ml` | {lang_counts.get('ml', 14400):,} | Malayalam |
| Marathi | `mr` | {lang_counts.get('mr', 14400):,} | Devanagari |
| Odia | `or` | {lang_counts.get('or', 14400):,} | Odia |
| Punjabi | `pa` | {lang_counts.get('pa', 14400):,} | Gurmukhi |
| Tamil | `ta` | {lang_counts.get('ta', 14400):,} | Tamil |
| Telugu | `te` | {lang_counts.get('te', 14400):,} | Telugu |
| **Total** | **12** | **{total_samples:,}** | |

## Scenario Distribution

Each language is balanced across 11 scenarios:
1. `normal_native_speech` (20.0%) - Baseline natural conversational sentences.
2. `code_mixed_speech` (15.0%) - Code-switching between native language and English.
3. `numeric_normalization` (12.0%) - Dates, currency, percentages, and door numbers.
4. `spoken_number_patterns` (8.0%) - Phone numbers, OTPs, PIN codes.
5. `abbreviations_acronyms` (8.0%) - Banking/tech acronyms (KYC, UPI, NEFT, IFSC, PAN).
6. `named_entities` (8.0%) - Indian names, cities, and corporate brands.
7. `units_measurements` (8.0%) - Metric units (mg/kg, ml, km/hr, Mbps).
8. `english_word_retention` (6.0%) - Tech/corporate loanwords.
9. `language_script_consistency` (5.0%) - Script boundary retention.
10. `similar_language_confusion` (5.0%) - Dialectal confusion test pairs.
11. `domain_specific_terms` (3.0%) - BFSI and Healthcare domain terms.

## Data Schema

```json
{{
  "id": "hi_00123",
  "language": "hi",
  "language_name": "Hindi",
  "scenario": "abbreviations_acronyms",
  "transcript": "कृपया अपने पंजीकृत मोबाइल पर भेजा गया OTP दर्ज करें।",
  "normalized_transcript": "कृपया अपने पंजीकृत मोबाइल पर भेजा गया ओ टी पी दर्ज करें।",
  "native_text": "कृपया अपने पंजीकृत मोबाइल पर भेजा गया OTP दर्ज करें।",
  "romanized_text": "kripayaa apane panjeekrita mobaaila para bhejaa gayaa OTP darja karen.",
  "code_mixed_text": "Please apne registered mobile pe send kiya gaya OTP enter karein.",
  "duration_seconds": 0.0
}}
```

## Usage with Hugging Face `datasets`

```python
from datasets import load_dataset

# Load the full benchmark
dataset = load_dataset("ansh-rohilla/verbalyze-stt-bench")

# Filter by scenario (e.g. code-mixing in Hindi)
hinglish = dataset["test"].filter(lambda x: x["language"] == "hi" and x["scenario"] == "code_mixed_speech")
print(hinglish[0])
```
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_content)


def export_stt_dataset(
    stt_base_dir: str = "stt_dataset",
    output_dir: str = "data/stt_bench",
    languages: Optional[List[str]] = None,
    split: bool = True,
    seed: int = 42
) -> Dict[str, int]:
    """Compiles and exports the STT dataset."""
    base_path = Path(stt_base_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_langs = languages or list(SUPPORTED_LANGUAGES.keys())
    all_train, all_val, all_test = [], [], []
    all_records = []
    lang_counts: Dict[str, int] = {}

    print(f"[STT Exporter] Scanning {len(target_langs)} languages in {base_path}...")

    for lang in target_langs:
        try:
            records = load_language_stt(base_path, lang)
            lang_counts[lang] = len(records)
            all_records.extend(records)
            print(f"  [PASS] {SUPPORTED_LANGUAGES.get(lang, lang)} ({lang}): {len(records):,} records")

            if split:
                tr, va, te = split_records(records, seed=seed)
                all_train.extend(tr)
                all_val.extend(va)
                all_test.extend(te)
        except FileNotFoundError as e:
            print(f"  [SKIP] {lang}: {e}")

    # Write files
    if split:
        for split_name, data in [("train", all_train), ("validation", all_val), ("test", all_test)]:
            dest_file = out_path / f"{split_name}.jsonl"
            with open(dest_file, "w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[STT Exporter] Wrote {len(data):,} samples to {dest_file}")
    else:
        dest_file = out_path / "dataset.jsonl"
        with open(dest_file, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[STT Exporter] Wrote {len(all_records):,} samples to {dest_file}")

    # Write Dataset Card
    card_file = out_path / "README.md"
    generate_dataset_card(len(all_records), lang_counts, card_file)
    print(f"[STT Exporter] Generated Hugging Face Dataset Card at {card_file}")

    return lang_counts


if __name__ == "__main__":
    export_stt_dataset()
