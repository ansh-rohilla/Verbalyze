---
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
| Assamese | `as` | 14,400 | Bengali-Assamese |
| Bengali | `bn` | 14,400 | Bengali |
| English (Indian) | `en` | 14,400 | Latin |
| Gujarati | `gu` | 14,400 | Gujarati |
| Hindi | `hi` | 14,400 | Devanagari |
| Kannada | `kn` | 14,400 | Kannada |
| Malayalam | `ml` | 14,400 | Malayalam |
| Marathi | `mr` | 14,400 | Devanagari |
| Odia | `or` | 14,400 | Odia |
| Punjabi | `pa` | 14,400 | Gurmukhi |
| Tamil | `ta` | 14,400 | Tamil |
| Telugu | `te` | 14,400 | Telugu |
| **Total** | **12** | **172,800** | |

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
{
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
}
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
