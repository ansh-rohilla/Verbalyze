# Indic ASR & Speech Benchmark Leaderboard

Benchmark evaluation across real-world Indic telephony scenarios: code-mixing, numeric normalization, abbreviations, and domain vocabulary.

| Model | Language | Samples | Overall WER (%) | CER (%) | Digit Accuracy (%) | Acronym Retention (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mock-Whisper-Base** | `hi` | 120 | 3.91% | 3.81% | 96.97% | 100.0% |

### Scenario Breakdown

| Scenario | Samples | Avg WER (%) | Digit Accuracy (%) |
|:---|:---:|:---:|:---:|
| `spoken_number_patterns` | 10 | 1.11% | 100.0% |
| `language_script_consistency` | 10 | 1.25% | 100.0% |
| `units_measurements` | 10 | 5.32% | 100.0% |
| `noisy_or_real_world_audio` | 10 | 6.56% | 100.0% |
| `normal_native_speech` | 10 | 1.25% | 100.0% |
| `code_mixed_speech` | 10 | 8.0% | 100.0% |
| `abbreviations_acronyms` | 10 | 6.0% | 100.0% |
| `similar_language_confusion` | 10 | 0.83% | 100.0% |
| `named_entities` | 10 | 4.54% | 100.0% |
| `english_word_retention` | 10 | 5.22% | 100.0% |
| `numeric_normalization` | 10 | 5.28% | 90.0% |
| `domain_specific_terms` | 10 | 1.6% | 100.0% |
