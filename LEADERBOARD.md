# Indic ASR & Speech Benchmark Leaderboard

Benchmark evaluation across real-world Indic telephony scenarios: code-mixing, numeric normalization, abbreviations, and domain vocabulary.

| Model | Language | Samples | Overall WER (%) | CER (%) | Digit Accuracy (%) | Acronym Retention (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mock-Whisper-Base** | `hi` | 180 | 3.83% | 3.51% | 94.12% | 87.5% |

### Scenario Breakdown

| Scenario | Samples | Avg WER (%) | Digit Accuracy (%) |
|:---|:---:|:---:|:---:|
| `spoken_number_patterns` | 15 | 2.07% | 100.0% |
| `language_script_consistency` | 15 | 2.88% | 100.0% |
| `units_measurements` | 15 | 7.09% | 86.67% |
| `noisy_or_real_world_audio` | 15 | 2.81% | 100.0% |
| `normal_native_speech` | 15 | 6.21% | 100.0% |
| `code_mixed_speech` | 15 | 4.17% | 100.0% |
| `abbreviations_acronyms` | 15 | 5.23% | 100.0% |
| `similar_language_confusion` | 15 | 1.3% | 100.0% |
| `named_entities` | 15 | 3.48% | 100.0% |
| `english_word_retention` | 15 | 0.83% | 100.0% |
| `numeric_normalization` | 15 | 6.73% | 93.33% |
| `domain_specific_terms` | 15 | 3.13% | 100.0% |
