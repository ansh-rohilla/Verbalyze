# Indic ASR & Speech Benchmark Leaderboard

Benchmark evaluation across real-world Indic telephony scenarios: code-mixing, numeric normalization, abbreviations, and domain vocabulary.

| Model | Language | Samples | Overall WER (%) | CER (%) | Digit Accuracy (%) | Acronym Retention (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mock-Whisper-Base** | `hi` | 240 | 3.26% | 3.09% | 97.14% | 100.0% |

### Scenario Breakdown

| Scenario | Samples | Avg WER (%) | Digit Accuracy (%) |
|:---|:---:|:---:|:---:|
| `spoken_number_patterns` | 20 | 1.56% | 100.0% |
| `language_script_consistency` | 20 | 5.77% | 100.0% |
| `units_measurements` | 20 | 2.94% | 95.0% |
| `noisy_or_real_world_audio` | 20 | 5.14% | 100.0% |
| `normal_native_speech` | 20 | 3.62% | 100.0% |
| `code_mixed_speech` | 20 | 4.75% | 100.0% |
| `abbreviations_acronyms` | 20 | 2.54% | 100.0% |
| `similar_language_confusion` | 20 | 3.56% | 100.0% |
| `named_entities` | 20 | 3.44% | 100.0% |
| `english_word_retention` | 20 | 0.62% | 100.0% |
| `numeric_normalization` | 20 | 2.8% | 95.0% |
| `domain_specific_terms` | 20 | 2.35% | 100.0% |
