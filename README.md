# Synthetic Conversational & STT Data Generation

A suite of unified Python tools to generate synthetic datasets for voice assistant training and Speech-to-Text (STT) validation across 12 regional Indian languages and English.

---

## Supported Languages (12)

You can generate data for any of the following languages:
* **Assamese** (`as`)
* **Bengali** (`bn`)
* **English** (`en`)
* **Gujarati** (`gu`)
* **Hindi** (`hi`)
* **Kannada** (`kn`)
* **Malayalam** (`ml`)
* **Marathi** (`mr`)
* **Odia** (`or`)
* **Punjabi** (`pa`)
* **Tamil** (`ta`)
* **Telugu** (`te`)

---

## File Structure

```
├── .gitignore
├── README.md
├── generate_dataset.py               # Unified conversation generator (10 languages)
├── generate_gujarati_dataset.py      # Standalone Gujarati conversation generator
├── generate_english_dataset.py       # Standalone English conversation generator
├── generate_stt_data.py              # Scenario-wise STT data and audio generator (12 languages)
├── dataset_En.json                   # Pre-generated English dataset (1,000 dialogues)
├── dataset_Gu.json                   # Pre-generated Gujarati dataset (1,000 dialogues)
├── dataset_Hi.json                   # Pre-generated Hindi dataset (1,000 dialogues)
├── dataset_Kn.json                   # Pre-generated Kannada dataset (1,000 dialogues)
├── dataset_Ml.json                   # Pre-generated Malayalam dataset (1,000 dialogues)
├── dataset_Mr.json                   # Pre-generated Marathi dataset (1,000 dialogues)
├── dataset_Or.json                   # Pre-generated Odia dataset (1,000 dialogues)
├── dataset_Pa.json                   # Pre-generated Punjabi dataset (1,000 dialogues)
├── dataset_Ta.json                   # Pre-generated Tamil dataset (1,000 dialogues)
└── dataset_Te.json                   # Pre-generated Telugu dataset (1,000 dialogues)
```

---

## Part 1: Voice Assistant Dialogues

Each generated conversation dataset outputs a JSON array matching the target **276-dialogue distribution**:
* **Core Scenarios (64.5%)**: Normal (71), Hinglish/Gujlish Code-Switch (30), Emotional (18), STT Errors (13), Interruptions (10), Barge-ins (12), Corrections (16), Multi-turn (8)
* **Switch & Confusion (19.2%)**: Switch to English (21), Switch to Hindi (9), Switch to any regional language (5), Switch back (8), Wrong language response (7), Ambiguous language detection (3)
* **Other Languages (16.3%)**: Replicates baseline data for 5 other regional languages (Totaling 45 dialogues: 26, 6, 4, 4, 3, 2). The script dynamically assigns these other languages from the remaining supported list.

### Usage

All scripts are completely self-contained, use Python's standard libraries, and connect directly via REST requests to the Google Gemini and OpenAI APIs (no external client package dependencies).

#### 1. Offline Test (Mock Mode)
Run a fast, local generation without using API credits (useful for checking output format):
* **Using the Unified Generator**:
  ```bash
  python3 generate_dataset.py --lang gu --run-mock --output dataset_Gu.json
  python3 generate_dataset.py --lang en --run-mock --output dataset_En.json
  ```
* **Using the Standalone Scripts**:
  ```bash
  python3 generate_gujarati_dataset.py --run-mock --output dataset_Gu.json
  python3 generate_english_dataset.py --run-mock --output dataset_En.json
  ```

#### 2. Generating with Gemini API
Set your key and select the target language code:
* **Unified**:
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  python3 generate_dataset.py --lang hi --provider gemini --model gemini-1.5-flash --output dataset_Hi.json
  ```
* **Standalone Gujarati**:
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  python3 generate_gujarati_dataset.py --provider gemini --model gemini-1.5-flash --output dataset_Gu.json
  ```

#### 3. Generating with OpenAI API
Set your key and select the target language code:
* **Unified**:
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  python3 generate_dataset.py --lang ta --provider openai --model gpt-4o-mini --output dataset_Ta.json
  ```
* **Standalone English**:
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  python3 generate_english_dataset.py --provider openai --model gpt-4o-mini --output dataset_En.json
  ```

#### Advanced Options

* `--scale <float>`: Multiplies the output size (e.g. `--scale 3.2` will scale the baseline 276 up to ~880 conversations while maintaining the exact scenario distribution proportions).
* `--delay <seconds>`: Adds a sleep interval between API calls (default is 1.5s) to stay within API rate limit quotas.
* **Auto-Resume**: If a run is interrupted, running the script again with the same parameters will pick up from the temporary `*_checkpoint.json` file.

---

## Part 2: Scenario-Wise STT Data & Audio Generator

The [`generate_stt_data.py`](file:///Users/anshrohilla/Documents/Verbalyze/generate_stt_data.py) script generates scenario-weighted, normalized speech transcripts mapped to synthesized audio files for model validation and training. It supports all 12 languages.

### Scenario Distribution Profiles
1. **normal_native_speech** (20%)
2. **code_mixed_speech** (15%)
3. **numeric_normalization** (12%)
4. **spoken_number_patterns** (8%)
5. **units_measurements** (8%)
6. **abbreviations_acronyms** (8%)
7. **named_entities** (8%)
8. **english_word_retention** (6%)
9. **language_script_consistency** (5%)
10. **similar_language_confusion** (5%)
11. **domain_specific_terms** (3%)
12. **noisy_or_real_world_audio** (2%)

### Usage

The script generates text transcripts instantly. Optional audio synthesis requires the `gTTS` library.

#### 1. Setup (Optional - for Audio Synthesis)
If you want the script to automatically generate synthesized audio files (`.mp3`), install `gTTS`:
```bash
pip install gTTS
```

#### 2. Generate Text Transcripts only (Fast)
To generate metadata containing formatted, scenario-weighted text data (without synthesizing audio):
```bash
python3 generate_stt_data.py --lang hi --limit 1000 --output-dir stt_dataset
```
This generates `stt_dataset/hi/metadata_hi.csv`.

#### 3. Generate Transcripts AND Audio (Synthesized)
To generate text data and automatically synthesize the audio files:
```bash
python3 generate_stt_data.py --lang gu --limit 100 --synthesize --output-dir stt_dataset
```
This creates:
- `stt_dataset/gu/metadata_gu.csv` containing metadata mapping text to audio files.
- `stt_dataset/gu/audio/` directory filled with synthesized audio files.

---

## Part 3: Indic Voice AI Suite (`verbalyze`)

A modular framework converting Verbalyze synthetic data into production benchmarks, fine-tuned Small Language Models (SLMs), and telephony voicebots.

```
verbalyze/
├── pipeline/
│   ├── stt_exporter.py       # Packages 172.8k STT dataset for Hugging Face Hub
│   └── dialogue_exporter.py  # Formats 16.3k conversations into ChatML & ShareGPT
├── benchmark/
│   ├── evaluator.py          # ASR evaluation harness across 11 scenarios
│   └── metrics.py            # WER, CER, Digit Accuracy, Acronym Retention
├── agent/
│   ├── voice_bot.py          # Telephony collection agent runtime
│   ├── tools.py              # Telephony tools (disconnect_tool, payment links)
│   └── audio_engine.py       # Edge-TTS neural audio synthesis & playback
└── telephony/
    └── server.py             # FastAPI Exotel/Twilio SIP webhook bridge
```

### 1. Phase 1: Benchmark Speech Models (`verbalyze benchmark`)
Evaluate any speech engine (Whisper, Sarvam, Google, Azure) against the 172,800 STT scenario dataset:
```bash
# Evaluate on Hindi across 11 edge scenarios
python3 -m verbalyze.cli benchmark --lang hi --samples 20 --output-report LEADERBOARD.md

# Package the 172.8k dataset for Hugging Face
python3 -m verbalyze.cli export-stt --output-dir data/stt_bench
```

### 2. Phase 2: Indic Voice SLM Fine-Tuning (`train_voice_slm.py`)
Compile 16,370 multi-turn voice conversations and fine-tune low-latency 1B–3B models (Llama 3.2 / Qwen 2.5):
```bash
# 1. Compile conversations into ChatML & ShareGPT splits
python3 -m verbalyze.cli export-dialogues --output-dir data/dialogues

# 2. Verify dataset and pipeline (Dry Run)
python3 scripts/train_voice_slm.py --dry-run

# 3. Start QLoRA Fine-Tuning on GPU
python3 scripts/train_voice_slm.py --model llama3.2-3b --epochs 3 --batch-size 4
```

### 3. Phase 3: Outbound Debt/EMI Telephony Voicebot (`verbalyze agent`)
Simulate real phone calls with the Muthoot Fincorp recovery bot with natural fillers, emotion handling, and automatic call hangup (`disconnect_tool`):
```bash
# Launch interactive terminal voicebot simulation
python3 -m verbalyze.cli agent --lang hi

# Start the FastAPI Telephony Webhook Server for Exotel / Twilio SIP trunks
python3 -m verbalyze.cli server --port 8000
```

---

## Live on Hugging Face Hub 🤗

Both datasets are publicly indexed and ready to use in the Hugging Face `datasets` library:

| Dataset | Samples | Formats | Link |
|---|:---:|:---:|:---:|
| **Verbalyze Dialogues** | **16,370** | ChatML, ShareGPT | [🤗 ansh-rohilla/verbalyze-dialogues](https://huggingface.co/datasets/ansh-rohilla/verbalyze-dialogues) |
| **Verbalyze STT Benchmark** | **172,800** | JSONL, Parquet | [🤗 ansh-rohilla/verbalyze-stt-bench](https://huggingface.co/datasets/ansh-rohilla/verbalyze-stt-bench) |

### Quickstart with Python:
```python
from datasets import load_dataset

# 1. Load multi-turn telephony conversations (12 languages)
dialogues = load_dataset("ansh-rohilla/verbalyze-dialogues", split="train")
print(f"Loaded {len(dialogues)} dialogues. Sample turn: {dialogues[0]['messages'][1]}")

# 2. Load STT scenario benchmark dataset (12 languages)
stt_bench = load_dataset("ansh-rohilla/verbalyze-stt-bench", split="test")
print(f"Loaded {len(stt_bench)} benchmark utterances. Sample: {stt_bench[0]['transcript']}")
```


