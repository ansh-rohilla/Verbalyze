# Synthetic Conversational Data Generation

A unified Python generator to build synthetic voice assistant datasets for regional Indian languages and English. 

The generation pipeline dynamically builds system prompts, scenario distributions, and turn flows for any of the 10 supported target languages, matching Verbalyze's voice assistant training and validation layout.

---

## Supported Languages

You can generate data for any of the following languages by specifying their language code:
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
├── generate_dataset.py               # Unified dataset generator (supports 10 languages)
├── generate_gujarati_dataset.py      # Standalone Gujarati generator script
├── generate_english_dataset.py       # Standalone English generator script
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

## Dataset Characteristics
Each generated dataset outputs a JSON array matching the target **276-dialogue distribution**:
* **Core Scenarios (64.5%)**: Normal (71), Hinglish/Gujlish Code-Switch (30), Emotional (18), STT Errors (13), Interruptions (10), Barge-ins (12), Corrections (16), Multi-turn (8)
* **Switch & Confusion (19.2%)**: Switch to English (21), Switch to Hindi (9), Switch to any regional language (5), Switch back (8), Wrong language response (7), Ambiguous language detection (3)
* **Other Languages (16.3%)**: Replicates baseline data for 5 other regional languages (Totaling 45 dialogues: 26, 6, 4, 4, 3, 2). The script dynamically assigns these other languages from the remaining supported list.

---

## Usage

All scripts are completely self-contained, use Python's standard libraries, and connect directly via REST requests to the Google Gemini and OpenAI APIs (no external client package dependencies).

### 1. Offline Test (Mock Mode)
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

### 2. Generating with Gemini API
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

### 3. Generating with OpenAI API
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

---

## Advanced Options

* `--scale <float>`: Multiplies the output size (e.g. `--scale 3.2` will scale the baseline 276 up to ~880 conversations while maintaining the exact scenario distribution proportions).
* `--delay <seconds>`: Adds a sleep interval between API calls (default is 1.5s) to stay within API rate limit quotas.
* **Auto-Resume**: If a run is interrupted, running the script again with the same parameters will pick up from the temporary `*_checkpoint.json` file.
