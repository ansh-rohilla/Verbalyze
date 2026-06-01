# Synthetic Conversational Data Generation

This repository contains robust, self-contained Python scripts to generate synthetic voice assistant datasets for regional languages, specifically tailored for **Gujarati** and **English** (including Hinglish code-switching).

The generation structure, scenario profiles, and metadata strictly match the formatting of Verbalyze's existing regional language datasets.

---

## Repository Structure

```
├── .gitignore
├── README.md
├── generate_gujarati_dataset.py       # Gujarati synthetic generator script
└── generate_english_dataset.py        # English synthetic generator script
```

---

## Dataset Characteristics
Each generated dataset outputs a JSON array matching the target **276-dialogue distribution**:
* **Core Scenarios (64.5%)**: Normal (71), Hinglish/Gujlish Code-Switch (30), Emotional (18), STT Errors (13), Interruptions (10), Barge-ins (12), Corrections (16), Multi-turn (8)
* **Switch & Confusion (19.2%)**: Switch to English (21), Switch to Hindi (9), Switch to any regional language (5), Switch back (8), Wrong language response (7), Ambiguous language detection (3)
* **Other Languages (16.3%)**: Baseline baseline data for other languages: Hindi (26), Hindi code-switch (4), Tamil (6), Telugu (4), Malayalam (3), Marathi (2)

---

## How to Run the Generators

Both scripts are completely self-contained and run out of the box using Python's standard libraries. They support direct REST integrations with Google Gemini API and OpenAI API.

### 1. Offline Validation (Mock Mode)
Run generation using pre-defined local templates without spending API credits:
```bash
# Gujarati Mock Generation
python3 generate_gujarati_dataset.py --run-mock --output dataset_Gu.json

# English Mock Generation
python3 generate_english_dataset.py --run-mock --output dataset_En.json
```

### 2. Live Generation using Google Gemini
Set your API key and run the script:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
python3 generate_gujarati_dataset.py --provider gemini --model gemini-1.5-flash --output dataset_Gu.json
```

### 3. Live Generation using OpenAI
Set your API key and run the script:
```bash
export OPENAI_API_KEY="your-openai-api-key"
python3 generate_english_dataset.py --provider openai --model gpt-4o-mini --output dataset_En.json
```

---

## Advanced Options

Both scripts support the following parameters:
* `--scale <float>`: Scales the total dataset size (e.g. `--scale 3.2` will scale the 276 baseline up to ~880 conversations while maintaining the exact scenario distribution proportions).
* `--delay <seconds>`: Configures sleep intervals between API calls (default is 1.5 seconds) to prevent triggering rate limits.
* **Auto-Resume**: If interrupted, running the script again with the same parameters will automatically resume from the `*_checkpoint.json` file.
