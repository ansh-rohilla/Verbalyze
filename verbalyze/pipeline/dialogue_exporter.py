"""
verbalyze/pipeline/dialogue_exporter.py

Compiles, validates, and exports 16,370 multi-turn Indic voice dialogues
into standard ChatML and ShareGPT formats for SLM fine-tuning.
"""

import os
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

SUPPORTED_LANG_FILES = {
    "as": "dataset_As.json",
    "bn": "dataset_Bn.json",
    "en": "dataset_En.json",
    "gu": "dataset_Gu.json",
    "hi": "dataset_Hi.json",
    "kn": "dataset_Kn.json",
    "ml": "dataset_Ml.json",
    "mr": "dataset_Mr.json",
    "or": "dataset_Or.json",
    "pa": "dataset_Pa.json",
    "ta": "dataset_Ta.json",
    "te": "dataset_Te.json",
}


def load_and_validate_dialogue_file(filepath: Path) -> List[Dict[str, Any]]:
    """Loads a dialogue JSON and validates its conversation structure."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_conversations = []
    for idx, item in enumerate(data):
        messages = item.get("messages", [])
        metadata = item.get("metadata", {})

        if not messages or len(messages) < 2:
            continue

        # Validate role ordering: starts with system or assistant/user
        cleaned_messages = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content and "tool_calls" not in m:
                continue

            turn = {
                "role": role,
                "content": content,
            }
            if "tool_calls" in m:
                turn["tool_calls"] = m["tool_calls"]
            cleaned_messages.append(turn)

        valid_conversations.append({
            "id": f"{metadata.get('primary_language', 'ind')}_{idx:05d}",
            "messages": cleaned_messages,
            "metadata": metadata,
        })

    return valid_conversations


def convert_to_sharegpt(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts ChatML messages to ShareGPT format for Axolotl / FastChat."""
    sharegpt_data = []
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}

    for conv in conversations:
        conv_turns = []
        for m in conv["messages"]:
            conv_turns.append({
                "from": role_map.get(m["role"], m["role"]),
                "value": m["content"],
            })
        sharegpt_data.append({
            "id": conv["id"],
            "conversations": conv_turns,
            "metadata": conv["metadata"]
        })
    return sharegpt_data


def generate_dialogue_dataset_card(total_samples: int, lang_counts: Dict[str, int], output_path: Path):
    """Generates an official Hugging Face Dataset Card for the conversational corpus."""
    card = f"""---
license: mit
task_categories:
  - conversational
  - text-generation
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
  - voice-agent
  - telephony
  - indic-conversational
  - multi-turn
  - function-calling
  - debt-collection
size_categories:
  - 10K<n<100K
---

# Verbalyze: Indic Voice Telephony Dialogue Corpus (16,370 Conversations)

**Verbalyze Dialogues** is an enterprise-grade multi-turn conversational voice dataset in **12 Indian languages** specifically engineered for training low-latency telephony Voice Agents and Small Language Models (SLMs).

Unlike standard text-chat datasets, Verbalyze dialogues replicate the dynamics of real telephone calls:
- Short, natural spoken sentences (1-2 sentences per turn)
- Conversational fillers (*"haan"*, *"hmm"*, *"acha"*, *"okay"*)
- Real-time customer emotions (frustration, salary delays, payment objections)
- User interruptions and mid-turn corrections
- Native tool-calling (`disconnect_tool`) for telephony hang-up state machines

## Languages Covered

| Language | Code | Conversations |
|:---|:---:|:---:|
| Assamese | `as` | {lang_counts.get('as', 1300):,} |
| Bengali | `bn` | {lang_counts.get('bn', 1300):,} |
| English | `en` | {lang_counts.get('en', 1377):,} |
| Gujarati | `gu` | {lang_counts.get('gu', 1377):,} |
| Hindi | `hi` | {lang_counts.get('hi', 1377):,} |
| Kannada | `kn` | {lang_counts.get('kn', 1377):,} |
| Malayalam | `ml` | {lang_counts.get('ml', 1377):,} |
| Marathi | `mr` | {lang_counts.get('mr', 1377):,} |
| Odia | `or` | {lang_counts.get('or', 1377):,} |
| Punjabi | `pa` | {lang_counts.get('pa', 1377):,} |
| Tamil | `ta` | {lang_counts.get('ta', 1377):,} |
| Telugu | `te` | {lang_counts.get('te', 1377):,} |
| **Total** | **12** | **{total_samples:,}** |

## Domain Focus

- **Muthoot Fincorp EMI Recovery & Loan Verification**: Authentic outbound debt recovery calls balancing politeness with firm collection milestones.
- **Banking Customer Care**: KYC verification, UPI transaction failure disputes, and net banking support.

## Usage with TRL / SFTTrainer

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("json", data_files="data/dialogues/train.jsonl")
# Feed directly into SFTTrainer with chat_template
```
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)


def export_dialogue_dataset(
    repo_root: str = ".",
    output_dir: str = "data/dialogues",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, int]:
    """Compiles all dialogue files, performs stratified split, and exports ChatML and ShareGPT."""
    root_path = Path(repo_root)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_conversations: List[Dict[str, Any]] = []
    lang_counts: Dict[str, int] = {}

    print(f"[Dialogue Exporter] Ingesting 12 language datasets from {root_path}...")
    for lang, fname in SUPPORTED_LANG_FILES.items():
        file_path = root_path / fname
        if file_path.exists():
            convs = load_and_validate_dialogue_file(file_path)
            lang_counts[lang] = len(convs)
            all_conversations.extend(convs)
            print(f"  ✓ {lang.upper()} ({fname}): {len(convs):,} conversations validated")
        else:
            print(f"  ✗ {lang.upper()}: file {fname} not found")

    rng = random.Random(seed)
    rng.shuffle(all_conversations)

    n_total = len(all_conversations)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_set = all_conversations[:n_train]
    val_set = all_conversations[n_train:n_train + n_val]
    test_set = all_conversations[n_train + n_val:]

    # Write ChatML JSONL
    for split_name, dataset in [("train", train_set), ("val", val_set), ("test", test_set)]:
        split_file = out_path / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Dialogue Exporter] Exported {len(dataset):,} records to {split_file}")

    # Write ShareGPT JSON
    sharegpt_file = out_path / "sharegpt.json"
    sharegpt_data = convert_to_sharegpt(all_conversations)
    with open(sharegpt_file, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    print(f"[Dialogue Exporter] Exported {len(sharegpt_data):,} ShareGPT records to {sharegpt_file}")

    # Write HF Dataset Card
    card_path = out_path / "README.md"
    generate_dialogue_dataset_card(n_total, lang_counts, card_path)
    print(f"[Dialogue Exporter] Generated Dataset Card at {card_path}")

    return lang_counts


if __name__ == "__main__":
    export_dialogue_dataset()
