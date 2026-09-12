---
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
| Assamese | `as` | 1,300 |
| Bengali | `bn` | 1,300 |
| English | `en` | 1,377 |
| Gujarati | `gu` | 1,377 |
| Hindi | `hi` | 1,377 |
| Kannada | `kn` | 1,377 |
| Malayalam | `ml` | 1,377 |
| Marathi | `mr` | 1,377 |
| Odia | `or` | 1,377 |
| Punjabi | `pa` | 1,377 |
| Tamil | `ta` | 1,377 |
| Telugu | `te` | 1,377 |
| **Total** | **12** | **16,370** |

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
