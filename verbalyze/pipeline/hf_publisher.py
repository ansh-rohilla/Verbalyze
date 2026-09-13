"""
verbalyze/pipeline/hf_publisher.py

Publishes Verbalyze datasets to the Hugging Face Hub:
- verbalyze-dialogues: 16,370 multi-turn voice conversations in 12 languages
- verbalyze-stt-bench: 172,800 STT scenario stress-test utterances in 12 languages
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from huggingface_hub import HfApi, get_token
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolves Hugging Face token from explicit arg, env, or cached login."""
    if explicit_token:
        return explicit_token.strip()
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()
    if HF_AVAILABLE:
        try:
            cached = get_token()
            if cached:
                return cached.strip()
        except Exception:
            pass
    return None


def publish_dataset(
    folder_path: str,
    repo_name: str,
    token: Optional[str] = None,
    private: bool = False,
    custom_repo_id: Optional[str] = None
) -> str:
    """Uploads a dataset folder to Hugging Face Hub."""
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub is required. Install via: pip install huggingface-hub")

    resolved_token = get_hf_token(token)
    if not resolved_token:
        raise ValueError(
            "Hugging Face token not found!\n"
            "Please provide a token by either:\n"
            "  1. Passing --token <your_hf_token>\n"
            "  2. Setting the environment variable: export HF_TOKEN=\"your_hf_token\"\n"
            "  3. Running: huggingface-cli login\n\n"
            "To get a free token, visit: https://huggingface.co/settings/tokens (ensure 'Write' role is checked)"
        )

    api = HfApi(token=resolved_token)

    # 1. Identify user account
    try:
        user_info = api.whoami(token=resolved_token)
        username = user_info.get("name", "user")
        print(f"[Hugging Face] Authenticated as: @{username}")
    except Exception as e:
        raise RuntimeError(f"Failed to authenticate with Hugging Face: {e}")

    # 2. Determine target repo ID
    repo_id = custom_repo_id or f"{username}/{repo_name}"
    local_dir = Path(folder_path)

    if not local_dir.exists():
        raise FileNotFoundError(f"Local dataset directory does not exist: {local_dir}")

    # 3. Create repo if needed
    print(f"[Hugging Face] Ensuring dataset repository exists: {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # 4. Upload folder
    print(f"[Hugging Face] Uploading {local_dir} to https://huggingface.co/datasets/{repo_id}...")
    upload_url = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload Verbalyze dataset: {repo_name}",
    )

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\n========================================================")
    print(f"🎉 Successfully published to Hugging Face Hub!")
    print(f"URL: {dataset_url}")
    print(f"Python usage:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{repo_id}')")
    print(f"========================================================\n")
    return dataset_url


def publish_all(
    token: Optional[str] = None,
    private: bool = False,
    dialogues_dir: str = "data/dialogues",
    stt_dir: str = "data/stt_bench"
) -> Dict[str, str]:
    """Publishes both dialogues and STT datasets to Hugging Face."""
    urls = {}

    # 1. Publish Dialogues
    print("\n--- Publishing 16.3k Conversational Telephony Dataset ---")
    urls["dialogues"] = publish_dataset(
        folder_path=dialogues_dir,
        repo_name="verbalyze-dialogues",
        token=token,
        private=private
    )

    # 2. Publish STT Benchmark
    print("\n--- Publishing 172.8k STT Benchmark Dataset ---")
    urls["stt_bench"] = publish_dataset(
        folder_path=stt_dir,
        repo_name="verbalyze-stt-bench",
        token=token,
        private=private
    )

    return urls
