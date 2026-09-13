"""
scripts/deploy_space.py

1-Click automated deployment of Verbalyze Interactive Web App to Hugging Face Spaces:
https://huggingface.co/spaces/ansh-rohilla/verbalyze-demo
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

SPACE_REPO_ID = "ansh-rohilla/verbalyze-demo"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = PROJECT_ROOT / "space"


def deploy(repo_id: str = SPACE_REPO_ID):
    print(f"🚀 Initializing Hugging Face Spaces Deployment for: {repo_id}")
    api = HfApi()

    # 1. Verify Hugging Face authentication
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"✓ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Error: Hugging Face authentication failed: {e}")
        print("Please ensure your token is set via 'huggingface-cli login' or HF_TOKEN environment variable.")
        sys.exit(1)

    # 2. Create or verify the Space repository (Static SDK is free forever for everyone)
    print(f"📦 Verifying Space repo: {repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
            private=False
        )
        print(f"✓ Space repository verified at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"⚠️ Repo creation notice: {e}")

    # 3. Upload space folder to Hugging Face Spaces
    print(f"⬆️ Uploading web app bundle from {SPACE_DIR} to {repo_id}...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="feat: deploy Verbalyze Indic Voice AI interactive web app"
    )

    print("\n" + "=" * 65)
    print("🎉 DEPLOYMENT COMPLETE!")
    print(f"🔗 Live Space URL: https://huggingface.co/spaces/{repo_id}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else SPACE_REPO_ID
    deploy(target)
