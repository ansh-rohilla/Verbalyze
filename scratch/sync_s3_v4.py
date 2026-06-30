import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.config import Config

S3_BUCKET = "ansh-rohilla"
S3_TRANSCRIPTS_BUCKET = "vighnesh-voice-data-2026"
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

# Sync Marathi, Odia, and Assamese
langs = ["mr", "or", "as"]
dest_base = Path("/Users/anshrohilla/Documents/Verbalyze/scratch/domain_specific_data_v4")
local_audio_base = Path("/Users/anshrohilla/Documents/Verbalyze/scratch/domain_specific_data_v4_audio")

s3_config = Config(
    max_pool_connections=100,
    retries={'max_attempts': 5}
)

def get_s3_client():
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            config=s3_config
        )
    return boto3.client('s3', region_name=AWS_REGION, config=s3_config)

def upload_file(local_path, bucket, s3_key, s3_client=None):
    if s3_client is None:
        s3_client = get_s3_client()
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        return True, local_path.name
    except Exception as e:
        return False, f"Error uploading {local_path.name}: {e}"

def main():
    s3_client = get_s3_client()
    
    # 1. Gather all local WAV files for the target languages
    upload_tasks = []
    for lang in langs:
        for folder in ["tts", "native"]:
            dir_path = local_audio_base / lang / folder
            if dir_path.exists():
                wavs = list(dir_path.glob("*.wav"))
                for wav in wavs:
                    s3_key = f"audio/{lang}/{wav.name}"
                    upload_tasks.append((wav, S3_BUCKET, s3_key))
                    
    total_files = len(upload_tasks)
    print(f"Found {total_files} local WAV files for {langs} to upload...")
    
    start_time = time.time()
    uploaded_count = 0
    error_count = 0
    
    # Upload WAVs in parallel using 64 workers
    max_workers = 64
    if total_files > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(upload_file, path, bucket, key, s3_client): path.name for path, bucket, key in upload_tasks}
            for future in as_completed(futures):
                success, msg = future.result()
                if success:
                    uploaded_count += 1
                    if uploaded_count % 200 == 0 or uploaded_count == total_files:
                        elapsed = time.time() - start_time
                        rate = uploaded_count / elapsed if elapsed > 0 else 0
                        print(f"  Uploaded {uploaded_count}/{total_files} files. Rate: {rate:.1f} files/sec")
                else:
                    error_count += 1
                    print(msg)
                    
    total_elapsed = time.time() - start_time
    print(f"WAV uploads complete! Successfully uploaded {uploaded_count}/{total_files} files (Errors: {error_count}) in {total_elapsed:.1f} seconds.")
    
    # 2. Upload transcript files
    print("\nStarting transcript JSONL uploads...")
    for lang in langs:
        local_transcript = dest_base / lang / f"transcriptsv1_{lang}_with_audio.jsonl"
        if local_transcript.exists():
            s3_transcript_key = f"asr-datasets/domain_specific_data_v4/transcripts_with_audio/{lang}/transcriptsv1_{lang}_with_audio.jsonl"
            print(f"Uploading {lang} transcript to s3://{S3_TRANSCRIPTS_BUCKET}/{s3_transcript_key}...")
            success, msg = upload_file(local_transcript, S3_TRANSCRIPTS_BUCKET, s3_transcript_key, s3_client)
            if success:
                print(f"  {lang} transcript upload successful!")
            else:
                print(f"  {lang} transcript upload failed: {msg}")
        else:
            print(f"  Warning: local transcript for {lang} does not exist at {local_transcript}")
            
    print("\nS3 sync completed!")

if __name__ == "__main__":
    main()
