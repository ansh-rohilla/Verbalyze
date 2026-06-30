import os
import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3

S3_BUCKET = "ansh-rohilla"
S3_TRANSCRIPTS_BUCKET = "vighnesh-voice-data-2026"
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

dest_base = Path("/Users/anshrohilla/Documents/Verbalyze/scratch/domain_specific_data_v4")
local_audio_base = Path("/Users/anshrohilla/Documents/Verbalyze/scratch/domain_specific_data_v4_audio")

# Queue of languages to process: (lang, steps)
indic_queue = [
    ("mr", 10),  # Marathi (1,880 records)
    ("or", 10),  # Odia (1,999 records)
    ("as", 15)   # Assamese (2,528 records - needs steps=15 for stability)
]

def get_s3_client():
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    return boto3.client('s3', region_name=AWS_REGION)

def upload_file(local_path, bucket, s3_key, s3_client=None):
    if s3_client is None:
        s3_client = get_s3_client()
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        return True, local_path.name
    except Exception as e:
        return False, f"Error uploading {local_path.name}: {e}"

def sync_language_to_s3(lang):
    s3_client = get_s3_client()
    print(f"\n[S3 Sync] Gathering files for {lang.upper()}...")
    
    upload_tasks = []
    for folder in ["tts", "native"]:
        dir_path = local_audio_base / lang / folder
        if dir_path.exists():
            wavs = list(dir_path.glob("*.wav"))
            for wav in wavs:
                s3_key = f"audio/{lang}/{wav.name}"
                upload_tasks.append((wav, S3_BUCKET, s3_key))
                
    total_files = len(upload_tasks)
    print(f"[S3 Sync] Found {total_files} WAV files to upload to S3 bucket '{S3_BUCKET}'...")
    
    start_time = time.time()
    uploaded_count = 0
    error_count = 0
    
    if total_files > 0:
        max_workers = 32
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(upload_file, path, bucket, key, s3_client): path.name for path, bucket, key in upload_tasks}
            for future in as_completed(futures):
                success, msg = future.result()
                if success:
                    uploaded_count += 1
                    if uploaded_count % 100 == 0 or uploaded_count == total_files:
                        elapsed = time.time() - start_time
                        rate = uploaded_count / elapsed if elapsed > 0 else 0
                        print(f"  Uploaded {uploaded_count}/{total_files} files. Rate: {rate:.1f} files/sec")
                else:
                    error_count += 1
                    print(msg)
                    
    print(f"[S3 Sync] WAV uploads complete for {lang.upper()}! Uploaded {uploaded_count}/{total_files} (Errors: {error_count}) in {time.time() - start_time:.1f}s")
    
    # Upload transcript
    local_transcript = dest_base / lang / f"transcriptsv1_{lang}_with_audio.jsonl"
    if local_transcript.exists():
        s3_transcript_key = f"asr-datasets/domain_specific_data_v4/transcripts_with_audio/{lang}/transcriptsv1_{lang}_with_audio.jsonl"
        print(f"[S3 Sync] Uploading transcript to s3://{S3_TRANSCRIPTS_BUCKET}/{s3_transcript_key}...")
        success, msg = upload_file(local_transcript, S3_TRANSCRIPTS_BUCKET, s3_transcript_key, s3_client)
        if success:
            print(f"  {lang.upper()} transcript upload successful!")
        else:
            print(f"  {lang.upper()} transcript upload failed: {msg}")
    else:
        print(f"  Warning: local transcript for {lang} not found.")

def main():
    print("==============================================")
    print("Starting Sequential Indic Datasets Pipeline")
    print("==============================================")
    
    for lang, steps in indic_queue:
        print(f"\n==============================================")
        print(f"STARTING GENERATION FOR {lang.upper()} (steps={steps})")
        print(f"==============================================")
        
        cmd = [
            "python3", "-u", "scratch/run_pipeline_wrapper.py",
            "--languages", lang,
            "--steps", str(steps),
            "--max-records", "200",
            "--local-only"
        ]
        
        print(f"Running generation command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        exit_code = process.wait()
        
        if exit_code != 0:
            print(f"Error: Generation wrapper failed for {lang.upper()} with exit code {exit_code}. Aborting pipeline.")
            sys.exit(exit_code)
            
        print(f"Successfully finished processing {lang.upper()}!")
        
    print("\n==============================================")
    print("All 4 Indic datasets successfully completed and synced!")
    print("==============================================")

if __name__ == "__main__":
    main()
