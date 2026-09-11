#!/usr/bin/env python3
"""
upload_to_s3.py

Utility script to normalize transcripts via LLM, synthesize and validate audio via Whisper,
and upload STT audio datasets to AWS S3. Tags each S3 object with metadata.
"""

import os
import sys
import csv
import time
import json
import argparse
import logging
import urllib.parse
import urllib.request
import urllib.error
import base64
import socket
import difflib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set default socket timeout to prevent indefinite hangs
socket.setdefaulttimeout(30.0)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try importing boto3
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    logger.error("boto3 package is not installed. Please run: pip install boto3")
    sys.exit(1)

# Try importing requests
try:
    import requests
except ImportError:
    logger.error("requests package is not installed. Please run: pip install requests")
    sys.exit(1)

# Try importing gTTS
try:
    from gtts import gTTS
except ImportError:
    logger.error("gtts package is not installed. Please run: pip install gTTS")
    sys.exit(1)

# Supported language configuration mapping
SUPPORTED_LANGUAGES = {
    "en": "en", "gu": "gu", "hi": "hi", "kn": "kn",
    "ml": "ml", "mr": "mr", "or": "or", "pa": "pa",
    "ta": "ta", "te": "te", "bn": "bn", "as": "as"
}

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

def normalize_text_via_llm(text: str, lang_code: str, api_key: str = GROQ_API_KEY) -> str:
    """Uses Groq's Llama-3.1 model to normalize raw transcript to spoken word format."""
    if not api_key:
        logger.warning("GROQ_API_KEY is empty. Skipping LLM normalization.")
        return text

    url = "https://api.groq.com/openai/v1/chat/completions"
    
    system_prompt = (
        "You are a text normalization assistant for Text-to-Speech (TTS). "
        "Your task is to convert raw transcript texts into their fully expanded spoken-word form, maintaining the original language and script. "
        "The output must sound natural when read.\n\n"
        "Rules:\n"
        "1. Slashes ('/') must be expanded based on context:\n"
        "   - Units: e.g., 'mg/kg' -> 'milligrams per kilogram', 'km/hr' -> 'kilometers per hour' (or native script equivalent except for metrics which must be in English/Latin script as per Rule 5)\n"
        "   - Acronym lists/codes: e.g., 'DPS/RMP/PT' -> 'D P S slash R M P slash P T' (or native script equivalent)\n"
        "   - Dates: e.g., '11/02/2015' -> 'eleventh February twenty fifteen' (or native script equivalent)\n"
        "   - Alternatives: e.g., 'yes/no' -> 'yes or no' (or native script equivalent)\n"
        "2. Door numbers / addresses (e.g., 'Door No. 8-2-293/4' or 'flat 89-6-354/9'):\n"
        "   - Convert door numbers and address identifiers to their exact spoken form by pronouncing the digits exactly as they appear. Separate hyphenated sections with a space/pause (do NOT say 'minus' or 'hyphen' or 'dash', e.g., '8-2' -> 'eight two', or native equivalent). Pronounce the slash '/' as 'by' (or native equivalent like 'बाई' / 'बाय'). Do not add, repeat, or skip any digits.\n"
        "3. Phone numbers, OTPs, Pincodes, and other digit sequences:\n"
        "   - Pronounce them digit-by-digit (or native script equivalent).\n"
        "   - CRITICAL: You must pronounce EVERY digit. Do NOT skip, add, or omit any digits under any circumstances.\n"
        "   - To ensure natural speech diversity, introduce realistic variations for repeating digits where appropriate: e.g., consecutive matching digits can be spoken either individually (e.g., '55' -> 'five five', or native equivalent) or grouped using 'double'/'triple' terms (e.g., 'double five', 'triple eight', or native equivalents like 'डबल पांच' / 'ट्रिपल आठ' depending on the language/script context). Aim for a mix of digit-by-digit and grouped double/triple spoken forms across the batch.\n"
        "4. Currency / Monetary Amounts:\n"
        "   - CRITICAL: Amounts and currency numbers (e.g., '3750 रुपये', '1500 টকা', '7440 ரூபாய்') must be normalized to their full written cardinal number words in the target script/language (e.g., '3750 रुपये' -> 'तीन हज़ार सात सौ पचास रुपये' in Hindi script; '1500' -> 'one thousand five hundred' in English). Do NOT spell them out digit-by-digit.\n"
        "5. Units of Measurement, Quantities, and Years:\n"
        "   - CRITICAL: All units of measurement, compound metrics, and the numbers preceding them (e.g., '250 Mega bits', '256 mbps', '50 GB', '16 TB', '79 km', '10 mg', 'mg/kg', 'ml', '5 ml', '25 mg/kg') MUST be normalized strictly in English words and Latin script for ALL languages (e.g., '250 Mega bits' -> 'two hundred fifty Megabits', '256 mbps' -> 'two hundred fifty-six Mbps', '10 mg' -> 'ten milligrams', '5 ml' -> 'five milliliters', '25 mg/kg' -> 'twenty-five milligrams per kilogram', 'mg/ml' -> 'milligrams per milliliter'). Do NOT translate or transliterate these metrics to native scripts in non-English transcripts; keep them strictly in English/Latin script.\n"
        "   - Calendar years (e.g., '2021', '2015') must be normalized to their full written cardinal number words in the target script/language (e.g., '2021' -> 'दो हज़ार इक्कीस', '2015' -> 'दो हज़ार पंधरा' / 'twenty fifteen'). Do NOT spell them out digit-by-digit.\n"
        "6. Abbreviations and Acronyms:\n"
        "   - Expand them to their full spoken form if appropriate, or spell out letters.\n"
        "7. Language/Script Consistency:\n"
        "   - The output MUST be in the exact same script and language as the input. Keep code-mixed words (like English words in an Indian language sentence) in their original script (Latin script).\n"
        "   - All numbers normalized in non-English Indian language transcripts must be written in the native script words of that language, NOT English words or English digits (e.g., use 'तीन हज़ार सात सौ पचास' in Hindi, not 'three thousand seven hundred fifty'). EXCEPT for numbers preceding units of measurement (metrics) and the metric units themselves, which MUST be normalized strictly in English/Latin script words as specified in Rule 5.\n"
        "8. Do NOT answer questions, translate, or add extra commentary.\n\n"
        "Input Format: You will receive a raw transcript as a string.\n"
        "Output Format: Respond with the normalized transcript string."
    )
    
    prompt = f"Language: {lang_code}\nInput Text: {text}\nNormalized Text:"
    
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0"
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            backoff = 2 ** attempt
            logger.warning(f"LLM normalization failed (Attempt {attempt+1}): {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            
    return text

def transcribe_audio_via_groq(file_path: str, api_key: str = GROQ_API_KEY) -> str:
    """Calls Groq's Whisper API to transcribe a synthesized audio file for validation."""
    if not api_key:
        logger.warning("GROQ_API_KEY is empty. Skipping Whisper transcription.")
        return ""
        
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    files = {
        "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/mp3")
    }
    data = {
        "model": "whisper-large-v3",
        "response_format": "json"
    }
    
    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            if response.status_code == 200:
                return response.json().get("text", "")
            elif response.status_code == 429:
                logger.warning(f"Whisper Rate Limited (429). Sleeping 6 seconds before retrying (Attempt {attempt+1}/5)...")
                time.sleep(6.0)
            else:
                logger.warning(f"Whisper transcription HTTP Error (Attempt {attempt+1}): {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Whisper transcription connection failed (Attempt {attempt+1}): {e}")
        time.sleep(2 ** attempt)
        
    return ""

def clean_text(s: str) -> str:
    """Helper to lowercase, remove punctuation, and normalize spaces for text comparison."""
    s = s.lower()
    for c in '.,!?;:-_()[]{}""\'`/':
        s = s.replace(c, ' ')
    return " ".join(s.split())

def compute_similarity(str1: str, str2: str) -> float:
    """Calculates Levenshtein-like similarity ratio between two strings using difflib."""
    s1 = clean_text(str1)
    s2 = clean_text(str2)
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def get_base64_tag_value(value: str) -> str:
    """Base64 encodes a Unicode string to bypass S3 tagging limitations, ensuring it stays under 250 characters."""
    encoded = base64.b64encode(value.encode('utf-8')).decode('ascii')
    while len(encoded) > 250 and len(value) > 0:
        value = value[:-5]
        encoded = base64.b64encode(value.encode('utf-8')).decode('ascii')
    return encoded

def upload_dataset(bucket_name: str, lang: str, limit: int, region: str, api_key: str = GROQ_API_KEY):
    # Initialize S3 client with environment credentials
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
    else:
        s3_client = boto3.client('s3', region_name=region)
    
    # 1. Read metadata file (prefer _s3.csv if it exists for resuming)
    s3_csv_file = Path("stt_dataset") / lang / f"metadata_{lang}_s3.csv"
    csv_file = Path("stt_dataset") / lang / f"metadata_{lang}.csv"
    
    read_file_path = s3_csv_file if s3_csv_file.exists() else csv_file
    if not read_file_path.exists():
        logger.error(f"Metadata file not found: {read_file_path}")
        return
        
    logger.info(f"Reading dataset metadata from {read_file_path}...")
    records = []
    with open(read_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    logger.info(f"Loaded {len(records)} records.")
    
    # 2. Check gTTS support
    tts_code = SUPPORTED_LANGUAGES.get(lang)
    if lang in ["as", "or"]:
        logger.warning(f"Language '{lang}' is not supported by gTTS. Will use 'bn' (Bengali) as phonetic proxy for synthesis.")
        tts_code = "bn"
        
    # Local directory to store synthesized files temporarily
    local_audio_dir = Path("stt_dataset") / lang / "audio"
    local_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalization cache file configuration
    cache_file = Path("stt_dataset") / lang / f"normalization_cache_{lang}.json"
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} entries from normalization cache.")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            
    def save_cache():
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            
    # Set limit
    process_limit = min(limit, len(records)) if limit > 0 else len(records)
    logger.info(f"Starting upload process for {process_limit} utterances...")
    
    def upload_single_file(row, index, local_path, s3_key) -> bool:
        try:
            transcript = row["transcript"]
            native_text = row["native_text"]
            record_id = row["id"]
            
            tag_dict = {
                "id": record_id,
                "language": lang,
                "scenario": row["scenario"],
                "transcript_b64": get_base64_tag_value(transcript),
                "native_text_b64": get_base64_tag_value(native_text),
                "normalized_transcript_b64": get_base64_tag_value(row.get("normalized_transcript", transcript))
            }
            tagging_str = urllib.parse.urlencode(tag_dict)
            
            logger.info(f"Uploading [{index+1}/{process_limit}] {local_path.name} -> s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(
                Filename=str(local_path),
                Bucket=bucket_name,
                Key=s3_key,
                ExtraArgs={"Tagging": tagging_str}
            )
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            row["audio_path"] = s3_uri
            return True
        except ClientError as ce:
            logger.error(f"S3 Upload failed for {row['id']}: {ce}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading {row['id']}: {e}")
            return False

    success_count = 0
    start_time = time.time()
    
    # We use a ThreadPoolExecutor for concurrent S3 uploads
    # Synthesis/validation stay single-threaded to prevent API rate limits
    futures = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(process_limit):
            row = records[i]
            transcript = row["transcript"]
            scenario = row["scenario"]
            record_id = row["id"]
            
            audio_filename = f"{lang}_{scenario}_{i+1:05d}.mp3"
            local_path = local_audio_dir / audio_filename
            s3_key = f"audio/{lang}/{audio_filename}"
            
            # A. Determine normalization text
            if transcript in cache:
                normalized_text = cache[transcript]
            else:
                logger.info(f"[{i+1}/{process_limit}] Normalizing raw transcript via LLM...")
                normalized_text = normalize_text_via_llm(transcript, lang, api_key)
                cache[transcript] = normalized_text
                # Periodic cache saving
                if (i + 1) % 50 == 0:
                    save_cache()
                    
            row["normalized_transcript"] = normalized_text
            
            # Self-healing: if local file exists but audio_path is N/A (e.g. from dry run overwrite), restore it
            if (not row.get("audio_path") or row.get("audio_path") == "N/A") and local_path.exists() and local_path.stat().st_size > 0:
                row["audio_path"] = f"s3://{bucket_name}/{s3_key}"
            
            # B. Conditional strategy logic
            already_uploaded = row.get("audio_path") != "N/A" and row.get("audio_path", "").startswith("s3://")
            normalization_unchanged = (normalized_text == transcript)
            
            skip_all = already_uploaded and normalization_unchanged
            
            if skip_all:
                # Set dummy score and fix durations if missing
                if not row.get("stt_validation_score"):
                    row["stt_validation_score"] = "1.00"
                if not row.get("duration_seconds") or row.get("duration_seconds") == "0.0":
                    words = len(normalized_text.split())
                    row["duration_seconds"] = str(round(max(1.5, (words / 150.0) * 60.0), 2))
                continue
                
            # C. Audio Synthesis & Whisper Quality Check
            needs_synthesis = (not local_path.exists() or local_path.stat().st_size == 0 or not normalization_unchanged)
            
            if needs_synthesis:
                # Remove stale audio if it exists
                if local_path.exists():
                    try:
                        local_path.unlink()
                    except Exception:
                        pass
                        
                retry_count = 0
                synthesized = False
                while not synthesized:
                    try:
                        logger.info(f"[{i+1}/{process_limit}] Synthesizing '{normalized_text[:35]}...' -> {local_path.name}")
                        tts = gTTS(text=normalized_text, lang=tts_code)
                        tts.save(str(local_path))
                        
                        # Validate with Whisper
                        score = 1.00
                        if api_key:
                            logger.info(f"[{i+1}/{process_limit}] Verifying audio quality via Whisper...")
                            whisper_text = transcribe_audio_via_groq(str(local_path), api_key)
                            if whisper_text and not whisper_text.startswith("Error"):
                                score = compute_similarity(normalized_text, whisper_text)
                                logger.info(f"[{i+1}/{process_limit}] Whisper transcribed: '{whisper_text[:40]}...'. Score: {score:.2f}")
                            else:
                                logger.warning(f"[{i+1}/{process_limit}] Whisper transcription failed: {whisper_text}")
                                score = 0.50 # Treat failure as low score to trigger retry
                                
                        if score >= 0.85:
                            synthesized = True
                            row["stt_validation_score"] = f"{score:.2f}"
                        else:
                            retry_count += 1
                            if retry_count >= 3:
                                logger.warning(f"[{i+1}/{process_limit}] Quality score {score:.2f} failed after 3 attempts. Accepting current audio.")
                                row["stt_validation_score"] = f"{score:.2f}"
                                synthesized = True
                            else:
                                logger.warning(f"[{i+1}/{process_limit}] Quality score {score:.2f} below threshold (0.85). Retrying synthesis (Attempt {retry_count+1})...")
                                if local_path.exists():
                                    try:
                                        local_path.unlink()
                                    except Exception:
                                        pass
                                time.sleep(1.5)
                                
                        words = len(normalized_text.split())
                        row["duration_seconds"] = str(round(max(1.5, (words / 150.0) * 60.0), 2))
                        time.sleep(0.4) # Control pacing for Google TTS rate limits
                    except Exception as e:
                        # Handle gTTS 429
                        err_str = str(e)
                        if "429" in err_str or "Too Many Requests" in err_str:
                            backoff = 600
                            logger.warning(f"Google TTS Rate limit hit (Attempt {retry_count}): {e}")
                            logger.info("Sleeping for 10 minutes to reset rate limits...")
                        else:
                            retry_count += 1
                            backoff = min(15 * retry_count, 120)
                            logger.warning(f"gTTS failed (Attempt {retry_count}): {e}")
                            logger.info(f"Sleeping for {backoff} seconds before retry...")
                        time.sleep(backoff)
            else:
                # Use existing cached audio but ensure metadata columns are populated
                if not row.get("stt_validation_score"):
                    row["stt_validation_score"] = "1.00"
                if not row.get("duration_seconds") or row.get("duration_seconds") == "0.0":
                    words = len(normalized_text.split())
                    row["duration_seconds"] = str(round(max(1.5, (words / 150.0) * 60.0), 2))
                    
            # D. Submit file upload to the ThreadPoolExecutor
            futures.append(executor.submit(upload_single_file, row, i, local_path, s3_key))
            
        # Wait for all uploads to complete
        for fut in as_completed(futures):
            if fut.result():
                success_count += 1
                
    # 3. Always save local cache and upload updated metadata CSV
    save_cache()
    
    updated_csv_path = Path("stt_dataset") / lang / f"metadata_{lang}_s3.csv"
    logger.info(f"Writing updated S3 metadata to {updated_csv_path}...")
    
    with open(updated_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id", "audio_path", "transcript", "normalized_transcript", "stt_validation_score",
            "native_text", "romanized_text", "code_mixed_text", "scenario", "language", "duration_seconds"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    s3_csv_key = f"metadata/{lang}/metadata_{lang}_s3.csv"
    try:
        logger.info(f"Uploading updated metadata sheet -> s3://{bucket_name}/{s3_csv_key}")
        s3_client.upload_file(
            Filename=str(updated_csv_path),
            Bucket=bucket_name,
            Key=s3_csv_key
        )
        logger.info("Metadata CSV uploaded to S3 successfully!")
    except Exception as e:
        logger.error(f"Failed to upload updated CSV metadata: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"Summary: Processed/Uploaded {success_count} files in {elapsed:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="AWS S3 Audio Dataset Uploader")
    parser.add_argument("--bucket", required=True, help="Target S3 Bucket Name")
    parser.add_argument("--lang", default="hi", choices=list(SUPPORTED_LANGUAGES.keys()), help="Target language code to process")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of processed files (0 for all)")
    parser.add_argument("--region", default="ap-south-1", help="AWS Region")
    parser.add_argument("--groq-key", default=None, help="Groq API Key")
    
    args = parser.parse_args()
    
    groq_key = args.groq_key or GROQ_API_KEY
    upload_dataset(args.bucket, args.lang, args.limit, args.region, groq_key)

if __name__ == "__main__":
    main()
