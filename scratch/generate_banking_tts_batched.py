import os
import sys
import re
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import soundfile as sf
from omnivoice import OmniVoice

WORKSPACE_DIR = Path("/Users/anshrohilla/Documents/Verbalyze")
DATA_DIR = WORKSPACE_DIR / "scratch" / "domain_specific_data_v4" / "banking_finance"
LOCAL_AUDIO_BASE = WORKSPACE_DIR / "scratch" / "domain_specific_data_v4_audio" / "banking_finance"
S3_BUCKET = "ansh-rohilla"

VOICE_CONFIGS = {
    "hi": {"female": "female, young adult, moderate pitch", "male": "male, middle-aged, low pitch"},
    "kn": {"female": "female, young adult, moderate pitch", "male": "male, middle-aged, low pitch"},
    "te": {"female": "female, young adult, moderate pitch", "male": "male, middle-aged, low pitch"}
}

DIGIT_MAPS_ENGLISH = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

DIGIT_MAPS_NATIVE = {
    "hi": {"0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार", "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ"},
    "kn": {"0": "ಶೂನ್ಯ", "1": "ಒಂದು", "2": "ಎರಡು", "3": "ಮೂರು", "4": "ನಾಲ್ಕು", "5": "ಐದು", "6": "ಆರು", "7": "ಏಳು", "8": "ಎಂಟು", "9": "ಒಂಬತ್ತು"},
    "te": {"0": "సున్నా", "1": "ఒకటి", "2": "రెండు", "3": "మూడు", "4": "నాలుగు", "5": "ఐదు", "6": "ఆరు", "7": "ఎనిమిది", "8": "తొమ్മിది"}
}

def post_process_digits(text: str, lang_code: str, is_native: bool = False) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    
    expansions = {}
    if not is_native:
        expansions = {
            r'\bms\b': 'milliseconds',
            r'\bmbps\b': 'M B P S',
            r'\bgbps\b': 'G B P S',
            r'\botp\b': 'O T P',
            r'\batm\b': 'A T M',
            r'\bkm\b': 'kilometers',
            r'\bhr\b': 'hours',
            r'\bmmhg\b': 'millimeters of mercury',
            r'\bkg\b': 'kilograms',
            r'\bcm\b': 'centimeters',
            r'\bml\b': 'milliliters',
            r'%': ' percent '
        }
    
    two_word = DIGIT_MAPS_NATIVE.get(lang_code, {}).get("2", "two") if is_native else "two"
    text = re.sub(r'\bSpO[ -]?2\b', f'SpO {two_word}', text, flags=re.IGNORECASE)
    
    for pattern, replacement in expansions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    digit_map = DIGIT_MAPS_NATIVE.get(lang_code, {}) if is_native else DIGIT_MAPS_ENGLISH
    
    def replace_isolated_digit(match):
        digit = match.group(0)
        return digit_map.get(digit, digit)
        
    text = re.sub(r'\b\d\b', replace_isolated_digit, text)
    
    def replace_word_digit(match):
        digit = match.group(1)
        return f" {digit_map.get(digit, digit)} "
    
    text = re.sub(r'(\d)', replace_word_digit, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_wav(audio_data, output_path, sr=24000):
    audio_data = np.array(audio_data, dtype=np.float32)
    max_val = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
    if max_val > 1.0:
        audio_data = audio_data / max_val
    sf.write(str(output_path), audio_data, sr)

def process_banking_language(lang: str, model, resume: bool = True, steps: int = 10, max_records: int = 0, batch_size: int = 4, sample: int = 0):
    print(f"\n==============================================")
    print(f"Processing Banking Language (BATCHED): {lang.upper()}")
    print(f"==============================================")
    
    jsonl_file = DATA_DIR / f"{lang}.jsonl"
    if not jsonl_file.exists():
        print(f"Error: banking transcript JSONL file {jsonl_file} does not exist.")
        return False
        
    local_tts_dir = LOCAL_AUDIO_BASE / lang / "tts"
    local_native_dir = LOCAL_AUDIO_BASE / lang / "native"
    local_tts_dir.mkdir(parents=True, exist_ok=True)
    local_native_dir.mkdir(parents=True, exist_ok=True)
    
    output_jsonl_file = DATA_DIR / f"{lang}_with_audio.jsonl"
    
    completed_records = {}
    if resume and output_jsonl_file.exists():
        print(f"Loading existing progress from {output_jsonl_file}...")
        with open(output_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_records[record["id"]] = record
                    except Exception:
                        pass
        print(f"Loaded {len(completed_records)} already completed records.")
        
    all_records = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass
                    
    total_records = len(all_records)
    print(f"Total records in dataset: {total_records}")
    
    voice_config = VOICE_CONFIGS.get(lang)
    
    pending_records = []
    for idx, row in enumerate(all_records):
        row_id = row["id"]
        
        native_text = row.get("native_tts_reading", "")
        is_currency = False
        if native_text:
            currency_keywords = [
                "टका", "ৰূপ", "টাকা", "রুপये", "रुपये", "रुपे", "రూపాయలు", "రూಪಾಯಿ", "rupees", "rupee", "rs"
            ]
            is_currency = any(kw in native_text.lower() for kw in currency_keywords)
        is_sample = (idx % 5 == 0)
        native_required = is_currency or is_sample
        
        has_native = bool(native_text) and native_required
        
        local_tts_path = local_tts_dir / f"{row_id}_tts.wav"
        local_native_path = local_native_dir / f"{row_id}_native.wav" if has_native else None
        
        tts_exists = local_tts_path.exists() and local_tts_path.stat().st_size > 0
        native_exists = not has_native or (local_native_path.exists() and local_native_path.stat().st_size > 0)
        
        if resume and row_id in completed_records and tts_exists and native_exists:
            continue
            
        row["tts_clean_text"] = post_process_digits(row["tts_reading"], lang, is_native=False)
        row["tts_audio_path"] = f"s3://{S3_BUCKET}/audio/banking_finance/{lang}/{row_id}_tts.wav"
        
        if has_native:
            row["native_clean_text"] = post_process_digits(native_text, lang, is_native=True)
            row["native_audio_path"] = f"s3://{S3_BUCKET}/audio/banking_finance/{lang}/{row_id}_native.wav"
        else:
            row.pop("native_tts_reading", None)
            row.pop("native_clean_text", None)
            row.pop("native_audio_path", None)
            
        row["gender"] = "female" if (idx % 2 == 0) else "male"
        row["has_native"] = has_native
        pending_records.append(row)
        
    print(f"Pending records to generate: {len(pending_records)}")
    if sample > 0:
        print(f"Sampling first {sample} records from pending queue.")
        pending_records = pending_records[:sample]
        print(f"New pending records count: {len(pending_records)}")
    
    if len(pending_records) == 0:
        print(f"Banking Language {lang.upper()} is already fully complete!")
        return True
        
    mode = "a" if (resume and output_jsonl_file.exists()) else "w"
    out_f = open(output_jsonl_file, mode, encoding="utf-8")
    
    synthesized_rows = 0
    start_time = time.time()
    
    for i in range(0, len(pending_records), batch_size):
        if max_records > 0 and synthesized_rows >= max_records:
            print(f"[Limit Reached] Exiting current subprocess run to recycle memory.")
            out_f.close()
            return False
            
        batch = pending_records[i : i + batch_size]
        
        # A. Synthesize TTS
        tts_texts = [row["tts_clean_text"] for row in batch]
        instructs = [voice_config[row["gender"]] for row in batch]
        try:
            results = model.generate(text=tts_texts, instruct=instructs, num_step=steps)
            for idx, row in enumerate(batch):
                audio = results[idx]
                if isinstance(audio, list):
                    audio = np.concatenate([t.detach().cpu().numpy() if torch.is_tensor(t) else np.array(t) for t in audio])
                elif torch.is_tensor(audio):
                    audio = audio.detach().cpu().numpy()
                else:
                    audio = np.array(audio, dtype=np.float32)
                audio = audio.flatten()
                save_wav(audio, local_tts_dir / f"{row['id']}_tts.wav")
        except Exception as e:
            print(f"Error in batch TTS synthesis: {e}")
            for row in batch:
                save_wav(np.array([]), local_tts_dir / f"{row['id']}_tts.wav")
                
        # B. Synthesize Native
        native_batch = [row for row in batch if row["has_native"]]
        if native_batch:
            native_texts = [row["native_clean_text"] for row in native_batch]
            native_instructs = [voice_config[row["gender"]] for row in native_batch]
            try:
                results = model.generate(text=native_texts, instruct=native_instructs, num_step=steps)
                for idx, row in enumerate(native_batch):
                    audio = results[idx]
                    if isinstance(audio, list):
                        audio = np.concatenate([t.detach().cpu().numpy() if torch.is_tensor(t) else np.array(t) for t in audio])
                    elif torch.is_tensor(audio):
                        audio = audio.detach().cpu().numpy()
                    else:
                        audio = np.array(audio, dtype=np.float32)
                    audio = audio.flatten()
                    save_wav(audio, local_native_dir / f"{row['id']}_native.wav")
            except Exception as e:
                print(f"Error in batch Native synthesis: {e}")
                for row in native_batch:
                    save_wav(np.array([]), local_native_dir / f"{row['id']}_native.wav")
                    
        for row in batch:
            row.pop("has_native", None)
            row.pop("gender", None)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
        out_f.flush()
        synthesized_rows += len(batch)
        
        if (synthesized_rows) % 40 == 0 or i + batch_size >= len(pending_records):
            elapsed = time.time() - start_time
            rate = synthesized_rows / elapsed if elapsed > 0 else 0
            print(f"  Generated {synthesized_rows} local records. Rate: {rate:.2f} records/sec. Elapsed: {elapsed:.1f}s")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
    out_f.close()
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    
    print(f"Loading OmniVoice on {device}...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", 
        device_map=device, 
        dtype=dtype,
        load_asr=False
    )
    
    langs = [l.strip() for l in args.languages.split(",")]
    for lang in langs:
        completed = process_banking_language(lang, model, resume=True, steps=args.steps, max_records=args.max_records, batch_size=args.batch_size, sample=args.sample)
        if not completed:
            sys.exit(10)

if __name__ == "__main__":
    main()
